#!/usr/bin/env python3
"""
Main remap script: Remap predictions by matching spatial bounds.

This script handles remapping of segmented predictions from source files to target files
by matching files based on their spatial boundaries, then using KDTree nearest neighbor
lookup to transfer attributes.

Usage:
    python main_remap.py --source_folder /path/to/segmented --target_folder /path/to/2cm --output_folder /path/to/output
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import laspy
from scipy.spatial import cKDTree  


def get_file_bounds(filepath: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get spatial bounds of a point cloud file using laspy header only (no point loading).
    
    Args:
        filepath: Path to LAZ file
    
    Returns:
        Tuple of (minx, maxx, miny, maxy) or None on error
    """
    try:
        # Use laspy.open() to read only header, not all points
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as las:
            return (las.header.x_min, las.header.x_max, las.header.y_min, las.header.y_max)
    except Exception:
        return None


def calculate_bounds_overlap(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    IoU is more robust than "overlap % of smaller file" because:
    - It's symmetric
    - Penalizes size mismatches
    - Prevents nested files from all getting 100%

    Args:
        bounds1: Tuple of (minx, maxx, miny, maxy) for first file
        bounds2: Tuple of (minx, maxx, miny, maxy) for second file

    Returns:
        IoU as percentage (0-100), where 100% = perfect overlap
    """
    if bounds1 is None or bounds2 is None:
        return 0.0

    minx1, maxx1, miny1, maxy1 = bounds1
    minx2, maxx2, miny2, maxy2 = bounds2

    # Calculate intersection region
    overlap_minx = max(minx1, minx2)
    overlap_maxx = min(maxx1, maxx2)
    overlap_miny = max(miny1, miny2)
    overlap_maxy = min(maxy1, maxy2)

    # Check if there's actual overlap
    if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
        return 0.0

    # Calculate intersection area
    intersection = (overlap_maxx - overlap_minx) * (overlap_maxy - overlap_miny)

    # Calculate union area
    area1 = (maxx1 - minx1) * (maxy1 - miny1)
    area2 = (maxx2 - minx2) * (maxy2 - miny2)
    union = area1 + area2 - intersection

    # Calculate IoU as percentage
    iou = (intersection / union) * 100 if union > 0 else 0.0

    return iou


def bounds_match_exact(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float],
    precision: float = 0.01
) -> bool:
    """
    Check if two bounds are identical (within floating point precision).

    Args:
        bounds1: Tuple of (minx, maxx, miny, maxy) for first file
        bounds2: Tuple of (minx, maxx, miny, maxy) for second file
        precision: Maximum difference for exact match (default: 0.01m = 1cm)

    Returns:
        True if bounds are essentially identical
    """
    if bounds1 is None or bounds2 is None:
        return False

    minx1, maxx1, miny1, maxy1 = bounds1
    minx2, maxx2, miny2, maxy2 = bounds2

    return (abs(minx1 - minx2) <= precision and
            abs(maxx1 - maxx2) <= precision and
            abs(miny1 - miny2) <= precision and
            abs(maxy1 - maxy2) <= precision)


def bounds_match_tolerance(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float],
    tolerance: float = 1.0
) -> bool:
    """
    Check if two bounds match within a strict tolerance.

    Args:
        bounds1: Tuple of (minx, maxx, miny, maxy) for first file
        bounds2: Tuple of (minx, maxx, miny, maxy) for second file
        tolerance: Maximum difference in meters for each bound component (default: 1.0m)

    Returns:
        True if all bound components match within tolerance
    """
    if bounds1 is None or bounds2 is None:
        return False

    minx1, maxx1, miny1, maxy1 = bounds1
    minx2, maxx2, miny2, maxy2 = bounds2

    return (abs(minx1 - minx2) <= tolerance and
            abs(maxx1 - maxx2) <= tolerance and
            abs(miny1 - miny2) <= tolerance and
            abs(maxy1 - maxy2) <= tolerance)


def bounds_match(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float]
) -> bool:
    """
    Check if two bounds match using two-stage approach:
    1. First try strict tolerance matching (1m) for exact matches
    2. If that fails, fallback to IoU matching (30%) for robust matching

    IoU threshold of 30% works for:
    - Dataset 257 (near-identical bounds): ~98% IoU ✓
    - GFZ correct matches (partial overlap): ~33% IoU ✓
    - GFZ wrong matches (minimal overlap): ~5% IoU ✗

    Args:
        bounds1: Tuple of (minx, maxx, miny, maxy) for first file
        bounds2: Tuple of (minx, maxx, miny, maxy) for second file

    Returns:
        True if bounds match by either method
    """
    if bounds1 is None or bounds2 is None:
        return False

    # Stage 1: Try strict tolerance matching (1m)
    if bounds_match_tolerance(bounds1, bounds2, tolerance=1.0):
        return True

    # Stage 2: Fallback to IoU matching (30%)
    iou = calculate_bounds_overlap(bounds1, bounds2)
    return iou >= 30.0


def remap_single_tile(
    segmented_file: Path,
    target_file: Path,
    output_file: Path
) -> Tuple[str, bool, str, int]:
    """
    Remap predictions from segmented file to target resolution file.
    
    Uses KDTree nearest neighbor search to transfer attributes from
    the segmented (coarse) file to the target (fine) file.
    
    Args:
        segmented_file: Path to segmented LAZ file (e.g., 10cm with predictions)
        target_file: Path to target resolution LAZ file (e.g., 2cm)
        output_file: Path for output LAZ file
    
    Returns:
        Tuple of (tile_id, success, message, point_count)
    """
    tile_id = segmented_file.stem.replace('_segmented', '').replace('_results', '')
    
    try:
        # Load segmented point cloud (source of predictions)
        print(f"    Loading segmented file...")
        segmented_las = laspy.read(
            str(segmented_file), 
            laz_backend=laspy.LazBackend.LazrsParallel
        )
        segmented_points = np.vstack((
            segmented_las.x, 
            segmented_las.y, 
            segmented_las.z
        )).T
        print(f"    Segmented file: {len(segmented_points):,} points")
        
        # Load target resolution point cloud
        print(f"    Loading target file...")
        target_las = laspy.read(
            str(target_file), 
            laz_backend=laspy.LazBackend.LazrsParallel
        )
        target_points = np.vstack((
            target_las.x, 
            target_las.y, 
            target_las.z
        )).T
        print(f"    Target file: {len(target_points):,} points")
        
        # Check for required attributes in segmented file
        extra_dims = {dim.name for dim in segmented_las.point_format.extra_dimensions}
        has_pred_instance = 'PredInstance' in extra_dims
        has_pred_semantic = 'PredSemantic' in extra_dims
        has_species_id = 'species_id' in extra_dims
        
        if not has_pred_instance:
            return (tile_id, False, "No PredInstance attribute in segmented file", 0)
        
        # Create KDTree from segmented points with progress indication
        print(f"    Building KDTree from {len(segmented_points):,} points...", end="", flush=True)
        tree = cKDTree(segmented_points)
        print(" ✓")
        
        # Query nearest neighbors
        print(f"    Querying nearest neighbors for {len(target_points):,} points...", end="", flush=True)
        distances, indices = tree.query(target_points, workers=-1)
        print(" ✓")
        
        # Create output with target resolution points
        # Add extra dimensions if they don't exist
        target_extra_dims = {dim.name for dim in target_las.point_format.extra_dimensions}
        
        if "PredInstance" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
            )
        
        if has_pred_semantic and "PredSemantic" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredSemantic", type=np.int32)
            )
        
        if has_species_id and "species_id" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="species_id", type=np.int32)
            )
        
        # Transfer attributes using nearest neighbor indices
        target_las.PredInstance = segmented_las.PredInstance[indices]
        
        if has_pred_semantic:
            target_las.PredSemantic = segmented_las.PredSemantic[indices]
        
        if has_species_id:
            target_las.species_id = segmented_las.species_id[indices]
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save output
        with open(str(output_file), "wb") as f:
            target_las.write(
                f, 
                do_compress=True, 
                laz_backend=laspy.LazBackend.LazrsParallel
            )
            f.flush()
            os.fsync(f.fileno())
        
        return (tile_id, True, "Success", len(target_points))
        
    except Exception as e:
        return (tile_id, False, str(e), 0)


def find_matching_files(
    source_folder: Path,
    target_folder: Path,
    overlap_threshold: float = 99.0,
    verbose: bool = False
) -> List[Tuple[Path, Path, str]]:
    """
    Find matching files between source and target folders using two-stage matching:
    1. First tries strict tolerance matching (1m) for exact matches
    2. Falls back to IoU matching (30%) for robust matching

    Args:
        source_folder: Directory containing source LAZ files (e.g., segmented files)
        target_folder: Directory containing target LAZ files (e.g., 2cm subsampled files)
        overlap_threshold: DEPRECATED - not used (kept for compatibility)
        verbose: If True, print detailed matching diagnostics

    Returns:
        List of (source_file, target_file, tile_id) tuples
    """
    matches = []

    # Get all LAZ files from both folders (flat structure)
    source_files = sorted(source_folder.glob("*.laz"))
    target_files = sorted(target_folder.glob("*.laz"))

    if not source_files:
        print(f"  Warning: No LAZ files found in source folder: {source_folder}")
        return matches

    if not target_files:
        print(f"  Warning: No LAZ files found in target folder: {target_folder}")
        return matches

    print(f"  Found {len(source_files)} source files and {len(target_files)} target files")
    print(f"  Two-stage matching: 1m tolerance → 30% IoU fallback")

    # Extract bounds for all target files once
    target_bounds_map = {}
    for target_file in target_files:
        bounds = get_file_bounds(target_file)
        if bounds:
            target_bounds_map[target_file] = bounds

    # Match each source file to target files using two-stage approach
    unmatched_count = 0
    tolerance_matches = 0
    iou_matches = 0

    for source_file in source_files:
        source_bounds = get_file_bounds(source_file)
        if source_bounds is None:
            print(f"  Warning: Could not extract bounds from {source_file.name}")
            continue

        # Find matching target file(s) and track overlap for each
        matched_targets = []

        for target_file, target_bounds in target_bounds_map.items():
            # Check both matching methods
            tolerance_match = bounds_match_tolerance(source_bounds, target_bounds, tolerance=1.0)
            iou = calculate_bounds_overlap(source_bounds, target_bounds)
            iou_match = iou >= 30.0

            # Determine which method succeeded (priority: tolerance > iou)
            if tolerance_match:
                match_method = 'tolerance'
            elif iou_match:
                match_method = 'iou'
            else:
                continue  # No match

            matched_targets.append((target_file, match_method, iou))

        if len(matched_targets) == 0:
            unmatched_count += 1
            print(f"  ⚠ Warning: No matching target file found for {source_file.name}")

            # Provide helpful diagnostics for best candidate
            if verbose:
                best_iou = 0.0
                best_target = None
                for target_file, target_bounds in target_bounds_map.items():
                    iou = calculate_bounds_overlap(source_bounds, target_bounds)
                    if iou > best_iou:
                        best_iou = iou
                        best_target = target_file
                if best_target:
                    print(f"    Best candidate: {best_target.name} (IoU: {best_iou:.2f}%)")

            continue
        # Sort by match quality (method priority, then IoU)
        def match_score(match_tuple):
            target_file, match_method, iou = match_tuple
            method_priority = {'tolerance': 2, 'iou': 1}
            return (method_priority[match_method], iou)

        matched_targets.sort(key=match_score, reverse=True)

        # Check for ambiguous matches (same method and IoU)
        if len(matched_targets) > 1:
            best_method = matched_targets[0][1]
            best_iou = matched_targets[0][2]

            # Count how many have the same score as the best
            ambiguous_matches = [
                m for m in matched_targets
                if m[1] == best_method and abs(m[2] - best_iou) < 0.01
            ]

            if len(ambiguous_matches) > 1:
                # Ambiguous match - cannot determine correct target
                ambiguous_names = [m[0].name for m in ambiguous_matches]
                raise ValueError(
                    f"Ambiguous match for {source_file.name}: "
                    f"Multiple targets with identical bounds and IoU ({best_iou:.2f}%): "
                    f"{', '.join(ambiguous_names)}. "
                    f"Cannot determine correct target file."
                )

        target_file, match_method, best_iou = matched_targets[0]

        # Update counters
        if match_method == 'tolerance':
            tolerance_matches += 1
        else:
            iou_matches += 1

        # Extract tile_id from filename if possible, otherwise use stem
        tile_id_match = re.search(r'(c\d+_r\d+)', source_file.stem)
        if tile_id_match:
            tile_id = tile_id_match.group(1)
        else:
            # Fallback: use filename stem without extension
            tile_id = source_file.stem.replace('_segmented', '').replace('_results', '')

        matches.append((source_file, target_file, tile_id))

        if verbose:
            if match_method == 'tolerance':
                method_str = "1m tolerance"
            else:
                method_str = f"IoU {best_iou:.1f}%"
            print(f"  ✓ Matched: {source_file.name} <-> {target_file.name} ({method_str})")
        else:
            print(f"  Matched: {source_file.name} <-> {target_file.name}")

    # Summary
    print()
    if tolerance_matches > 0:
        print(f"  ✓ {tolerance_matches} file(s) matched by tolerance (1m)")
    if iou_matches > 0:
        print(f"  ✓ {iou_matches} file(s) matched by IoU fallback (30%)")
    if unmatched_count > 0:
        print(f"  ⚠ {unmatched_count} file(s) could not be matched")

    return matches


def remap_all_tiles(
    source_folder: Path,
    target_folder: Path,
    output_folder: Path,
    overlap_threshold: float = 99.0,
    verbose: bool = False
) -> Path:
    """
    Remap predictions from source files to target files for all tiles.

    Matches files between source and target folders by spatial overlap.

    Args:
        source_folder: Path to folder containing source LAZ files (e.g., segmented files)
        target_folder: Path to folder containing target LAZ files (e.g., 2cm subsampled files)
        output_folder: Output folder for remapped files
        overlap_threshold: Minimum spatial overlap percentage required (default: 99.0%)
        verbose: If True, print detailed matching diagnostics

    Returns:
        Path to output folder
    """
    print("=" * 60)
    print("3DTrees Remap Pipeline")
    print("=" * 60)
    print(f"Source folder: {source_folder}")
    print(f"Target folder: {target_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Matching: Two-stage (1m tolerance → 30% IoU)")
    print()

    # Validate directories exist
    if not source_folder.exists():
        raise ValueError(f"Source directory not found: {source_folder}")

    if not target_folder.exists():
        raise ValueError(f"Target directory not found: {target_folder}")

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find matching files by spatial overlap
    print("Matching files by spatial bounds...")
    matches = find_matching_files(
        source_folder, target_folder, overlap_threshold, verbose
    )

    if not matches:
        raise ValueError(f"No matching source/target file pairs found")
    
    print(f"Found {len(matches)} matching file pairs")
    print()
    
    # Process each tile
    successful = 0
    failed = 0
    total_points = 0
    
    for i, (source_file, target_file, tile_id) in enumerate(matches, 1):
        # Get point count from target file header for display
        target_bounds = get_file_bounds(target_file)
        target_header = None
        try:
            with laspy.open(str(target_file), laz_backend=laspy.LazBackend.LazrsParallel) as las:
                target_header = las.header
        except Exception:
            pass
        
        point_count_display = f"{target_header.point_count:,} points" if target_header else "points"
        print(f"[{i}/{len(matches)}] Processing tile {tile_id} ({point_count_display})...")
        
        output_file = output_folder / f"{tile_id}_segmented_remapped.laz"
        
        # Skip if already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"  Skipping (already exists)")
            successful += 1
            continue
        
        tile_id_result, success, message, point_count = remap_single_tile(
            source_file,
            target_file,
            output_file
        )
        
        if success:
            successful += 1
            total_points += point_count
            print(f"  ✓ Completed: {point_count:,} points")
        else:
            failed += 1
            print(f"  ✗ {message}")
    
    # Summary
    print()
    print("=" * 60)
    print("Remap Pipeline Complete")
    print("=" * 60)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total points: {total_points:,}")
    print(f"  Output: {output_folder}")
    
    return output_folder


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Remap Pipeline - Remap predictions by matching spatial bounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remap files matching by bounds
  python main_remap.py --source_folder /path/to/segmented --target_folder /path/to/2cm --output_folder /path/to/output
  
  # Remap with custom tolerance
  python main_remap.py --source_folder /path/to/segmented --target_folder /path/to/2cm --output_folder /path/to/output --tolerance 10.0
        """
    )
    
    parser.add_argument(
        "--source_folder",
        type=Path,
        required=True,
        help="Path to folder containing source LAZ files (e.g., segmented files with predictions)"
    )
    
    parser.add_argument(
        "--target_folder",
        type=Path,
        required=True,
        help="Path to folder containing target LAZ files (e.g., 2cm subsampled files)"
    )
    
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Output folder for remapped files"
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Maximum difference in meters for bounds matching (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        output_folder = remap_all_tiles(
            source_folder=args.source_folder,
            target_folder=args.target_folder,
            output_folder=args.output_folder,
            tolerance=args.tolerance
        )
        print(f"\nRemapped files ready: {output_folder}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

