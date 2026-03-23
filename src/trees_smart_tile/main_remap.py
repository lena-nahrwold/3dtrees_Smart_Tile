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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import laspy
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree

# For JSON-based file matching (same grid as merge)
from merge_tiles import (
    build_neighbor_graph_from_bounds_json,
    _match_tiles_to_json_bounds,
    extra_bytes_params_from_dimension_info,
)  


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
    output_file: Path,
    threedtrees_dims: Optional[Set[str]] = None,
    threedtrees_suffix: str = "SAT",
) -> Tuple[str, bool, str, int]:
    """
    Remap predictions from segmented file to target resolution file.

    Uses KDTree nearest neighbor search to transfer attributes from
    the segmented (coarse) file to the target (fine) file.

    Args:
        segmented_file: Path to segmented LAZ file (e.g., 10cm with predictions)
        target_file: Path to target resolution LAZ file (e.g., 2cm)
        output_file: Path for output LAZ file
        threedtrees_dims: If set, only transfer these dims, branded as 3DT_{name}_{suffix}
        threedtrees_suffix: Suffix for branding (default: "SAT")

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
        
        # Create KDTree from segmented points with progress indication
        print(f"    Building KDTree from {len(segmented_points):,} points...", end="", flush=True)
        tree = cKDTree(segmented_points)
        print(" ✓")
        
        # Query nearest neighbors
        print(f"    Querying nearest neighbors for {len(target_points):,} points...", end="", flush=True)
        distances, indices = tree.query(target_points, workers=-1)
        print(" ✓")
        
        source_extra_dims = list(segmented_las.point_format.extra_dimensions)
        target_extra_dim_names = set(target_las.point_format.dimension_names)

        if len(source_extra_dims) == 0:
            print(f"    Warning: No extra dimensions found in segmented file")

        # Resolve names and collect params for batch add
        dims_to_add = []  # (extra_params, source_dim_name)
        for dim_info in source_extra_dims:
            dim_name = dim_info.name
            # If branding is active, only transfer 3DTrees dims with branded names
            if threedtrees_dims is not None:
                if dim_name not in threedtrees_dims:
                    continue
                out_name = f"3DT_{dim_name}_{threedtrees_suffix}" if threedtrees_suffix else f"3DT_{dim_name}"
            else:
                # No branding — use collision-safe naming
                out_name = dim_name
                if out_name in target_extra_dim_names:
                    suffix = 1
                    while f"{dim_name}_{suffix}" in target_extra_dim_names:
                        suffix += 1
                    out_name = f"{dim_name}_{suffix}"
            dims_to_add.append((extra_bytes_params_from_dimension_info(dim_info, name=out_name), dim_name))
            target_extra_dim_names.add(out_name)
        
        if dims_to_add:
            target_las.add_extra_dims([params for params, _ in dims_to_add])
            for params, src_name in dims_to_add:
                setattr(target_las, params.name, getattr(segmented_las, src_name)[indices])
        
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


def _match_files_via_json(
    tile_bounds_json: Path,
    source_folder: Path,
    target_folder: Path,
    verbose: bool = False,
) -> List[Tuple[Path, Path, str]]:
    """
    Match source and target files using tile_bounds_tindex.json.
    Both source and target files are matched to JSON entries (stepwise bounds/centroid);
    pairs are formed by shared JSON index. Uses the same stable matching as merge.
    """
    source_files = sorted(source_folder.glob("*.laz")) or sorted(source_folder.glob("*.las"))
    target_files = sorted(target_folder.glob("*.laz")) or sorted(target_folder.glob("*.las"))
    if not source_files or not target_files:
        return []

    # Single-file shortcut: 1 source and 1 target -> pair directly
    if len(source_files) == 1 and len(target_files) == 1:
        src = source_files[0]
        tgt = target_files[0]
        tile_id = re.sub(r"_segmented$|_results$|_subsampled[\d.]+m$", "", src.stem)
        if not tile_id:
            tile_id = src.stem
        print(f"  Single file pair: {src.name} <-> {tgt.name}")
        return [(src, tgt, tile_id)]

    json_bounds, centers, _ = build_neighbor_graph_from_bounds_json(tile_bounds_json)

    source_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    stem_to_source_path: Dict[str, Path] = {}
    for f in source_files:
        b = get_file_bounds(f)
        if b is not None:
            stem = f.stem
            source_boundaries[stem] = b
            stem_to_source_path[stem] = f

    target_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    stem_to_target_path: Dict[str, Path] = {}
    for f in target_files:
        b = get_file_bounds(f)
        if b is not None:
            stem = f.stem
            target_boundaries[stem] = b
            stem_to_target_path[stem] = f

    if not source_boundaries:
        raise ValueError("Could not read bounds from any source file")
    if not target_boundaries:
        raise ValueError("Could not read bounds from any target file")

    source_to_json, json_to_source = _match_tiles_to_json_bounds(
        source_boundaries, json_bounds, centers
    )
    target_to_json, json_to_target = _match_tiles_to_json_bounds(
        target_boundaries, json_bounds, centers
    )

    matches: List[Tuple[Path, Path, str]] = []
    for j in range(len(json_bounds)):
        src_stem = json_to_source.get(j)
        tgt_stem = json_to_target.get(j)
        if src_stem is None or tgt_stem is None:
            if src_stem is not None:
                raise ValueError(
                    f"Source file {stem_to_source_path[src_stem].name} matched JSON tile index {j} "
                    "but no target file matched that tile. Cannot remap."
                )
            continue
        src_path = stem_to_source_path[src_stem]
        tgt_path = stem_to_target_path[tgt_stem]
        tile_id = re.sub(r"_segmented$|_results$|_subsampled[\d.]+m$", "", src_stem)
        if not tile_id:
            tile_id = src_stem
        matches.append((src_path, tgt_path, tile_id))
        if verbose:
            print(f"  ✓ Matched (JSON): {src_path.name} <-> {tgt_path.name} ({tile_id})")

    return matches


def find_matching_files(
    source_folder: Path,
    target_folder: Path,
    overlap_threshold: float = 99.0,
    verbose: bool = False,
    tile_bounds_json: Optional[Path] = None,
) -> List[Tuple[Path, Path, str]]:
    """
    Find matching files between source and target folders.
    If tile_bounds_json is provided and exists, uses JSON-based matching (stepwise
    bounds/centroid, same as merge). Otherwise uses two-stage matching:
    1. Strict tolerance matching (1m)
    2. IoU matching (30%) fallback

    Args:
        source_folder: Directory containing source LAZ files (e.g., segmented files)
        target_folder: Directory containing target LAZ files (e.g., 2cm subsampled files)
        overlap_threshold: DEPRECATED - not used (kept for compatibility)
        verbose: If True, print detailed matching diagnostics
        tile_bounds_json: Optional path to tile_bounds_tindex.json for grid-based matching

    Returns:
        List of (source_file, target_file, tile_id) tuples
    """
    if tile_bounds_json is not None and tile_bounds_json.exists():
        print(f"  Using tile_bounds_tindex.json for matching: {tile_bounds_json}")
        return _match_files_via_json(tile_bounds_json, source_folder, target_folder, verbose)

    matches = []

    # Get all LAZ/LAS files from both folders (flat structure)
    source_files = sorted(source_folder.glob("*.laz")) or sorted(source_folder.glob("*.las"))
    target_files = sorted(target_folder.glob("*.laz")) or sorted(target_folder.glob("*.las"))

    if not source_files:
        print(f"  Warning: No LAZ/LAS files found in source folder: {source_folder}")
        return matches

    if not target_files:
        print(f"  Warning: No LAZ/LAS files found in target folder: {target_folder}")
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


def _remap_worker_item(item):
    """Unpack work item and call remap_single_tile; must be at module level for ProcessPoolExecutor pickle."""
    src, tgt, out, _tid = item
    return remap_single_tile(src, tgt, out)


def remap_all_tiles(
    source_folder: Path,
    target_folder: Path,
    output_folder: Path,
    overlap_threshold: float = 99.0,
    verbose: bool = False,
    tile_bounds_json: Optional[Path] = None,
    num_workers: int = 4,
) -> Path:
    """
    Remap predictions from source files to target files for all tiles.

    Matches files between source and target folders by spatial overlap.
    If tile_bounds_json is provided, uses JSON-based matching (same grid as merge).

    Args:
        source_folder: Path to folder containing source LAZ files (e.g., segmented files)
        target_folder: Path to folder containing target LAZ files (e.g., 2cm subsampled files)
        output_folder: Output folder for remapped files
        overlap_threshold: Minimum spatial overlap percentage required (default: 99.0%)
        verbose: If True, print detailed matching diagnostics
        tile_bounds_json: Optional path to tile_bounds_tindex.json for grid-based matching
        num_workers: Number of parallel processes (default: 4); use parameters.workers in run.py

    Returns:
        Path to output folder
    """
    print("=" * 60)
    print("3DTrees Remap Pipeline")
    print("=" * 60)
    print(f"Source folder: {source_folder}")
    print(f"Target folder: {target_folder}")
    print(f"Output folder: {output_folder}")
    if tile_bounds_json and tile_bounds_json.exists():
        print(f"Matching: tile_bounds_tindex.json ({tile_bounds_json})")
    else:
        print("Matching: Two-stage (1m tolerance → 30% IoU)")
    print()

    # Validate directories exist
    if not source_folder.exists():
        raise ValueError(f"Source directory not found: {source_folder}")

    if not target_folder.exists():
        raise ValueError(f"Target directory not found: {target_folder}")

    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find matching files by spatial overlap (or JSON when provided)
    print("Matching files by spatial bounds...")
    matches = find_matching_files(
        source_folder, target_folder, overlap_threshold, verbose, tile_bounds_json
    )

    if not matches:
        n_src = len(list(source_folder.glob("*.laz")) + list(source_folder.glob("*.las")))
        n_tgt = len(list(target_folder.glob("*.laz")) + list(target_folder.glob("*.las")))
        msg = (
            "No matching source/target file pairs found. "
            f"Source folder has {n_src} LAZ/LAS file(s), target folder has {n_tgt}. "
            "With tile_bounds_json, both folders must contain one file per tile (same grid); "
            "file bounds are matched to the JSON tile bounds."
        )
        raise ValueError(msg)
    
    print(f"Found {len(matches)} matching file pairs")
    print()
    
    # Build work items, skipping already-processed tiles
    successful = 0
    failed = 0
    total_points = 0
    work_items = []

    for source_file, target_file, tile_id in matches:
        output_file = output_folder / f"{tile_id}_segmented_remapped.laz"
        if output_file.exists() and output_file.stat().st_size > 0:
            successful += 1
            continue
        work_items.append((source_file, target_file, output_file, tile_id))

    if successful > 0:
        print(f"  Skipping {successful} already processed tiles")

    if len(work_items) == 0:
        print(f"  All tiles already processed!")
    else:
        n_procs = min(max(1, num_workers), len(work_items))
        print(f"  Processing {len(work_items)} tiles with {n_procs} workers...")

        with ProcessPoolExecutor(max_workers=n_procs) as executor:
            for i, result in enumerate(executor.map(_remap_worker_item, work_items)):
                tile_id_result, success, message, point_count = result
                tile_id = work_items[i][3]
                if success:
                    successful += 1
                    total_points += point_count
                    print(f"  [{i+1}/{len(work_items)}] ✓ {tile_id}: {point_count:,} points")
                else:
                    failed += 1
                    print(f"  [{i+1}/{len(work_items)}] ✗ {tile_id}: {message}")
    
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
        help="Maximum difference in meters for bounds matching when not using --tile_bounds_json (default: 5.0)"
    )
    parser.add_argument(
        "--tile_bounds_json",
        type=Path,
        default=None,
        help="Path to tile_bounds_tindex.json for grid-based matching (same as merge); optional"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed matching diagnostics"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)"
    )

    args = parser.parse_args()

    # Run pipeline
    try:
        output_folder = remap_all_tiles(
            source_folder=args.source_folder,
            target_folder=args.target_folder,
            output_folder=args.output_folder,
            overlap_threshold=99.0,
            verbose=args.verbose,
            tile_bounds_json=args.tile_bounds_json,
            num_workers=args.workers,
        )
        print(f"\nRemapped files ready: {output_folder}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
