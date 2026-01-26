#!/usr/bin/env python3
"""
Main merge script: Merge segmented tiles with instance matching.

This script wraps the merge_tiles.py functionality to provide a clean interface
for the pipeline orchestrator.

Pipeline:
1. Load and filter (centroid-based buffer zone filtering)
2. Assign global IDs
3. Cross-tile instance matching
4. Merge and deduplicate
5. Small volume merging
6. Retile to original files (required)

Usage:
    python main_merge.py --segmented_folder /path/to/segmented_remapped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Import parameters and core merge function
from parameters import MERGE_PARAMS
from merge_tiles import merge_tiles as core_merge_tiles


def run_merge(
    segmented_dir: Path,
    output_tiles_dir: Path,
    original_tiles_dir: Path,
    original_input_dir: Optional[Path] = None,
    output_merged: Optional[Path] = None,
    buffer: float = 10.0,
    overlap_threshold: float = 0.3,
    max_centroid_distance: float = 3.0,
    correspondence_tolerance: float = 0.05,
    max_volume_for_merge: float = 4.0,
    border_zone_width: float = 10.0,
    min_cluster_size: int = 300,
    num_threads: int = 4,
    enable_matching: bool = True,
    require_overlap: bool = True,
    enable_volume_merge: bool = True,
    skip_merged_file: bool = False,
    verbose: bool = True,
    retile_buffer: float = 1.0,
    retile_max_radius: float = 0.1,
) -> Path:
    """
    Run the tile merge pipeline.
    
    Args:
        segmented_dir: Directory containing segmented LAZ tiles
        output_tiles_dir: Output directory for retiled files
        original_tiles_dir: Directory with original tile files for retiling
        original_input_dir: Directory with original input LAZ files for final remap (optional)
        output_merged: Output path for merged LAZ file (auto-derived if None)
        buffer: Buffer zone distance in meters
        overlap_threshold: Overlap ratio threshold for instance matching
        max_centroid_distance: Max distance between centroids to merge instances
        correspondence_tolerance: Max distance for point correspondence
        max_volume_for_merge: Max convex hull volume for small instance merging
        num_threads: Number of workers for parallel processing
        enable_matching: Enable cross-tile instance matching
        require_overlap: Require overlap ratio check (vs centroid distance only)
        enable_volume_merge: Enable small volume instance merging
        skip_merged_file: Skip creating merged LAZ file (only retile)
        verbose: Print detailed merge decisions
        retile_buffer: Spatial buffer expansion in meters for filtering merged points during retiling
        retile_max_radius: Maximum distance threshold in meters for cKDTree nearest neighbor matching during retiling
    
    Returns:
        Path to merged output file
    """
    print("=" * 60)
    print("3DTrees Merge Pipeline")
    print("=" * 60)
    
    # Validate input
    if not segmented_dir.exists():
        raise ValueError(f"Segmented directory not found: {segmented_dir}")
    
    if not original_tiles_dir.exists():
        raise ValueError(f"Original tiles directory not found: {original_tiles_dir}")
    
    if not output_tiles_dir.exists():
        output_tiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-derive output path if not provided
    if output_merged is None:
        output_merged = segmented_dir.parent / "merged.laz"
    
    print(f"Input: {segmented_dir}")
    print(f"Output merged: {output_merged}" + (" (SKIPPED)" if skip_merged_file else ""))
    print(f"Buffer: {buffer}m")
    print(f"Instance matching: {'ENABLED' if enable_matching else 'DISABLED'}")
    if enable_matching:
        print(f"  Overlap threshold: {overlap_threshold}")
        print(f"  Max centroid distance: {max_centroid_distance}m")
    print(f"Small cluster reassignment: ENABLED")
    print(f"  Min cluster size: {min_cluster_size} points")
    print(f"Volume merge: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    if enable_volume_merge:
        print(f"  Max volume: {max_volume_for_merge} m³")
    print(f"Workers: {num_threads}")
    if original_input_dir:
        print(f"Original input dir: {original_input_dir} (Stage 7 enabled)")
    print()
    
    # Run the core merge function
    core_merge_tiles(
        input_dir=segmented_dir,
        original_tiles_dir=original_tiles_dir,
        output_merged=output_merged,
        output_tiles_dir=output_tiles_dir,
        original_input_dir=original_input_dir,
        buffer=buffer,
        overlap_threshold=overlap_threshold,
        correspondence_tolerance=correspondence_tolerance,
        max_volume_for_merge=max_volume_for_merge,
        border_zone_width=border_zone_width,
        min_cluster_size=min_cluster_size,
        num_threads=num_threads,
        enable_matching=enable_matching,
        enable_volume_merge=enable_volume_merge,
        skip_merged_file=skip_merged_file,
        verbose=verbose,
        retile_buffer=retile_buffer,
    )
    
    return output_merged


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Merge Pipeline - Merge segmented tiles with instance matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--segmented_dir", "--segmented_folder", "-i",
        type=Path,
        required=True,
        dest="segmented_dir",
        help="Directory containing segmented LAZ tiles"
    )
    
    parser.add_argument(
        "--output_merged", "-o",
        type=Path,
        default=None,
        help="Output path for merged LAZ file (auto-derived if not specified)"
    )
    
    parser.add_argument(
        "--output_tiles_dir",
        type=Path,
        required=True,
        help="Output directory for retiled files (required)"
    )
    
    parser.add_argument(
        "--original_tiles_dir",
        type=Path,
        required=True,
        help="Directory with original tile files for retiling (required)"
    )
    
    parser.add_argument(
        "--original_input_dir",
        type=Path,
        default=None,
        help="Directory with original input LAZ files for final remap (optional, enables Stage 7)"
    )
    
    parser.add_argument(
        "--buffer",
        type=float,
        default=MERGE_PARAMS.get('buffer', 10.0),
        help=f"Buffer zone distance in meters (default: {MERGE_PARAMS.get('buffer', 10.0)})"
    )
    
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=MERGE_PARAMS.get('overlap_threshold', 0.3),
        help=f"Overlap ratio threshold (default: {MERGE_PARAMS.get('overlap_threshold', 0.3)})"
    )
    
    parser.add_argument(
        "--max_centroid_distance",
        type=float,
        default=MERGE_PARAMS.get('max_centroid_distance', 3.0),
        help=f"Max centroid distance (default: {MERGE_PARAMS.get('max_centroid_distance', 3.0)})"
    )
    
    parser.add_argument(
        "--correspondence_tolerance",
        type=float,
        default=MERGE_PARAMS.get('correspondence_tolerance', 0.05),
        help=f"Point correspondence tolerance (default: {MERGE_PARAMS.get('correspondence_tolerance', 0.05)})"
    )
    
    parser.add_argument(
        "--max_volume_for_merge",
        type=float,
        default=MERGE_PARAMS.get('max_volume_for_merge', 4.0),
        help=f"Max volume for small instance merge (default: {MERGE_PARAMS.get('max_volume_for_merge', 4.0)})"
    )
    
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=MERGE_PARAMS.get('min_cluster_size', 300),
        help=f"Minimum cluster size in points for reassignment (default: {MERGE_PARAMS.get('min_cluster_size', 300)})"
    )
    
    parser.add_argument(
        "--num_threads", "--workers",
        type=int,
        default=MERGE_PARAMS.get('workers', 4),
        dest="num_threads",
        help=f"Number of workers (default: {MERGE_PARAMS.get('workers', 4)})"
    )

    parser.add_argument(
        "--border_zone_width",
        type=float,
        default=MERGE_PARAMS.get('border_zone_width', 10.0),
        help=f"Width of border zone beyond buffer for instance matching (default: {MERGE_PARAMS.get('border_zone_width', 10.0)})"
    )

    parser.add_argument(
        "--retile_buffer",
        type=float,
        default=MERGE_PARAMS.get('retile_buffer', 1.0),
        help=f"Spatial buffer expansion in meters for retiling (default: {MERGE_PARAMS.get('retile_buffer', 1.0)})"
    )

    parser.add_argument(
        "--retile_max_radius",
        type=float,
        default=MERGE_PARAMS.get('retile_max_radius', 0.1),
        help=f"Max distance for nearest neighbor matching during retiling (default: {MERGE_PARAMS.get('retile_max_radius', 0.1)})"
    )

    parser.add_argument(
        "--disable_matching",
        action="store_true",
        help="Disable cross-tile instance matching"
    )
    
    parser.add_argument(
        "--disable_overlap_check",
        action="store_true",
        help="Disable overlap ratio check (centroid distance only)"
    )
    
    parser.add_argument(
        "--disable_volume_merge",
        action="store_true",
        help="Disable small volume instance merging"
    )
    
    parser.add_argument(
        "--skip_merged_file",
        action="store_true",
        help="Skip creating merged LAZ file (only create retiled outputs)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed merge decisions"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        output_file = run_merge(
            segmented_dir=args.segmented_dir,
            output_tiles_dir=args.output_tiles_dir,
            original_tiles_dir=args.original_tiles_dir,
            original_input_dir=args.original_input_dir,
            output_merged=args.output_merged,
            buffer=args.buffer,
            overlap_threshold=args.overlap_threshold,
            max_centroid_distance=args.max_centroid_distance,
            correspondence_tolerance=args.correspondence_tolerance,
            max_volume_for_merge=args.max_volume_for_merge,
            min_cluster_size=args.min_cluster_size,
            num_threads=args.num_threads,
            enable_matching=not args.disable_matching,
            require_overlap=not args.disable_overlap_check,
            enable_volume_merge=not args.disable_volume_merge,
            skip_merged_file=args.skip_merged_file,
            verbose=args.verbose,
            border_zone_width=args.border_zone_width,
            retile_buffer=args.retile_buffer,
            retile_max_radius=args.retile_max_radius,
        )
        if not args.skip_merged_file:
            print(f"\nMerged output: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
