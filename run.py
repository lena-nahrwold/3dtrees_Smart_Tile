#!/usr/bin/env python3
"""
Main orchestrator script for the 3DTrees smart tiling pipeline.

Routes to appropriate task scripts based on --task parameter:
- tile: Tiling with overlap and subsampling (2cm and 10cm)
- remap: Remap predictions from 10cm to target resolution (default 2cm)
- merge: Merge tiles, remove buffer instances, reassign small instances

Usage:
    python run.py --task tile --input_dir /path/to/input --output_dir /path/to/output
    python run.py --task remap --subsampled_10cm_folder /path/to/subsampled_10cm
    python run.py --task merge --segmented_remapped_folder /path/to/segmented_remapped
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import default parameters
try:
    from parameters import TILE_PARAMS, REMAP_PARAMS, MERGE_PARAMS
except ImportError:
    print("Error: Could not import parameters.py. Make sure it exists in the same directory.")
    sys.exit(1)


def run_tile_task(args, params):
    """Run the tile task: retiling with overlap and subsampling."""
    # Required arguments
    if not args.input_dir:
        print("Error: --input_dir is required for tile task")
        sys.exit(1)
    if not args.output_dir:
        print("Error: --output_dir is required for tile task")
        sys.exit(1)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Get parameters (command-line overrides defaults)
    tile_length = args.tile_length if args.tile_length else params['tile_length']
    tile_buffer = args.tile_buffer if args.tile_buffer else params['tile_buffer']
    threads = args.threads if args.threads else params['threads']
    num_threads = args.num_threads if args.num_threads else params['num_threads']
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    main_tile_script = script_dir / "main_tile.sh"
    
    if not main_tile_script.exists():
        print(f"Error: main_tile.sh not found at {main_tile_script}")
        sys.exit(1)
    
    # Build command
    cmd = [
        "bash",
        str(main_tile_script),
        args.input_dir,
        args.output_dir,
        str(tile_length),
        str(tile_buffer),
        str(threads),
        str(num_threads)
    ]
    
    print("=" * 60)
    print("Running Tile Task")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tile length: {tile_length}m")
    print(f"Tile buffer: {tile_buffer}m")
    print(f"Threads: {threads} (retiling), {num_threads} (subsampling)")
    print("")
    
    # Run the script
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def run_remap_task(args, params):
    """Run the remap task: remap predictions from 10cm to target resolution."""
    # Required arguments
    if not args.subsampled_10cm_folder:
        print("Error: --subsampled_10cm_folder is required for remap task")
        sys.exit(1)
    
    # Validate input directory
    if not os.path.isdir(args.subsampled_10cm_folder):
        print(f"Error: Input directory does not exist: {args.subsampled_10cm_folder}")
        sys.exit(1)
    
    # Get parameters
    target_resolution = args.target_resolution if args.target_resolution else params['target_resolution_cm']
    num_threads = args.num_threads if args.num_threads else params['num_threads']
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    main_remap_script = script_dir / "main_remap.sh"
    
    if not main_remap_script.exists():
        print(f"Error: main_remap.sh not found at {main_remap_script}")
        sys.exit(1)
    
    # Build command
    cmd = [
        "bash",
        str(main_remap_script),
        args.subsampled_10cm_folder,
        "--target_resolution",
        str(target_resolution)
    ]
    
    # Add optional subsampled target folder
    if args.subsampled_target_folder:
        cmd.extend(["--subsampled_target_folder", args.subsampled_target_folder])
    
    # Add optional output folder
    if args.output_folder:
        cmd.extend(["--output_folder", args.output_folder])
    
    print("=" * 60)
    print("Running Remap Task")
    print("=" * 60)
    print(f"Input folder (10cm): {args.subsampled_10cm_folder}")
    print(f"Target resolution: {target_resolution}cm")
    if args.subsampled_target_folder:
        print(f"Target folder: {args.subsampled_target_folder}")
    if args.output_folder:
        print(f"Output folder: {args.output_folder}")
    print("")
    
    # Run the script
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def run_merge_task(args, params):
    """Run the merge task: merge tiles with instance matching and species preservation."""
    # Required arguments
    if not args.segmented_remapped_folder:
        print("Error: --segmented_remapped_folder is required for merge task")
        sys.exit(1)
    
    # Validate input directory
    if not os.path.isdir(args.segmented_remapped_folder):
        print(f"Error: Input directory does not exist: {args.segmented_remapped_folder}")
        sys.exit(1)
    
    # Get parameters
    buffer = args.buffer if args.buffer is not None else params['buffer']
    min_cluster_size = args.min_cluster_size if args.min_cluster_size else params['min_cluster_size']
    overlap_threshold = args.overlap_threshold if args.overlap_threshold is not None else params.get('overlap_threshold', 0.1)
    num_threads = args.num_threads if args.num_threads else params['num_threads']
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    main_merge_script = script_dir / "main_merge.sh"
    
    if not main_merge_script.exists():
        print(f"Error: main_merge.sh not found at {main_merge_script}")
        sys.exit(1)
    
    # Build command
    cmd = [
        "bash",
        str(main_merge_script),
        args.segmented_remapped_folder,
        "--buffer",
        str(buffer),
        "--min-cluster-size",
        str(min_cluster_size),
        "--overlap-threshold",
        str(overlap_threshold),
        "--num-threads",
        str(num_threads)
    ]
    
    # Add optional paths
    if args.output_merged_laz:
        cmd.extend(["--output-merged", args.output_merged_laz])
    if args.output_tiles_folder:
        cmd.extend(["--output-tiles-dir", args.output_tiles_folder])
    if args.original_tiles_dir:
        cmd.extend(["--original-tiles-dir", args.original_tiles_dir])
    if args.disable_matching:
        cmd.append("--disable-matching")
    
    print("=" * 60)
    print("Running Merge Task")
    print("=" * 60)
    print(f"Input folder: {args.segmented_remapped_folder}")
    print(f"Buffer: {buffer}m")
    print(f"Min cluster size: {min_cluster_size} points")
    print(f"Overlap threshold: {overlap_threshold}")
    print(f"Instance matching: {'DISABLED' if args.disable_matching else 'ENABLED'}")
    print(f"Threads: {num_threads}")
    if args.output_merged_laz:
        print(f"Output merged LAZ: {args.output_merged_laz}")
    if args.output_tiles_folder:
        print(f"Output tiles folder: {args.output_tiles_folder}")
    if args.original_tiles_dir:
        print(f"Original tiles dir: {args.original_tiles_dir}")
    print("")
    
    # Run the script
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="3DTrees Smart Tiling Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tile task
  python run.py --task tile --input_dir /path/to/input --output_dir /path/to/output
  
  # Remap task
  python run.py --task remap --subsampled_10cm_folder /path/to/subsampled_10cm --target_resolution 2
  
  # Merge task
  python run.py --task merge --segmented_remapped_folder /path/to/segmented_remapped
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["tile", "remap", "merge"],
        help="Task to run: tile, remap, or merge"
    )
    
    # Tile task arguments
    parser.add_argument("--input_dir", type=str, help="Input directory with LAZ/COPC files (required for tile)")
    parser.add_argument("--output_dir", type=str, help="Output directory (required for tile)")
    parser.add_argument("--tile_length", type=float, help=f"Tile size in meters (default: {TILE_PARAMS['tile_length']})")
    parser.add_argument("--tile_buffer", type=float, help=f"Buffer size in meters (default: {TILE_PARAMS['tile_buffer']})")
    parser.add_argument("--threads", type=int, help=f"Threads per COPC writer (default: {TILE_PARAMS['threads']})")
    parser.add_argument("--num_threads", type=int, help=f"Number of parallel threads (default: varies by task)")
    
    # Remap task arguments
    parser.add_argument("--subsampled_10cm_folder", type=str, help="Path to subsampled_10cm folder (required for remap)")
    parser.add_argument("--target_resolution", type=int, help=f"Target resolution in cm (default: {REMAP_PARAMS['target_resolution_cm']})")
    parser.add_argument("--subsampled_target_folder", type=str, help="Path to target resolution subsampled folder (optional, auto-derived if not specified)")
    parser.add_argument("--output_folder", type=str, help="Output folder for remapped files (optional, auto-derived if not specified)")
    
    # Merge task arguments
    parser.add_argument("--segmented_remapped_folder", type=str, help="Path to segmented_remapped folder (required for merge)")
    parser.add_argument("--output_merged_laz", type=str, help="Output path for merged LAZ file (optional)")
    parser.add_argument("--output_tiles_folder", type=str, help="Output folder for per-tile results (optional)")
    parser.add_argument("--original_tiles_dir", type=str, help="Directory with original tile files for retiling (optional)")
    parser.add_argument("--buffer", type=float, help=f"Buffer distance for filtering in meters (default: {MERGE_PARAMS['buffer']})")
    parser.add_argument("--min_cluster_size", type=int, help=f"Minimum cluster size in points (default: {MERGE_PARAMS['min_cluster_size']})")
    parser.add_argument("--overlap_threshold", type=float, help=f"Overlap ratio threshold for instance matching (default: {MERGE_PARAMS.get('overlap_threshold', 0.1)})")
    parser.add_argument("--disable_matching", action="store_true", help="Disable cross-tile instance matching")
    
    args = parser.parse_args()
    
    # Route to appropriate task function
    if args.task == "tile":
        run_tile_task(args, TILE_PARAMS)
    elif args.task == "remap":
        run_remap_task(args, REMAP_PARAMS)
    elif args.task == "merge":
        run_merge_task(args, MERGE_PARAMS)
    else:
        print(f"Error: Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()

