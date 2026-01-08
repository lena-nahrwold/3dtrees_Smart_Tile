#!/usr/bin/env python3
"""
Main orchestrator script for the 3DTrees smart tiling pipeline.

Routes to appropriate task modules based on --task parameter:
- tile: XYZ reduction, COPC conversion, tiling, and subsampling (2cm and 10cm)
- merge: Remap predictions and merge tiles with instance matching

Usage:
    python run.py --task tile --input_dir /path/to/input --output_dir /path/to/output
    python run.py --task merge --subsampled_10cm_folder /path/to/subsampled_10cm
"""

import argparse
import os
import sys
from pathlib import Path

# Import parameter loading functions
try:
    from parameters import load_params, print_params, TILE_PARAMS, REMAP_PARAMS, MERGE_PARAMS
except ImportError:
    print("Error: Could not import parameters.py")
    sys.exit(1)


def run_tile_task(args, params):
    """
    Run the tile task: XYZ reduction, COPC conversion, tiling, and subsampling.
    
    Pipeline:
    1. Convert LAZ to XYZ-only COPC (via main_tile.py)
    2. Build spatial index
    3. Calculate tile bounds
    4. Create overlapping tiles
    5. Subsample to resolution 1 (2cm)
    6. Subsample to resolution 2 (10cm)
    """
    # Import Python modules
    try:
        from main_tile import run_tiling_pipeline
        from main_subsample import run_subsample_pipeline
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure main_tile.py and main_subsample.py exist.")
        sys.exit(1)
    
    # Required arguments
    if not args.input_dir:
        print("Error: --input_dir is required for tile task")
        sys.exit(1)
    if not args.output_dir:
        print("Error: --output_dir is required for tile task")
        sys.exit(1)
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Get parameters (command-line overrides defaults)
    tile_length = args.tile_length if args.tile_length else params.get('tile_length', 100)
    tile_buffer = args.tile_buffer if args.tile_buffer else params.get('tile_buffer', 5)
    threads = args.threads if args.threads else params.get('threads', 5)
    workers = args.workers if args.workers else params.get('workers', 4)
    grid_offset = args.grid_offset if args.grid_offset else params.get('grid_offset', 1.0)
    skip_dimension_reduction = args.skip_dimension_reduction if hasattr(args, 'skip_dimension_reduction') and args.skip_dimension_reduction else params.get('skip_dimension_reduction', False)
    # num_spatial_chunks defaults to workers (one chunk per worker)
    num_spatial_chunks = args.num_spatial_chunks if args.num_spatial_chunks else None  # None means auto = workers
    res1 = params.get('resolution_1', 0.02)
    res2 = params.get('resolution_2', 0.1)
    
    print("=" * 60)
    print("Running Tile Task (Python Pipeline)")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tile length: {tile_length}m")
    print(f"Tile buffer: {tile_buffer}m")
    print(f"Grid offset: {grid_offset}m")
    print(f"Workers: {workers}")
    print(f"Threads per writer: {threads}")
    print(f"Skip dimension reduction: {skip_dimension_reduction}")
    print(f"Resolutions: {res1}m ({int(res1*100)}cm), {res2}m ({int(res2*100)}cm)")
    print()
    
    try:
        # Step 1-4: Tiling pipeline
        tiles_dir = run_tiling_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            tile_length=tile_length,
            tile_buffer=tile_buffer,
            grid_offset=grid_offset,
            num_workers=workers,
            threads=threads,
            max_tile_procs=workers,
            skip_dimension_reduction=skip_dimension_reduction
        )
        
        # Step 5-6: Subsampling pipeline
        output_prefix = f"{output_dir.name}_{int(tile_length)}m"
        res1_dir, res2_dir = run_subsample_pipeline(
            tiles_dir=tiles_dir,
            res1=res1,
            res2=res2,
            num_cores=workers,
            num_spatial_chunks=num_spatial_chunks,
            output_prefix=output_prefix
        )
        
        print()
        print("=" * 60)
        print("Tile Task Complete")
        print("=" * 60)
        print(f"Tiles: {tiles_dir}")
        print(f"Subsampled {int(res1*100)}cm: {res1_dir}")
        print(f"Subsampled {int(res2*100)}cm: {res2_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_merge_task(args, params):
    """
    Run the merge task: remap predictions and merge tiles.
    
    Pipeline:
    1. Remap predictions from 10cm to target resolution (via main_remap.py)
    2. Merge tiles with instance matching (via main_merge.py)
    """
    # Import Python modules
    try:
        from main_remap import remap_all_tiles
        from main_merge import run_merge
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure main_remap.py and main_merge.py exist.")
        sys.exit(1)
    
    # Required arguments - need either subsampled_10cm_folder or segmented_remapped_folder
    if not args.subsampled_10cm_folder and not args.segmented_remapped_folder:
        print("Error: --subsampled_10cm_folder or --segmented_remapped_folder is required for merge task")
        sys.exit(1)
    
    # Get parameters
    target_resolution = args.target_resolution if args.target_resolution else params.get('target_resolution_cm', 2)
    workers = args.workers if args.workers else MERGE_PARAMS.get('workers', 4)
    buffer = args.buffer if args.buffer is not None else MERGE_PARAMS.get('buffer', 10.0)
    overlap_threshold = args.overlap_threshold if args.overlap_threshold is not None else MERGE_PARAMS.get('overlap_threshold', 0.3)
    max_centroid_distance = MERGE_PARAMS.get('max_centroid_distance', 3.0)
    correspondence_tolerance = MERGE_PARAMS.get('correspondence_tolerance', 0.05)
    max_volume_for_merge = MERGE_PARAMS.get('max_volume_for_merge', 4.0)
    
    print("=" * 60)
    print("Running Merge Task (Python Pipeline)")
    print("=" * 60)
    
    try:
        # Step 1: Remap predictions (if subsampled_10cm_folder provided)
        segmented_remapped_folder = None
        
        if args.subsampled_10cm_folder:
            subsampled_10cm_dir = Path(args.subsampled_10cm_folder)
            
            if not subsampled_10cm_dir.exists():
                print(f"Error: Input directory does not exist: {subsampled_10cm_dir}")
                sys.exit(1)
            
            print(f"Input (10cm): {subsampled_10cm_dir}")
            print(f"Target resolution: {target_resolution}cm")
            print()
            
            # Remap
            segmented_remapped_folder = remap_all_tiles(
                subsampled_10cm_dir=subsampled_10cm_dir,
                target_resolution_cm=target_resolution,
                subsampled_target_folder=Path(args.subsampled_target_folder) if args.subsampled_target_folder else None,
                output_folder=Path(args.output_folder) if args.output_folder else None,
                num_threads=workers
            )
        
        # Step 2: Merge tiles
        if args.segmented_remapped_folder:
            segmented_remapped_folder = Path(args.segmented_remapped_folder)
        
        if segmented_remapped_folder is None:
            print("Error: No segmented remapped folder available for merge")
            sys.exit(1)
        
        if not segmented_remapped_folder.exists():
            print(f"Error: Segmented folder does not exist: {segmented_remapped_folder}")
            sys.exit(1)
        
        print()
        print(f"Segmented folder: {segmented_remapped_folder}")
        print(f"Buffer: {buffer}m")
        print(f"Overlap threshold: {overlap_threshold}")
        print(f"Workers: {workers}")
        print()
        
        output_merged = Path(args.output_merged_laz) if args.output_merged_laz else None
        output_tiles_dir = Path(args.output_tiles_folder) if args.output_tiles_folder else None
        original_tiles_dir = Path(args.original_tiles_dir) if args.original_tiles_dir else None
        
        merged_output = run_merge(
            segmented_dir=segmented_remapped_folder,
            output_merged=output_merged,
            output_tiles_dir=output_tiles_dir,
            original_tiles_dir=original_tiles_dir,
            buffer=buffer,
            overlap_threshold=overlap_threshold,
            max_centroid_distance=max_centroid_distance,
            correspondence_tolerance=correspondence_tolerance,
            max_volume_for_merge=max_volume_for_merge,
            num_threads=workers,
            enable_matching=not args.disable_matching,
            require_overlap=True,
            enable_volume_merge=True,
            verbose=True,   
        )
        
        print()
        print("=" * 60)
        print("Merge Task Complete")
        print("=" * 60)
        print(f"Merged output: {merged_output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="3DTrees Smart Tiling Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tile task: XYZ reduction, COPC conversion, tiling, and subsampling
  python run.py --task tile --input_dir /path/to/input --output_dir /path/to/output
  
  # Tile task with custom parameters
  python run.py --task tile --input_dir /path/to/input --output_dir /path/to/output \\
    --tile_length 150 --resolution_1 0.03 --workers 8
  
  # Tile task with config file
  python run.py --task tile --input_dir /path/to/input --output_dir /path/to/output \\
    --config my_params.py
  
  # Tile task with config + parameter overrides
  python run.py --task tile --input_dir /path/to/input --output_dir /path/to/output \\
    --config my_params.py --tile_length 200
  
  # Merge task: remap + merge (provide 10cm folder to run both)
  python run.py --task merge --subsampled_10cm_folder /path/to/subsampled_10cm
  
  # Merge task with custom parameters
  python run.py --task merge --subsampled_10cm_folder /path/to/subsampled_10cm \\
    --buffer 15.0 --overlap_threshold 0.4
  
  # Merge task: merge only (provide segmented folder)
  python run.py --task merge --segmented_remapped_folder /path/to/segmented_remapped
  
  # View current parameters
  python run.py --show-params
        """
    )
    
    # Parameter configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Custom config file (Python file with TILE_PARAMS, REMAP_PARAMS, MERGE_PARAMS)"
    )
    
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="Show current parameter configuration and exit"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["tile", "merge"],
        help="Task to run: tile (tiling+subsampling) or merge (remap+merge)"
    )
    
    # Tile task arguments
    parser.add_argument("--input_dir", type=str, help="Input directory with LAZ files (required for tile)")
    parser.add_argument("--output_dir", type=str, help="Output directory (required for tile)")
    
    # Tile parameters
    parser.add_argument("--tile_length", type=float, help="Tile size in meters (default: 100)")
    parser.add_argument("--tile_buffer", type=float, help="Buffer size in meters (default: 5)")
    parser.add_argument("--grid_offset", type=float, help="Grid offset in meters (default: 1.0)")
    parser.add_argument("--threads", type=int, help="Threads per COPC writer (default: 5)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: 4)")
    parser.add_argument("--num_spatial_chunks", type=int, help="Number of spatial chunks per tile for subsampling (default: equals workers)")
    parser.add_argument("--resolution_1", type=float, help="First subsampling resolution in meters (default: 0.02 = 2cm)")
    parser.add_argument("--resolution_2", type=float, help="Second subsampling resolution in meters (default: 0.1 = 10cm)")
    parser.add_argument("--skip_dimension_reduction", action="store_true", help="Skip XYZ-only reduction and keep all point dimensions")
    
    # Remap task arguments (used in merge task)
    parser.add_argument("--subsampled_10cm_folder", type=str, help="Path to subsampled_10cm folder with segmented results")
    parser.add_argument("--target_resolution", type=int, help="Target resolution in cm (default: 2)")
    parser.add_argument("--subsampled_target_folder", type=str, help="Path to target resolution subsampled folder (auto-derived if not specified)")
    parser.add_argument("--output_folder", type=str, help="Output folder for remapped files (auto-derived if not specified)")
    
    # Merge task arguments
    parser.add_argument("--segmented_remapped_folder", type=str, help="Path to segmented_remapped folder")
    parser.add_argument("--output_merged_laz", type=str, help="Output path for merged LAZ file (optional)")
    parser.add_argument("--output_tiles_folder", type=str, help="Output folder for per-tile results (optional)")
    parser.add_argument("--original_tiles_dir", type=str, help="Directory with original tile files for retiling (optional)")
    
    # Merge parameters
    parser.add_argument("--buffer", type=float, help="Buffer distance for filtering in meters (default: 10.0)")
    parser.add_argument("--overlap_threshold", type=float, help="Overlap ratio threshold for instance matching (default: 0.3)")
    parser.add_argument("--max_centroid_distance", type=float, help="Max centroid distance to merge instances (default: 3.0)")
    parser.add_argument("--correspondence_tolerance", type=float, help="Point correspondence tolerance in meters (default: 0.05)")
    parser.add_argument("--max_volume_for_merge", type=float, help="Max volume for small instance merge in m³ (default: 4.0)")
    parser.add_argument("--disable_matching", action="store_true", help="Disable cross-tile instance matching")
    
    args = parser.parse_args()
    
    # Build parameter overrides from CLI arguments
    param_overrides = []
    
    # Tile parameters
    if args.tile_length is not None:
        param_overrides.append(f"tile_length={args.tile_length}")
    if args.tile_buffer is not None:
        param_overrides.append(f"tile_buffer={args.tile_buffer}")
    if args.grid_offset is not None:
        param_overrides.append(f"grid_offset={args.grid_offset}")
    if args.threads is not None:
        param_overrides.append(f"threads={args.threads}")
    if args.workers is not None:
        param_overrides.append(f"workers={args.workers}")
    if args.resolution_1 is not None:
        param_overrides.append(f"resolution_1={args.resolution_1}")
    if args.resolution_2 is not None:
        param_overrides.append(f"resolution_2={args.resolution_2}")
    if args.skip_dimension_reduction:
        param_overrides.append(f"skip_dimension_reduction=True")
    
    # Remap parameters
    if args.target_resolution is not None:
        param_overrides.append(f"target_resolution_cm={args.target_resolution}")
    
    # Merge parameters
    if args.buffer is not None:
        param_overrides.append(f"buffer={args.buffer}")
    if args.overlap_threshold is not None:
        param_overrides.append(f"overlap_threshold={args.overlap_threshold}")
    if args.max_centroid_distance is not None:
        param_overrides.append(f"max_centroid_distance={args.max_centroid_distance}")
    if args.correspondence_tolerance is not None:
        param_overrides.append(f"correspondence_tolerance={args.correspondence_tolerance}")
    if args.max_volume_for_merge is not None:
        param_overrides.append(f"max_volume_for_merge={args.max_volume_for_merge}")
    
    # Load parameters with overrides
    params = load_params(
        config_file=args.config,
        param_overrides=param_overrides if param_overrides else None,
        use_env=True
    )
    
    # Extract specific parameter sets
    TILE_PARAMS = params['TILE_PARAMS']
    REMAP_PARAMS = params['REMAP_PARAMS']
    MERGE_PARAMS = params['MERGE_PARAMS']
    
    # Show parameters if requested
    if args.show_params:
        print_params(params)
        sys.exit(0)
    
    # Task is required if not showing params
    if not args.task:
        parser.error("--task is required (unless using --show-params)")
    
    # Show parameter summary if config or CLI overrides were provided
    if args.config or param_overrides:
        print()
        print_params(params)
        print()
    
    # Route to appropriate task function
    if args.task == "tile":
        run_tile_task(args, TILE_PARAMS)
    elif args.task == "merge":
        run_merge_task(args, MERGE_PARAMS)
    else:
        print(f"Error: Unknown task: {args.task}")
        sys.exit(1)


if __name__ == "__main__":
    main()
