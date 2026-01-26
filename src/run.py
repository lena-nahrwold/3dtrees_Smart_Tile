#!/usr/bin/env python3
"""
Main orchestrator script for the 3DTrees smart tiling pipeline.

Routes to appropriate task modules based on --task parameter:
- tile: XYZ reduction, COPC conversion, tiling, and subsampling (2cm and 10cm)
- remap: Remap predictions from source files to target resolution
- merge: Remap predictions and merge tiles with instance matching
- remap_merge: Remap predictions then merge tiles (combines remap + merge)

Usage:
    python src/run.py --task tile --input-dir /path/to/input --output-dir /path/to/output
    python src/run.py --task remap --source-folder /path/to/segmented --target-folder /path/to/subsampled
    python src/run.py --task merge --subsampled-10cm-folder /path/to/subsampled_10cm --original-input-dir /path/to/input
    python src/run.py --task remap_merge --source-folder /path/to/segmented --target-folder /path/to/subsampled --original-tiles-dir /path/to/tiles
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports when run from project root
_src_dir = Path(__file__).parent.resolve()
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import Pydantic-based parameters
try:
    from parameters import Parameters, print_params, get_tile_params, get_merge_params, get_remap_params
except ImportError as e:
    print(f"Error: Could not import parameters.py: {e}")
    print("Please install required dependencies: pip install pydantic pydantic-settings")
    sys.exit(1)


def run_tile_task(params: Parameters):
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
    if not params.input_dir:
        print("Error: --input-dir is required for tile task")
        sys.exit(1)
    if not params.output_dir:
        print("Error: --output-dir is required for tile task")
        sys.exit(1)
    
    # Validate input directory
    input_dir = Path(params.input_dir)
    output_dir = Path(params.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Get parameters from Pydantic model
    tile_length = params.tile_length
    tile_buffer = params.tile_buffer
    threads = params.threads
    workers = params.workers
    grid_offset = params.grid_offset
    skip_dimension_reduction = params.skip_dimension_reduction
    num_spatial_chunks = params.num_spatial_chunks
    res1 = params.resolution_1
    res2 = params.resolution_2
    tiling_threshold = params.tiling_threshold

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
    if tiling_threshold is not None:
        print(f"Tiling threshold: {tiling_threshold} MB")
    print()
    
    try:
        # Step 1-4: Tiling pipeline
        # Convert skip_dimension_reduction to dimension_reduction (inverted logic)
        dimension_reduction = not skip_dimension_reduction
        tiles_dir = run_tiling_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            tile_length=tile_length,
            tile_buffer=tile_buffer,
            grid_offset=grid_offset,
            num_workers=workers,
            threads=threads,
            max_tile_procs=workers,
            dimension_reduction=dimension_reduction,
            tiling_threshold=tiling_threshold
        )

        # Check if tiling was skipped (returns copc_dir instead of tiles_dir)
        tiling_skipped = tiles_dir.name.startswith("copc_")

        if tiling_skipped:
            # Single file case - use simpler output prefix
            output_prefix = f"{output_dir.name}_single"
            print(f"  Note: Tiling was skipped, subsampling directly from COPC")
        else:
            # Normal tiled case
            output_prefix = f"{output_dir.name}_{int(tile_length)}m"

        # Step 5-6: Subsampling pipeline
        res1_dir, res2_dir = run_subsample_pipeline(
            tiles_dir=tiles_dir,
            res1=res1,
            res2=res2,
            num_cores=workers,
            num_threads=num_spatial_chunks,  # num_spatial_chunks maps to num_threads in the function
            output_prefix=output_prefix
        )
        
        print()
        print("=" * 60)
        print("Tile Task Complete")
        print("=" * 60)
        print(f"Tiles: {tiles_dir}")
        print(f"Subsampled {int(res1*100)}cm: {res1_dir}")
        print(f"Subsampled {int(res2*100)}cm: {res2_dir}")
        
        # Return the input_dir for use in merge task if needed
        return input_dir
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_remap_task(params: Parameters):
    """
    Run the remap task: remap predictions from source to target resolution.

    Pipeline:
    1. Remap predictions from source files to target files (via main_remap.py)
    """
    # Import Python modules
    try:
        from main_remap import remap_all_tiles
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure main_remap.py exists.")
        sys.exit(1)

    # Required arguments
    if not params.source_folder:
        print("Error: --source-folder is required for remap task")
        sys.exit(1)
    if not params.target_folder:
        print("Error: --target-folder is required for remap task")
        sys.exit(1)

    # Validate directories
    source_folder = Path(params.source_folder)
    target_folder = Path(params.target_folder)

    if not source_folder.exists():
        print(f"Error: Source folder does not exist: {source_folder}")
        sys.exit(1)
    if not target_folder.exists():
        print(f"Error: Target folder does not exist: {target_folder}")
        sys.exit(1)

    # Output folder
    output_folder = params.output_folder
    if output_folder is None:
        output_folder = source_folder.parent / "segmented_remapped"
    else:
        output_folder = Path(output_folder)

    tolerance = params.tolerance if params.tolerance else 5.0

    print("=" * 60)
    print("Running Remap Task")
    print("=" * 60)
    print(f"Source folder: {source_folder}")
    print(f"Target folder: {target_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Tolerance: {tolerance}m")
    print()

    try:
        # Remap predictions
        remapped_folder = remap_all_tiles(
            source_folder=source_folder,
            target_folder=target_folder,
            output_folder=output_folder,
            tolerance=tolerance
        )

        print()
        print("=" * 60)
        print("Remap Task Complete")
        print("=" * 60)
        print(f"Remapped output: {remapped_folder}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_merge_task(params: Parameters):
    """
    Run the merge task: remap predictions and merge tiles.

    Pipeline:
    1. Remap predictions from 10cm to target resolution (via main_remap.py)
    2. Merge tiles with instance matching (via main_merge.py)
    3. Remap to original input files (if original_input_dir provided)
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
    if not params.subsampled_10cm_folder and not params.segmented_remapped_folder:
        print("Error: --subsampled-10cm-folder or --segmented-remapped-folder is required for merge task")
        sys.exit(1)
    
    # Get parameters from Pydantic model
    target_resolution = params.target_resolution
    workers = params.workers
    buffer = params.buffer
    overlap_threshold = params.overlap_threshold
    max_centroid_distance = params.max_centroid_distance
    correspondence_tolerance = params.correspondence_tolerance
    max_volume_for_merge = params.max_volume_for_merge
    border_zone_width = params.border_zone_width
    min_cluster_size = params.min_cluster_size
    retile_buffer = params.retile_buffer

    print("=" * 60)
    print("Running Merge Task (Python Pipeline)")
    print("=" * 60)
    
    try:
        # Step 1: Remap predictions (if subsampled_10cm_folder provided)
        segmented_remapped_folder = None
        
        if params.subsampled_10cm_folder:
            subsampled_10cm_dir = Path(params.subsampled_10cm_folder)
            
            if not subsampled_10cm_dir.exists():
                print(f"Error: Input directory does not exist: {subsampled_10cm_dir}")
                sys.exit(1)
            
            print(f"Input (10cm): {subsampled_10cm_dir}")
            print(f"Target resolution: {target_resolution}cm")
            print()
            
            # Derive target folder (2cm) and output folder
            # The 10cm folder is typically at: tiles_100m/subsampled_10cm
            # The 2cm folder would be at: tiles_100m/subsampled_2cm
            parent_dir = subsampled_10cm_dir.parent
            target_folder = params.subsampled_target_folder
            if target_folder is None:
                target_folder = parent_dir / f"subsampled_{target_resolution}cm"
            
            output_folder = params.output_folder
            if output_folder is None:
                output_folder = parent_dir / "segmented_remapped"
            
            if not target_folder.exists():
                print(f"Error: Target resolution folder does not exist: {target_folder}")
                print(f"Expected folder for {target_resolution}cm subsampled tiles")
                sys.exit(1)
            
            # Remap - source is 10cm segmented, target is 2cm subsampled
            segmented_remapped_folder = remap_all_tiles(
                source_folder=subsampled_10cm_dir,
                target_folder=target_folder,
                output_folder=output_folder,
                tolerance=5.0
            )
        
        # Step 2: Merge tiles
        if params.segmented_remapped_folder:
            segmented_remapped_folder = Path(params.segmented_remapped_folder)
        
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
        if params.original_input_dir:
            print(f"Original input dir: {params.original_input_dir}")
        print()
        
        output_merged = params.output_merged_laz
        output_tiles_dir = params.output_tiles_folder
        original_tiles_dir = params.original_tiles_dir
        original_input_dir = params.original_input_dir
        
        # Auto-derive paths if not provided
        parent_dir = segmented_remapped_folder.parent
        if output_tiles_dir is None:
            output_tiles_dir = parent_dir / "output_tiles"
        if original_tiles_dir is None:
            # Try to find the tiles directory (parent of subsampled folders)
            original_tiles_dir = parent_dir
        
        merged_output = run_merge(
            segmented_dir=segmented_remapped_folder,
            output_tiles_dir=output_tiles_dir,
            original_tiles_dir=original_tiles_dir,
            original_input_dir=original_input_dir,
            output_merged=output_merged,
            buffer=buffer,
            overlap_threshold=overlap_threshold,
            max_centroid_distance=max_centroid_distance,
            correspondence_tolerance=correspondence_tolerance,
            max_volume_for_merge=max_volume_for_merge,
            border_zone_width=border_zone_width,
            min_cluster_size=min_cluster_size,
            num_threads=workers,
            enable_matching=not params.disable_matching,
            require_overlap=True,
            enable_volume_merge=not params.disable_volume_merge,
            skip_merged_file=params.skip_merged_file,
            verbose=params.verbose,
            retile_buffer=retile_buffer,
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


def run_remap_merge_task(params: Parameters):
    """
    Run the remap_merge task: remap predictions then merge tiles.

    Pipeline:
    1. Remap predictions from source files to target files (via main_remap.py)
    2. Merge tiles with instance matching (via main_merge.py)
    3. Remap to original input files (if original_input_dir provided)

    This combines remap and merge into a single workflow.
    """
    # Import Python modules
    try:
        from main_remap import remap_all_tiles
        from main_merge import run_merge
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure main_remap.py and main_merge.py exist.")
        sys.exit(1)

    # Required arguments
    if not params.source_folder:
        print("Error: --source-folder is required for remap_merge task")
        sys.exit(1)
    if not params.target_folder:
        print("Error: --target-folder is required for remap_merge task")
        sys.exit(1)
    if not params.original_tiles_dir:
        print("Error: --original-tiles-dir is required for remap_merge task")
        sys.exit(1)

    # Validate directories
    source_folder = Path(params.source_folder)
    target_folder = Path(params.target_folder)
    original_tiles_dir = Path(params.original_tiles_dir)

    if not source_folder.exists():
        print(f"Error: Source folder does not exist: {source_folder}")
        sys.exit(1)
    if not target_folder.exists():
        print(f"Error: Target folder does not exist: {target_folder}")
        sys.exit(1)
    if not original_tiles_dir.exists():
        print(f"Error: Original tiles directory does not exist: {original_tiles_dir}")
        sys.exit(1)

    # Output folder for remap step
    output_folder = params.output_folder
    if output_folder is None:
        output_folder = source_folder.parent / "segmented_remapped"
    else:
        output_folder = Path(output_folder)

    tolerance = params.tolerance if params.tolerance else 5.0

    # Get merge parameters
    workers = params.workers
    buffer = params.buffer
    overlap_threshold = params.overlap_threshold
    max_centroid_distance = params.max_centroid_distance
    correspondence_tolerance = params.correspondence_tolerance
    max_volume_for_merge = params.max_volume_for_merge
    border_zone_width = params.border_zone_width
    min_cluster_size = params.min_cluster_size
    retile_buffer = params.retile_buffer

    print("=" * 60)
    print("Running Remap-Merge Task")
    print("=" * 60)
    print(f"Source folder: {source_folder}")
    print(f"Target folder: {target_folder}")
    print(f"Original tiles dir: {original_tiles_dir}")
    print(f"Output folder (remap): {output_folder}")
    print(f"Tolerance: {tolerance}m")
    print()

    try:
        # Step 1: Remap predictions
        print("=" * 60)
        print("Step 1: Remapping predictions")
        print("=" * 60)
        
        remapped_folder = remap_all_tiles(
            source_folder=source_folder,
            target_folder=target_folder,
            output_folder=output_folder,
            tolerance=tolerance
        )

        print()
        print("=" * 60)
        print("Remap Step Complete")
        print("=" * 60)
        print(f"Remapped output: {remapped_folder}")
        print()

        # Step 2: Merge tiles
        print("=" * 60)
        print("Step 2: Merging tiles")
        print("=" * 60)
        print(f"Segmented folder: {remapped_folder}")
        print(f"Buffer: {buffer}m")
        print(f"Overlap threshold: {overlap_threshold}")
        print(f"Workers: {workers}")
        if params.original_input_dir:
            print(f"Original input dir: {params.original_input_dir}")
        print()

        output_merged = params.output_merged_laz
        output_tiles_dir = params.output_tiles_folder
        original_input_dir = params.original_input_dir

        # Auto-derive paths if not provided
        parent_dir = remapped_folder.parent
        if output_tiles_dir is None:
            output_tiles_dir = parent_dir / "output_tiles"

        merged_output = run_merge(
            segmented_dir=remapped_folder,
            output_tiles_dir=output_tiles_dir,
            original_tiles_dir=original_tiles_dir,
            original_input_dir=original_input_dir,
            output_merged=output_merged,
            buffer=buffer,
            overlap_threshold=overlap_threshold,
            max_centroid_distance=max_centroid_distance,
            correspondence_tolerance=correspondence_tolerance,
            max_volume_for_merge=max_volume_for_merge,
            border_zone_width=border_zone_width,
            min_cluster_size=min_cluster_size,
            num_threads=workers,
            enable_matching=not params.disable_matching,
            require_overlap=True,
            enable_volume_merge=not params.disable_volume_merge,
            skip_merged_file=params.skip_merged_file,
            verbose=params.verbose,
            retile_buffer=retile_buffer,
        )

        print()
        print("=" * 60)
        print("Remap-Merge Task Complete")
        print("=" * 60)
        print(f"Remapped output: {remapped_folder}")
        print(f"Merged output: {merged_output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def preprocess_boolean_flags(args_list):
    """
    Preprocess CLI args to convert boolean flags to explicit True/False for Pydantic.
    Pydantic expects --flag True/False, but we want --flag to work like argparse.
    """
    boolean_flags = [
        '--show-params', '--show_params',
        '--skip-dimension-reduction', '--skip_dimension_reduction',
        '--disable-matching', '--disable_matching',
        '--disable-volume-merge', '--disable_volume_merge',
        '--skip-merged-file', '--skip_merged_file',
        '--verbose', '-v'
    ]
    
    processed = []
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg in boolean_flags:
            # Check if next arg is already True/False
            if i + 1 < len(args_list) and args_list[i + 1] in ['True', 'False']:
                processed.extend([arg, args_list[i + 1]])
                i += 2
            else:
                # Add explicit True for boolean flag
                processed.extend([arg, 'True'])
                i += 1
        else:
            processed.append(arg)
            i += 1
    return processed


def main():
    # Handle --show-params flag first using argparse (before Pydantic parsing)
    # This avoids Pydantic's boolean flag parsing issues
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--show-params', '--show_params', action='store_true')
    pre_args, remaining_args = pre_parser.parse_known_args()
    
    # If --show-params was found, add it back to remaining_args for Pydantic
    if pre_args.show_params:
        remaining_args = ['--show-params'] + remaining_args
    
    # Preprocess boolean flags for Pydantic
    processed_args = [sys.argv[0]] + preprocess_boolean_flags(remaining_args)
    
    # Temporarily replace sys.argv for Pydantic parsing
    original_argv = sys.argv
    sys.argv = processed_args
    
    # Parse parameters using Pydantic (handles CLI automatically)
    try:
        params = Parameters()
    except Exception as e:
        print(f"Error parsing parameters: {e}")
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    # Show parameters if requested
    if params.show_params:
        print_params(params)
        sys.exit(0)
    
    # Task is required if not showing params
    if not params.task:
        print("Error: --task is required (unless using --show-params)")
        print("Usage: python run.py --task tile --input-dir /path/to/input --output-dir /path/to/output")
        print("       python run.py --task remap --source-folder /path/to/segmented --target-folder /path/to/subsampled")
        print("       python run.py --task merge --subsampled-10cm-folder /path/to/10cm")
        print("       python run.py --task remap_merge --source-folder /path/to/segmented --target-folder /path/to/subsampled --original-tiles-dir /path/to/tiles")
        print("       python run.py --show-params")
        sys.exit(1)

    # Route to appropriate task function
    if params.task == "tile":
        run_tile_task(params)
    elif params.task == "remap":
        run_remap_task(params)
    elif params.task == "merge":
        run_merge_task(params)
    elif params.task == "remap_merge":
        run_remap_merge_task(params)
    else:
        print(f"Error: Unknown task: {params.task}")
        print("Valid tasks: tile, remap, merge, remap_merge")
        sys.exit(1)


if __name__ == "__main__":
    main()
