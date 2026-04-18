from __future__ import annotations

import sys
from pathlib import Path

from parameters import Parameters


def run_tile_task(params: Parameters):
    """Run the public tile task (tiling plus two-stage subsampling)."""
    try:
        from main_tile import run_tiling_pipeline
        from main_subsample import run_subsample_pipeline
    except ImportError as e:
        print(f"Error: Could not import required tile modules: {e}")
        sys.exit(1)

    if not params.input_dir:
        print("Error: --input-dir is required for tile task")
        sys.exit(1)
    if not params.output_dir:
        print("Error: --output-dir is required for tile task")
        sys.exit(1)

    input_dir = Path(params.input_dir)
    output_dir = Path(params.output_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    tile_length = params.tile_length
    tile_buffer = params.tile_buffer
    threads = params.threads
    workers = params.workers
    dimension_reduction = bool(params.dimension_reduction)
    if params.skip_dimension_reduction is not None:
        dimension_reduction = not bool(params.skip_dimension_reduction)
        print(
            "  Note: deprecated --skip-dimension-reduction was provided; "
            "using its inverse for dimension_reduction",
            flush=True,
        )
    num_spatial_chunks = params.num_spatial_chunks or threads
    res1 = params.resolution_1
    res2 = params.resolution_2
    tiling_threshold = params.tiling_threshold
    chunk_size = params.chunk_size
    chunkwise_copc_source_creation = params.chunkwise_copc_source_creation
    output_copc_res1 = bool(params.output_copc_res1)

    print("=" * 60)
    print("Running Tile Task")
    print("=" * 60)
    print(f"Input directory:           {input_dir}")
    print(f"Output directory:          {output_dir}")
    print(f"Tile length:              {tile_length}m")
    print(f"Tile buffer:              {tile_buffer}m")
    print(f"Workers:                  {workers}")
    print(f"Threads per writer:       {threads}")
    print(f"Spatial chunks:           {num_spatial_chunks}")
    print(
        "Dimension reduction:      "
        f"{dimension_reduction} ({'standard dims only' if dimension_reduction else 'keep all dims'})"
    )
    print(f"Resolution 1:             {res1}m")
    print(f"Resolution 2:             {res2}m")
    print(f"Res1 output COPC:         {output_copc_res1}")
    if tiling_threshold is not None:
        print(f"Tiling threshold:         {tiling_threshold} MB")
    print(f"Chunk size:               {chunk_size:,} points")
    print(f"Chunkwise COPC creation:  {chunkwise_copc_source_creation}")
    print()

    try:
        tiles_dir = run_tiling_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            tile_length=tile_length,
            tile_buffer=tile_buffer,
            num_workers=workers,
            threads=threads,
            max_tile_procs=workers,
            dimension_reduction=dimension_reduction,
            tiling_threshold=tiling_threshold,
            chunk_size=chunk_size,
            chunkwise_copc_source_creation=chunkwise_copc_source_creation,
        )

        tiling_skipped = tiles_dir.name.startswith("copc_")
        if tiling_skipped:
            import shutil

            normalized_tiles_dir = output_dir / f"tiles_{int(tile_length)}m"
            normalized_tiles_dir.mkdir(exist_ok=True)
            for copc_file in tiles_dir.glob("*.copc.laz"):
                dest_file = normalized_tiles_dir / copc_file.name
                if not dest_file.exists():
                    shutil.copy2(copc_file, dest_file)
            tiles_dir = normalized_tiles_dir
            print(
                "  Note: tiling threshold skipped tile generation; "
                f"using normalized tile directory {tiles_dir}",
                flush=True,
            )

        output_prefix = f"{output_dir.name}_{int(tile_length)}m"
        res1_dir, res2_dir = run_subsample_pipeline(
            tiles_dir=tiles_dir,
            res1=res1,
            res2=res2,
            num_cores=workers,
            num_threads=num_spatial_chunks,
            output_prefix=output_prefix,
            output_base_dir=output_dir,
            dimension_reduction=dimension_reduction,
            output_copc_res1=output_copc_res1,
        )

        bounds_json = output_dir / "tile_bounds_tindex.json"
        if bounds_json.exists():
            from main_tile import update_tile_bounds_json_from_files

            num_updated = update_tile_bounds_json_from_files(bounds_json, res1_dir)
            if num_updated > 0:
                print(
                    "  Added actual_bounds to tile_bounds_tindex.json for "
                    f"{num_updated} tile(s) in {res1_dir.name}",
                    flush=True,
                )

        print()
        print("=" * 60)
        print("Tile Task Complete")
        print("=" * 60)
        print(f"Tiles:                  {tiles_dir}")
        print(f"Subsampled res1:        {res1_dir}")
        print(f"Subsampled res2:        {res2_dir}")
        return input_dir
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
