#!/usr/bin/env python3
"""
Main orchestrator script for the 3DTrees smart tiling pipeline.

Routes to appropriate task modules based on --task parameter:
- tile:   XYZ reduction, COPC conversion, tiling, and subsampling (2cm and 10cm)
- merge:  Remap predictions and merge tiles with instance matching
- remap:  Remap merged file dimensions to original input files
- filter: Remove buffer-zone instances per tile (no cross-tile merging)

Usage:
    python src/run.py --task tile --input-dir /path/to/input --output-dir /path/to/output
    python src/run.py --task merge --subsampled-10cm-folder /path/to/10cm --original-input-dir /path/to/input
    python src/run.py --task remap --merged-laz /path/to/merged.laz --original-input-dir /path/to/originals --output-dir /path/to/output
    python src/run.py --task filter --segmented-remapped-folder /path/to/tiles --tile-bounds-json /path/to/tile_bounds_tindex.json --output-dir /path/to/output
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

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
    1. Normalize source inputs to COPC while keeping all dimensions
    2. Build spatial index from COPC sources
    3. Calculate tile bounds
    4. Create overlapping COPC tiles
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
    # Coerce to bool so CLI/env string "True"/"true"/"1" is respected (Pydantic usually does this; be explicit)
    skip_dimension_reduction = bool(
        params.skip_dimension_reduction if isinstance(params.skip_dimension_reduction, bool)
        else str(params.skip_dimension_reduction).strip().lower() in ("true", "1", "yes")
    )
    num_spatial_chunks = params.num_spatial_chunks
    res1 = params.resolution_1
    res2 = params.resolution_2
    tiling_threshold = params.tiling_threshold
    chunk_size = params.chunk_size
    chunkwise_copc_source_creation = params.chunkwise_copc_source_creation
    print("=" * 60)
    print("Running Tile Task (Python Pipeline)")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tile length: {tile_length}m")
    print(f"Tile buffer: {tile_buffer}m")
    print(f"Workers: {workers}")
    print(f"Threads per writer: {threads}")
    print(f"Skip dimension reduction: {skip_dimension_reduction}")
    dimension_reduction = not skip_dimension_reduction
    print(f"Subsampling dimensions: {'minimal (standard dims only)' if dimension_reduction else 'keep all (including extra_dims)'}")
    print(f"Resolutions: {res1}m ({int(res1*100)}cm), {res2}m ({int(res2*100)}cm)")
    if tiling_threshold is not None:
        print(f"Tiling threshold: {tiling_threshold} MB")
    print(f"Chunk size: {chunk_size:,} points")
    print(f"Chunkwise source COPC creation: {chunkwise_copc_source_creation}")
    print()

    try:
        # Step 1-4: COPC-first tiling pipeline.
        # The COPC normalization always preserves all dimensions; dimension_reduction
        # only changes the downstream subsampled outputs below.
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

        # Check if tiling was skipped (returns copc_dir instead of tiles_dir)
        tiling_skipped = tiles_dir.name.startswith("copc_")

        if tiling_skipped:
            # Single file case - create tiles_* directory structure for consistency
            # Move COPC files to tiles_* directory so subsampling creates consistent structure
            tiles_dir_normalized = output_dir / f"tiles_{int(tile_length)}m"
            tiles_dir_normalized.mkdir(exist_ok=True)

            # Copy/move COPC files to tiles directory
            import shutil
            for copc_file in tiles_dir.glob("*.copc.laz"):
                dest_file = tiles_dir_normalized / copc_file.name
                if not dest_file.exists():
                    shutil.copy2(copc_file, dest_file)

            # Update tiles_dir to use normalized structure
            tiles_dir = tiles_dir_normalized
            output_prefix = f"{output_dir.name}_{int(tile_length)}m"
            print(f"  Note: Tiling was skipped, using normalized directory structure: {tiles_dir}")
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
            output_prefix=output_prefix,
            output_base_dir=output_dir,  # Output directly to output_dir, not under tiles_dir
            dimension_reduction=dimension_reduction,  # True = minimal (standard dims only); False = keep extra_dims
        )

        # Step 7: Attach actual bounds from created tiles while preserving the
        # planned tile geometry from the original tiling stage.
        bounds_json = output_dir / "tile_bounds_tindex.json"
        if bounds_json.exists():
            from main_tile import update_tile_bounds_json_from_files
            num_updated = update_tile_bounds_json_from_files(bounds_json, res1_dir)
            if num_updated > 0:
                print(
                    f"  Added actual_bounds to tile_bounds_tindex.json for "
                    f"{num_updated} tile(s) in {res1_dir.name}"
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
    # Note: subsampled_10cm_folder is populated by --subsampled-segmented-folder via alias
    if not params.subsampled_10cm_folder and not params.segmented_remapped_folder:
        print("Error: --subsampled-segmented-folder (or --subsampled-10cm-folder) or --segmented-remapped-folder is required for merge task")
        sys.exit(1)

    # Get parameters from Pydantic model
    workers = params.workers
    overlap_threshold = params.overlap_threshold
    max_centroid_distance = params.max_centroid_distance
    max_volume_for_merge = params.max_volume_for_merge
    border_zone_width = params.border_zone_width
    min_cluster_size = params.min_cluster_size
    retile_buffer = 2.0  # Fixed to 2.0m
    merge_chunk_size = params.merge_chunk_size

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
            print()

            # Derive target folder and output folder
            # The resolution folders are now at: tiles_*/subsampled_res1 and tiles_*/subsampled_res2
            # For backward compatibility, also check old naming: subsampled_{resolution}cm
            parent_dir = subsampled_10cm_dir.parent
            target_folder = params.subsampled_target_folder

            if target_folder is None:
                # Try new naming first (subsampled_res1) as default target
                target_folder_res1 = parent_dir / "subsampled_res1"
                if target_folder_res1.exists():
                    target_folder = target_folder_res1
                else:
                    # Fallback or error
                    pass

            output_folder = params.output_folder
            if output_folder is None:
                output_folder = parent_dir / "segmented_remapped"

            if target_folder is None or not target_folder.exists():
                print(f"Error: Target resolution folder does not exist or not specified")
                if target_folder:
                    print(f"Path: {target_folder}")
                print(f"Please provide --subsampled-target-folder")
                sys.exit(1)

            # Optional: tile_bounds_tindex.json for remap matching (use --tile_bounds_json first)
            remap_tile_bounds_json = None
            if params.tile_bounds_json and params.tile_bounds_json.exists():
                remap_tile_bounds_json = params.tile_bounds_json
            if remap_tile_bounds_json is None:
                remap_json_candidates = [
                    parent_dir / "tile_bounds_tindex.json",
                    subsampled_10cm_dir / "tile_bounds_tindex.json",
                ]
                if params.original_tiles_dir:
                    remap_json_candidates.insert(0, Path(params.original_tiles_dir) / "tile_bounds_tindex.json")
                for p in remap_json_candidates:
                    if p.exists():
                        remap_tile_bounds_json = p
                        break

            # Remap - source is 10cm segmented, target is 2cm subsampled
            segmented_remapped_folder = remap_all_tiles(
                source_folder=subsampled_10cm_dir,
                target_folder=target_folder,
                output_folder=output_folder,
                tile_bounds_json=remap_tile_bounds_json,
                verbose=bool(params.verbose),
                num_workers=workers,
                merge_chunk_size=merge_chunk_size,
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
            # Use segmented folder's parent, but ensure it's writable
            # If parent is root or not writable, use segmented folder itself
            if parent_dir == Path('/') or not os.access(parent_dir, os.W_OK):
                output_tiles_dir = segmented_remapped_folder / "output_tiles"
            else:
                output_tiles_dir = parent_dir / "output_tiles"
        if original_tiles_dir is None:
            # Try to find the tiles directory (parent of subsampled folders)
            original_tiles_dir = parent_dir

        # tile_bounds_json is required for merge (no fallback)
        tile_bounds_json = params.tile_bounds_json
        if tile_bounds_json is None:
            raise ValueError(
                "Merge task requires --tile_bounds_json /path/to/tile_bounds_tindex.json (e.g. from Tile task output)."
            )
        if not tile_bounds_json.exists():
            raise FileNotFoundError(
                f"tile_bounds_tindex.json not found: {tile_bounds_json}. Pass a valid --tile_bounds_json path."
            )

        # Parse 3DTrees dimension branding params
        threedtrees_dims = [d.strip() for d in params.threedtrees_dims.split(",") if d.strip()] if params.threedtrees_dims else None
        threedtrees_suffix = params.threedtrees_suffix

        # Derive boolean flags for the streaming merge pipeline
        enable_matching = not params.disable_matching
        enable_volume_merge = not params.disable_volume_merge
        enable_orphan_recovery = True

        merged_output = run_merge(
            segmented_dir=segmented_remapped_folder,
            output_tiles_dir=output_tiles_dir,
            original_tiles_dir=original_tiles_dir,
            tile_bounds_json=tile_bounds_json,
            original_input_dir=original_input_dir,
            output_merged=output_merged,
            overlap_threshold=overlap_threshold,
            max_centroid_distance=max_centroid_distance,
            max_volume_for_merge=max_volume_for_merge,
            border_zone_width=border_zone_width,
            min_cluster_size=min_cluster_size,
            num_threads=workers,
            enable_matching=enable_matching,
            require_overlap=True,
            enable_volume_merge=enable_volume_merge,
            skip_merged_file=params.skip_merged_file,
            verbose=params.verbose,
            retile_buffer=retile_buffer,
            instance_dimension=params.instance_dimension,
            transfer_original_dims_to_merged=params.transfer_original_dims_to_merged,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            save_filtered_tiles_flag=params.save_filtered_tiles,
            enable_orphan_recovery=enable_orphan_recovery,
            standardization_json=params.standardization_json,
            merge_chunk_size=merge_chunk_size,
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


def _canonical_tile_name_for_json_index(
    json_idx: int,
    json_labels,
) -> str:
    """Return a stable tile name sourced from the bounds JSON."""
    if json_idx < len(json_labels):
        label = json_labels[json_idx]
        if label:
            return label
    return f"tile_{json_idx:04d}"


def _preferred_json_bounds_field(json_tiles) -> str:
    """Use actual_bounds for matching when available, otherwise fall back."""
    if any("actual_bounds" in tile for tile in json_tiles):
        return "actual_bounds"
    return "planned_bounds"


def update_trees_files_with_global_ids(
    tile_results,
    global_to_merged,
    trees_by_tile,
    trees_output_dir,
    tile_offset: int,
):
    """
    For each tile that has a matching QSM trees .txt file, write an updated trees file to
    trees_output_dir with:
      - Buffer-zone trees removed (instances not in global_to_merged are skipped).
      - A new leading column 'predinstance' holding the global cross-tile ID.

    Trees file format (line_number is 1-indexed):
      Line 1 : comment      → kept as-is
      Line 2 : column header → becomes 'predinstance,<original_header>'
      Line N (N>=3): one tree → local predinstance = N-2; if it survived filtering,
                                written as '{new_global_id},{original_line}'

    Matching is provided explicitly via trees_by_tile so input filenames remain
    irrelevant; tile identity comes from the matched bounds JSON entry.
    """
    trees_output_dir = Path(trees_output_dir)
    trees_output_dir.mkdir(parents=True, exist_ok=True)

    if not trees_by_tile:
        print("  Warning: no co-located .txt files found for the matched tiles")
        return

    n_written = 0
    for result in tile_results:
        tile_name = result.tile_name
        trees_file = trees_by_tile.get(tile_name)
        if trees_file is None:
            print(f"  No trees file for tile {tile_name}, skipping")
            continue

        # Build local_id -> new_global_id for this tile's surviving instances
        local_to_new = {}
        for gid, meta in result.instances.items():
            if not meta.is_filtered and gid in global_to_merged:
                local_id = gid - result.tile_idx * tile_offset
                local_to_new[local_id] = global_to_merged[gid]

        with open(trees_file, "r") as fh:
            lines = fh.readlines()

        out_lines = []
        data_idx = 0  # counts data lines seen so far (0-based)
        for line_no, line in enumerate(lines):
            if line_no == 0:
                out_lines.append(line)
            elif line_no == 1:
                out_lines.append("predinstance," + line.lstrip())
            else:
                local_id = data_idx + 1   # line 3 => local_id 1
                data_idx += 1
                new_id = local_to_new.get(local_id)
                if new_id is not None:
                    out_lines.append(f"{new_id}," + line.lstrip())
                # filtered instance: drop the line

        out_path = trees_output_dir / f"{tile_name}_trees.txt"
        with open(out_path, "w") as fh:
            fh.writelines(out_lines)
        n_written += 1
        print(
            f"  Trees {trees_file.name}: "
            f"{len(local_to_new)}/{data_idx} trees kept → {out_path.name}"
        )

    print(f"  Trees files written: {n_written}")


def run_filter_task(params: Parameters):
    """
    Run the filter task: remove buffer-zone instances per tile, no cross-tile merging.

    Instances whose centroid OR highest point falls in the buffer region on a side
    that has a neighboring tile are removed. Output is one LAZ file per input tile,
    written to output_dir/filtered_tiles/.

    Required parameters:
        --segmented-remapped-folder or --subsampled-10cm-folder  (input tiles)
        --tile-bounds-json                                        (neighbor info)
        --output-dir                                              (output location)
    """
    try:
        import json as _json
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from merge_tiles import (
            build_neighbor_graph_from_bounds_json,
            get_tile_bounds_from_header,
        )
        from merge_tiles_streaming import (
            _extract_tile_metadata_wrapper,
            _match_tiles_to_json_bounds,
            write_filtered_tiles_streaming,
            redistribute_small_instances,
            TILE_OFFSET,
        )
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        sys.exit(1)

    # ── Validate inputs ──────────────────────────────────────────────────────
    # Support N comma-separated input directories (--segmented-folders) so that
    # collections organised in separate folders are filtered in one unified pass.
    # Falls back to the single-folder params for backwards compatibility.
    raw_folders = params.segmented_folders or ""
    input_dirs = [Path(p.strip()) for p in raw_folders.split(",") if p.strip()]
    if not input_dirs:
        fallback = params.segmented_remapped_folder or params.subsampled_10cm_folder
        if fallback:
            input_dirs = [Path(fallback)]
    if not input_dirs:
        print("Error: --segmented-folders or --segmented-remapped-folder is required for filter task")
        sys.exit(1)
    for _d in input_dirs:
        if not _d.exists():
            print(f"Error: Input directory not found: {_d}")
            sys.exit(1)

    tile_bounds_json = params.tile_bounds_json
    if not tile_bounds_json:
        print("Error: --tile-bounds-json is required for filter task (needed for neighbor info)")
        sys.exit(1)
    tile_bounds_json = Path(tile_bounds_json)
    if not tile_bounds_json.exists():
        print(f"Error: tile_bounds_tindex.json not found: {tile_bounds_json}")
        sys.exit(1)

    output_dir = params.output_dir
    if not output_dir:
        output_dir = input_dirs[0].parent / "filtered"
    output_dir = Path(output_dir)
    filtered_output_dir = output_dir / "filtered_tiles"

    workers = params.workers
    border_zone_width = params.border_zone_width
    instance_dimension = params.instance_dimension
    merge_chunk_size = params.merge_chunk_size

    print("=" * 60)
    print("Running Filter Task")
    print("=" * 60)
    print(f"  Input dirs:      {[str(d) for d in input_dirs]}")
    print(f"  Tile bounds JSON: {tile_bounds_json}")
    print(f"  Output:          {filtered_output_dir}")
    print(f"  Border zone:     {border_zone_width}m")
    print(f"  Instance dim:    {instance_dimension}")
    print(f"  Filter anchor:   {params.filter_anchor}")
    print(f"  Workers:         {workers}")

    # ── Find input tiles (all dirs combined into one flat list) ───────────────
    laz_files = []
    for _d in input_dirs:
        _found = sorted(_d.glob("*.laz")) or sorted(_d.glob("*.las"))
        laz_files.extend(_found)
    if not laz_files:
        print(f"Error: No LAZ/LAS files found in any of: {[str(d) for d in input_dirs]}")
        sys.exit(1)
    print(f"\nFound {len(laz_files)} tiles across {len(input_dirs)} input director(ies)")

    # ── Load neighbor graph from JSON ────────────────────────────────────────
    print("  Loading neighbor graph from tile_bounds_tindex.json...")
    with tile_bounds_json.open() as f:
        json_data = _json.load(f)
    json_labels = [
        f"c{int(tile['col']):02d}_r{int(tile['row']):02d}"
        if "col" in tile and "row" in tile else None
        for tile in json_data.get("tiles", [])
    ]
    bounds_field = _preferred_json_bounds_field(json_data.get("tiles", []))
    json_bounds, centers, neighbors_idx = build_neighbor_graph_from_bounds_json(
        tile_bounds_json,
        bounds_field=bounds_field,
    )

    file_boundaries = {}
    file_key_to_path: Dict[str, Path] = {}
    for f in laz_files:
        bounds = get_tile_bounds_from_header(f)
        if bounds:
            file_key = str(f)
            file_boundaries[file_key] = bounds
            file_key_to_path[file_key] = f

    tile_to_json, json_to_tile = _match_tiles_to_json_bounds(
        file_boundaries, json_bounds, centers, json_labels=json_labels
    )

    tile_name_by_key = {
        file_key: _canonical_tile_name_for_json_index(json_idx, json_labels)
        for file_key, json_idx in tile_to_json.items()
    }
    tile_name_by_path = {
        file_key_to_path[file_key]: tile_name
        for file_key, tile_name in tile_name_by_key.items()
        if file_key in file_key_to_path
    }
    json_idx_to_tile_name = {
        json_idx: tile_name_by_key[file_key]
        for file_key, json_idx in tile_to_json.items()
    }

    tile_boundaries = {
        json_idx_to_tile_name[json_idx]: json_bounds[json_idx]
        for json_idx in json_idx_to_tile_name
    }

    core_bounds_by_tile = {}
    _json_tiles = json_data.get("tiles", [])
    for json_idx, tile_name in json_idx_to_tile_name.items():
        if json_idx < len(_json_tiles) and "core" in _json_tiles[json_idx]:
            core = _json_tiles[json_idx]["core"]
            core_bounds_by_tile[tile_name] = (
                float(core[0][0]), float(core[0][1]),
                float(core[1][0]), float(core[1][1]),
            )

    neighbors_by_tile = {}
    for json_idx, tile_name in json_idx_to_tile_name.items():
        nbrs = {"east": None, "west": None, "north": None, "south": None}
        for direction in ("east", "west", "north", "south"):
            n_idx = neighbors_idx[json_idx].get(direction)
            if n_idx is not None:
                nbrs[direction] = json_idx_to_tile_name.get(n_idx)
        neighbors_by_tile[tile_name] = nbrs

    trees_by_tile: Dict[str, Path] = {}
    for filepath, tile_name in tile_name_by_path.items():
        txt_files = sorted(filepath.parent.glob("*.txt"))
        if len(txt_files) == 1:
            trees_by_tile[tile_name] = txt_files[0]
        elif len(txt_files) > 1:
            print(
                f"  Warning: multiple .txt files next to {filepath}; "
                f"skipping tree-file update for tile {tile_name}"
            )

    # ── Phase A: Metadata extraction (parallel) ──────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase A: Metadata Extraction ({workers} workers)")
    print(f"{'=' * 60}")

    work_items = []
    for tile_idx, filepath in enumerate(laz_files):
        tile_name = tile_name_by_path.get(filepath)
        if tile_name is None:
            print(f"  Warning: no matched bounds entry for {filepath}, skipping")
            continue
        nbrs = neighbors_by_tile.get(tile_name)
        core_boundary = core_bounds_by_tile.get(tile_name)
        work_items.append((
            filepath, tile_idx, tile_boundaries, tile_name, nbrs,
            border_zone_width, 0.05, instance_dimension, core_boundary, merge_chunk_size,
        ))

    tile_results = []
    if workers <= 1:
        for item in work_items:
            result = _extract_tile_metadata_wrapper(item)
            if result is not None:
                tile_results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_tile_metadata_wrapper, item): item for item in work_items}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    tile_results.append(result)
    tile_results.sort(key=lambda r: r.tile_idx)

    # ── Sort tiles north-to-south for ID assignment ───────────────────────────
    # core_bounds_by_tile stores (mn_x, mx_x, mn_y, mx_y); sort by descending
    # center Y (northernmost first), then ascending center X (west→east).
    def _ns_sort_key(r):
        bounds = core_bounds_by_tile.get(r.tile_name)
        if bounds is None:
            bounds = tile_boundaries.get(r.tile_name)
        if bounds:
            mn_x, mx_x, mn_y, mx_y = bounds
            return (-(mn_y + mx_y) / 2, (mn_x + mx_x) / 2)
        return (0.0, 0.0)

    tile_results_ns = sorted(tile_results, key=_ns_sort_key)

    # ── Phase B.5: Small instance redistribution (optional) ──────────────────
    # Build global_to_merged in north-to-south order: each kept instance gets
    # a unique sequential ID starting at 1.
    global_to_merged: dict = {}
    next_id = 1
    for result in tile_results_ns:
        for gid, meta in result.instances.items():
            if not meta.is_filtered:
                global_to_merged[gid] = next_id
                next_id += 1

    if not params.disable_volume_merge:
        print(f"\n{'=' * 60}")
        print("Phase B.5: Small Instance Redistribution")
        print(f"{'=' * 60}")
        redistribute_small_instances(
            tile_results=tile_results,
            global_to_merged=global_to_merged,
            min_points_reassign=params.min_cluster_size or 300,
            max_volume_for_merge=params.max_volume_for_merge or 5.0,
            instance_dimension=instance_dimension,
            chunk_size=max(100_000, merge_chunk_size // 4),
        )

    # ── Write filtered tiles ─────────────────────────────────────────────────
    write_filtered_tiles_streaming(
        tile_results=tile_results,
        neighbors_by_tile=neighbors_by_tile,
        core_bounds_by_tile=core_bounds_by_tile,
        output_dir=filtered_output_dir,
        instance_dimension=instance_dimension,
        chunk_size=merge_chunk_size,
        filter_anchor=params.filter_anchor,
        global_to_merged=global_to_merged,
    )

    # ── Update trees files (auto-discovered from input dirs) ─────────────────
    filtered_trees_dir = output_dir / "filtered_trees"
    print(f"\n{'=' * 60}")
    print("Updating trees files with global IDs")
    print(f"{'=' * 60}")
    print(f"  Scanning for .txt in: {input_dirs}")
    print(f"  Output:               {filtered_trees_dir}")
    update_trees_files_with_global_ids(
        tile_results=tile_results,
        global_to_merged=global_to_merged,
        trees_by_tile=trees_by_tile,
        trees_output_dir=filtered_trees_dir,
        tile_offset=TILE_OFFSET,
    )

    # ── Optional: remap filtered tiles to subsampled_res1 (if folder provided) ─
    if getattr(params, "subsampled_target_folder", None):
        try:
            from merge_tiles import remap_collections_to_original_files
        except ImportError as e:
            print(f"  Warning: could not import remap module, skipping subsampled remap: {e}")
        else:
            sub_target = Path(params.subsampled_target_folder)
            sub_output = output_dir / "subsampled_with_predictions"
            remap_dims_set = (
                {d.strip() for d in params.remap_dims.split(",") if d.strip()}
                if getattr(params, "remap_dims", None) else None
            )
            print(f"\n{'=' * 60}")
            print("Remapping filtered tiles to subsampled target")
            print(f"{'=' * 60}")
            print(f"  Source: {filtered_output_dir}")
            print(f"  Target: {sub_target}")
            print(f"  Output: {sub_output}")
            remap_collections_to_original_files(
                collections=[filtered_output_dir],
                original_input_dir=sub_target,
                output_dir=sub_output,
                tolerance=0.1,
                retile_buffer=2.0,
                chunk_size=merge_chunk_size,
                target_dims=remap_dims_set,
            )

    print()
    print("=" * 60)
    print("Filter Task Complete")
    print("=" * 60)
    print(f"Output: {filtered_output_dir}")
    print(f"Trees:  {filtered_trees_dir}")

    # Return internal state so filter_remap can reuse it without re-running Phase A
    return dict(
        tile_results=tile_results,
        global_to_merged=global_to_merged,
        neighbors_by_tile=neighbors_by_tile,
        core_bounds_by_tile=core_bounds_by_tile,
        filtered_output_dir=filtered_output_dir,
        instance_dimension=instance_dimension,
        merge_chunk_size=merge_chunk_size,
    )


def run_filter_remap_task(params: Parameters):
    """
    Filter (buffer removal + optional small instance redistribution) then remap to
    original input files. Optionally also produces a single merged.laz and
    merged_with_originals.laz.

    Required parameters:
        --segmented-folders or --segmented-remapped-folder  (input tile collections)
        --tile-bounds-json                                  (neighbor info)
        --original-input-dir                               (pre-tiling original files)
        --output-dir                                       (output location)
    Optional:
        --produce-merged-file                              (also write merged.laz)
        --segmented-folders /col1,/col2                    (multiple collections)
    """
    try:
        from merge_tiles import (
            add_original_dimensions_to_merged,
            load_standardization_dims,
            remap_collections_to_original_files,
        )
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        sys.exit(1)

    if not params.original_input_dir:
        print("Error: --original-input-dir is required for filter_remap task")
        sys.exit(1)
    original_input_dir = Path(params.original_input_dir)
    if not original_input_dir.exists():
        print(f"Error: original-input-dir not found: {original_input_dir}")
        sys.exit(1)

    # ── Parse segmented collections ──────────────────────────────────────────
    raw_folders = params.segmented_folders or ""
    collections = [Path(p.strip()) for p in raw_folders.split(",") if p.strip()]
    # Fallback to legacy single-folder params
    if not collections:
        fallback = params.segmented_remapped_folder or params.subsampled_10cm_folder
        if fallback:
            collections = [Path(fallback)]
    if not collections:
        print("Error: --segmented-folders or --segmented-remapped-folder is required for filter_remap task")
        sys.exit(1)
    for c in collections:
        if not c.exists():
            print(f"Error: segmented folder not found: {c}")
            sys.exit(1)

    output_dir = Path(params.output_dir) if params.output_dir else collections[0].parent / "filter_remap_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    remap_dims_set = (
        {d.strip() for d in params.remap_dims.split(",") if d.strip()}
        if getattr(params, "remap_dims", None) else None
    )

    print("=" * 60)
    print("Running Filter+Remap Task")
    print("=" * 60)
    print(f"  Collections:      {[str(c) for c in collections]}")
    print(f"  Original input:   {original_input_dir}")
    print(f"  Output:           {output_dir}")
    print(f"  Produce merged:   {params.produce_merged_file}")
    print(f"  Remap dims:       {remap_dims_set or 'all extra dims'}")

    # ── Step 1: Filter ALL collections together, or reuse existing output ───
    filtered_output_dir = output_dir / "filtered_tiles"
    filtered_trees_dir = output_dir / "filtered_trees"
    if _dir_has_pointcloud_outputs(filtered_output_dir):
        print(f"\n{'=' * 60}")
        print("Step 1: Reusing existing filtered tiles")
        print(f"{'=' * 60}")
        print(f"  Reusing: {filtered_output_dir}")
        if filtered_trees_dir.exists():
            print(f"  Existing filtered trees: {filtered_trees_dir}")
        merge_chunk_size = params.merge_chunk_size
    else:
        import types
        filter_params = types.SimpleNamespace(**{
            k: getattr(params, k) for k in vars(params) if not k.startswith("_")
        })
        filter_params.segmented_folders = ",".join(str(c) for c in collections)
        filter_params.segmented_remapped_folder = None
        filter_params.subsampled_10cm_folder = None
        filter_params.subsampled_target_folder = None
        filter_params.output_dir = str(output_dir)

        filter_state = run_filter_task(filter_params)
        if filter_state is None:
            print("Error: filter task failed")
            sys.exit(1)

        filtered_output_dir = filter_state["filtered_output_dir"]
        merge_chunk_size = filter_state["merge_chunk_size"]

    # ── Step 2a: Remap filtered tiles to subsampled_res1 (if provided) ───────
    sub_output = None
    if getattr(params, "subsampled_target_folder", None):
        sub_target = Path(params.subsampled_target_folder)
        sub_output = output_dir / "subsampled_with_predictions"
        print(f"\n{'=' * 60}")
        print(f"Step 2a: Remapping filtered tiles to subsampled target")
        print(f"{'=' * 60}")
        print(f"  Source: {filtered_output_dir}")
        print(f"  Target: {sub_target}")
        print(f"  Output: {sub_output}")
        if _dir_has_pointcloud_outputs(sub_output):
            print(f"  Existing remap outputs detected in {sub_output}; existing files will be skipped.")
        remap_collections_to_original_files(
            collections=[filtered_output_dir],
            original_input_dir=sub_target,
            output_dir=sub_output,
            tolerance=0.1,
            retile_buffer=2.0,
            chunk_size=merge_chunk_size,
            target_dims=remap_dims_set,
        )

    # ── Step 2b: Produce merged from subsampled output (if requested) ────────
    merged_laz = None
    if params.produce_merged_file and sub_output:
        out_files = sorted(sub_output.glob("*.laz")) + sorted(sub_output.glob("*.las"))
        if out_files:
            merged_laz = output_dir / "merged.laz"
            if merged_laz.exists():
                print(f"\nMerged output already exists, skipping concatenation: {merged_laz}")
            else:
                print(f"\nConcatenating {len(out_files)} subsampled files → {merged_laz.name}")
                _concat_laz_files(out_files, merged_laz, chunk_size=merge_chunk_size)

            if params.transfer_original_dims_to_merged:
                output_merged_with_originals = (
                    Path(params.output_merged_with_originals)
                    if params.output_merged_with_originals is not None
                    else output_dir / "merged_with_originals.laz"
                )
                target_dims = None
                if params.standardization_json is not None:
                    target_dims = load_standardization_dims(params.standardization_json)
                    print(
                        f"  Standardization: filtering merged enrichment to "
                        f"{len(target_dims)} dims from {params.standardization_json.name}"
                    )
                if output_merged_with_originals.exists():
                    print(
                        f"  Merged-with-originals output already exists, skipping enrichment: "
                        f"{output_merged_with_originals}"
                    )
                else:
                    print(
                        f"\nEnriching merged file with original dims → "
                        f"{output_merged_with_originals.name}"
                    )
                    add_original_dimensions_to_merged(
                        merged_laz,
                        original_input_dir,
                        output_merged_with_originals,
                        tolerance=0.1,
                        retile_buffer=2.0,
                        num_threads=max(1, params.workers),
                        target_dims=target_dims,
                        merge_chunk_size=merge_chunk_size,
                    )
        else:
            print("  Warning: no subsampled output files found; skipping merged output")

    # ── Step 3: Remap filtered tiles to original input files ─────────────────
    orig_predictions_dir = output_dir / "original_with_predictions"
    print(f"\n{'=' * 60}")
    print(f"Step 3: Remapping filtered tiles to original files")
    print(f"{'=' * 60}")
    print(f"  Source: {filtered_output_dir}")
    print(f"  Target: {original_input_dir}")
    print(f"  Output: {orig_predictions_dir}")
    if _dir_has_pointcloud_outputs(orig_predictions_dir):
        print(f"  Existing remap outputs detected in {orig_predictions_dir}; existing files will be skipped.")
    remap_collections_to_original_files(
        collections=[filtered_output_dir],
        original_input_dir=original_input_dir,
        output_dir=orig_predictions_dir,
        tolerance=0.1,
        retile_buffer=2.0,
        chunk_size=merge_chunk_size,
        target_dims=remap_dims_set,
    )

    print()
    print("=" * 60)
    print("Filter+Remap Task Complete")
    print("=" * 60)
    if sub_output:
        print(f"  Subsampled files:   {sub_output}")
    if merged_laz:
        print(f"  Merged:             {merged_laz}")
    print(f"  Original files:     {orig_predictions_dir}")


def _concat_laz_files(input_files, output_path, chunk_size=2_000_000):
    """
    Stream-concatenate multiple LAZ files that share the same point schema into
    a single output file.  All files must have been produced by the same pipeline
    step (e.g. remap_collections_to_original_files) so their extra-dimension VLRs
    are identical.
    """
    import laspy

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"  Output already exists, skipping concatenation: {output_path}")
        return

    with laspy.open(str(input_files[0])) as src0:
        out_header = src0.header

    written = 0
    with laspy.open(str(output_path), mode="w", header=out_header) as writer:
        for f in input_files:
            with laspy.open(str(f)) as reader:
                for chunk in reader.chunk_iterator(chunk_size):
                    writer.write_points(chunk)
                    written += len(chunk)

    print(f"  Concatenated {len(input_files)} files ({written:,} points) → {output_path.name}")


def _concat_segmented_collections_with_dim_union(
    collections,
    output_path,
    chunk_size=2_000_000,
    target_dims=None,
):
    """
    Stream-concatenate segmented collections into one LAZ with a harmonized schema.

    Unlike ``_concat_laz_files``, this helper supports multiple collections with
    colliding extra-dimension names by renaming later occurrences to ``name_2``,
    ``name_3``, etc., mirroring the multi-collection remap logic.
    """
    import laspy
    import numpy as np

    from merge_tiles import (
        extra_bytes_params_from_dimension_info,
        list_pointcloud_files,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"  Output already exists, skipping segmented merge: {output_path}")
        return

    collection_files = []
    for coll_path in collections:
        files = [coll_path] if Path(coll_path).is_file() else list_pointcloud_files(Path(coll_path))
        collection_files.append(files)

    all_input_files = [f for files in collection_files for f in files]
    if not all_input_files:
        print("  Warning: no segmented files found; skipping merged output")
        return

    seen_dim_names = {}
    collection_dim_maps = []
    for coll_path, coll_files in zip(collections, collection_files):
        dim_entries = []
        scanned_names = set()
        for cf in coll_files:
            try:
                with laspy.open(str(cf), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                    for dim in reader.header.point_format.extra_dimensions:
                        if dim.name in scanned_names:
                            continue
                        if target_dims is not None and dim.name not in target_dims:
                            continue
                        scanned_names.add(dim.name)
                        out_name = dim.name
                        count = seen_dim_names.get(dim.name, 0)
                        if count > 0:
                            out_name = f"{dim.name}_{count + 1}"
                        seen_dim_names[dim.name] = count + 1
                        dim_entries.append((
                            dim.name,
                            out_name,
                            dim.dtype,
                            extra_bytes_params_from_dimension_info(dim, name=out_name),
                        ))
            except Exception:
                continue
            if scanned_names:
                break
        collection_dim_maps.append(dim_entries)

        if dim_entries:
            dim_str = ", ".join(
                out_name if orig_name == out_name else f"{orig_name}->{out_name}"
                for orig_name, out_name, _, _ in dim_entries
            )
        else:
            dim_str = "(no extra dimensions discovered)"
        print(f"  Collection merge dims from {coll_path}: {dim_str}")

    with laspy.open(str(all_input_files[0]), laz_backend=laspy.LazBackend.LazrsParallel) as src0:
        base_header = src0.header

    out_header = laspy.LasHeader(
        point_format=base_header.point_format.id,
        version=base_header.version,
    )
    out_header.offsets = base_header.offsets
    out_header.scales = base_header.scales

    existing_out_names = set(out_header.point_format.dimension_names)
    extra_params = []
    for dim_entries in collection_dim_maps:
        for _, out_name, _, params in dim_entries:
            if out_name in existing_out_names:
                continue
            extra_params.append(params)
            existing_out_names.add(out_name)
    if extra_params:
        out_header.add_extra_dims(extra_params)

    written = 0
    out_dim_names = set(out_header.point_format.dimension_names)
    std_out_names = {
        name for name in out_dim_names
        if not any(name == params.name for params in extra_params)
    }

    with laspy.open(
        str(output_path), mode="w", header=out_header,
        laz_backend=laspy.LazBackend.LazrsParallel, do_compress=True,
    ) as writer:
        for coll_idx, (coll_path, coll_files, dim_entries) in enumerate(
            zip(collections, collection_files, collection_dim_maps),
            start=1,
        ):
            if not coll_files:
                print(f"  Collection {coll_idx}: no files found in {coll_path}, skipping")
                continue
            print(f"  Collection {coll_idx}: concatenating {len(coll_files)} file(s)")
            for input_file in coll_files:
                with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                    chunk_dims = set(reader.header.point_format.dimension_names)
                    common_std_dims = chunk_dims & std_out_names
                    for chunk in reader.chunk_iterator(chunk_size):
                        out_record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=out_header)
                        for dim_name in common_std_dims:
                            try:
                                out_record[dim_name] = chunk[dim_name]
                            except Exception:
                                pass
                        for orig_name, out_name, dtype, _ in dim_entries:
                            values = getattr(chunk, orig_name, None)
                            if values is None:
                                out_record[out_name] = np.zeros(len(chunk), dtype=dtype)
                            else:
                                out_record[out_name] = np.asarray(values).astype(dtype, copy=False)
                        writer.write_points(out_record)
                        written += len(chunk)

    print(
        f"  Concatenated {len(all_input_files)} segmented file(s) "
        f"({written:,} points) → {output_path.name}"
    )


def _dir_has_pointcloud_outputs(directory: Path) -> bool:
    """Return True when a directory already contains LAS/LAZ outputs."""
    return directory.exists() and any(directory.glob("*.la[sz]"))


def _describe_laz_dimensions(laz_path: Path) -> str:
    """Return a compact string of LAS dimension names for logging."""
    import laspy

    with laspy.open(str(laz_path)) as f:
        dims = list(f.header.point_format.dimension_names)
    return ", ".join(str(d) for d in dims)


def run_remap_task(params: Parameters):
    """
    Run the remap task.

    Two modes:
      1. Single merged LAZ → original files (legacy, requires --merged-laz)
      2. N segmented folders → original files (multi-collection, requires --segmented-folders
         or --segmented-remapped-folder and --original-input-dir)
    """
    try:
        from merge_tiles import (
            add_original_dimensions_to_merged,
            load_standardization_dims,
            remap_to_original_input_files_streaming,
            remap_collections_to_original_files,
        )
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        sys.exit(1)

    # ── Multi-collection mode ─────────────────────────────────────────────────
    raw_folders = params.segmented_folders or ""
    collections = [Path(p.strip()) for p in raw_folders.split(",") if p.strip()]
    if not collections and (params.segmented_remapped_folder or params.subsampled_10cm_folder):
        fallback = params.segmented_remapped_folder or params.subsampled_10cm_folder
        collections = [Path(fallback)]

    if collections:
        if not params.original_input_dir:
            print("Error: --original-input-dir is required for multi-collection remap")
            sys.exit(1)
        original_input_dir = Path(params.original_input_dir)
        if not original_input_dir.exists():
            print(f"Error: original-input-dir not found: {original_input_dir}")
            sys.exit(1)
        output_dir = Path(params.output_dir) if params.output_dir else original_input_dir.parent / "remap_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        merge_chunk_size = params.merge_chunk_size

        remap_dims_set = (
            {d.strip() for d in params.remap_dims.split(",") if d.strip()}
            if getattr(params, "remap_dims", None) else None
        )

        print("=" * 60)
        print("Remap: segmented collections -> subsampled + original files")
        print("=" * 60)
        print(f"  Collections:  {[str(c) for c in collections]}")
        print(f"  Originals:    {original_input_dir}")
        print(f"  Output:       {output_dir}")
        print(f"  Remap dims:   {remap_dims_set or 'all extra dims'}")

        # ── Step 1: Remap to subsampled_res1 (if provided) ───────────────────
        sub_output = None
        if getattr(params, "subsampled_target_folder", None):
            sub_target = Path(params.subsampled_target_folder)
            sub_output = output_dir / "subsampled_with_predictions"
            print(f"\nStep 1: Remapping collections to subsampled target → {sub_output}")
            if _dir_has_pointcloud_outputs(sub_output):
                print(f"  Existing remap outputs detected in {sub_output}; existing files will be skipped.")
            remap_collections_to_original_files(
                collections=collections,
                original_input_dir=sub_target,
                output_dir=sub_output,
                tolerance=0.1,
                retile_buffer=2.0,
                chunk_size=merge_chunk_size,
                target_dims=remap_dims_set,
            )

        # ── Step 2: Produce merged file from subsampled output ───────────────
        if params.produce_merged_file and sub_output:
            out_files = sorted(sub_output.glob("*.laz")) + sorted(sub_output.glob("*.las"))
            if out_files:
                merged_all = output_dir / "merged_with_all_dims.laz"
                if merged_all.exists():
                    print(f"\nStep 2: merged output already exists, skipping concatenation: {merged_all}")
                else:
                    print(f"\nStep 2: Concatenating {len(out_files)} subsampled files → {merged_all.name}")
                    _concat_laz_files(out_files, merged_all, chunk_size=merge_chunk_size)
                print(f"  Final merged dims: {_describe_laz_dimensions(merged_all)}")
            else:
                print("  Warning: no subsampled output files found; skipping merged output")

        # ── Step 3: Remap to original input files ────────────────────────────
        orig_output = output_dir / "original_with_predictions"
        print(f"\nStep 3: Remapping collections to original files → {orig_output}")
        if _dir_has_pointcloud_outputs(orig_output):
            print(f"  Existing remap outputs detected in {orig_output}; existing files will be skipped.")
        remap_collections_to_original_files(
            collections=collections,
            original_input_dir=original_input_dir,
            output_dir=orig_output,
            tolerance=0.1,
            retile_buffer=2.0,
            chunk_size=merge_chunk_size,
            target_dims=remap_dims_set,
        )

        # ── Step 4: Fallback merged file directly from segmented collections ──
        if params.produce_merged_file and not sub_output:
            merged_all = output_dir / "merged_with_all_dims.laz"
            if merged_all.exists():
                print(f"\nStep 4: merged output already exists, skipping segmented concatenation: {merged_all}")
            else:
                print(
                    f"\nStep 4: No subsampled target provided; "
                    f"concatenating segmented collections → {merged_all.name}"
                )
                _concat_segmented_collections_with_dim_union(
                    collections=collections,
                    output_path=merged_all,
                    chunk_size=merge_chunk_size,
                    target_dims=remap_dims_set,
                )
            if merged_all.exists():
                print(f"  Final merged dims: {_describe_laz_dimensions(merged_all)}")

        # ── Option: enrich a pre-existing merged LAZ with original dims ──────
        if params.transfer_original_dims_to_merged and params.merged_laz:
            merged_laz_path = Path(params.merged_laz)
            if merged_laz_path.exists():
                output_merged_with_originals = (
                    Path(params.output_merged_with_originals)
                    if params.output_merged_with_originals
                    else output_dir / "merged_with_originals.laz"
                )
                _target_dims = None
                if params.standardization_json:
                    _target_dims = load_standardization_dims(params.standardization_json)
                if output_merged_with_originals.exists():
                    print(
                        f"\nMerged-with-originals output already exists, skipping enrichment: "
                        f"{output_merged_with_originals}"
                    )
                else:
                    print(f"\nEnriching existing merged LAZ with original dims → {output_merged_with_originals.name}")
                    add_original_dimensions_to_merged(
                        merged_laz_path, original_input_dir, output_merged_with_originals,
                        tolerance=0.1, retile_buffer=2.0,
                        num_threads=max(1, params.workers),
                        target_dims=_target_dims,
                        merge_chunk_size=merge_chunk_size,
                    )

        print()
        print("=" * 60)
        print("Remap complete")
        print("=" * 60)
        if sub_output:
            print(f"  Subsampled files:   {sub_output}")
        elif params.produce_merged_file:
            print(f"  Merged:             {output_dir / 'merged_with_all_dims.laz'}")
        print(f"  Original files:     {orig_output}")
        return

    # ── Legacy single merged-LAZ mode ────────────────────────────────────────
    if not params.merged_laz:
        print("Error: --merged-laz (or --segmented-folders) is required for remap task")
        sys.exit(1)
    if not params.original_input_dir:
        print("Error: --original-input-dir is required for remap task")
        sys.exit(1)

    merged_laz = Path(params.merged_laz)
    original_input_dir = Path(params.original_input_dir)
    if not merged_laz.exists():
        print(f"Error: Merged file not found: {merged_laz}")
        sys.exit(1)
    if not original_input_dir.exists():
        print(f"Error: Original input directory not found: {original_input_dir}")
        sys.exit(1)

    output_dir = params.output_dir
    if output_dir is None:
        output_dir = original_input_dir.parent / "original_with_predictions"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, params.workers)
    retile_buffer = 2.0
    tolerance = 0.1

    print("=" * 60)
    print("Remap: merged file -> original files")
    print("=" * 60)
    print(f"Merged file: {merged_laz}")
    print(f"Original input dir: {original_input_dir}")
    print(f"Output dir: {output_dir}")
    print()

    # Parse 3DTrees dimension branding params
    threedtrees_dims = [d.strip() for d in params.threedtrees_dims.split(",") if d.strip()] if params.threedtrees_dims else None
    threedtrees_suffix = params.threedtrees_suffix

    _target_dims = None
    if params.standardization_json is not None:
        _target_dims = load_standardization_dims(params.standardization_json)
        print(f"  Standardization: filtering to {len(_target_dims)} dims from {params.standardization_json.name}")
    remap_to_original_input_files_streaming(
        merged_file=merged_laz,
        original_input_dir=original_input_dir,
        output_dir=output_dir,
        tolerance=tolerance,
        retile_buffer=retile_buffer,
        threedtrees_dims=threedtrees_dims,
        threedtrees_suffix=threedtrees_suffix,
        target_dims=_target_dims,
    )

    # Add dimensions from original files to the merged file (optional)
    if params.transfer_original_dims_to_merged:
        # Compute target dims from standardization JSON if provided
        output_merged_with_originals = (
            Path(params.output_merged_with_originals)
            if params.output_merged_with_originals is not None
            else output_dir / "merged_with_originals.laz"
        )
        add_original_dimensions_to_merged(
            merged_laz,
            original_input_dir,
            output_merged_with_originals,
            tolerance=tolerance,
            retile_buffer=retile_buffer,
            num_threads=max(1, params.workers),
            target_dims=_target_dims,
        )
    else:
        print("  Skipping transfer of original dimensions to merged file (disabled).")

    print()
    print("Remap complete.")


def preprocess_boolean_flags(args_list):
    """
    Preprocess CLI args to convert boolean flags to explicit True/False for Pydantic.
    Pydantic expects --flag True/False, but we want --flag to work like argparse.
    """
    boolean_flags = [
        '--show-params', '--show_params',
        '--skip-dimension-reduction', '--skip_dimension_reduction',
        '--chunkwise-copc-source-creation', '--chunkwise_copc_source_creation',
        '--disable-matching', '--disable_matching',
        '--disable-volume-merge', '--disable_volume_merge',
        '--skip-merged-file', '--skip_merged_file',
        '--produce-merged-file', '--produce_merged_file',
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

    # Manually map aliases that Pydantic might not generate flags for
    # --subsampled-segmented-folder -> --subsampled-10cm-folder
    # --no-transfer-original-dims-to-merged -> --transfer-original-dims-to-merged False
    mapped_args = []
    for arg in remaining_args:
        if arg == '--subsampled-segmented-folder':
            mapped_args.append('--subsampled-10cm-folder')
        elif arg == '--no-transfer-original-dims-to-merged':
            mapped_args.extend(['--transfer-original-dims-to-merged', 'False'])
        else:
            mapped_args.append(arg)
    remaining_args = mapped_args

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

    # Show parameters if requested (flag handled by pre-parser; not in Parameters)
    if pre_args.show_params:
        print_params(params)
        sys.exit(0)

    # Task is required if not showing params
    if not params.task:
        print("Error: --task is required (unless using --show-params)")
        print("Usage: python run.py --task tile --input-dir /path/to/input --output-dir /path/to/output")
        print("       python run.py --task merge --subsampled-10cm-folder /path/to/10cm")
        print("       python run.py --task remap --merged-laz /path/to/merged.laz --original-input-dir /path/to/originals")
        print("       python run.py --task filter --segmented-remapped-folder /path/to/tiles --tile-bounds-json /path/to/tindex.json --output-dir /path/to/output")
        print("       python run.py --task filter_remap --segmented-folders /col1,/col2 --tile-bounds-json /path/to/tindex.json --original-input-dir /path/to/originals --output-dir /path/to/output")
        print("       python run.py --show-params")
        sys.exit(1)

    # Route to appropriate task function
    if params.task == "tile":
        run_tile_task(params)
    elif params.task == "merge":
        run_merge_task(params)
    elif params.task == "remap":
        run_remap_task(params)
    elif params.task == "filter":
        run_filter_task(params)
    elif params.task == "filter_remap":
        run_filter_remap_task(params)
    else:
        print(f"Error: Unknown task: {params.task}")
        print("Valid tasks: tile, merge, remap, filter, filter_remap")
        sys.exit(1)


if __name__ == "__main__":
    main()
