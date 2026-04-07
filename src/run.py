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
from typing import Dict, List, Optional, Set, Tuple

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
    """Thin dispatcher for the extracted filter task orchestration."""
    try:
        from filter_task import FilterTaskDependencies, run_filter_task as _run_filter_task_impl
    except ImportError as e:
        print(f"Error: Could not import filter_task.py: {e}")
        sys.exit(1)

    deps = FilterTaskDependencies(
        update_trees_files_with_global_ids=update_trees_files_with_global_ids,
        pointcloud_key_fn=_pointcloud_file_key,
        convert_collection_file_to_copc=_convert_collection_file_to_copc,
    )
    return _run_filter_task_impl(params, deps)


def run_filter_remap_task(params: Parameters):
    """Thin dispatcher for the extracted filter_remap task orchestration."""
    try:
        from filter_remap_task import (
            FilterRemapTaskDependencies,
            run_filter_remap_task as _run_filter_remap_task_impl,
        )
    except ImportError as e:
        print(f"Error: Could not import filter_remap_task.py: {e}")
        sys.exit(1)

    deps = FilterRemapTaskDependencies(
        run_filter_task=run_filter_task,
        concat_laz_files=_concat_laz_files,
        describe_laz_dimensions=_describe_laz_dimensions,
        pointcloud_key_fn=_pointcloud_file_key,
        convert_collection_file_to_copc=_convert_collection_file_to_copc,
    )
    return _run_filter_remap_task_impl(params, deps)


def _concat_laz_files(input_files, output_path, chunk_size=2_000_000):
    """
    Stream-concatenate multiple LAZ files that share the same point schema into
    a single output file.  All files must have been produced by the same pipeline
    step (e.g. remap_collections_to_original_files) so their extra-dimension VLRs
    are identical.
    """
    import laspy
    from merge_tiles import extra_bytes_params_from_dimension_info

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"  Output already exists, skipping concatenation: {output_path}")
        return

    with laspy.open(str(input_files[0])) as src0:
        src_header = src0.header
        out_header = laspy.LasHeader(
            point_format=src_header.point_format.id,
            version=src_header.version,
        )
        out_header.offsets = src_header.offsets
        out_header.scales = src_header.scales
        if src_header.point_format.extra_dimensions:
            out_header.add_extra_dims([
                extra_bytes_params_from_dimension_info(dim)
                for dim in src_header.point_format.extra_dimensions
            ])

        existing_vlr_keys = {
            (getattr(v, "user_id", ""), getattr(v, "record_id", None))
            for v in out_header.vlrs
        }
        for vlr in src_header.vlrs:
            user_id = getattr(vlr, "user_id", "")
            record_id = getattr(vlr, "record_id", None)
            # Skip COPC and ExtraBytes VLRs; laspy regenerates the latter from the header.
            if (record_id in (1, 2) and user_id == "copc") or (record_id == 4 and user_id == "LASF_Spec"):
                continue
            key = (user_id, record_id)
            if key not in existing_vlr_keys:
                out_header.vlrs.append(vlr)
                existing_vlr_keys.add(key)

    written = 0
    with laspy.open(
        str(output_path), mode="w", header=out_header,
        do_compress=output_path.suffix.lower() == ".laz",
        laz_backend=laspy.LazBackend.LazrsParallel,
    ) as writer:
        for f in input_files:
            with laspy.open(str(f), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
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


def _rewrite_invalid_extra_dim_names_for_copc(
    input_path,
    output_path,
    chunk_size=2_000_000,
):
    """
    Rewrite a LAS/LAZ file so invalid extra-dimension names become COPC-safe.

    Current sanitation rules:
    - replace non ``[A-Za-z0-9_]`` chars with ``_``
    - require the first char to be a letter; otherwise prefix ``TDT_``
    - if the source name starts with ``3DT_``, rewrite that prefix to ``TDT_``
    - de-duplicate sanitized names with numeric suffixes
    """
    import laspy
    import numpy as np
    import re

    from merge_tiles import extra_bytes_params_from_dimension_info

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(input_path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        header = reader.header
        extra_dims = list(header.point_format.extra_dimensions)
        existing_names = {dim.name for dim in extra_dims}
        used_names = set(existing_names)
        rename_map = {}

        def _sanitize_dim_name(name):
            candidate = name
            if candidate.startswith("3DT_"):
                candidate = f"TDT_{candidate[4:]}"
            candidate = re.sub(r"[^A-Za-z0-9_]", "_", candidate)
            if not candidate or not candidate[0].isalpha():
                candidate = f"TDT_{candidate.lstrip('_')}" if candidate else "TDT_dim"
            candidate = re.sub(r"_+", "_", candidate)
            candidate = candidate.rstrip("_") or "TDT_dim"

            if candidate == name:
                return candidate

            base = candidate
            suffix = 2
            while candidate in used_names:
                candidate = f"{base}_{suffix}"
                suffix += 1
            return candidate

        for dim in extra_dims:
            sanitized = _sanitize_dim_name(dim.name)
            if sanitized != dim.name:
                rename_map[dim.name] = sanitized
                used_names.add(sanitized)

        if not rename_map:
            print(f"      Dimension-name scan: no invalid extra dims in {input_path.name}")
            return False, {}

        print(
            f"      Dimension-name scan: found {len(rename_map)} invalid/copc-unsafe "
            f"extra dim(s) in {input_path.name}",
            flush=True,
        )
        for src_name, dst_name in sorted(rename_map.items()):
            print(f"        rename: {src_name} -> {dst_name}", flush=True)
        print(f"      Writing temporary sanitized copy: {output_path}", flush=True)

        out_header = laspy.LasHeader(
            point_format=header.point_format.id,
            version=header.version,
        )
        out_header.offsets = header.offsets
        out_header.scales = header.scales

        existing_vlr_keys = {
            (getattr(v, "user_id", ""), getattr(v, "record_id", None))
            for v in out_header.vlrs
        }
        for vlr in header.vlrs:
            user_id = getattr(vlr, "user_id", "")
            record_id = getattr(vlr, "record_id", None)
            # Skip COPC and ExtraBytes VLRs; laspy regenerates the latter from the header.
            if (record_id in (1, 2) and user_id == "copc") or (record_id == 4 and user_id == "LASF_Spec"):
                continue
            key = (user_id, record_id)
            if key not in existing_vlr_keys:
                out_header.vlrs.append(vlr)
                existing_vlr_keys.add(key)

        for dim in extra_dims:
            out_header.add_extra_dim(
                extra_bytes_params_from_dimension_info(
                    dim,
                    name=rename_map.get(dim.name, dim.name),
                )
            )

        with laspy.open(
            str(output_path), mode="w", header=out_header,
            do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel,
        ) as writer:
            std_dims = list(header.point_format.dimension_names)
            for chunk in reader.chunk_iterator(chunk_size):
                out_record = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=out_header)
                for dim_name in std_dims:
                    try:
                        out_record[dim_name] = np.asarray(chunk[dim_name])
                    except Exception:
                        pass
                for dim in extra_dims:
                    src_name = dim.name
                    dst_name = rename_map.get(src_name, src_name)
                    out_record[dst_name] = np.asarray(getattr(chunk, src_name))
                writer.write_points(out_record)

    print(f"      Temporary sanitized copy ready: {output_path.name}", flush=True)
    return True, rename_map


def _convert_collection_file_to_copc(
    input_path,
    output_copc,
    chunk_size,
):
    """Convert one collection file to COPC, sanitizing invalid dim names first."""
    import tempfile

    from main_tile import _convert_laz_to_copc

    input_path = Path(input_path)
    output_copc = Path(output_copc)
    print(f"      COPC conversion prep: scanning {input_path.name}", flush=True)
    with tempfile.TemporaryDirectory(prefix=f"{input_path.stem}_tdt_") as tmp_dir_str:
        sanitized_input = Path(tmp_dir_str) / input_path.name
        renamed, rename_map = _rewrite_invalid_extra_dim_names_for_copc(
            input_path=input_path,
            output_path=sanitized_input,
            chunk_size=chunk_size,
        )
        source_for_conversion = sanitized_input if renamed else input_path
        if renamed:
            print(
                f"      COPC conversion input: temporary sanitized copy "
                f"({source_for_conversion.name})",
                flush=True,
            )
            chunkwise_source_creation = True
        else:
            print(f"      COPC conversion input: original file ({input_path.name})", flush=True)
            print("      No rename needed: using direct LAZ->COPC conversion", flush=True)
            chunkwise_source_creation = False
        success, retry_message = _convert_laz_to_copc(
            input_laz=source_for_conversion,
            output_copc=output_copc,
            chunk_size=chunk_size,
            chunkwise_source_creation=chunkwise_source_creation,
        )
        if success:
            if renamed:
                mapped = ", ".join(f"{src}->{dst}" for src, dst in sorted(rename_map.items()))
                message = f"{retry_message}; renamed dims for COPC: {mapped}"
                print(f"      COPC conversion OK: {message}", flush=True)
                return True, message
            print(f"      COPC conversion OK: {retry_message}", flush=True)
            return True, retry_message
        print(f"      COPC conversion FAILED: {retry_message}", flush=True)
        return False, retry_message


def _ensure_collections_copc(
    collections,
    output_dir,
    chunk_size=20_000_000,
):
    """
    Ensure each segmented collection is available as COPC.

    Returns a list of normalized collection paths. Existing COPC collections are
    reused directly; mixed/plain collections are converted into a persistent
    cache under ``output_dir / source_collections_copc``.
    """
    import shutil

    from merge_tiles import list_pointcloud_files

    cache_root = Path(output_dir) / "source_collections_copc"
    cache_root.mkdir(parents=True, exist_ok=True)

    normalized = []
    for idx, coll in enumerate(collections, start=1):
        coll_path = Path(coll)
        if coll_path.is_file():
            if coll_path.name.endswith(".copc.laz"):
                print(f"  Collection {idx}: using existing COPC file {coll_path}")
                normalized.append(coll_path)
                continue

            dest_dir = cache_root / f"collection_{idx:02d}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{coll_path.stem}.copc.laz"
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(f"  Collection {idx}: reusing cached COPC {dest_path.name}")
            else:
                print(f"  Collection {idx}: converting {coll_path.name} -> {dest_path.name}")
                success, message = _convert_collection_file_to_copc(
                    input_path=coll_path,
                    output_copc=dest_path,
                    chunk_size=chunk_size,
                )
                if not success:
                    print(
                        f"  Collection {idx}: COPC conversion failed for {coll_path.name}; "
                        f"falling back to original source scan ({message})"
                    )
                    normalized.append(coll_path)
                    continue
            normalized.append(dest_path)
            continue

        coll_files = list_pointcloud_files(coll_path)
        if not coll_files:
            raise ValueError(f"No point cloud files found in collection: {coll_path}")

        if all(path.name.endswith(".copc.laz") for path in coll_files):
            print(f"  Collection {idx}: using existing COPC collection {coll_path}")
            normalized.append(coll_path)
            continue

        dest_dir = cache_root / f"collection_{idx:02d}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Collection {idx}: normalizing {len(coll_files)} file(s) to COPC in {dest_dir}")
        conversion_failed = False
        reused_count = 0
        converted_count = 0
        copied_copc_count = 0

        for src in coll_files:
            if src.name.endswith(".copc.laz"):
                dest_path = dest_dir / src.name
                if not dest_path.exists() or dest_path.stat().st_size == 0:
                    print(f"    reusing source COPC by copying {src.name} -> {dest_path.name}")
                    shutil.copy2(src, dest_path)
                    copied_copc_count += 1
                else:
                    print(f"    already converted/copied: {dest_path.name}")
                    reused_count += 1
                continue

            dest_path = dest_dir / f"{src.stem}.copc.laz"
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(f"    already converted: {src.name} -> {dest_path.name}")
                reused_count += 1
                continue

            print(f"    {src.name} -> {dest_path.name}")
            success, message = _convert_collection_file_to_copc(
                input_path=src,
                output_copc=dest_path,
                chunk_size=chunk_size,
            )
            if not success:
                print(
                    f"  Collection {idx}: COPC conversion failed for {src.name}; "
                    f"falling back to original collection scan ({message})"
                )
                conversion_failed = True
                break
            converted_count += 1

        if conversion_failed:
            normalized.append(coll_path)
        else:
            print(
                f"  Collection {idx} COPC cache summary: "
                f"{converted_count} converted, {reused_count} reused, "
                f"{copied_copc_count} copied from source COPC",
                flush=True,
            )
            normalized.append(dest_dir)

    return normalized


def _pointcloud_file_key(path: Path) -> str:
    """Return a stable tile key shared by `.laz` and `.copc.laz` variants."""
    path = Path(path)
    return path.name[:-9] if path.name.endswith(".copc.laz") else path.stem


def _fusion_key_dims() -> List[str]:
    """Return the point-identity key used for attribute fusion."""
    return ["X", "Y", "Z"]


def _compute_alignment_order(las_data, key_dims: List[str]):
    """Compute a stable row order for order-insensitive aligned fusion."""
    import numpy as np

    if not key_dims:
        return np.arange(len(las_data.points), dtype=np.int64)
    sort_keys = [np.asarray(las_data[dim_name]) for dim_name in key_dims]
    return np.lexsort(tuple(reversed(sort_keys)))


def _build_fused_tile_header(ref_header, collection_meta):
    """Build the output header for one fused source tile."""
    import laspy

    out_header = laspy.LasHeader(
        point_format=ref_header.point_format.id,
        version=ref_header.version,
    )
    out_header.offsets = ref_header.offsets
    out_header.scales = ref_header.scales

    existing_out_names = set(out_header.point_format.dimension_names)
    extra_params = []
    for coll_meta in collection_meta:
        for _, out_name, dtype, ebp in coll_meta["dim_entries"]:
            if out_name in existing_out_names:
                continue
            extra_params.append(laspy.ExtraBytesParams(
                name=out_name,
                type=dtype,
                description=ebp.description or "",
            ))
            existing_out_names.add(out_name)
    if extra_params:
        out_header.add_extra_dims(extra_params)

    base_copy_dims = [
        dim_name for dim_name in ref_header.point_format.dimension_names
        if dim_name in existing_out_names
    ]
    return out_header, base_copy_dims


def _merge_1d_intervals(
    intervals: List[Tuple[float, float]],
    abs_tol: float = 1e-9,
) -> List[Tuple[float, float]]:
    """Merge overlapping/touching 1D intervals."""
    if not intervals:
        return []
    ordered = sorted((float(lo), float(hi)) for lo, hi in intervals)
    merged: List[Tuple[float, float]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        cur_lo, cur_hi = merged[-1]
        if lo <= cur_hi + abs_tol:
            merged[-1] = (cur_lo, max(cur_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _annotate_canonical_tile_edge_neighbors(
    canonical_tiles,
    abs_tol: float = 1e-9,
):
    """Annotate canonical tiles with local edge spans covered by east/north neighbors."""
    import math

    for tile_info in canonical_tiles:
        xmin, xmax, ymin, ymax = tile_info["bounds"]
        east_spans: List[Tuple[float, float]] = []
        north_spans: List[Tuple[float, float]] = []
        for other in canonical_tiles:
            if other is tile_info:
                continue
            oxmin, oxmax, oymin, oymax = other["bounds"]
            if math.isclose(oxmin, xmax, rel_tol=0.0, abs_tol=abs_tol):
                lo = max(ymin, oymin)
                hi = min(ymax, oymax)
                if hi >= lo - abs_tol:
                    east_spans.append((lo, hi))
            if math.isclose(oymin, ymax, rel_tol=0.0, abs_tol=abs_tol):
                lo = max(xmin, oxmin)
                hi = min(xmax, oxmax)
                if hi >= lo - abs_tol:
                    north_spans.append((lo, hi))
        tile_info["east_neighbor_spans"] = _merge_1d_intervals(east_spans, abs_tol=abs_tol)
        tile_info["north_neighbor_spans"] = _merge_1d_intervals(north_spans, abs_tol=abs_tol)
    return canonical_tiles


def _upper_edge_membership_mask(
    coords,
    upper_bound: float,
    neighbor_spans: List[Tuple[float, float]],
    orthogonal_coords,
    abs_tol: float = 1e-9,
):
    """Return membership on an upper tile edge with local-neighbor ownership."""
    import numpy as np

    coords = np.asarray(coords, dtype=np.float64)
    orthogonal_coords = np.asarray(orthogonal_coords, dtype=np.float64)
    inside_mask = coords < (float(upper_bound) - abs_tol)
    on_edge_mask = np.abs(coords - float(upper_bound)) <= abs_tol
    if not np.any(on_edge_mask):
        return inside_mask
    if not neighbor_spans:
        return inside_mask | on_edge_mask

    covered_by_neighbor = np.zeros(len(coords), dtype=bool)
    for lo, hi in neighbor_spans:
        covered_by_neighbor |= (
            (orthogonal_coords >= float(lo) - abs_tol)
            & (orthogonal_coords <= float(hi) + abs_tol)
        )
    return inside_mask | (on_edge_mask & ~covered_by_neighbor)


class _StreamingCopcTilePartsWriter:
    """Stage LAS parts for one fused tile, then finalize directly to COPC."""

    def __init__(
        self,
        final_copc: Path,
        header_snapshot,
        label: str,
        parts_root: Path,
    ) -> None:
        from main_tile import _make_tile_header

        self.final_copc = Path(final_copc)
        self.label = label
        self._make_tile_header = _make_tile_header
        self.parts_dir = Path(parts_root) / label
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.header_snapshot = header_snapshot
        self.part_paths: List[Path] = []
        self.part_idx = 0

    def write_points(self, points) -> int:
        import laspy

        if len(points) == 0:
            return 0
        part_path = self.parts_dir / f"part_{self.part_idx:05d}.las"
        part_header = self._make_tile_header(self.header_snapshot)
        with laspy.open(str(part_path), mode="w", header=part_header) as writer:
            writer.write_points(points)
        self.part_paths.append(part_path)
        self.part_idx += 1
        return len(points)

    def finalize(self) -> Tuple[bool, str]:
        from main_tile import _finalize_tile_to_copc_untwine

        if not self.part_paths:
            return (False, "no staged LAS parts")
        success, message = _finalize_tile_to_copc_untwine(
            parts=self.part_paths,
            final_tile=self.final_copc,
            log_dir=self.parts_dir,
            label=self.label,
        )
        if not success:
            self.final_copc.unlink(missing_ok=True)
        return (success, message)


def _try_streaming_fuse_aligned_tile(
    source_files,
    collection_meta,
    temp_las,
    chunk_size,
):
    """
    Try low-memory streaming fusion assuming all collections already share row order.

    Returns `(did_fuse, ref_count, reason)`. When `did_fuse` is False, the caller
    can fall back to the more expensive order-insensitive alignment path.
    """
    import laspy
    import numpy as np

    readers = []
    try:
        readers = [
            laspy.open(str(src), laz_backend=laspy.LazBackend.LazrsParallel)
            for src in source_files
        ]
        headers = [reader.header for reader in readers]
        ref_header = headers[0]
        ref_count = int(ref_header.point_count)

        for idx, header in enumerate(headers[1:], start=2):
            if int(header.point_count) != ref_count:
                raise ValueError(
                    f"Point-count mismatch for tile {Path(source_files[0]).stem}: "
                    f"collection 1 has {ref_count:,}, collection {idx} has {int(header.point_count):,}"
                )

        out_header, base_copy_dims = _build_fused_tile_header(ref_header, collection_meta)

        with laspy.open(
            str(temp_las), mode="w", header=out_header,
            do_compress=False,
        ) as writer:
            chunk_iters = [reader.chunk_iterator(chunk_size) for reader in readers]
            total_written = 0
            chunk_index = 0

            for ref_chunk in chunk_iters[0]:
                chunk_index += 1
                chunks = [ref_chunk]
                for coll_idx, chunk_iter in enumerate(chunk_iters[1:], start=2):
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration as exc:
                        raise ValueError(
                            f"Collection {coll_idx} ended early at chunk {chunk_index}"
                        ) from exc
                    if len(chunk) != len(ref_chunk):
                        raise ValueError(
                            f"Chunk-size mismatch at chunk {chunk_index} "
                            f"(collection 1={len(ref_chunk):,}, collection {coll_idx}={len(chunk):,})"
                        )
                    if (
                        not np.array_equal(np.asarray(ref_chunk.X), np.asarray(chunk.X))
                        or not np.array_equal(np.asarray(ref_chunk.Y), np.asarray(chunk.Y))
                        or not np.array_equal(np.asarray(ref_chunk.Z), np.asarray(chunk.Z))
                    ):
                        return False, ref_count, f"row-order mismatch at chunk {chunk_index}"
                    chunks.append(chunk)

                out_record = laspy.ScaleAwarePointRecord.zeros(len(ref_chunk), header=out_header)
                for dim_name in base_copy_dims:
                    try:
                        out_record[dim_name] = ref_chunk[dim_name]
                    except Exception:
                        pass

                for chunk, coll_meta in zip(chunks, collection_meta):
                    for orig_name, out_name, dtype, _ in coll_meta["dim_entries"]:
                        values = getattr(chunk, orig_name, None)
                        if values is None:
                            out_record[out_name] = np.zeros(len(chunk), dtype=dtype)
                        else:
                            out_record[out_name] = np.asarray(values).astype(dtype, copy=False)

                writer.write_points(out_record)
                total_written += len(ref_chunk)

            for coll_idx, chunk_iter in enumerate(chunk_iters[1:], start=2):
                try:
                    extra_chunk = next(chunk_iter)
                    raise ValueError(
                        f"Collection {coll_idx} has trailing extra data after {total_written:,} "
                        f"fused points ({len(extra_chunk):,} extra points)"
                    )
                except StopIteration:
                    pass

        return True, ref_count, "streaming exact-order fusion"
    finally:
        for reader in readers:
            try:
                reader.close()
            except Exception:
                pass


def _try_copc_spatial_fuse_aligned_tile(
    source_files,
    collection_meta,
    temp_las,
    chunk_size,
    slice_target_points,
    work_dir=None,
):
    """
    Try low-RAM aligned fusion by spatially slicing COPC tiles.

    This path is intended for tiles whose point sets are identical but whose row
    order no longer matches. It uses COPC spatial queries to load one spatial
    slice at a time, aligns points within that slice by exact integer ``X/Y/Z``,
    and writes the fused output incrementally.

    Returns ``(did_fuse, ref_count, reason)``.
    """
    import laspy
    import numpy as np

    from merge_tiles import (
        _build_spatial_slices,
        _load_copc_subset_all_dims,
        _snap_spatial_slices_to_header_grid,
    )

    if not source_files or not all(Path(src).name.endswith(".copc.laz") for src in source_files):
        return False, 0, "COPC spatial alignment requires COPC tiles"

    headers = []
    for src in source_files:
        with laspy.open(str(src), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
            headers.append(reader.header)

    ref_header = headers[0]
    ref_count = int(ref_header.point_count)
    for idx, header in enumerate(headers[1:], start=2):
        if int(header.point_count) != ref_count:
            raise ValueError(
                f"Point-count mismatch for tile {Path(source_files[0]).stem}: "
                f"collection 1 has {ref_count:,}, collection {idx} has {int(header.point_count):,}"
            )

    key_dims = _fusion_key_dims()
    for idx, header in enumerate(headers[1:], start=2):
        other_key_dims = [d for d in key_dims if d in header.point_format.dimension_names]
        if other_key_dims != key_dims:
            raise ValueError(
                f"XYZ-dimension mismatch for tile {Path(source_files[0]).stem}: "
                f"collection 1 dims {key_dims} vs collection {idx} dims {other_key_dims}"
            )

    out_header, base_copy_dims = _build_fused_tile_header(ref_header, collection_meta)
    ref_bounds = (
        float(ref_header.x_min),
        float(ref_header.x_max),
        float(ref_header.y_min),
        float(ref_header.y_max),
    )
    slice_specs = _build_spatial_slices(
        ref_bounds,
        slice_count=50,
        target_points=max(int(slice_target_points), 1),
        total_points=ref_count,
    )
    slice_specs = _snap_spatial_slices_to_header_grid(slice_specs, ref_header)

    def _core_mask(x_vals, y_vals, slice_bounds, axis, include_upper):
        if axis == "x":
            upper_mask = x_vals <= slice_bounds[1] if include_upper else x_vals < slice_bounds[1]
            return (
                (x_vals >= slice_bounds[0]) & upper_mask
                & (y_vals >= slice_bounds[2]) & (y_vals <= slice_bounds[3])
            )
        upper_mask = y_vals <= slice_bounds[3] if include_upper else y_vals < slice_bounds[3]
        return (
            (x_vals >= slice_bounds[0]) & (x_vals <= slice_bounds[1])
            & (y_vals >= slice_bounds[2]) & upper_mask
        )

    with laspy.open(
        str(temp_las), mode="w", header=out_header,
        do_compress=False,
    ) as writer:
        total_written = 0
        for slice_idx, (slice_bounds, axis, include_upper) in enumerate(slice_specs, start=1):
            slice_datasets = []
            for coll_idx, (src, coll_meta) in enumerate(zip(source_files, collection_meta), start=1):
                subset, subset_count = _load_copc_subset_all_dims(
                    Path(src),
                    slice_bounds,
                    halo=0.0,
                    work_dir=work_dir,
                )
                if subset is None or subset_count == 0:
                    slice_datasets.append({
                        "collection_index": coll_idx,
                        "point_count": 0,
                        "data": {},
                    })
                    continue

                sx = np.asarray(subset.x)
                sy = np.asarray(subset.y)
                mask = _core_mask(sx, sy, slice_bounds, axis, include_upper)
                if not np.any(mask):
                    slice_datasets.append({
                        "collection_index": coll_idx,
                        "point_count": 0,
                        "data": {},
                    })
                    del subset
                    continue

                data = {
                    "X": np.asarray(subset.X)[mask],
                    "Y": np.asarray(subset.Y)[mask],
                    "Z": np.asarray(subset.Z)[mask],
                }
                for dim_name in base_copy_dims:
                    values = getattr(subset, dim_name, None)
                    if values is not None:
                        data[dim_name] = np.asarray(values)[mask]
                for orig_name, _, _, _ in coll_meta["dim_entries"]:
                    values = getattr(subset, orig_name, None)
                    if values is not None:
                        data[orig_name] = np.asarray(values)[mask]

                slice_datasets.append({
                    "collection_index": coll_idx,
                    "point_count": int(len(data["X"])),
                    "data": data,
                })
                del subset

            slice_counts = [entry["point_count"] for entry in slice_datasets]
            if len(set(slice_counts)) != 1:
                counts_str = ", ".join(
                    f"collection {entry['collection_index']}={entry['point_count']:,}"
                    for entry in slice_datasets
                )
                return False, ref_count, (
                    f"point-count mismatch after COPC spatial query at slice {slice_idx} "
                    f"({counts_str})"
                )

            slice_point_count = slice_counts[0]
            if slice_point_count == 0:
                print(
                    f"    Slice {slice_idx}/{len(slice_specs)}: empty after COPC crop",
                    flush=True,
                )
                continue

            ref_data = slice_datasets[0]["data"]
            exact_order = True
            for entry in slice_datasets[1:]:
                cur = entry["data"]
                if (
                    not np.array_equal(ref_data["X"], cur["X"])
                    or not np.array_equal(ref_data["Y"], cur["Y"])
                    or not np.array_equal(ref_data["Z"], cur["Z"])
                ):
                    exact_order = False
                    break

            orders = []
            if exact_order:
                identity = np.arange(slice_point_count, dtype=np.int64)
                orders = [identity for _ in slice_datasets]
            else:
                ref_order = np.lexsort((ref_data["Z"], ref_data["Y"], ref_data["X"]))
                orders.append(ref_order)
                ref_sorted = {dim: ref_data[dim][ref_order] for dim in key_dims}
                for entry in slice_datasets[1:]:
                    cur = entry["data"]
                    cur_order = np.lexsort((cur["Z"], cur["Y"], cur["X"]))
                    orders.append(cur_order)
                    for dim_name in key_dims:
                        cur_sorted = cur[dim_name][cur_order]
                        if not np.array_equal(ref_sorted[dim_name], cur_sorted):
                            return False, ref_count, (
                                f"point-set mismatch after COPC spatial alignment at slice {slice_idx} "
                                f"(collection {entry['collection_index']}, differing dim: {dim_name})"
                            )

            out_record = laspy.ScaleAwarePointRecord.zeros(slice_point_count, header=out_header)
            ref_order = orders[0]
            for dim_name in base_copy_dims:
                values = ref_data.get(dim_name)
                if values is not None:
                    out_record[dim_name] = values[ref_order]

            for entry, order, coll_meta in zip(slice_datasets, orders, collection_meta):
                data = entry["data"]
                for orig_name, out_name, dtype, _ in coll_meta["dim_entries"]:
                    values = data.get(orig_name)
                    if values is None:
                        out_record[out_name] = np.zeros(slice_point_count, dtype=dtype)
                    else:
                        out_record[out_name] = values[order].astype(dtype, copy=False)

            writer.write_points(out_record)
            total_written += slice_point_count
            print(
                f"    Slice {slice_idx}/{len(slice_specs)}: fused {slice_point_count:,} pts "
                f"({total_written:,}/{ref_count:,}) via COPC spatial alignment",
                flush=True,
            )

    if total_written != ref_count:
        return False, ref_count, (
            f"COPC spatial alignment wrote {total_written:,} of {ref_count:,} expected points"
        )

    return True, ref_count, "COPC spatial-slice alignment"


def _fuse_one_aligned_tile(
    tile_key,
    source_files,
    output_dir,
    chunk_size,
    copc_chunk_size,
    target_dims: Optional[Set[str]] = None,
):
    """
    Fuse one aligned tile from multiple collections into a fused COPC/LAZ output.

    Returns ``(status, tile_key)`` where status is one of ``"copc"``, ``"las"``,
    or ``"reused"``.
    """
    import shutil
    import tempfile

    import laspy
    import numpy as np

    from merge_tiles import _prepare_collection_remap_metadata

    output_dir = Path(output_dir)
    source_files = [Path(src) for src in source_files]
    collection_meta = _prepare_collection_remap_metadata(source_files, target_dims=target_dims)
    fused_dir = output_dir / "fused_aligned_collection"
    fused_dir.mkdir(parents=True, exist_ok=True)
    fused_copc = fused_dir / f"{tile_key}.copc.laz"
    fused_las = fused_dir / f"{tile_key}.las"

    if fused_copc.exists() and fused_copc.stat().st_size > 0:
        print(f"  Fused tile already available (COPC): {fused_copc.name}", flush=True)
        return "reused", tile_key
    if fused_las.exists() and fused_las.stat().st_size > 0:
        print(f"  Fused tile already available (LAS fallback): {fused_las.name}", flush=True)
        return "reused", tile_key

    print(
        f"  Fusing tile {tile_key} from {len(source_files)} collection(s)",
        flush=True,
    )

    datasets = []
    with tempfile.TemporaryDirectory(prefix=f"3dtrees_fused_{tile_key}_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        temp_las = tmp_dir / f"{tile_key}.las"
        try:
            did_stream, ref_count, stream_reason = _try_streaming_fuse_aligned_tile(
                source_files=source_files,
                collection_meta=collection_meta,
                temp_las=temp_las,
                chunk_size=chunk_size,
            )
            if did_stream:
                print(
                    f"    Tile {tile_key}: fused via low-memory streaming path",
                    flush=True,
                )
            else:
                print(
                    f"    Tile {tile_key}: {stream_reason}; falling back to order-insensitive alignment",
                    flush=True,
                )
                if temp_las.exists():
                    temp_las.unlink()
                if all(Path(src).name.endswith(".copc.laz") for src in source_files):
                    did_copc_align, ref_count, copc_reason = _try_copc_spatial_fuse_aligned_tile(
                        source_files=source_files,
                        collection_meta=collection_meta,
                        temp_las=temp_las,
                        chunk_size=chunk_size,
                        slice_target_points=copc_chunk_size,
                        work_dir=tmp_dir,
                    )
                    if did_copc_align:
                        print(
                            f"    Tile {tile_key}: fused via {copc_reason}",
                            flush=True,
                        )
                    else:
                        raise ValueError(f"Tile {tile_key}: {copc_reason}")
                else:
                    datasets = [
                        laspy.read(str(src), laz_backend=laspy.LazBackend.LazrsParallel)
                        for src in source_files
                    ]
                    headers = [dataset.header for dataset in datasets]
                    ref_header = headers[0]
                    ref_count = int(ref_header.point_count)

                    for idx, header in enumerate(headers[1:], start=2):
                        if int(header.point_count) != ref_count:
                            raise ValueError(
                                f"Point-count mismatch for tile {tile_key}: "
                                f"collection 1 has {ref_count:,}, collection {idx} has {int(header.point_count):,}"
                            )

                    key_dims = _fusion_key_dims()
                    for idx, header in enumerate(headers[1:], start=2):
                        other_key_dims = [d for d in _fusion_key_dims() if d in header.point_format.dimension_names]
                        if other_key_dims != key_dims:
                            raise ValueError(
                                f"XYZ-dimension mismatch for tile {tile_key}: "
                                f"collection 1 dims {key_dims} vs collection {idx} dims {other_key_dims}"
                            )

                    ref_order = _compute_alignment_order(datasets[0], key_dims)
                    orders = [ref_order]

                    for coll_idx, dataset in enumerate(datasets[1:], start=2):
                        order = _compute_alignment_order(dataset, key_dims)
                        orders.append(order)
                        for dim_name in key_dims:
                            ref_sorted = np.asarray(datasets[0][dim_name])[ref_order]
                            cur_sorted = np.asarray(dataset[dim_name])[order]
                            if not np.array_equal(ref_sorted, cur_sorted):
                                raise ValueError(
                                    f"Tile {tile_key}: point-set mismatch between collection 1 and "
                                    f"collection {coll_idx} after order-insensitive alignment "
                                    f"(first differing dim: {dim_name})"
                                )

                    print(
                        f"    Tile {tile_key}: source row order differs across collections; "
                        f"aligning by XYZ",
                        flush=True,
                    )

                    out_header, base_copy_dims = _build_fused_tile_header(ref_header, collection_meta)

                    with laspy.open(
                        str(temp_las), mode="w", header=out_header,
                        do_compress=False,
                    ) as writer:
                        total_written = 0
                        for start in range(0, ref_count, chunk_size):
                            stop = min(start + chunk_size, ref_count)
                            ref_order_slice = ref_order[start:stop]
                            out_record = laspy.ScaleAwarePointRecord.zeros(stop - start, header=out_header)
                            for dim_name in base_copy_dims:
                                try:
                                    out_record[dim_name] = np.asarray(datasets[0][dim_name])[ref_order_slice]
                                except Exception:
                                    pass

                            for dataset, order, coll_meta in zip(datasets, orders, collection_meta):
                                order_slice = order[start:stop]
                                for orig_name, out_name, dtype, _ in coll_meta["dim_entries"]:
                                    values = getattr(dataset, orig_name, None)
                                    if values is None:
                                        out_record[out_name] = np.zeros(stop - start, dtype=dtype)
                                    else:
                                        out_record[out_name] = np.asarray(values)[order_slice].astype(dtype, copy=False)

                            writer.write_points(out_record)
                            total_written += stop - start

            success, message = _convert_collection_file_to_copc(
                input_path=temp_las,
                output_copc=fused_copc,
                chunk_size=copc_chunk_size,
            )
            if success:
                if temp_las.exists():
                    temp_las.unlink()
                if fused_las.exists():
                    fused_las.unlink()
                print(
                    f"    Fused tile ready as COPC: {fused_copc.name} ({ref_count:,} pts)",
                    flush=True,
                )
                return "copc", tile_key

            if fused_copc.exists() and fused_copc.stat().st_size == 0:
                fused_copc.unlink()
            shutil.move(str(temp_las), str(fused_las))
            print(
                f"    Fused tile kept as LAS fallback: {fused_las.name} "
                f"({ref_count:,} pts; COPC conversion failed: {message})",
                flush=True,
            )
            return "las", tile_key
        finally:
            if datasets:
                del datasets


def _fuse_one_aligned_tile_worker(args):
    """Pickle-friendly wrapper for process-pool tile fusion."""
    return _fuse_one_aligned_tile(*args)


def _fuse_aligned_collections_to_copc(
    collections,
    output_dir,
    chunk_size=2_000_000,
    copc_chunk_size=20_000_000,
    target_dims: Optional[Set[str]] = None,
    workers: int = 1,
):
    """
    Fuse aligned collection tiles into one source collection before remap.

    This path assumes each collection contains the same points tile-by-tile, with
    different attributes attached. Files are matched by tile key and aligned by a
    stable sort over XYZ so row-order differences do not
    block fusion. Each fused tile is then converted to COPC when possible, falling
    back to LAZ if COPC conversion fails.

    Returns the fused collection directory, or ``None`` when aligned fusion is not
    possible and the caller should fall back to independent per-collection remap.
    """
    import shutil
    import tempfile

    import laspy
    import numpy as np

    from merge_tiles import _prepare_collection_remap_metadata, list_pointcloud_files

    collections = [Path(c) for c in collections]
    if len(collections) <= 1:
        return collections[0] if collections else None

    collection_file_maps = []
    reference_keys = None
    mismatch_detected = False

    for idx, coll_path in enumerate(collections, start=1):
        files = [coll_path] if coll_path.is_file() else list_pointcloud_files(coll_path)
        if not files:
            print(f"  Fusion check: collection {idx} has no files in {coll_path}")
            return None

        key_map = {}
        for src in files:
            key = _pointcloud_file_key(src)
            if key in key_map:
                print(
                    f"  Fusion check: duplicate tile key '{key}' in collection {idx} ({coll_path}); "
                    f"falling back to separate remap",
                    flush=True,
                )
                return None
            key_map[key] = src
        collection_file_maps.append(key_map)

        if reference_keys is None:
            reference_keys = set(key_map.keys())
            continue

        current_keys = set(key_map.keys())
        if current_keys != reference_keys:
            only_ref = sorted(reference_keys - current_keys)
            only_cur = sorted(current_keys - reference_keys)
            print(
                f"  Fusion check: collection {idx} tile keys differ from collection 1; "
                f"falling back to separate remap",
                flush=True,
            )
            if only_ref:
                print(f"    Missing keys in collection {idx}: {', '.join(only_ref[:5])}", flush=True)
            if only_cur:
                print(f"    Extra keys in collection {idx}: {', '.join(only_cur[:5])}", flush=True)
            mismatch_detected = True

    if mismatch_detected or not reference_keys:
        return None

    fused_dir = Path(output_dir) / "fused_aligned_collection"
    fused_dir.mkdir(parents=True, exist_ok=True)

    collection_meta = _prepare_collection_remap_metadata(collections, target_dims=target_dims)
    print("\nFusing aligned collections into one source collection before remap...")
    for ci, coll_meta in enumerate(collection_meta, start=1):
        if coll_meta["dim_entries"]:
            dim_str = ", ".join(
                out_name if orig_name == out_name else f"{orig_name}->{out_name}"
                for orig_name, out_name, _, _ in coll_meta["dim_entries"]
            )
        else:
            dim_str = "(no extra dimensions discovered)"
        print(f"  Collection {ci} fused dims: {dim_str}", flush=True)

    copc_ready = 0
    las_ready = 0
    reused = 0

    tile_jobs = []
    for tile_key in sorted(reference_keys):
        source_files = [file_map[tile_key] for file_map in collection_file_maps]
        tile_jobs.append((
            tile_key,
            [str(src) for src in source_files],
            str(output_dir),
            chunk_size,
            copc_chunk_size,
            target_dims,
        ))

    max_fusion_workers = max(1, int(workers or 1))
    if max_fusion_workers > 1 and len(tile_jobs) > 1:
        print(f"  Parallel aligned fusion workers: {min(max_fusion_workers, len(tile_jobs))}", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=min(max_fusion_workers, len(tile_jobs))) as executor:
            futures = {
                executor.submit(_fuse_one_aligned_tile_worker, job): job[0]
                for job in tile_jobs
            }
            for future in as_completed(futures):
                tile_key = futures[future]
                try:
                    status, _done_tile = future.result()
                except Exception as exc:
                    print(
                        f"  Fusion failed for tile {tile_key}: {exc}. "
                        f"Falling back to separate multi-collection remap.",
                        flush=True,
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    return None
                if status == "copc":
                    copc_ready += 1
                elif status == "las":
                    las_ready += 1
                else:
                    reused += 1
    else:
        for job in tile_jobs:
            tile_key = job[0]
            try:
                status, _done_tile = _fuse_one_aligned_tile_worker(job)
            except Exception as exc:
                print(
                    f"  Fusion failed for tile {tile_key}: {exc}. "
                    f"Falling back to separate multi-collection remap.",
                    flush=True,
                )
                return None
            if status == "copc":
                copc_ready += 1
            elif status == "las":
                las_ready += 1
            else:
                reused += 1

    print(
        f"  Fused source collection ready: {fused_dir} "
        f"({copc_ready} COPC, {las_ready} LAS fallback, {reused} reused)",
        flush=True,
    )
    return fused_dir


def _build_merged_copc_from_fused_tiles(
    fused_dir,
    output_dir,
    chunk_size=2_000_000,
    copc_chunk_size=20_000_000,
):
    """
    Build one large merged COPC from a directory of schema-aligned fused tiles.

    Returns the preferred source path for remap:
    - merged COPC when conversion succeeds
    - ``None`` if no fused tiles exist
    """
    from main_tile import _finalize_tile_to_copc_untwine
    from merge_tiles import list_pointcloud_files

    fused_dir = Path(fused_dir)
    output_dir = Path(output_dir)
    fused_files = list_pointcloud_files(fused_dir)
    if not fused_files:
        print(f"  Warning: no fused tiles found in {fused_dir}; cannot build merged source COPC", flush=True)
        return None

    merged_copc = output_dir / "fused_aligned_collection_merged.copc.laz"

    if merged_copc.exists() and merged_copc.stat().st_size > 0:
        print(f"  Reusing fused merged source COPC: {merged_copc}", flush=True)
        return merged_copc

    print(
        f"  Building one merged fused source from {len(fused_files)} tile(s) "
        f"for spatial COPC queries via untwine...",
        flush=True,
    )
    success, message = _finalize_tile_to_copc_untwine(
        parts=fused_files,
        final_tile=merged_copc,
        log_dir=output_dir,
        label="fused_aligned_collection_merged",
    )
    if success:
        print(f"  Fused merged source COPC ready: {merged_copc}", flush=True)
        return merged_copc

    print(
        f"  Warning: merged fused source COPC conversion failed; "
        f"using fused tile collection directly for remap ({message})",
        flush=True,
    )
    return None


def _build_merged_copc_per_collection(
    collections,
    output_dir,
    chunk_size=2_000_000,
    copc_chunk_size=20_000_000,
):
    """
    Build one merged COPC per provided collection.

    This is useful when collections represent the same global point cloud but
    differ in tile boundary assignment, so per-tile fusion can fail even though
    a global XYZ-based fusion is still possible.
    """
    from main_tile import _finalize_tile_to_copc_untwine
    from merge_tiles import list_pointcloud_files

    output_dir = Path(output_dir)
    merged_collections: List[Path] = []

    for coll_idx, coll in enumerate(collections, start=1):
        coll_path = Path(coll)
        coll_files = [coll_path] if coll_path.is_file() else list_pointcloud_files(coll_path)
        if not coll_files:
            print(f"  Warning: collection {coll_idx} has no point-cloud files; cannot build merged COPC", flush=True)
            return None

        if coll_path.is_file() and coll_path.name.endswith(".copc.laz"):
            merged_collections.append(coll_path)
            continue

        merged_copc = output_dir / f"collection_{coll_idx:02d}_merged_source.copc.laz"
        if merged_copc.exists() and merged_copc.stat().st_size > 0:
            print(f"  Reusing merged source COPC for collection {coll_idx}: {merged_copc}", flush=True)
            merged_collections.append(merged_copc)
            continue

        print(
            f"  Building merged source COPC for collection {coll_idx} from "
            f"{len(coll_files)} tile(s)...",
            flush=True,
        )
        success, message = _finalize_tile_to_copc_untwine(
            parts=coll_files,
            final_tile=merged_copc,
            log_dir=output_dir,
            label=f"collection_{coll_idx:02d}_merged_source",
        )
        if not success:
            print(
                f"  Warning: could not build merged source COPC for collection {coll_idx} "
                f"({message})",
                flush=True,
            )
            return None
        print(f"  Collection {coll_idx} merged source COPC ready: {merged_copc}", flush=True)
        merged_collections.append(merged_copc)

    return merged_collections


def _fuse_collections_by_spatial_chunks_to_copc(
    collections,
    output_dir,
    chunk_size=2_000_000,
    copc_chunk_size=20_000_000,
    target_dims: Optional[Set[str]] = None,
    spatial_slices: int = 50,
    spatial_chunk_length: Optional[float] = None,
    spatial_target_points: Optional[int] = None,
):
    """
    Fuse multiple collections globally by spatial chunks rather than tile-local equality.

    The first collection defines the canonical output tiling/layout. Each spatial
    chunk is loaded from all collections, aligned by XYZ only, and then written
    into the canonical fused tile set.
    """
    import gc
    import math
    import tempfile

    import laspy
    import numpy as np

    from merge_tiles import (
        _bounds_overlap_2d,
        _build_spatial_slices,
        _load_copc_subset_all_dims,
        _prepare_collection_remap_metadata,
    )

    collections = [Path(c) for c in collections]
    if len(collections) <= 1:
        return collections[0] if collections else None

    collection_meta = _prepare_collection_remap_metadata(collections, target_dims=target_dims)
    if not collection_meta or not collection_meta[0]["files"]:
        raise RuntimeError("Spatial fusion requires at least one readable file in collection 1")

    fused_dir = Path(output_dir) / "fused_spatial_collection"
    fused_dir.mkdir(parents=True, exist_ok=True)

    canonical_entries = list(collection_meta[0]["files"])
    expected_outputs = [
        fused_dir / f"{_pointcloud_file_key(entry['path'])}.copc.laz"
        for entry in canonical_entries
    ]
    if expected_outputs and all(path.exists() and path.stat().st_size > 0 for path in expected_outputs):
        print(f"  Reusing spatially fused collection: {fused_dir}", flush=True)
        return fused_dir

    for stale in list(fused_dir.glob("*.copc.laz")) + list(fused_dir.glob("*.laz")):
        stale.unlink(missing_ok=True)

    with laspy.open(
        str(canonical_entries[0]["path"]),
        laz_backend=laspy.LazBackend.LazrsParallel,
    ) as ref_reader:
        ref_header = ref_reader.header
    out_header, base_copy_dims = _build_fused_tile_header(ref_header, collection_meta)

    total_points_per_collection = [
        sum(int(entry.get("point_count", 0)) for entry in coll_meta["files"])
        for coll_meta in collection_meta
    ]
    ref_total_points = total_points_per_collection[0] if total_points_per_collection else 0

    global_bounds = (
        min(entry["bounds"][0] for coll_meta in collection_meta for entry in coll_meta["files"]),
        max(entry["bounds"][1] for coll_meta in collection_meta for entry in coll_meta["files"]),
        min(entry["bounds"][2] for coll_meta in collection_meta for entry in coll_meta["files"]),
        max(entry["bounds"][3] for coll_meta in collection_meta for entry in coll_meta["files"]),
    )
    slice_specs = _build_spatial_slices(
        global_bounds,
        spatial_slices,
        spatial_chunk_length,
        spatial_target_points,
        ref_total_points,
    )

    canonical_tiles = []
    for entry in canonical_entries:
        tile_key = _pointcloud_file_key(entry["path"])
        bounds = entry["bounds"]
        canonical_tiles.append({
            "name": tile_key,
            "bounds": bounds,
            "final_copc": fused_dir / f"{tile_key}.copc.laz",
        })
    canonical_tiles = _annotate_canonical_tile_edge_neighbors(canonical_tiles, abs_tol=1e-9)

    print("\nSpatial global fusion of provided collections", flush=True)
    for ci, (coll_path, coll_meta, total_points) in enumerate(
        zip(collections, collection_meta, total_points_per_collection),
        start=1,
    ):
        if coll_meta["dim_entries"]:
            dim_str = ", ".join(
                out_name if orig_name == out_name else f"{orig_name}->{out_name}"
                for orig_name, out_name, _, _ in coll_meta["dim_entries"]
            )
        else:
            dim_str = "(no extra dimensions discovered)"
        print(
            f"  Collection {ci}: {coll_path} ({total_points:,} pts) dims: {dim_str}",
            flush=True,
        )
    if spatial_target_points:
        print(f"  Chunk basis: target points ≈ {int(spatial_target_points):,} per chunk", flush=True)
    elif spatial_chunk_length:
        print(f"  Chunk basis: fixed spatial length {spatial_chunk_length} m", flush=True)
    else:
        print(f"  Chunk basis: fixed slice count {len(slice_specs)}", flush=True)
    print(
        f"  Fused output tiles: {len(canonical_tiles)} in {fused_dir}",
        flush=True,
    )

    staged_tile_writers: Dict[str, _StreamingCopcTilePartsWriter] = {}

    def _get_writer(tile_info, parts_root: Path):
        writer = staged_tile_writers.get(tile_info["name"])
        if writer is None:
            writer = _StreamingCopcTilePartsWriter(
                final_copc=tile_info["final_copc"],
                header_snapshot=out_header,
                label=tile_info["name"],
                parts_root=parts_root,
            )
            staged_tile_writers[tile_info["name"]] = writer
        return writer

    def _core_mask(x_vals, y_vals, slice_bounds, axis, include_upper):
        if axis == "x":
            upper_mask = x_vals <= slice_bounds[1] if include_upper else x_vals < slice_bounds[1]
            return (
                (x_vals >= slice_bounds[0]) & upper_mask
                & (y_vals >= slice_bounds[2]) & (y_vals <= slice_bounds[3])
            )
        upper_mask = y_vals <= slice_bounds[3] if include_upper else y_vals < slice_bounds[3]
        return (
            (x_vals >= slice_bounds[0]) & (x_vals <= slice_bounds[1])
            & (y_vals >= slice_bounds[2]) & upper_mask
        )

    def _alignment_order(data_dict):
        return np.lexsort((data_dict["Z"], data_dict["Y"], data_dict["X"]))

    try:
        with tempfile.TemporaryDirectory(prefix="3dtrees_spatial_fuse_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            parts_root = tmp_dir / "fused_tile_parts"

            for chunk_idx, (slice_bounds, axis, include_upper) in enumerate(slice_specs, start=1):
                chunk_desc = (
                    f"x=[{slice_bounds[0]:.2f}, {slice_bounds[1]:.2f}] "
                    f"y=[{slice_bounds[2]:.2f}, {slice_bounds[3]:.2f}]"
                )
                print(
                    f"  Chunk {chunk_idx}/{len(slice_specs)}: {chunk_desc}",
                    flush=True,
                )

                chunk_datasets = []
                for coll_idx, coll_meta in enumerate(collection_meta, start=1):
                    needed_dim_names = {"X", "Y", "Z"}
                    if coll_idx == 1:
                        needed_dim_names.update(base_copy_dims)
                    needed_dim_names.update(orig_name for orig_name, _, _, _ in coll_meta["dim_entries"])

                    candidate_entries = [
                        entry for entry in coll_meta["files"]
                        if _bounds_overlap_2d(entry["bounds"], slice_bounds, buffer=0.0)
                    ]
                    arrays_by_dim: Dict[str, List[np.ndarray]] = {}
                    subset_points = 0
                    for entry in candidate_entries:
                        subset, subset_count = _load_copc_subset_all_dims(
                            entry["path"],
                            slice_bounds,
                            halo=0.0,
                            work_dir=tmp_dir,
                        )
                        if subset is None or subset_count == 0:
                            continue
                        subset_points += subset_count
                        for dim_name in needed_dim_names:
                            values = getattr(subset, dim_name, None)
                            if values is None:
                                continue
                            arrays_by_dim.setdefault(dim_name, []).append(np.asarray(values))
                        arrays_by_dim.setdefault("__x__", []).append(np.asarray(subset.x, dtype=np.float64))
                        arrays_by_dim.setdefault("__y__", []).append(np.asarray(subset.y, dtype=np.float64))
                        del subset

                    if subset_points == 0:
                        chunk_datasets.append({
                            "collection_index": coll_idx,
                            "point_count": 0,
                            "candidate_count": len(candidate_entries),
                            "data": {},
                        })
                        continue

                    merged_data = {
                        name: (
                            arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
                        )
                        for name, arrays in arrays_by_dim.items()
                    }
                    core_mask = _core_mask(
                        merged_data["__x__"],
                        merged_data["__y__"],
                        slice_bounds,
                        axis,
                        include_upper,
                    )
                    if not np.any(core_mask):
                        chunk_datasets.append({
                            "collection_index": coll_idx,
                            "point_count": 0,
                            "candidate_count": len(candidate_entries),
                            "data": {},
                        })
                        continue

                    cropped = {
                        name: values[core_mask]
                        for name, values in merged_data.items()
                    }
                    chunk_datasets.append({
                        "collection_index": coll_idx,
                        "point_count": int(len(cropped["X"])),
                        "candidate_count": len(candidate_entries),
                        "data": cropped,
                    })
                    del merged_data, cropped, core_mask

                chunk_counts = [entry["point_count"] for entry in chunk_datasets]
                if len(set(chunk_counts)) != 1:
                    counts_str = ", ".join(
                        f"collection {entry['collection_index']}={entry['point_count']:,}"
                        for entry in chunk_datasets
                    )
                    raise RuntimeError(
                        f"Spatial fusion chunk mismatch at {chunk_desc}: point counts differ "
                        f"({counts_str})"
                    )

                chunk_point_count = chunk_counts[0]
                if chunk_point_count == 0:
                    print(f"    Empty chunk after crop across all collections, skipping", flush=True)
                    continue

                ref_data = chunk_datasets[0]["data"]
                exact_order = True
                for entry in chunk_datasets[1:]:
                    cur = entry["data"]
                    if (
                        not np.array_equal(ref_data["X"], cur["X"])
                        or not np.array_equal(ref_data["Y"], cur["Y"])
                        or not np.array_equal(ref_data["Z"], cur["Z"])
                    ):
                        exact_order = False
                        break

                orders = []
                if exact_order:
                    identity = np.arange(chunk_point_count, dtype=np.int64)
                    orders = [identity for _ in chunk_datasets]
                else:
                    ref_order = _alignment_order(ref_data)
                    orders.append(ref_order)
                    ref_sorted = {dim: ref_data[dim][ref_order] for dim in ("X", "Y", "Z")}
                    for entry in chunk_datasets[1:]:
                        cur = entry["data"]
                        cur_order = _alignment_order(cur)
                        orders.append(cur_order)
                        for dim_name in ("X", "Y", "Z"):
                            cur_sorted = cur[dim_name][cur_order]
                            if not np.array_equal(ref_sorted[dim_name], cur_sorted):
                                raise RuntimeError(
                                    f"Spatial fusion chunk mismatch at {chunk_desc}: "
                                    f"collection 1 and collection {entry['collection_index']} differ on {dim_name} "
                                    f"after XYZ alignment "
                                    f"(counts: {chunk_point_count:,} vs {entry['point_count']:,})"
                                )

                aligned_arrays: Dict[str, np.ndarray] = {}
                ref_order = orders[0]
                ref_x = ref_data["__x__"][ref_order]
                ref_y = ref_data["__y__"][ref_order]
                for dim_name in base_copy_dims:
                    values = ref_data.get(dim_name)
                    if values is not None:
                        aligned_arrays[dim_name] = values[ref_order]

                for entry, order, coll_meta in zip(chunk_datasets, orders, collection_meta):
                    data = entry["data"]
                    for orig_name, out_name, dtype, _ in coll_meta["dim_entries"]:
                        values = data.get(orig_name)
                        if values is None:
                            aligned_arrays[out_name] = np.zeros(chunk_point_count, dtype=dtype)
                        else:
                            aligned_arrays[out_name] = values[order].astype(dtype, copy=False)

                candidate_tiles = [
                    tile_info for tile_info in canonical_tiles
                    if _bounds_overlap_2d(tile_info["bounds"], slice_bounds, buffer=0.0)
                ]
                assigned = np.zeros(chunk_point_count, dtype=bool)
                written = 0
                eps = 1e-9
                for tile_info in candidate_tiles:
                    xmin, xmax, ymin, ymax = tile_info["bounds"]
                    upper_x = _upper_edge_membership_mask(
                        ref_x,
                        xmax,
                        tile_info.get("east_neighbor_spans", []),
                        ref_y,
                        abs_tol=eps,
                    )
                    upper_y = _upper_edge_membership_mask(
                        ref_y,
                        ymax,
                        tile_info.get("north_neighbor_spans", []),
                        ref_x,
                        abs_tol=eps,
                    )
                    tile_mask = (
                        ~assigned
                        & (ref_x >= xmin - eps) & upper_x
                        & (ref_y >= ymin - eps) & upper_y
                    )
                    if not np.any(tile_mask):
                        continue

                    out_record = laspy.ScaleAwarePointRecord.zeros(int(np.sum(tile_mask)), header=out_header)
                    for dim_name, values in aligned_arrays.items():
                        try:
                            out_record[dim_name] = values[tile_mask]
                        except Exception:
                            pass
                    _get_writer(tile_info, parts_root).write_points(out_record)
                    assigned[tile_mask] = True
                    written += len(out_record)
                    del out_record

                if not np.all(assigned):
                    missing = int(np.count_nonzero(~assigned))
                    raise RuntimeError(
                        f"Spatial fusion chunk assignment failed at {chunk_desc}: "
                        f"{missing:,} points could not be assigned to canonical output tiles"
                    )

                summary = " ".join(
                    f"c{entry['collection_index']}:{entry['point_count']:,}/{entry['candidate_count']}"
                    for entry in chunk_datasets
                )
                print(
                    f"    Fused {written:,} pts [{summary}]",
                    flush=True,
                )

                del chunk_datasets, aligned_arrays, ref_x, ref_y, assigned
                gc.collect()
            converted = 0
            finalized_parts = 0
            for tile_info in canonical_tiles:
                final_copc = tile_info["final_copc"]
                writer = staged_tile_writers.get(tile_info["name"])
                if writer is None:
                    final_copc.unlink(missing_ok=True)
                    continue
                success, message = writer.finalize()
                finalized_parts += len(writer.part_paths)
                if not success:
                    raise RuntimeError(
                        f"Failed to finalize fused COPC tile {final_copc.name}: {message}"
                    )
                converted += 1
    finally:
        staged_tile_writers.clear()

    print(
        f"  Spatially fused source collection ready: {fused_dir} "
        f"({converted} COPC tile(s), {finalized_parts} staged LAS part(s))",
        flush=True,
    )
    return fused_dir


def _dir_has_pointcloud_outputs(directory: Path) -> bool:
    """Return True when a directory already contains LAS/LAZ outputs."""
    return directory.exists() and any(directory.glob("*.la[sz]"))


def _resolve_remap_target_dims(params):
    """
    Resolve the dimension set to transfer during remap-style operations.

    This intentionally only considers ``--remap-dims``. The standardization JSON
    is reserved for filtering original dimensions during merged-file enrichment
    (``merged_with_originals``), not for filtering prediction dimensions during
    remap itself.
    """
    return (
        {d.strip() for d in params.remap_dims.split(",") if d.strip()}
        if getattr(params, "remap_dims", None) else None
    )


def _describe_laz_dimensions(laz_path: Path) -> str:
    """Return a compact string of LAS dimension names for logging."""
    import laspy

    with laspy.open(str(laz_path)) as f:
        dims = list(f.header.point_format.dimension_names)
    return ", ".join(str(d) for d in dims)


def _prepare_single_collection_remap_source(
    source_collection,
    output_dir,
    merge_chunk_size,
    copc_chunk_size,
    prefer_merged_source: bool = True,
):
    """
    Normalize one remap source collection to COPC when needed and build one
    merged source COPC for spatial-query remap.

    Returns ``(remap_sources, merged_source_copc, normalized_collection)`` where
    ``remap_sources`` is the preferred input list for remap calls.
    """
    try:
        from merge_tiles import list_pointcloud_files
    except ImportError as e:
        print(f"  Warning: could not import pointcloud helpers: {e}")
        return [Path(source_collection)], None, Path(source_collection)

    source_collection = Path(source_collection)
    normalized_collection = source_collection
    merged_source_copc = None

    if prefer_merged_source and source_collection.is_dir():
        existing_files = list_pointcloud_files(source_collection)
        if existing_files and all(path.name.endswith(".copc.laz") for path in existing_files):
            print(f"  Existing COPC tiles detected; building merged source COPC from {source_collection}")
            merged_source_copc = _build_merged_copc_from_fused_tiles(
                fused_dir=source_collection,
                output_dir=output_dir,
                chunk_size=merge_chunk_size,
                copc_chunk_size=copc_chunk_size,
            )
            if merged_source_copc is not None:
                print(f"  Using merged source COPC for remap: {merged_source_copc}", flush=True)
                return [merged_source_copc], merged_source_copc, source_collection

    print("  Normalizing remap source collection to COPC (reuse cache when available)...")
    normalized_list = _ensure_collections_copc(
        collections=[source_collection],
        output_dir=output_dir,
        chunk_size=copc_chunk_size,
    )
    normalized_collection = Path(normalized_list[0])
    remap_sources = [normalized_collection]

    if prefer_merged_source and normalized_collection.is_dir():
        print(f"  Building merged source COPC from normalized tiles in {normalized_collection}...")
        merged_source_copc = _build_merged_copc_from_fused_tiles(
            fused_dir=normalized_collection,
            output_dir=output_dir,
            chunk_size=merge_chunk_size,
            copc_chunk_size=copc_chunk_size,
        )
        if merged_source_copc is not None:
            print(f"  Using merged source COPC for remap: {merged_source_copc}", flush=True)
            remap_sources = [merged_source_copc]

    return remap_sources, merged_source_copc, normalized_collection


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
            list_pointcloud_files,
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
        import time as _time

        def _fmt_elapsed(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.1f}s"
            return f"{seconds / 60:.1f} min"

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
        remap_spatial_slices = params.remap_spatial_slices
        remap_spatial_chunk_length = params.remap_spatial_chunk_length
        remap_spatial_target_points = params.remap_spatial_target_points
        force_spatial_chunked_fusion = bool(getattr(params, "force_spatial_chunked_fusion", False))
        copc_chunk_size = max(int(getattr(params, "chunk_size", 20_000_000) or 20_000_000), merge_chunk_size)

        remap_dims_set = _resolve_remap_target_dims(params)

        print("=" * 60)
        print("Remap: provided collections -> subsampled + original files")
        print("=" * 60)
        print(f"  Collections:  {[str(c) for c in collections]}")
        print(f"  Originals:    {original_input_dir}")
        print(f"  Output:       {output_dir}")
        print(f"  Remap dims:   {remap_dims_set or 'all extra dims'}")
        if remap_spatial_target_points:
            print(f"  Spatial target points: {remap_spatial_target_points:,}")
        elif remap_spatial_chunk_length:
            print(f"  Spatial chunk length: {remap_spatial_chunk_length} m")
        else:
            print(f"  Spatial slices: {remap_spatial_slices}")
        print(f"  Force spatial fusion: {force_spatial_chunked_fusion}")

        raw_collections = list(collections)
        remap_sources = raw_collections
        fused_source_collection = None
        fused_merged_source = None
        canonical_projected_collection = None
        if len(raw_collections) > 1:
            projection_t0 = _time.monotonic()
            collections = list(raw_collections)

            aligned_fusion_ok = False
            if force_spatial_chunked_fusion:
                print(
                    "\nSkipping aligned-fusion verification and forcing chunked spatial fusion "
                    "for the provided collections...",
                    flush=True,
                )
            else:
                print(
                    "\nChecking whether the provided collections already share the same "
                    "tile-local point sets...",
                    flush=True,
                )
                try:
                    fused_source_collection = _fuse_aligned_collections_to_copc(
                        collections=raw_collections,
                        output_dir=output_dir,
                        chunk_size=merge_chunk_size,
                        copc_chunk_size=copc_chunk_size,
                        target_dims=remap_dims_set,
                        workers=params.workers,
                    )
                    if fused_source_collection is not None:
                        remap_sources, fused_merged_source, _normalized_fused_collection = _prepare_single_collection_remap_source(
                            source_collection=fused_source_collection,
                            output_dir=output_dir,
                            merge_chunk_size=merge_chunk_size,
                            copc_chunk_size=copc_chunk_size,
                        )
                        remap_source = remap_sources[0]
                        aligned_fusion_ok = True
                        print(
                            f"  Collections verified as aligned; using fused source for remap: "
                            f"{remap_source}",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"  Aligned-fusion verification failed: {exc}",
                        flush=True,
                    )

            spatial_fusion_ok = False
            if not aligned_fusion_ok:
                normalize_t0 = _time.monotonic()
                print("\nNormalizing provided collections to COPC for canonical-geometry projection...")
                collections = _ensure_collections_copc(
                    collections=raw_collections,
                    output_dir=output_dir,
                    chunk_size=copc_chunk_size,
                )
                print(f"  Normalized collections: {[str(c) for c in collections]}")
                print(
                    f"  Collection normalization duration: "
                    f"{_fmt_elapsed(_time.monotonic() - normalize_t0)}",
                    flush=True,
                )
                print(
                    "\nCollections are not tile-identical; attempting chunked spatial fusion "
                    "with the configured remap chunking settings...",
                    flush=True,
                )
                try:
                    fused_source_collection = _fuse_collections_by_spatial_chunks_to_copc(
                        collections=collections,
                        output_dir=output_dir,
                        chunk_size=merge_chunk_size,
                        copc_chunk_size=copc_chunk_size,
                        target_dims=remap_dims_set,
                        spatial_slices=remap_spatial_slices,
                        spatial_chunk_length=remap_spatial_chunk_length,
                        spatial_target_points=remap_spatial_target_points,
                    )
                    if fused_source_collection is not None:
                        remap_sources, fused_merged_source, _normalized_fused_collection = _prepare_single_collection_remap_source(
                            source_collection=fused_source_collection,
                            output_dir=output_dir,
                            merge_chunk_size=merge_chunk_size,
                            copc_chunk_size=copc_chunk_size,
                        )
                        remap_source = remap_sources[0]
                        spatial_fusion_ok = True
                        print(
                            f"  Chunked spatial fusion succeeded; using fused source for remap: "
                            f"{remap_source}",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"  Chunked spatial fusion failed: {exc}",
                        flush=True,
                    )

            if not aligned_fusion_ok and not spatial_fusion_ok:
                canonical_collection = Path(collections[-1])
                source_collections = list(collections[:-1])
                canonical_projected_collection = output_dir / "canonical_collection_with_predictions"

                print(
                    f"\nProjecting {len(source_collections)} collection(s) onto canonical geometry: "
                    f"{canonical_collection}",
                    flush=True,
                )
                if _dir_has_pointcloud_outputs(canonical_projected_collection):
                    print(
                        f"  Existing projected canonical collection detected in "
                        f"{canonical_projected_collection}; existing files will be skipped.",
                        flush=True,
                    )
                remap_collections_to_original_files(
                    collections=source_collections,
                    original_input_dir=canonical_collection,
                    output_dir=canonical_projected_collection,
                    tolerance=0.1,
                    retile_buffer=2.0,
                    chunk_size=merge_chunk_size,
                    target_dims=remap_dims_set,
                    spatial_slices=remap_spatial_slices,
                    spatial_chunk_length=remap_spatial_chunk_length,
                    spatial_target_points=remap_spatial_target_points,
                )

                fused_source_collection = canonical_projected_collection
                remap_sources, fused_merged_source, _normalized_canonical_collection = _prepare_single_collection_remap_source(
                    source_collection=canonical_projected_collection,
                    output_dir=output_dir,
                    merge_chunk_size=merge_chunk_size,
                    copc_chunk_size=copc_chunk_size,
                )
                remap_source = remap_sources[0]
                print(f"  Using projected canonical source for remap: {remap_source}", flush=True)
            print(
                f"  Multi-collection source-prep duration: "
                f"{_fmt_elapsed(_time.monotonic() - projection_t0)}",
                flush=True,
            )
        elif len(raw_collections) == 1 and Path(raw_collections[0]).is_dir():
            copc_candidate_files = list_pointcloud_files(Path(raw_collections[0]))
            if copc_candidate_files and all(path.name.endswith(".copc.laz") for path in copc_candidate_files):
                merged_source_t0 = _time.monotonic()
                print("\nExisting COPC tiles detected in single collection; building merged source COPC...")
                fused_merged_source = _build_merged_copc_from_fused_tiles(
                    fused_dir=raw_collections[0],
                    output_dir=output_dir,
                    chunk_size=merge_chunk_size,
                    copc_chunk_size=copc_chunk_size,
                )
                if fused_merged_source is not None:
                    remap_sources = [fused_merged_source]
                    print(f"  Using merged source COPC for remap: {fused_merged_source}", flush=True)
                else:
                    print(
                        "  Warning: merged source COPC could not be built from existing COPC tiles; "
                        "falling back to the collection directory.",
                        flush=True,
                    )
                print(
                    f"  Existing-COPC merged-source duration: "
                    f"{_fmt_elapsed(_time.monotonic() - merged_source_t0)}",
                    flush=True,
                )
        if len(raw_collections) == 1 and fused_source_collection is None and fused_merged_source is None:
            normalize_t0 = _time.monotonic()
            print("\nNormalizing provided collections to COPC (reuse cache when available)...")
            collections = _ensure_collections_copc(
                collections=raw_collections,
                output_dir=output_dir,
                chunk_size=copc_chunk_size,
            )
            print(f"  Normalized collections: {[str(c) for c in collections]}")
            remap_sources = collections
            print(f"  Collection normalization duration: {_fmt_elapsed(_time.monotonic() - normalize_t0)}", flush=True)
            if len(collections) == 1 and Path(collections[0]).is_dir():
                merged_source_t0 = _time.monotonic()
                print("\nBuilding one merged source COPC from normalized collection tiles...")
                single_merged_source = _build_merged_copc_from_fused_tiles(
                    fused_dir=collections[0],
                    output_dir=output_dir,
                    chunk_size=merge_chunk_size,
                    copc_chunk_size=copc_chunk_size,
                )
                if single_merged_source is not None:
                    fused_merged_source = single_merged_source
                    remap_sources = [single_merged_source]
                    print(f"  Using merged source COPC for remap: {single_merged_source}", flush=True)
                else:
                    print(
                        "  Warning: merged source COPC could not be built; "
                        "using normalized collection tiles directly.",
                        flush=True,
                    )
                print(
                    f"  Single-collection merged-source duration: "
                    f"{_fmt_elapsed(_time.monotonic() - merged_source_t0)}",
                    flush=True,
                )
        elif len(raw_collections) == 1:
            collections = raw_collections
        merged_all = None

        # ── Step 1: Remap to subsampled_res1 (if provided) ───────────────────
        sub_output = None
        if getattr(params, "subsampled_target_folder", None):
            sub_target = Path(params.subsampled_target_folder)
            sub_output = output_dir / "subsampled_with_predictions"
            step_t0 = _time.monotonic()
            print(f"\nStep 1: Remapping collections to subsampled target → {sub_output}")
            if _dir_has_pointcloud_outputs(sub_output):
                print(f"  Existing remap outputs detected in {sub_output}; existing files will be skipped.")
            remap_collections_to_original_files(
                collections=remap_sources,
                original_input_dir=sub_target,
                output_dir=sub_output,
                tolerance=0.1,
                retile_buffer=2.0,
                chunk_size=merge_chunk_size,
                target_dims=remap_dims_set,
                spatial_slices=remap_spatial_slices,
                spatial_chunk_length=remap_spatial_chunk_length,
                spatial_target_points=remap_spatial_target_points,
            )
            print(f"  Step 1 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

        # ── Step 2: Produce merged file from subsampled output ───────────────
        if params.produce_merged_file and sub_output:
            step_t0 = _time.monotonic()
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
            print(f"  Step 2 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

        # ── Step 3: Remap to original input files ────────────────────────────
        orig_output = output_dir / "original_with_predictions"
        step_t0 = _time.monotonic()
        print(f"\nStep 3: Remapping collections to original files → {orig_output}")
        if _dir_has_pointcloud_outputs(orig_output):
            print(f"  Existing remap outputs detected in {orig_output}; existing files will be skipped.")
        remap_collections_to_original_files(
            collections=remap_sources,
            original_input_dir=original_input_dir,
            output_dir=orig_output,
            tolerance=0.1,
            retile_buffer=2.0,
            chunk_size=merge_chunk_size,
            target_dims=remap_dims_set,
            spatial_slices=remap_spatial_slices,
            spatial_chunk_length=remap_spatial_chunk_length,
            spatial_target_points=remap_spatial_target_points,
        )
        print(f"  Step 3 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

        # ── Step 4: Fallback merged file directly from provided collections ───
        if params.produce_merged_file and not sub_output:
            step_t0 = _time.monotonic()
            merged_all = output_dir / "merged_with_all_dims.laz"
            if merged_all.exists():
                print(f"\nStep 4: merged output already exists, skipping collection concatenation: {merged_all}")
            else:
                if canonical_projected_collection is not None and Path(canonical_projected_collection).exists():
                    from merge_tiles import list_pointcloud_files

                    prepared_input = fused_merged_source or canonical_projected_collection
                    prepared_files = (
                        [prepared_input]
                        if Path(prepared_input).is_file()
                        else list_pointcloud_files(Path(prepared_input))
                    )
                    print(
                        f"\nStep 4: No subsampled target provided; "
                        f"concatenating canonical projected collection → {merged_all.name}"
                    )
                    _concat_laz_files(prepared_files, merged_all, chunk_size=merge_chunk_size)
                elif fused_source_collection is not None:
                    from merge_tiles import list_pointcloud_files

                    fused_input = fused_merged_source or fused_source_collection
                    fused_files = (
                        [fused_input]
                        if Path(fused_input).is_file()
                        else list_pointcloud_files(Path(fused_input))
                    )
                    print(
                        f"\nStep 4: No subsampled target provided; "
                        f"concatenating fused aligned source collection → {merged_all.name}"
                    )
                    _concat_laz_files(fused_files, merged_all, chunk_size=merge_chunk_size)
                elif len(raw_collections) > 1:
                    raise RuntimeError(
                        "Multi-collection remap expected a fused source collection for merged output creation, "
                        "but none was available."
                    )
                else:
                    print(
                        f"\nStep 4: No subsampled target provided; "
                        f"concatenating provided collections → {merged_all.name}"
                    )
                    _concat_segmented_collections_with_dim_union(
                        collections=collections,
                        output_path=merged_all,
                        chunk_size=merge_chunk_size,
                        target_dims=remap_dims_set,
                    )
            if merged_all.exists():
                print(f"  Final merged dims: {_describe_laz_dimensions(merged_all)}")
            print(f"  Step 4 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

        # ── Option: enrich merged LAZ with original dims ──────────────────────
        if params.transfer_original_dims_to_merged:
            merged_laz_path = merged_all
            if merged_laz_path is None and params.merged_laz:
                merged_laz_path = Path(params.merged_laz)
            direct_enrichment_source = None
            if fused_merged_source is not None and Path(fused_merged_source).exists():
                direct_enrichment_source = Path(fused_merged_source)
            enrichment_source = direct_enrichment_source or merged_laz_path
            if enrichment_source is not None and Path(enrichment_source).exists():
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
                    step_t0 = _time.monotonic()
                    if direct_enrichment_source is not None:
                        print(
                            f"\nEnriching fused merged source with original dims in one step → "
                            f"{output_merged_with_originals.name}"
                        )
                    else:
                        print(f"\nEnriching merged LAZ with original dims → {output_merged_with_originals.name}")
                    add_original_dimensions_to_merged(
                        enrichment_source, original_input_dir, output_merged_with_originals,
                        tolerance=0.1, retile_buffer=2.0,
                        num_threads=max(1, params.workers),
                        target_dims=_target_dims,
                        merge_chunk_size=merge_chunk_size,
                        spatial_slices=remap_spatial_slices,
                        spatial_chunk_length=remap_spatial_chunk_length,
                        spatial_target_points=remap_spatial_target_points,
                    )
                    print(f"  Merged enrichment duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

        print()
        print("=" * 60)
        print("Remap complete")
        print("=" * 60)
        if sub_output:
            print(f"  Subsampled files:   {sub_output}")
        if merged_all is not None:
            print(f"  Merged:             {merged_all}")
        if params.transfer_original_dims_to_merged:
            output_merged_with_originals = (
                Path(params.output_merged_with_originals)
                if params.output_merged_with_originals
                else output_dir / "merged_with_originals.laz"
            )
            if output_merged_with_originals.exists():
                print(f"  Merged+originals:   {output_merged_with_originals}")
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

    remap_to_original_input_files_streaming(
        merged_file=merged_laz,
        original_input_dir=original_input_dir,
        output_dir=output_dir,
        tolerance=tolerance,
        retile_buffer=retile_buffer,
        spatial_slices=params.remap_spatial_slices,
        spatial_chunk_length=params.remap_spatial_chunk_length,
        spatial_target_points=params.remap_spatial_target_points,
        threedtrees_dims=threedtrees_dims,
        threedtrees_suffix=threedtrees_suffix,
    )

    # Add dimensions from original files to the merged file (optional)
    if params.transfer_original_dims_to_merged:
        # Compute target dims from standardization JSON if provided
        output_merged_with_originals = (
            Path(params.output_merged_with_originals)
            if params.output_merged_with_originals is not None
            else output_dir / "merged_with_originals.laz"
        )
        _target_dims = None
        if params.standardization_json is not None:
            _target_dims = load_standardization_dims(params.standardization_json)
            print(f"  Standardization: filtering merged enrichment to {len(_target_dims)} dims from {params.standardization_json.name}")
        add_original_dimensions_to_merged(
            merged_laz,
            original_input_dir,
            output_merged_with_originals,
            tolerance=tolerance,
            retile_buffer=retile_buffer,
            num_threads=max(1, params.workers),
            target_dims=_target_dims,
            spatial_slices=params.remap_spatial_slices,
            spatial_chunk_length=params.remap_spatial_chunk_length,
            spatial_target_points=params.remap_spatial_target_points,
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
        '--force-spatial-chunked-fusion', '--force_spatial_chunked_fusion',
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
