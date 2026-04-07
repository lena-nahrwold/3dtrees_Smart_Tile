from __future__ import annotations

import json as _json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from parameters import Parameters

from filter_task_support import (
    ACTIVE_FILTER_PARAM_NAMES,
    describe_collection_source_mode,
    effective_copc_chunk_size,
    format_active_param_names,
    prepare_filtered_outputs_for_remap,
    resolve_remap_target_dims,
)
from merge_tiles import (
    build_neighbor_graph_from_bounds_json,
    get_tile_bounds_from_header,
    remap_collections_to_original_files,
)
from merge_tiles_streaming import (
    TILE_OFFSET,
    _canonical_tile_name_for_json_index,
    _extract_tile_metadata_wrapper,
    _match_tiles_to_json_bounds,
    _preferred_json_bounds_field,
    redistribute_small_instances,
    write_filtered_tiles_streaming,
)


@dataclass(frozen=True)
class FilterTaskDependencies:
    update_trees_files_with_global_ids: Callable
    pointcloud_key_fn: Callable[[Path], str]
    convert_collection_file_to_copc: Callable


def run_filter_task(params: Parameters, deps: FilterTaskDependencies):
    """
    Run the buffer-instance filter stage.

    Active knobs for this task live in ``ACTIVE_FILTER_PARAM_NAMES``. Other
    shared CLI flags are accepted through the central ``Parameters`` model for
    backwards compatibility but are not interpreted here.
    """
    raw_folders = params.segmented_folders or ""
    input_dirs = [Path(p.strip()) for p in raw_folders.split(",") if p.strip()]
    if not input_dirs:
        fallback = params.segmented_remapped_folder or params.subsampled_10cm_folder
        if fallback:
            input_dirs = [Path(fallback)]
    if not input_dirs:
        print("Error: --segmented-folders or --segmented-remapped-folder is required for filter task")
        sys.exit(1)
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

    tile_bounds_json = params.tile_bounds_json
    if not tile_bounds_json:
        print("Error: --tile-bounds-json is required for filter task (needed for neighbor info)")
        sys.exit(1)
    tile_bounds_json = Path(tile_bounds_json)
    if not tile_bounds_json.exists():
        print(f"Error: tile_bounds_tindex.json not found: {tile_bounds_json}")
        sys.exit(1)

    output_dir = Path(params.output_dir) if params.output_dir else input_dirs[0].parent / "filtered"
    filtered_output_dir = output_dir / "filtered_tiles"
    filtered_copc_dir = output_dir / "filtered_tiles_copc"
    workers = params.workers
    border_zone_width = params.border_zone_width
    instance_dimension = params.instance_dimension
    merge_chunk_size = params.merge_chunk_size
    copc_chunk_size = effective_copc_chunk_size(params, merge_chunk_size)

    print("=" * 60)
    print("Running Filter Task")
    print("=" * 60)
    print(f"  Input dirs:            {[str(d) for d in input_dirs]}")
    print(f"  Tile bounds JSON:      {tile_bounds_json}")
    print(f"  Output:                {filtered_output_dir}")
    print(f"  Border zone:           {border_zone_width}m")
    print(f"  Instance dim:          {instance_dimension}")
    print(f"  Filter anchor:         {params.filter_anchor}")
    print(f"  Workers:               {workers}")
    print(f"  Merge chunk size:      {merge_chunk_size:,} points")
    print(f"  COPC prep chunk size:  {copc_chunk_size:,} points")
    print(f"  Active task params:    {format_active_param_names(ACTIVE_FILTER_PARAM_NAMES)}")

    laz_files = []
    for input_dir in input_dirs:
        found = sorted(input_dir.glob("*.laz")) or sorted(input_dir.glob("*.las"))
        laz_files.extend(found)
    if not laz_files:
        print(f"Error: No LAZ/LAS files found in any of: {[str(d) for d in input_dirs]}")
        sys.exit(1)
    print(f"\nFound {len(laz_files)} tiles across {len(input_dirs)} input director(ies)")

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
    for path in laz_files:
        bounds = get_tile_bounds_from_header(path)
        if bounds:
            file_key = str(path)
            file_boundaries[file_key] = bounds
            file_key_to_path[file_key] = path

    tile_to_json, _json_to_tile = _match_tiles_to_json_bounds(
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
    json_tiles = json_data.get("tiles", [])
    for json_idx, tile_name in json_idx_to_tile_name.items():
        if json_idx < len(json_tiles) and "core" in json_tiles[json_idx]:
            core = json_tiles[json_idx]["core"]
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
                f"skipping tree-file update for tile {tile_name}",
                flush=True,
            )

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

    def _ns_sort_key(result):
        bounds = core_bounds_by_tile.get(result.tile_name)
        if bounds is None:
            bounds = tile_boundaries.get(result.tile_name)
        if bounds:
            mn_x, mx_x, mn_y, mx_y = bounds
            return (-(mn_y + mx_y) / 2, (mn_x + mx_x) / 2)
        return (0.0, 0.0)

    tile_results_ns = sorted(tile_results, key=_ns_sort_key)

    global_to_merged = {}
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

    write_filtered_tiles_streaming(
        tile_results=tile_results,
        neighbors_by_tile=neighbors_by_tile,
        core_bounds_by_tile=core_bounds_by_tile,
        output_dir=filtered_output_dir,
        copc_output_dir=filtered_copc_dir,
        copc_chunk_size=copc_chunk_size,
        instance_dimension=instance_dimension,
        chunk_size=merge_chunk_size,
        filter_anchor=params.filter_anchor,
        global_to_merged=global_to_merged,
    )

    filtered_remap_source = prepare_filtered_outputs_for_remap(
        source_collection=filtered_output_dir,
        output_dir=output_dir,
        chunk_size=copc_chunk_size,
        convert_collection_file_to_copc=deps.convert_collection_file_to_copc,
        pointcloud_key_fn=deps.pointcloud_key_fn,
    )
    print(
        f"  Preferred remap source: {filtered_remap_source} "
        f"[{describe_collection_source_mode(Path(filtered_remap_source))}]",
        flush=True,
    )

    filtered_trees_dir = output_dir / "filtered_trees"
    print(f"\n{'=' * 60}")
    print("Updating trees files with global IDs")
    print(f"{'=' * 60}")
    print(f"  Scanning for .txt in: {input_dirs}")
    print(f"  Output:               {filtered_trees_dir}")
    deps.update_trees_files_with_global_ids(
        tile_results=tile_results,
        global_to_merged=global_to_merged,
        trees_by_tile=trees_by_tile,
        trees_output_dir=filtered_trees_dir,
        tile_offset=TILE_OFFSET,
    )

    if getattr(params, "subsampled_target_folder", None):
        sub_target = Path(params.subsampled_target_folder)
        sub_output = output_dir / "subsampled_with_predictions"
        remap_dims_set = resolve_remap_target_dims(params)
        print(f"\n{'=' * 60}")
        print("Remapping filtered tiles to subsampled target")
        print(f"{'=' * 60}")
        print(f"  Source: {filtered_remap_source}")
        print(f"  Source mode: {describe_collection_source_mode(Path(filtered_remap_source))}")
        print(f"  Target: {sub_target}")
        print(f"  Output: {sub_output}")
        if params.remap_spatial_target_points:
            print(f"  Spatial target points: {params.remap_spatial_target_points:,}")
        elif params.remap_spatial_chunk_length:
            print(f"  Spatial chunk length: {params.remap_spatial_chunk_length} m")
        else:
            print(f"  Spatial slices: {params.remap_spatial_slices}")
        remap_collections_to_original_files(
            collections=[Path(filtered_remap_source)],
            original_input_dir=sub_target,
            output_dir=sub_output,
            tolerance=0.1,
            retile_buffer=2.0,
            chunk_size=merge_chunk_size,
            target_dims=remap_dims_set,
            spatial_slices=params.remap_spatial_slices,
            spatial_chunk_length=params.remap_spatial_chunk_length,
            spatial_target_points=params.remap_spatial_target_points,
        )

    print()
    print("=" * 60)
    print("Filter Task Complete")
    print("=" * 60)
    print(f"Output: {filtered_output_dir}")
    if Path(filtered_remap_source) != filtered_output_dir:
        print(f"COPC:   {filtered_remap_source}")
    print(f"Trees:  {filtered_trees_dir}")

    return dict(
        tile_results=tile_results,
        global_to_merged=global_to_merged,
        neighbors_by_tile=neighbors_by_tile,
        core_bounds_by_tile=core_bounds_by_tile,
        filtered_output_dir=filtered_output_dir,
        filtered_remap_source=Path(filtered_remap_source),
        filtered_trees_dir=filtered_trees_dir,
        instance_dimension=instance_dimension,
        merge_chunk_size=merge_chunk_size,
    )
