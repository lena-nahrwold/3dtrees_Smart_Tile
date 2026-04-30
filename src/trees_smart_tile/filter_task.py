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
    ACTIVE_FILTER_REMAP_PARAM_NAMES,
    _all_tree_xy_centroids,
    _count_tree_instances,
    _fraction_in_bounds,
    _read_laz_point_count,
    classify_tree_sidecar_file,
    collect_pointcloud_files,
    copy_additional_collection_files,
    copy_mesh_sidecars_for_tiles,
    derive_border_zone_width_from_json,
    describe_collection_source_mode,
    discover_tree_sidecars_for_pointcloud,
    format_active_param_names,
    is_tree_sidecar_file,
    prepare_filtered_outputs_for_remap,
    resolve_filter_chunk_size,
    resolve_filter_input_paths,
    write_filtered_tile_manifest,
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

_INTERNAL_REMAP_SPATIAL_SLICES = 10


@dataclass(frozen=True)
class FilterTaskDependencies:
    update_trees_files_with_global_ids: Callable
    pointcloud_key_fn: Callable[[Path], str]
    convert_collection_file_to_copc: Callable
    run_remap_task: Callable


def run_filter_task(params: Parameters, deps: FilterTaskDependencies):
    """
    Run the buffer-instance filter stage.

    Active knobs for this task live in ``ACTIVE_FILTER_PARAM_NAMES``. Other
    shared CLI flags are accepted through the central ``Parameters`` model for
    backwards compatibility but are not interpreted here.
    """
    input_paths = resolve_filter_input_paths(params)
    if not input_paths:
        print("Error: --segmented-folders is required for filter task")
        sys.exit(1)
    for input_path in input_paths:
        if not input_path.exists():
            print(f"Error: Input path not found: {input_path}")
            sys.exit(1)

    tile_bounds_json = params.tile_bounds_json
    if tile_bounds_json:
        tile_bounds_json = Path(tile_bounds_json)
        if not tile_bounds_json.exists():
            print(f"Error: tile_bounds_tindex.json not found: {tile_bounds_json}")
            sys.exit(1)

    output_dir = Path(params.output_dir) if params.output_dir else input_paths[0].parent / "filtered"
    filtered_output_dir = output_dir / "filtered_tiles"
    filtered_copc_dir = output_dir / "filtered_tiles_copc"
    workers = params.workers
    remap_merge_requested = bool(params.remap_merge)
    has_original_inputs = bool(getattr(params, "original_input_dir", None))
    has_subsampled_target = bool(getattr(params, "subsampled_target_folder", None))
    remap_merge_effective = remap_merge_requested and has_original_inputs
    remap_to_subsampled_only = remap_merge_requested and (not has_original_inputs) and has_subsampled_target
    if remap_merge_requested and not has_original_inputs:
        if remap_to_subsampled_only:
            print(
                "  Note: --remap-merge was requested but --original-input-dir is missing; "
                "will remap filtered tiles to --subsampled-target-folder only.",
                flush=True,
            )
        else:
            print(
                "  Note: --remap-merge was requested but --original-input-dir is missing; "
                "skipping remap-merge tail.",
                flush=True,
            )

    instance_dimension = params.instance_dimension
    chunk_size = resolve_filter_chunk_size(params)
    active_param_names = (
        ACTIVE_FILTER_REMAP_PARAM_NAMES if remap_merge_effective else ACTIVE_FILTER_PARAM_NAMES
    )

    laz_files = []
    for input_path in input_paths:
        laz_files.extend(collect_pointcloud_files(input_path))
    if not laz_files:
        print(f"Error: No LAZ/LAS files found in any of: {[str(d) for d in input_paths]}")
        sys.exit(1)

    use_json_neighbors = tile_bounds_json is not None

    print("=" * 60)
    print("Running Filter Task")
    print("=" * 60)
    print(f"  Input paths:           {[str(d) for d in input_paths]}")
    print(f"  Tile bounds JSON:      {tile_bounds_json or '(not provided — using filename-based neighbors)'}")
    print(f"  Instance dim:          {instance_dimension}")
    print(f"  Filter anchor:         {params.filter_anchor}")
    print(f"  Workers:               {workers}")
    print(f"  Chunk size:            {chunk_size:,} points")
    print(f"  Remap merge:           {remap_merge_effective}")
    print(f"  Active task params:    {format_active_param_names(active_param_names)}")

    print(f"\nFound {len(laz_files)} tiles across {len(input_paths)} input path(s)")

    estimated_buffer = None  # set by filename-based path when overlap is detected

    if use_json_neighbors:
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
    else:
        print("  Building neighbor graph from cXX_rYY filename coordinates...")
        import re as _re
        _TILE_COORD_RE = _re.compile(r"c(\d+)_r(\d+)")

        file_boundaries = {}
        file_key_to_path: Dict[str, Path] = {}
        tile_name_by_path: Dict[Path, str] = {}
        coord_to_tile_name: Dict[tuple, str] = {}
        tile_name_to_coord: Dict[str, tuple] = {}

        for path in laz_files:
            m = _TILE_COORD_RE.search(path.stem)
            if m:
                col, row = int(m.group(1)), int(m.group(2))
                tile_name = f"c{col:02d}_r{row:02d}"
            else:
                tile_name = path.stem
                col, row = None, None

            tile_name_by_path[path] = tile_name
            bounds = get_tile_bounds_from_header(path)
            if bounds:
                file_key = str(path)
                file_boundaries[file_key] = bounds
                file_key_to_path[file_key] = path

            if col is not None:
                coord_to_tile_name[(col, row)] = tile_name
                tile_name_to_coord[tile_name] = (col, row)

        tile_boundaries = {}
        for path, tname in tile_name_by_path.items():
            bounds = get_tile_bounds_from_header(path)
            if bounds:
                tile_boundaries[tname] = bounds

        neighbors_by_tile = {}
        for tname, (col, row) in tile_name_to_coord.items():
            neighbors_by_tile[tname] = {
                "east":  coord_to_tile_name.get((col + 1, row)),
                "west":  coord_to_tile_name.get((col - 1, row)),
                "north": coord_to_tile_name.get((col, row + 1)),
                "south": coord_to_tile_name.get((col, row - 1)),
            }
        for path, tname in tile_name_by_path.items():
            if tname not in neighbors_by_tile:
                neighbors_by_tile[tname] = {
                    "east": None, "west": None, "north": None, "south": None,
                }

        # Estimate buffer from spatial overlap between adjacent tiles.
        # For east/west neighbors the X ranges overlap; for north/south the Y
        # ranges overlap.  The overlap width equals 2× the tile buffer.
        overlap_samples = []
        for tname, nbrs in neighbors_by_tile.items():
            tb = tile_boundaries.get(tname)
            if tb is None:
                continue
            minx, maxx, miny, maxy = tb
            east_name = nbrs.get("east")
            if east_name and east_name in tile_boundaries:
                nb = tile_boundaries[east_name]
                overlap_x = maxx - nb[0]  # this_maxx - neighbor_minx
                if overlap_x > 0:
                    overlap_samples.append(overlap_x / 2.0)
            north_name = nbrs.get("north")
            if north_name and north_name in tile_boundaries:
                nb = tile_boundaries[north_name]
                overlap_y = maxy - nb[2]  # this_maxy - neighbor_miny
                if overlap_y > 0:
                    overlap_samples.append(overlap_y / 2.0)

        if overlap_samples:
            estimated_buffer = float(sorted(overlap_samples)[len(overlap_samples) // 2])
        else:
            estimated_buffer = 0.0

        # Compute core bounds by shrinking header bounds by the estimated buffer.
        core_bounds_by_tile = {}
        for tname, (minx, maxx, miny, maxy) in tile_boundaries.items():
            core_bounds_by_tile[tname] = (
                minx + estimated_buffer,
                maxx - estimated_buffer,
                miny + estimated_buffer,
                maxy - estimated_buffer,
            )

        n_with_neighbors = sum(
            1 for nbrs in neighbors_by_tile.values()
            if any(v is not None for v in nbrs.values())
        )
        print(
            f"  Identified {len(neighbors_by_tile)} tiles from filenames, "
            f"{n_with_neighbors} with at least one neighbor",
        )
        if estimated_buffer > 0:
            print(f"  Estimated tile buffer from spatial overlap: {estimated_buffer:.1f}m")

    # ── Resolve border zone width ──────────────────────────────────────
    if params.border_zone_width is not None:
        border_zone_width = float(params.border_zone_width)
        border_zone_label = f"{border_zone_width}m (explicit parameter)"
    elif use_json_neighbors:
        border_zone_width = derive_border_zone_width_from_json(tile_bounds_json)
        border_zone_label = f"{border_zone_width}m (derived from tile_bounds_json)"
    elif estimated_buffer is not None and estimated_buffer > 0:
        border_zone_width = estimated_buffer
        border_zone_label = f"{border_zone_width:.1f}m (estimated from tile overlap)"
    else:
        border_zone_width = 0.0
        border_zone_label = "0.0m (single tile, no overlap detected)"

    print(f"  Output:                {filtered_output_dir}")
    print(f"  Border zone:           {border_zone_label}")

    pointcloud_count_by_parent: Dict[Path, int] = {}
    txt_files_by_parent: Dict[Path, list[Path]] = {}
    for filepath in laz_files:
        pointcloud_count_by_parent[filepath.parent] = pointcloud_count_by_parent.get(filepath.parent, 0) + 1
    for parent in pointcloud_count_by_parent:
        txt_files_by_parent[parent] = sorted(
            child for child in parent.glob("*.txt") if child.is_file()
        )

    # Fallback matching for tree files not resolved by name (e.g. Galaxy-flattened collections
    # where element identifiers don't carry the tile coordinate).
    #
    # Strategy 2 – instance-count ratio: sort tree files by row count and LAZ files by
    # header point count; the relative ordering is stable when density is uniform across
    # tiles and requires no point data to be loaded.
    #
    # Strategy 3 – coordinate overlap: when counts are tied, compute the fraction of tree
    # XY positions (one per row in the TXT) that fall inside each tile's core bounds and
    # assign greedily by highest overlap.
    inferred_tree_sidecars_by_pointcloud: Dict[Path, Dict[str, Path]] = {}
    for parent, txt_files in txt_files_by_parent.items():
        tree_candidates = [p for p in txt_files if is_tree_sidecar_file(p)]
        if not tree_candidates:
            continue

        tiles_in_parent = [
            (pc_path, tile_name_by_path.get(pc_path))
            for pc_path in laz_files
            if pc_path.parent == parent and tile_name_by_path.get(pc_path) is not None
        ]
        if not tiles_in_parent:
            continue

        laz_paths = [pc_path for pc_path, _ in tiles_in_parent]

        # Group tree candidates by kind so each kind is matched independently.
        trees_by_kind: Dict[str, list] = {}
        for tree_txt in tree_candidates:
            kind = classify_tree_sidecar_file(tree_txt)
            if kind:
                trees_by_kind.setdefault(kind, []).append(tree_txt)

        for kind, kind_trees in trees_by_kind.items():
            if len(kind_trees) != len(laz_paths):
                continue  # can't form a 1-to-1 assignment

            # --- Strategy 2: sort by count ratio ---
            tree_counts = {t: _count_tree_instances(t) for t in kind_trees}
            laz_pt_counts = {l: _read_laz_point_count(l) or 0 for l in laz_paths}

            sorted_trees = sorted(kind_trees, key=lambda t: tree_counts[t])
            sorted_lazs = sorted(laz_paths, key=lambda l: laz_pt_counts[l])

            tc_list = [tree_counts[t] for t in sorted_trees]
            lc_list = [laz_pt_counts[l] for l in sorted_lazs]
            has_ties = (
                any(tc_list[i] == tc_list[i + 1] for i in range(len(tc_list) - 1))
                or any(lc_list[i] == lc_list[i + 1] for i in range(len(lc_list) - 1))
            )

            if not has_ties:
                for tree_txt, laz_path in zip(sorted_trees, sorted_lazs):
                    inferred_tree_sidecars_by_pointcloud.setdefault(laz_path, {}).setdefault(kind, tree_txt)
                continue

            # --- Strategy 3: fraction of tree XY within tile core bounds ---
            tile_bounds_map = {
                pc_path: (core_bounds_by_tile.get(tname) or tile_boundaries.get(tname))
                for pc_path, tname in tiles_in_parent
            }
            tile_bounds_map = {k: v for k, v in tile_bounds_map.items() if v is not None}
            if not tile_bounds_map:
                continue

            tree_points = {t: _all_tree_xy_centroids(t) for t in kind_trees}

            scored = []
            for tree_txt in kind_trees:
                pts = tree_points[tree_txt]
                for laz_path in laz_paths:
                    bounds = tile_bounds_map.get(laz_path)
                    score = _fraction_in_bounds(pts, bounds) if bounds and pts else 0.0
                    scored.append((score, tree_txt, laz_path))
            scored.sort(key=lambda x: x[0], reverse=True)

            assigned_trees: set = set()
            assigned_lazs: set = set()
            for score, tree_txt, laz_path in scored:
                if score <= 0:
                    break
                if tree_txt in assigned_trees or laz_path in assigned_lazs:
                    continue
                inferred_tree_sidecars_by_pointcloud.setdefault(laz_path, {}).setdefault(kind, tree_txt)
                assigned_trees.add(tree_txt)
                assigned_lazs.add(laz_path)

    tree_texts_by_input_file: Dict[Path, Dict[str, Path]] = {}
    matched_tree_files = set()
    for filepath, tile_name in tile_name_by_path.items():
        matched = discover_tree_sidecars_for_pointcloud(
            pointcloud_path=filepath,
            tile_name=tile_name,
            sibling_txt_files=txt_files_by_parent.get(filepath.parent, []),
            pointcloud_key_fn=deps.pointcloud_key_fn,
            sibling_pointcloud_count=pointcloud_count_by_parent.get(filepath.parent, 1),
        )
        if not matched:
            inferred = inferred_tree_sidecars_by_pointcloud.get(filepath)
            if inferred:
                matched = dict(inferred)
        if matched:
            tree_texts_by_input_file[filepath] = matched
            matched_tree_files.update(matched.values())

    unmatched_tree_candidates = sorted(
        txt_path
        for txt_files in txt_files_by_parent.values()
        for txt_path in txt_files
        if is_tree_sidecar_file(txt_path) and txt_path not in matched_tree_files
    )
    for txt_path in unmatched_tree_candidates:
        print(
            f"  Warning: could not associate tree sidecar with a unique tile: {txt_path}",
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
            border_zone_width, 0.05, instance_dimension, core_boundary, chunk_size,
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

    print(f"  Volume merge:          {params.enable_volume_merge}")

    if params.enable_volume_merge:
        print(f"\n{'=' * 60}")
        print("Phase B.5: Small Instance Redistribution")
        print(f"{'=' * 60}")
        redistribute_small_instances(
            tile_results=tile_results,
            global_to_merged=global_to_merged,
            min_points_reassign=params.min_cluster_size or 300,
            max_volume_for_merge=params.max_volume_for_merge or 5.0,
            instance_dimension=instance_dimension,
            chunk_size=max(100_000, chunk_size // 4),
        )

    tile_output_states = write_filtered_tiles_streaming(
        tile_results=tile_results,
        neighbors_by_tile=neighbors_by_tile,
        core_bounds_by_tile=core_bounds_by_tile,
        output_dir=filtered_output_dir,
        copc_output_dir=filtered_copc_dir,
        copc_chunk_size=chunk_size,
        instance_dimension=instance_dimension,
        chunk_size=chunk_size,
        filter_anchor=params.filter_anchor,
        global_to_merged=global_to_merged,
    )
    created_tile_names = {
        tile_name
        for tile_name, state in tile_output_states.items()
        if state.get("created")
    }
    manifest_path, bounds_copy_path = write_filtered_tile_manifest(
        output_dir=output_dir,
        tile_output_states=tile_output_states,
        tile_bounds_json=tile_bounds_json,
    )
    print(f"  Tile manifest:         {manifest_path}", flush=True)
    if bounds_copy_path is not None:
        print(f"  Filtered bounds JSON:  {bounds_copy_path}", flush=True)

    filtered_remap_source = prepare_filtered_outputs_for_remap(
        source_collection=filtered_output_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        convert_collection_file_to_copc=deps.convert_collection_file_to_copc,
        pointcloud_key_fn=deps.pointcloud_key_fn,
    )
    print(
        f"  Preferred remap source: {filtered_remap_source} "
        f"[{describe_collection_source_mode(Path(filtered_remap_source))}]",
        flush=True,
    )
    remap_source_files = collect_pointcloud_files(Path(filtered_remap_source))
    if not remap_source_files:
        print(
            "  Note: no filtered point-cloud outputs were written; remap targets will be skipped.",
            flush=True,
        )

    filtered_trees_dir = output_dir / "filtered_trees"
    print(f"\n{'=' * 60}")
    print("Updating trees files with global IDs")
    print(f"{'=' * 60}")
    print(f"  Scanning for .txt in: {input_paths}")
    print(f"  Output:               {filtered_trees_dir}")
    deps.update_trees_files_with_global_ids(
        tile_results=tile_results,
        global_to_merged=global_to_merged,
        tree_texts_by_input_file=tree_texts_by_input_file,
        trees_output_dir=filtered_trees_dir,
        tile_offset=TILE_OFFSET,
        included_tile_names=created_tile_names,
    )
    # Tree sources already tied to a processed tile are rewritten in-place above.
    # Do not copy them again (same basename would become *_2.txt and break Galaxy outputs).
    # Resolve LAZ paths when matching dict keys (relative vs absolute Path keys).
    processed_laz_paths = {Path(r.filepath).resolve() for r in tile_results}
    tree_sources_written_with_global_ids = {
        Path(trees_file).resolve()
        for laz_path, trees_map in tree_texts_by_input_file.items()
        if Path(laz_path).resolve() in processed_laz_paths
        for trees_file in trees_map.values()
    }
    all_tree_sidecars = sorted(set(matched_tree_files) | set(unmatched_tree_candidates))
    tree_sidecars_to_copy = [
        txt_path
        for txt_path in all_tree_sidecars
        if txt_path.resolve() not in tree_sources_written_with_global_ids
    ]
    if tree_sidecars_to_copy:
        print("  Copying tree sidecars not already rewritten above into filtered_trees", flush=True)
        for txt_path in tree_sidecars_to_copy:
            dest = filtered_trees_dir / txt_path.name
            counter = 2
            while dest.exists():
                dest = filtered_trees_dir / f"{txt_path.stem}_{counter}{txt_path.suffix}"
                counter += 1
            dest.write_text(txt_path.read_text())
            print(f"  Tree sidecar copy: {txt_path.name} -> {dest.name}", flush=True)

    print(f"\n{'=' * 60}")
    print("Copying additional collection sidecars")
    print(f"{'=' * 60}")
    copied_sidecars = copy_additional_collection_files(
        input_paths=input_paths,
        destination_dir=filtered_output_dir,
        excluded_files=all_tree_sidecars,
    )
    print(f"  Additional sidecar files copied: {copied_sidecars}")

    print(f"\n{'=' * 60}")
    print("Copying mesh sidecars")
    print(f"{'=' * 60}")
    copied_mesh = copy_mesh_sidecars_for_tiles(
        tile_name_by_path={
            path: tile_name
            for path, tile_name in tile_name_by_path.items()
            if tile_name in created_tile_names
        },
        destination_dir=filtered_output_dir,
    )
    print(f"  Mesh sidecar files copied: {copied_mesh}")

    if getattr(params, "subsampled_target_folder", None) and not remap_merge_effective and remap_source_files:
        sub_target = Path(params.subsampled_target_folder)
        sub_output = output_dir / "subsampled_with_predictions"
        print(f"\n{'=' * 60}")
        print("Remapping filtered tiles to subsampled target")
        print(f"{'=' * 60}")
        print(f"  Source: {filtered_remap_source}")
        print(f"  Source mode: {describe_collection_source_mode(Path(filtered_remap_source))}")
        print(f"  Target: {sub_target}")
        print(f"  Output: {sub_output}")
        print(f"  Spatial slicing: automatic ({_INTERNAL_REMAP_SPATIAL_SLICES} internal slices)")
        # Match remap task semantics: --remap-dims is a strict allowlist, otherwise
        # transfer all extra dimensions. In subsampled-only mode, standardization
        # JSON is ignored (it filters original attributes for merged enrichment).
        from remap_task_support import _concat_laz_files, _resolve_remap_target_dims
        remap_dims_set = _resolve_remap_target_dims(params)
        remap_collections_to_original_files(
            collections=[Path(filtered_remap_source)],
            original_input_dir=sub_target,
            output_dir=sub_output,
            tolerance=0.1,
            retile_buffer=2.0,
            chunk_size=chunk_size,
            target_dims=remap_dims_set,
            spatial_slices=_INTERNAL_REMAP_SPATIAL_SLICES,
        )
        if params.produce_merged_file:
            out_files = sorted(sub_output.glob("*.laz")) + sorted(sub_output.glob("*.las"))
            if out_files:
                merged_all = output_dir / "merged_with_all_dims.laz"
                if merged_all.exists():
                    print(f"  Merged output already exists, skipping: {merged_all}", flush=True)
                else:
                    print(f"  Concatenating {len(out_files)} subsampled outputs -> {merged_all.name}", flush=True)
                    _concat_laz_files(out_files, merged_all, chunk_size=chunk_size)

    if remap_merge_effective and remap_source_files:
        if not params.original_input_dir:
            print("Error: --original-input-dir is required when --remap-merge is enabled")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("Running Remap/Merge Tail on Filtered Collection")
        print(f"{'=' * 60}")
        print(f"  Remap source: {filtered_remap_source}")
        print(f"  Source mode:  {describe_collection_source_mode(Path(filtered_remap_source))}")
        remap_params = params.model_copy(
            update={
                "task": "remap",
                "segmented_folders": str(filtered_remap_source),
                "output_dir": output_dir,
            }
        )
        deps.run_remap_task(remap_params)
    elif remap_merge_effective:
        print(
            "  Note: remap-merge requested, but no filtered tile outputs were written; skipping remap tail.",
            flush=True,
        )

    print()
    print("=" * 60)
    print("Filter Task Complete")
    print("=" * 60)
    print(f"Output: {filtered_output_dir}")
    if Path(filtered_remap_source) != filtered_output_dir:
        print(f"COPC:   {filtered_remap_source}")
    print(f"Trees:  {filtered_trees_dir}")
    print(f"Sidecars copied into filtered collection: {copied_sidecars}")

    return dict(
        tile_results=tile_results,
        global_to_merged=global_to_merged,
        neighbors_by_tile=neighbors_by_tile,
        core_bounds_by_tile=core_bounds_by_tile,
        filtered_output_dir=filtered_output_dir,
        filtered_remap_source=Path(filtered_remap_source),
        filtered_trees_dir=filtered_trees_dir,
        tile_output_states=tile_output_states,
        filtered_tile_manifest=manifest_path,
        filtered_tile_bounds_json=bounds_copy_path,
        instance_dimension=instance_dimension,
        merge_chunk_size=chunk_size,
    )
