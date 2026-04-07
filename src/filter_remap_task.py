from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from parameters import Parameters

from filter_task_support import (
    ACTIVE_FILTER_REMAP_PARAM_NAMES,
    describe_collection_source_mode,
    dir_has_pointcloud_outputs,
    effective_copc_chunk_size,
    format_active_param_names,
    prepare_filtered_outputs_for_remap,
    resolve_remap_target_dims,
)
from merge_tiles import (
    add_original_dimensions_to_merged,
    load_standardization_dims,
    remap_collections_to_original_files,
)


@dataclass(frozen=True)
class FilterRemapTaskDependencies:
    run_filter_task: Callable
    concat_laz_files: Callable
    describe_laz_dimensions: Callable
    pointcloud_key_fn: Callable[[Path], str]
    convert_collection_file_to_copc: Callable


def run_filter_remap_task(params: Parameters, deps: FilterRemapTaskDependencies):
    """
    Filter and remap orchestration.

    The shared ``Parameters`` model still accepts many cross-task flags for CLI
    compatibility, but this task only interprets the subset listed in
    ``ACTIVE_FILTER_REMAP_PARAM_NAMES``.
    """
    if not params.original_input_dir:
        print("Error: --original-input-dir is required for filter_remap task")
        sys.exit(1)
    original_input_dir = Path(params.original_input_dir)
    if not original_input_dir.exists():
        print(f"Error: original-input-dir not found: {original_input_dir}")
        sys.exit(1)

    raw_folders = params.segmented_folders or ""
    collections = [Path(p.strip()) for p in raw_folders.split(",") if p.strip()]
    if not collections:
        fallback = params.segmented_remapped_folder or params.subsampled_10cm_folder
        if fallback:
            collections = [Path(fallback)]
    if not collections:
        print("Error: --segmented-folders or --segmented-remapped-folder is required for filter_remap task")
        sys.exit(1)
    for collection in collections:
        if not collection.exists():
            print(f"Error: segmented folder not found: {collection}")
            sys.exit(1)

    output_dir = Path(params.output_dir) if params.output_dir else collections[0].parent / "filter_remap_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    remap_dims_set = resolve_remap_target_dims(params)
    copc_chunk_size = effective_copc_chunk_size(params, params.merge_chunk_size)

    print("=" * 60)
    print("Running Filter+Remap Task")
    print("=" * 60)
    print(f"  Collections:           {[str(c) for c in collections]}")
    print(f"  Original input:        {original_input_dir}")
    print(f"  Output:                {output_dir}")
    print(f"  Produce merged:        {params.produce_merged_file}")
    print(f"  Remap dims:            {remap_dims_set or 'all extra dims'}")
    print(f"  Merge chunk size:      {params.merge_chunk_size:,} points")
    print(f"  COPC prep chunk size:  {copc_chunk_size:,} points")
    print(f"  Active task params:    {format_active_param_names(ACTIVE_FILTER_REMAP_PARAM_NAMES)}")
    if params.remap_spatial_target_points:
        print(f"  Spatial target points: {params.remap_spatial_target_points:,}")
    elif params.remap_spatial_chunk_length:
        print(f"  Spatial chunk length:  {params.remap_spatial_chunk_length} m")
    else:
        print(f"  Spatial slices:        {params.remap_spatial_slices}")

    filtered_output_dir = output_dir / "filtered_tiles"
    filtered_trees_dir = output_dir / "filtered_trees"
    if dir_has_pointcloud_outputs(filtered_output_dir):
        print(f"\n{'=' * 60}")
        print("Step 1: Reusing existing filtered tiles")
        print(f"{'=' * 60}")
        print(f"  Reusing: {filtered_output_dir}")
        if filtered_trees_dir.exists():
            print(f"  Existing filtered trees: {filtered_trees_dir}")
        merge_chunk_size = params.merge_chunk_size
        filtered_remap_source = prepare_filtered_outputs_for_remap(
            source_collection=filtered_output_dir,
            output_dir=output_dir,
            chunk_size=copc_chunk_size,
            convert_collection_file_to_copc=deps.convert_collection_file_to_copc,
            pointcloud_key_fn=deps.pointcloud_key_fn,
        )
    else:
        filter_params = types.SimpleNamespace(**{
            k: getattr(params, k) for k in vars(params) if not k.startswith("_")
        })
        filter_params.segmented_folders = ",".join(str(c) for c in collections)
        filter_params.segmented_remapped_folder = None
        filter_params.subsampled_10cm_folder = None
        filter_params.subsampled_target_folder = None
        filter_params.output_dir = str(output_dir)

        filter_state = deps.run_filter_task(filter_params)
        if filter_state is None:
            print("Error: filter task failed")
            sys.exit(1)

        filtered_output_dir = filter_state["filtered_output_dir"]
        filtered_remap_source = filter_state["filtered_remap_source"]
        merge_chunk_size = filter_state["merge_chunk_size"]

    filtered_remap_source = Path(filtered_remap_source)
    print(
        f"  Preferred remap source: {filtered_remap_source} "
        f"[{describe_collection_source_mode(filtered_remap_source)}]",
        flush=True,
    )
    remap_sources = [filtered_remap_source]

    sub_output = None
    if getattr(params, "subsampled_target_folder", None):
        sub_target = Path(params.subsampled_target_folder)
        sub_output = output_dir / "subsampled_with_predictions"
        print(f"\n{'=' * 60}")
        print("Step 2a: Remapping filtered tiles to subsampled target")
        print(f"{'=' * 60}")
        print(f"  Source: {filtered_remap_source}")
        print(f"  Source mode: {describe_collection_source_mode(filtered_remap_source)}")
        print(f"  Target: {sub_target}")
        print(f"  Output: {sub_output}")
        if dir_has_pointcloud_outputs(sub_output):
            print(f"  Existing remap outputs detected in {sub_output}; existing files will be skipped.")
        remap_collections_to_original_files(
            collections=remap_sources,
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

    merged_laz = None
    if params.produce_merged_file and sub_output:
        out_files = sorted(sub_output.glob("*.laz")) + sorted(sub_output.glob("*.las"))
        if out_files:
            merged_laz = output_dir / "merged_with_all_dims.laz"
            if merged_laz.exists():
                print(f"\nMerged output already exists, skipping concatenation: {merged_laz}")
            else:
                print(f"\nConcatenating {len(out_files)} subsampled files → {merged_laz.name}")
                deps.concat_laz_files(out_files, merged_laz, chunk_size=merge_chunk_size)

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
                    print(f"\nEnriching merged file with original dims → {output_merged_with_originals.name}")
                    add_original_dimensions_to_merged(
                        merged_laz,
                        original_input_dir,
                        output_merged_with_originals,
                        tolerance=0.1,
                        retile_buffer=2.0,
                        num_threads=max(1, params.workers),
                        target_dims=target_dims,
                        merge_chunk_size=merge_chunk_size,
                        spatial_slices=params.remap_spatial_slices,
                        spatial_chunk_length=params.remap_spatial_chunk_length,
                        spatial_target_points=params.remap_spatial_target_points,
                    )
        else:
            print("  Warning: no subsampled output files found; skipping merged output")

    orig_predictions_dir = output_dir / "original_with_predictions"
    print(f"\n{'=' * 60}")
    print("Step 3: Remapping filtered tiles to original files")
    print(f"{'=' * 60}")
    print(f"  Source: {filtered_remap_source}")
    print(f"  Source mode: {describe_collection_source_mode(filtered_remap_source)}")
    print(f"  Target: {original_input_dir}")
    print(f"  Output: {orig_predictions_dir}")
    if dir_has_pointcloud_outputs(orig_predictions_dir):
        print(f"  Existing remap outputs detected in {orig_predictions_dir}; existing files will be skipped.")
    remap_collections_to_original_files(
        collections=remap_sources,
        original_input_dir=original_input_dir,
        output_dir=orig_predictions_dir,
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
    print("Filter+Remap Task Complete")
    print("=" * 60)
    print(f"  Filtered tiles:   {filtered_output_dir}")
    print(f"  Remap source:     {filtered_remap_source}")
    if merged_laz is not None and merged_laz.exists():
        print(f"  Merged:           {merged_laz}")
        print(f"  Final merged dims: {deps.describe_laz_dimensions(merged_laz)}")
    print(f"  Original files:   {orig_predictions_dir}")
