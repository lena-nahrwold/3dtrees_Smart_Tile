from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from parameters import Parameters


@dataclass(frozen=True)
class RemapTaskDependencies:
    concat_laz_files: Callable
    describe_laz_dimensions: Callable
    dir_has_pointcloud_outputs: Callable
    collect_pointcloud_outputs: Callable
    resolve_remap_target_dims: Callable
    ensure_collections_copc: Callable
    fuse_aligned_collections_to_copc: Callable
    fuse_collections_by_spatial_chunks_to_copc: Callable


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f} min"


def run_remap_task(params: Parameters, deps: RemapTaskDependencies):
    """Run the collection-based remap task."""
    try:
        from merge_tiles import (
            enrich_collection_tiles_with_original_dimensions,
            load_standardization_dims,
            remap_collections_to_original_files,
        )
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        sys.exit(1)

    raw_folders = params.segmented_folders or ""
    collections = [Path(p.strip()) for p in raw_folders.split(",") if p.strip()]
    if not collections:
        print("Error: --segmented-folders is required for remap task")
        sys.exit(1)
    if not params.original_input_dir:
        print("Error: --original-input-dir is required for remap task")
        sys.exit(1)

    original_input_dir = Path(params.original_input_dir)
    if not original_input_dir.exists():
        print(f"Error: original-input-dir not found: {original_input_dir}")
        sys.exit(1)

    output_dir = Path(params.output_dir) if params.output_dir else original_input_dir.parent / "remap_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_chunk_size = int(params.chunk_size)
    copc_chunk_size = max(merge_chunk_size, 20_000_000)
    remap_dims_set = deps.resolve_remap_target_dims(params)

    print("=" * 60)
    print("Remap: provided collections -> subsampled + original files")
    print("=" * 60)
    print(f"  Collections:           {[str(c) for c in collections]}")
    print(f"  Originals:             {original_input_dir}")
    print(f"  Output:                {output_dir}")
    print(f"  Remap dims:            {sorted(remap_dims_set) if remap_dims_set else 'all extra dims'}")
    print(f"  Produce merged file:   {params.produce_merged_file}")
    print(f"  Chunk size:            {merge_chunk_size:,} points")
    print(
        "  Spatial slicing:       automatic "
        "(ceil(points per cloud / chunk-size) internal slices)",
        flush=True,
    )

    raw_collections = list(collections)
    remap_sources = raw_collections
    fused_source_collection = None

    if len(raw_collections) > 1:
        projection_t0 = _time.monotonic()
        print(
            "\nChecking whether the provided collections already share the same tile-local point sets...",
            flush=True,
        )
        try:
            fused_source_collection = deps.fuse_aligned_collections_to_copc(
                collections=raw_collections,
                output_dir=output_dir,
                chunk_size=merge_chunk_size,
                copc_chunk_size=copc_chunk_size,
                target_dims=remap_dims_set,
                workers=params.workers,
            )
            if fused_source_collection is not None:
                remap_sources = [Path(fused_source_collection)]
                print(
                    "  Collections verified as aligned; using tiled fused COPC source "
                    f"for remap: {remap_sources[0]}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  Aligned-fusion verification failed: {exc}", flush=True)

        if fused_source_collection is None:
            normalize_t0 = _time.monotonic()
            print("\nNormalizing provided collections to COPC for canonical-geometry projection...")
            collections = deps.ensure_collections_copc(
                collections=raw_collections,
                output_dir=output_dir,
                chunk_size=copc_chunk_size,
            )
            print(f"  Normalized collections: {[str(c) for c in collections]}")
            print(
                f"  Collection normalization duration: {_fmt_elapsed(_time.monotonic() - normalize_t0)}",
                flush=True,
            )
            print(
                "\nCollections are not tile-identical; attempting automatic chunked spatial fusion...",
                flush=True,
            )
            try:
                fused_source_collection = deps.fuse_collections_by_spatial_chunks_to_copc(
                    collections=collections,
                    output_dir=output_dir,
                    chunk_size=merge_chunk_size,
                    copc_chunk_size=copc_chunk_size,
                    target_dims=remap_dims_set,
                    spatial_target_points=merge_chunk_size,
                )
                if fused_source_collection is not None:
                    remap_sources = [Path(fused_source_collection)]
                    print(
                        "  Chunked spatial fusion succeeded; using tiled fused COPC source "
                        f"for remap: {remap_sources[0]}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"  Chunked spatial fusion failed: {exc}", flush=True)

        if fused_source_collection is None:
            canonical_collection = Path(collections[-1])
            source_collections = list(collections[:-1])
            canonical_projected_collection = output_dir / "canonical_collection_with_predictions"

            print(
                f"\nProjecting {len(source_collections)} collection(s) onto canonical geometry: "
                f"{canonical_collection}",
                flush=True,
            )
            if deps.dir_has_pointcloud_outputs(canonical_projected_collection):
                print(
                    f"  Existing projected canonical collection detected in "
                    f"{canonical_projected_collection}; skipping reprojection step.",
                    flush=True,
                )
            else:
                remap_collections_to_original_files(
                    collections=source_collections,
                    original_input_dir=canonical_collection,
                    output_dir=canonical_projected_collection,
                    tolerance=0.1,
                    retile_buffer=2.0,
                    chunk_size=merge_chunk_size,
                    target_dims=remap_dims_set,
                    spatial_target_points=merge_chunk_size,
                )

            print("\nNormalizing projected canonical collection to COPC for tiled remap...", flush=True)
            normalized_canonical = deps.ensure_collections_copc(
                collections=[canonical_projected_collection],
                output_dir=output_dir,
                chunk_size=copc_chunk_size,
            )
            fused_source_collection = Path(normalized_canonical[0])
            remap_sources = [fused_source_collection]
            print(
                "  Using projected canonical tiled source for remap: "
                f"{remap_sources[0]}",
                flush=True,
            )

        print(
            f"  Multi-collection source-prep duration: "
            f"{_fmt_elapsed(_time.monotonic() - projection_t0)}",
            flush=True,
        )

    if len(raw_collections) == 1 and fused_source_collection is None:
        normalize_t0 = _time.monotonic()
        print("\nNormalizing provided collections to COPC for tiled remap (reuse cache when available)...")
        collections = deps.ensure_collections_copc(
            collections=raw_collections,
            output_dir=output_dir,
            chunk_size=copc_chunk_size,
        )
        print(f"  Normalized collections: {[str(c) for c in collections]}")
        remap_sources = collections
        print(
            f"  Collection normalization duration: {_fmt_elapsed(_time.monotonic() - normalize_t0)}",
            flush=True,
        )

    merged_all = None
    tile_enriched_output = output_dir / "tiles_with_original_dimensions"
    merge_enrichment_target_dims = None
    if params.standardization_json is not None:
        merge_enrichment_target_dims = load_standardization_dims(params.standardization_json)
        print(
            f"  Standardization: filtering tile enrichment to "
            f"{len(merge_enrichment_target_dims)} dims from {params.standardization_json.name}",
            flush=True,
        )
    if params.produce_merged_file and params.transfer_original_dims_to_merged:
        if merge_enrichment_target_dims is not None and remap_dims_set is not None:
            print(
                "  Merged output policy: standardization-json original attrs + "
                "remap-dims-selected source dims",
                flush=True,
            )
        elif merge_enrichment_target_dims is not None:
            print(
                "  Merged output policy: standardization-json original attrs + "
                "all source extra dims",
                flush=True,
            )
        elif remap_dims_set is not None:
            print(
                "  Merged output policy: all original attrs + "
                "remap-dims-selected source dims",
                flush=True,
            )
        else:
            print(
                "  Merged output policy: all original attrs + all source extra dims",
                flush=True,
            )
    tile_source_collection = remap_sources[0] if len(remap_sources) == 1 else None
    tiled_merge_inputs = deps.collect_pointcloud_outputs(remap_sources)

    sub_output = None
    if params.subsampled_target_folder:
        sub_target = Path(params.subsampled_target_folder)
        sub_output = output_dir / "subsampled_with_predictions"
        step_t0 = _time.monotonic()
        print(f"\nStep 1: Remapping collections to subsampled target -> {sub_output}")
        if deps.dir_has_pointcloud_outputs(sub_output):
            print(f"  Existing remap outputs detected in {sub_output}; skipping step.")
        else:
            remap_collections_to_original_files(
                collections=remap_sources,
                original_input_dir=sub_target,
                output_dir=sub_output,
                tolerance=0.1,
                retile_buffer=2.0,
                chunk_size=merge_chunk_size,
                target_dims=remap_dims_set,
                spatial_target_points=merge_chunk_size,
            )
        print(f"  Step 1 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

    if params.produce_merged_file and sub_output:
        step_t0 = _time.monotonic()
        out_files = sorted(sub_output.glob("*.laz")) + sorted(sub_output.glob("*.las"))
        if out_files:
            merged_all = output_dir / "merged_with_all_dims.laz"
            if merged_all.exists():
                print(f"\nStep 2: merged output already exists, skipping concatenation: {merged_all}")
            else:
                print(f"\nStep 2: Concatenating {len(out_files)} subsampled files -> {merged_all.name}")
                deps.concat_laz_files(out_files, merged_all, chunk_size=merge_chunk_size)
            print(f"  Final merged dims: {deps.describe_laz_dimensions(merged_all)}")
        else:
            print("  Warning: no subsampled output files found; skipping merged output")
        print(f"  Step 2 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

    orig_output = output_dir / "original_with_predictions"
    step_t0 = _time.monotonic()
    print(f"\nStep 3: Remapping collections to original files -> {orig_output}")
    if deps.dir_has_pointcloud_outputs(orig_output):
        print(f"  Existing remap outputs detected in {orig_output}; skipping step.")
    else:
        remap_collections_to_original_files(
            collections=remap_sources,
            original_input_dir=original_input_dir,
            output_dir=orig_output,
            tolerance=0.1,
            retile_buffer=2.0,
            chunk_size=merge_chunk_size,
            target_dims=remap_dims_set,
            spatial_target_points=merge_chunk_size,
        )
    print(f"  Step 3 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

    build_plain_tile_merge = (
        params.produce_merged_file
        and not sub_output
        and not params.transfer_original_dims_to_merged
    )
    if params.produce_merged_file and not sub_output and params.transfer_original_dims_to_merged:
        print(
            "\nStep 4: Skipping standalone merged_with_all_dims.laz because "
            "tile enrichment is enabled; the final merged output will be built "
            "from enriched tiles instead.",
            flush=True,
        )
    if build_plain_tile_merge:
        step_t0 = _time.monotonic()
        merged_all = output_dir / "merged_with_all_dims.laz"
        if merged_all.exists():
            print(f"\nStep 4: merged output already exists, skipping tiled concatenation: {merged_all}")
        else:
            if not tiled_merge_inputs:
                print("  Warning: no duplicate-free remap tiles found; skipping final merged output", flush=True)
            else:
                print(
                    "\nStep 4: No subsampled target provided; "
                    f"concatenating duplicate-free remap tiles -> {merged_all.name}"
                )
                deps.concat_laz_files(
                    tiled_merge_inputs,
                    merged_all,
                    chunk_size=merge_chunk_size,
                )
        if merged_all.exists():
            print(f"  Final merged dims: {deps.describe_laz_dimensions(merged_all)}")
        print(f"  Step 4 duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

    if params.produce_merged_file and params.transfer_original_dims_to_merged:
        output_merged_with_originals = (
            Path(params.output_merged_with_originals)
            if params.output_merged_with_originals
            else output_dir / "merged_with_originals.laz"
        )
        if output_merged_with_originals.exists():
            print(
                "\nMerged-with-originals output already exists, skipping concatenation: "
                f"{output_merged_with_originals}"
            )
        else:
            step_t0 = _time.monotonic()
            if deps.dir_has_pointcloud_outputs(tile_enriched_output):
                print(
                    f"\nStep 5: Existing tile enrichment outputs detected in "
                    f"{tile_enriched_output}; existing files will be skipped.",
                    flush=True,
                )
            else:
                print(
                    "\nStep 5: Enriching duplicate-free remap tiles with original attributes "
                    f"-> {tile_enriched_output}",
                    flush=True,
                )
            if not tile_source_collection:
                print(
                    "  Warning: remap tile source did not resolve to a single collection; "
                    "skipping merged-with-originals output",
                    flush=True,
                )
            elif tiled_merge_inputs:
                enrich_collection_tiles_with_original_dimensions(
                    tile_collection=tile_source_collection,
                    original_input_dir=original_input_dir,
                    output_dir=tile_enriched_output,
                    tolerance=0.1,
                    retile_buffer=2.0,
                    num_threads=max(1, params.workers),
                    target_dims=merge_enrichment_target_dims,
                    source_extra_dims_to_keep=remap_dims_set,
                    merge_chunk_size=merge_chunk_size,
                    spatial_target_points=merge_chunk_size,
                )
                enriched_tile_files = deps.collect_pointcloud_outputs([tile_enriched_output])
                if not enriched_tile_files:
                    print(
                        "  Warning: no enriched remap tiles found; skipping merged-with-originals output",
                        flush=True,
                    )
                else:
                    print(
                        "\nConcatenating enriched remap tiles with original attributes "
                        f"+ prediction dims -> {output_merged_with_originals.name}",
                        flush=True,
                    )
                    deps.concat_laz_files(
                        enriched_tile_files,
                        output_merged_with_originals,
                        chunk_size=merge_chunk_size,
                    )
            else:
                print(
                    "  Warning: no duplicate-free remap tiles found; skipping merged-with-originals output",
                    flush=True,
                )
            print(f"  Merged+originals duration: {_fmt_elapsed(_time.monotonic() - step_t0)}", flush=True)

    print()
    print("=" * 60)
    print("Remap complete")
    print("=" * 60)
    if sub_output:
        print(f"  Subsampled files:   {sub_output}")
    if merged_all is not None:
        print(f"  Merged:             {merged_all}")
    if params.produce_merged_file and params.transfer_original_dims_to_merged and deps.dir_has_pointcloud_outputs(tile_enriched_output):
        print(f"  Enriched tiles:     {tile_enriched_output}")
    if params.produce_merged_file and params.transfer_original_dims_to_merged:
        output_merged_with_originals = (
            Path(params.output_merged_with_originals)
            if params.output_merged_with_originals
            else output_dir / "merged_with_originals.laz"
        )
        if output_merged_with_originals.exists():
            print(f"  Merged+originals:   {output_merged_with_originals}")
    print(f"  Original files:     {orig_output}")
