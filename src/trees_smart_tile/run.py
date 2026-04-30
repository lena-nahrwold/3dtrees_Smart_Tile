#!/usr/bin/env python3
"""
CLI orchestrator for the 3DTrees smart tile pipeline.

This file intentionally stays thin: it parses CLI arguments, builds task
dependencies, and dispatches to the dedicated task modules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from parameters import Parameters, print_params
except ImportError as e:
    print(f"Error: Could not import parameters.py: {e}")
    print("Please install required dependencies: pip install pydantic pydantic-settings")
    sys.exit(1)

from filter_task import FilterTaskDependencies, run_filter_task as _run_filter_task_impl
from filter_tree_files import update_trees_files_with_global_ids
from remap_task import RemapTaskDependencies, run_remap_task as _run_remap_task_impl
from remap_task_support import (
    _collect_pointcloud_outputs,
    _concat_laz_files,
    _convert_collection_file_to_copc,
    _describe_laz_dimensions,
    _dir_has_pointcloud_outputs,
    _ensure_collections_copc,
    _fuse_aligned_collections_to_copc,
    _fuse_collections_by_spatial_chunks_to_copc,
    _pointcloud_file_key,
    _resolve_remap_target_dims,
)
from tile_task import run_tile_task


def run_filter_task(params: Parameters):
    """Build dependencies and dispatch the filter task."""
    deps = FilterTaskDependencies(
        update_trees_files_with_global_ids=update_trees_files_with_global_ids,
        pointcloud_key_fn=_pointcloud_file_key,
        convert_collection_file_to_copc=_convert_collection_file_to_copc,
        run_remap_task=run_remap_task,
    )
    return _run_filter_task_impl(params, deps)


def run_remap_task(params: Parameters):
    """Build dependencies and dispatch the remap task."""
    deps = RemapTaskDependencies(
        concat_laz_files=_concat_laz_files,
        describe_laz_dimensions=_describe_laz_dimensions,
        dir_has_pointcloud_outputs=_dir_has_pointcloud_outputs,
        collect_pointcloud_outputs=_collect_pointcloud_outputs,
        resolve_remap_target_dims=_resolve_remap_target_dims,
        ensure_collections_copc=_ensure_collections_copc,
        fuse_aligned_collections_to_copc=_fuse_aligned_collections_to_copc,
        fuse_collections_by_spatial_chunks_to_copc=_fuse_collections_by_spatial_chunks_to_copc,
    )
    return _run_remap_task_impl(params, deps)


def preprocess_boolean_flags(args_list):
    """
    Convert bare boolean flags into explicit True/False values for Pydantic.
    """
    boolean_flags = [
        "--show-params", "--show_params",
        "--dimension-reduction", "--dimension_reduction",
        "--skip-dimension-reduction", "--skip_dimension_reduction",
        "--output-copc-res1", "--output_copc_res1",
        "--output-copc-res2", "--output_copc_res2",
        "--chunkwise-copc-source-creation", "--chunkwise_copc_source_creation",
        "--enable-volume-merge", "--enable_volume_merge",
        "--produce-merged-file", "--produce_merged_file",
        "--transfer-original-dims-to-merged", "--transfer_original_dims_to_merged",
        "--remap-merge", "--remap_merge",
    ]

    processed = []
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg in boolean_flags:
            if i + 1 < len(args_list) and args_list[i + 1] in ["True", "False"]:
                processed.extend([arg, args_list[i + 1]])
                i += 2
            else:
                processed.extend([arg, "True"])
                i += 1
        else:
            processed.append(arg)
            i += 1
    return processed


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--show-params", "--show_params", action="store_true")
    pre_args, remaining_args = pre_parser.parse_known_args()

    processed_args = [sys.argv[0]] + preprocess_boolean_flags(remaining_args)
    original_argv = sys.argv
    sys.argv = processed_args
    try:
        params = Parameters()
    except Exception as e:
        print(f"Error parsing parameters: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv

    if pre_args.show_params:
        print_params(params)
        sys.exit(0)

    if not params.task:
        print("Error: --task is required (unless using --show-params)")
        print("       python run.py --task tile --input-dir /path/to/input --output-dir /path/to/output")
        print("       python run.py --task filter --segmented-folders /path/to/tiles --tile-bounds-json /path/to/tindex.json --output-dir /path/to/output")
        print("       python run.py --task remap --segmented-folders /path/to/tiles --original-input-dir /path/to/originals --output-dir /path/to/output")
        print("       python run.py --task filter --segmented-folders /path/to/tiles --tile-bounds-json /path/to/tindex.json --remap-merge --original-input-dir /path/to/originals --output-dir /path/to/output")
        print("       python run.py --show-params")
        sys.exit(1)

    if params.task == "tile":
        run_tile_task(params)
    elif params.task == "remap":
        run_remap_task(params)
    elif params.task == "filter":
        run_filter_task(params)
    else:
        print(f"Error: Unknown task: {params.task}")
        print("Valid tasks: tile, filter, remap")
        sys.exit(1)


if __name__ == "__main__":
    main()
