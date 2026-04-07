from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence, Set, Tuple

import laspy


ACTIVE_FILTER_PARAM_NAMES: Tuple[str, ...] = (
    "segmented_folders",
    "segmented_remapped_folder",
    "subsampled_10cm_folder",
    "tile_bounds_json",
    "output_dir",
    "instance_dimension",
    "border_zone_width",
    "filter_anchor",
    "merge_chunk_size",
    "chunk_size",
    "workers",
    "min_cluster_size",
    "max_volume_for_merge",
    "disable_volume_merge",
    "subsampled_target_folder",
    "remap_dims",
    "remap_spatial_slices",
    "remap_spatial_chunk_length",
    "remap_spatial_target_points",
)

ACTIVE_FILTER_REMAP_PARAM_NAMES: Tuple[str, ...] = (
    "segmented_folders",
    "segmented_remapped_folder",
    "subsampled_10cm_folder",
    "tile_bounds_json",
    "original_input_dir",
    "output_dir",
    "instance_dimension",
    "border_zone_width",
    "filter_anchor",
    "merge_chunk_size",
    "chunk_size",
    "workers",
    "min_cluster_size",
    "max_volume_for_merge",
    "disable_volume_merge",
    "subsampled_target_folder",
    "produce_merged_file",
    "transfer_original_dims_to_merged",
    "output_merged_with_originals",
    "standardization_json",
    "remap_dims",
    "remap_spatial_slices",
    "remap_spatial_chunk_length",
    "remap_spatial_target_points",
)


def dir_has_pointcloud_outputs(directory: Path) -> bool:
    """Return True when a directory already contains LAS/LAZ outputs."""
    return directory.exists() and any(directory.glob("*.la[sz]"))


def resolve_remap_target_dims(params) -> Optional[Set[str]]:
    """
    Resolve the extra dimensions to transfer during filter/filter_remap remap
    stages.

    Only ``--remap-dims`` affects prediction-dimension transfer. Other shared
    task parameters are accepted for CLI compatibility but are not interpreted
    here.
    """
    return (
        {d.strip() for d in params.remap_dims.split(",") if d.strip()}
        if getattr(params, "remap_dims", None)
        else None
    )


def effective_copc_chunk_size(params, merge_chunk_size: int) -> int:
    """Return the COPC-oriented chunk size while respecting the streaming floor."""
    return max(int(getattr(params, "chunk_size", 20_000_000) or 20_000_000), merge_chunk_size)


def describe_collection_source_mode(source_collection: Path) -> str:
    """Return the remap source mode label used by multi-collection remap."""
    from merge_tiles import list_pointcloud_files

    source_collection = Path(source_collection)
    if source_collection.is_file():
        return "COPC subsets" if source_collection.name.endswith(".copc.laz") else "stream scan fallback"

    source_files = list_pointcloud_files(source_collection)
    if source_files and all(path.name.endswith(".copc.laz") for path in source_files):
        return "COPC subsets"
    return "stream scan fallback"


def validate_pointcloud_header(path: Path) -> Tuple[bool, Optional[str]]:
    """Return whether a LAS/LAZ/COPC file has a readable laspy header."""
    path = Path(path)
    try:
        with laspy.open(str(path), laz_backend=laspy.LazBackend.LazrsParallel):
            return True, None
    except Exception as exc:
        return False, str(exc)


def prepare_filtered_outputs_for_remap(
    source_collection,
    output_dir,
    chunk_size: int,
    *,
    convert_collection_file_to_copc: Callable,
    pointcloud_key_fn: Callable[[Path], str],
):
    """
    Validate and prepare filtered outputs as the preferred remap source.

    The returned path is the canonical collection to use for later remap calls:
    a validated ``filtered_tiles_copc`` directory when complete, otherwise the
    original filtered LAZ directory as a safe fallback.
    """
    from merge_tiles import list_pointcloud_files

    source_collection = Path(source_collection)
    output_dir = Path(output_dir)

    if source_collection.is_file():
        ok, reason = validate_pointcloud_header(source_collection)
        if not ok:
            raise ValueError(f"Invalid filtered point-cloud source {source_collection}: {reason}")
        return source_collection

    source_files = list_pointcloud_files(source_collection)
    if not source_files:
        return source_collection

    for src in source_files:
        ok, reason = validate_pointcloud_header(src)
        if not ok:
            raise ValueError(f"Invalid filtered output {src}: {reason}")

    if all(path.name.endswith(".copc.laz") for path in source_files):
        print(f"  Filtered outputs already in COPC: {source_collection}", flush=True)
        return source_collection

    dest_dir = output_dir / "filtered_tiles_copc"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPreparing filtered outputs as COPC in {dest_dir}...", flush=True)
    print(f"  Filtered COPC prep chunk size: {chunk_size:,} points", flush=True)

    expected_keys = {pointcloud_key_fn(src) for src in source_files}
    existing_dest_files = list_pointcloud_files(dest_dir)
    valid_dest_keys: Set[str] = set()
    invalid_dest_keys: Set[str] = set()
    invalid_dest_files = []
    for existing in existing_dest_files:
        key = pointcloud_key_fn(existing)
        ok, reason = validate_pointcloud_header(existing)
        if ok:
            valid_dest_keys.add(key)
        else:
            invalid_dest_keys.add(key)
            invalid_dest_files.append((existing, reason or "invalid header"))

    regenerated = 0
    for bad_path, reason in invalid_dest_files:
        print(f"  Removing invalid filtered COPC: {bad_path.name} ({reason})", flush=True)
        bad_path.unlink(missing_ok=True)
        regenerated += 1

    reused = 0
    converted = 0
    copied = 0
    failures = []
    for src in source_files:
        key = pointcloud_key_fn(src)
        dest_path = dest_dir / (src.name if src.name.endswith(".copc.laz") else f"{src.stem}.copc.laz")

        if key in valid_dest_keys and dest_path.exists():
            print(f"  Reusing filtered COPC: {dest_path.name}", flush=True)
            reused += 1
            continue

        if dest_path.exists():
            dest_path.unlink(missing_ok=True)

        if src.name.endswith(".copc.laz"):
            shutil.copy2(src, dest_path)
            copied += 1
        else:
            print(f"  Filtered COPC: {src.name} -> {dest_path.name}", flush=True)
            success, message = convert_collection_file_to_copc(
                input_path=src,
                output_copc=dest_path,
                chunk_size=chunk_size,
            )
            if not success:
                failures.append((src.name, message))
                print(
                    f"  Warning: failed to convert filtered tile {src.name} to COPC "
                    f"({message}); keeping LAZ filtered outputs as fallback",
                    flush=True,
                )
                dest_path.unlink(missing_ok=True)
                continue
            converted += 1

        ok, reason = validate_pointcloud_header(dest_path)
        if ok:
            valid_dest_keys.add(key)
        else:
            failures.append((dest_path.name, reason or "invalid header after conversion"))
            print(
                f"  Warning: generated filtered COPC is invalid for {dest_path.name} "
                f"({reason}); keeping LAZ filtered outputs as fallback",
                flush=True,
            )
            dest_path.unlink(missing_ok=True)

    print(
        f"  Filtered COPC summary: {converted} converted, {copied} copied, "
        f"{reused} reused, {regenerated} regenerated",
        flush=True,
    )

    if failures:
        for name, reason in failures:
            print(f"    Failed COPC prep: {name} ({reason})", flush=True)

    available_dest_files = list_pointcloud_files(dest_dir)
    available_valid_keys = {
        pointcloud_key_fn(path)
        for path in available_dest_files
        if validate_pointcloud_header(path)[0]
    }
    if expected_keys.issubset(available_valid_keys):
        print(f"  Using validated filtered COPC outputs for remap: {dest_dir}", flush=True)
        return dest_dir

    missing = sorted(expected_keys - available_valid_keys)
    print(
        f"  Warning: filtered COPC set is incomplete after validation/conversion "
        f"(missing {len(missing)} tile(s)); using LAZ filtered outputs instead",
        flush=True,
    )
    return source_collection


def format_active_param_names(names: Sequence[str]) -> str:
    """Compact formatter for task-local active parameter logging."""
    return ", ".join(names)
