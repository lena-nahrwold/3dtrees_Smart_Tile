from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import laspy


ACTIVE_FILTER_PARAM_NAMES: Tuple[str, ...] = (
    "segmented_folders",
    "tile_bounds_json",
    "output_dir",
    "instance_dimension",
    "border_zone_width",
    "filter_anchor",
    "chunk_size",
    "workers",
    "min_cluster_size",
    "max_volume_for_merge",
    "enable_volume_merge",
    "remap_merge",
    "subsampled_target_folder",
)

ACTIVE_FILTER_REMAP_PARAM_NAMES: Tuple[str, ...] = (
    "segmented_folders",
    "tile_bounds_json",
    "original_input_dir",
    "output_dir",
    "instance_dimension",
    "border_zone_width",
    "filter_anchor",
    "chunk_size",
    "workers",
    "min_cluster_size",
    "max_volume_for_merge",
    "enable_volume_merge",
    "remap_merge",
    "subsampled_target_folder",
    "produce_merged_file",
    "transfer_original_dims_to_merged",
    "output_merged_with_originals",
    "standardization_json",
)

_TREE_OUTPUT_SUFFIXES: Tuple[str, ...] = (
    "_trees.txt",
    "_trees_info.txt",
)

_TREE_FILE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^input_trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^input_trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^.+_trees(?:_[^.]+)?\.txt$", re.IGNORECASE),
    re.compile(r"^.+_trees_info(?:_[^.]+)?\.txt$", re.IGNORECASE),
)


def dir_has_pointcloud_outputs(directory: Path) -> bool:
    """Return True when a directory already contains LAS/LAZ outputs."""
    return directory.exists() and any(directory.glob("*.la[sz]"))


def is_pointcloud_file(path: Path) -> bool:
    """Return True for LAS/LAZ/COPC point-cloud files."""
    name = Path(path).name.lower()
    return name.endswith(".copc.laz") or name.endswith(".laz") or name.endswith(".las")


def is_tree_sidecar_file(path: Path) -> bool:
    """Return True for supported tree sidecar text files."""
    name = Path(path).name
    return any(pattern.match(name) for pattern in _TREE_FILE_PATTERNS)


def classify_tree_sidecar_file(path: Path) -> Optional[str]:
    """Return the canonical output suffix for a supported tree sidecar text file."""
    if not is_tree_sidecar_file(path):
        return None
    lower_name = Path(path).name.lower()
    if "trees_info" in lower_name:
        return "_trees_info.txt"
    return "_trees.txt"


def _collection_file_key(path: Path, pointcloud_key_fn: Callable[[Path], str]) -> str:
    """Return a normalized tile/file key used to match collection sidecars."""
    from merge_tiles import normalize_tile_id

    raw_key = pointcloud_key_fn(Path(path))
    return normalize_tile_id(raw_key)


def discover_tree_sidecars_for_pointcloud(
    pointcloud_path: Path,
    tile_name: str,
    sibling_txt_files: Sequence[Path],
    *,
    pointcloud_key_fn: Callable[[Path], str],
    sibling_pointcloud_count: int,
) -> Dict[str, Path]:
    """
    Match supported tree sidecars for one point-cloud tile/file.

    Supported patterns:
    - ``<tile>_trees.txt``
    - ``<tile>_trees_info.txt``
    - ``trees.txt`` / ``trees_info.txt`` only when the directory contains a
      single point-cloud file, so the association is unambiguous.
    """
    pointcloud_path = Path(pointcloud_path)
    sibling_txt_files = [Path(path) for path in sibling_txt_files]

    def _dedupe_extend(bases: List[str], *values: str) -> None:
        for v in values:
            v = str(v or "").strip()
            if v and v not in bases:
                bases.append(v)

    # Galaxy often appends `_<id>` to element identifiers when staging
    # collections. Also, segmented outputs frequently include a suffix like
    # `_segmented` (or `_segmented_<id>`). Tree sidecars may instead be keyed to
    # a "base" stem such as `<tile>_subsampled_10cm` and won’t match unless we
    # generate candidate bases that strip those processing/id suffixes.
    raw_key = pointcloud_key_fn(pointcloud_path)
    stem = pointcloud_path.stem
    # Remove a single trailing `_<digits>` (Galaxy dataset id) when present.
    stem_no_id = re.sub(r"_\d+$", "", stem)
    raw_key_no_id = re.sub(r"_\d+$", "", str(raw_key))
    # Remove trailing processing suffixes with optional `_id`.
    stem_no_proc = re.sub(r"(?:_(?:segmented_remapped|segmented|remapped|results)(?:_\d+)?)$", "", stem, flags=re.I)
    raw_key_no_proc = re.sub(r"(?:_(?:segmented_remapped|segmented|remapped|results)(?:_\d+)?)$", "", str(raw_key), flags=re.I)

    candidate_bases: List[str] = []
    _dedupe_extend(
        candidate_bases,
        tile_name,
        raw_key,
        raw_key_no_id,
        raw_key_no_proc,
        stem,
        stem_no_id,
        stem_no_proc,
        _collection_file_key(pointcloud_path, pointcloud_key_fn),
    )

    matched: Dict[str, Path] = {}
    pointcloud_lower = pointcloud_path.name.lower()
    is_segmented_pointcloud = (
        "segmented" in pointcloud_lower
        or pointcloud_lower.endswith("segmented_pc.laz")
        or pointcloud_lower.endswith("segmented_pc.las")
    )
    for txt_path in sibling_txt_files:
        lower_name = txt_path.name.lower()
        tree_kind = classify_tree_sidecar_file(txt_path)
        if tree_kind is None:
            continue
        if is_segmented_pointcloud:
            if lower_name.startswith("input_trees_info"):
                matched.setdefault("_trees_info.txt", txt_path)
                continue
            if lower_name.startswith("input_trees"):
                matched.setdefault("_trees.txt", txt_path)
                continue

        if sibling_pointcloud_count == 1:
            if lower_name.startswith("trees_info"):
                matched.setdefault("_trees_info.txt", txt_path)
                continue
            if lower_name.startswith("trees"):
                matched.setdefault("_trees.txt", txt_path)
                continue

        # Check _trees_info before _trees to avoid prefix collision.
        # Also accept an optional trailing _<id> segment (e.g. Galaxy appends _${f.id})
        # so that c00_r00_trees_67890.txt matches tile-name base c00_r00.
        for suffix in ("_trees_info.txt", "_trees.txt"):
            suffix_stem = suffix[:-4]  # "_trees_info" or "_trees"
            if any(
                lower_name == f"{b.lower()}{suffix}"
                or (
                    lower_name.startswith(f"{b.lower()}{suffix_stem}_")
                    and lower_name.endswith(".txt")
                )
                for b in candidate_bases
            ):
                matched.setdefault(suffix, txt_path)
                break

    return matched


def collect_pointcloud_files(
    input_path: Path,
    *,
    recursive_max_depth: int = 4,
) -> List[Path]:
    """
    Collect point-cloud inputs from a path that may be a file or directory.

    Behavior (directory inputs):
      - Prefer `input_segmented.laz` / `segmented_pc.laz` when present.
      - If no LAZ/LAS files exist at the directory root, fall back to a depth-limited
        recursive search (useful for per-tile subfolders).

    This keeps the filter task usable with Galaxy-style collections (flat folders)
    and with nested outputs from upstream tools (e.g. raycloudtools runs).
    """
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if is_pointcloud_file(input_path) else []
    if not input_path.is_dir():
        return []

    direct = (
        sorted(input_path.glob("*.copc.laz"))
        + sorted(input_path.glob("*.laz"))
        + sorted(input_path.glob("*.las"))
    )
    if direct:
        preferred = [
            path
            for path in direct
            if path.name.lower() in ("input_segmented.laz", "segmented_pc.laz", "input_segmented.las", "segmented_pc.las")
        ]
        if preferred:
            return preferred

        segmented_named = [path for path in direct if "segmented" in path.name.lower()]
        return segmented_named or direct

    segmented: List[Path] = []
    other: List[Path] = []

    root = input_path.resolve()
    for current_root, dirs, files in os.walk(root):
        current_path = Path(current_root)
        depth = len(current_path.relative_to(root).parts)
        if depth >= recursive_max_depth:
            dirs[:] = []
        for filename in files:
            lower = filename.lower()
            path = current_path / filename
            if (
                lower in ("input_segmented.laz", "segmented_pc.laz", "input_segmented.las", "segmented_pc.las")
                or ("segmented" in lower and lower.endswith((".laz", ".las", ".copc.laz")))
            ):
                segmented.append(path)
            elif lower.endswith((".laz", ".las")) or lower.endswith(".copc.laz"):
                other.append(path)

    if segmented:
        return sorted(segmented)
    return sorted(other)


def copy_additional_collection_files(
    input_paths: Sequence[Path],
    destination_dir: Path,
    *,
    excluded_files: Optional[Iterable[Path]] = None,
) -> int:
    """
    Copy non-pointcloud, non-tree sidecar files from collection directories into
    ``destination_dir``.

    When multiple source collections contain the same sidecar filename, later
    copies are renamed with a source-directory prefix to avoid overwriting.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    excluded = {Path(path).resolve() for path in (excluded_files or [])}
    copied = 0

    def _unique_dest_path(src_path: Path, source_root: Path) -> Path:
        candidate = destination_dir / src_path.name
        if not candidate.exists():
            return candidate

        prefix = source_root.name or src_path.parent.name or "collection"
        candidate = destination_dir / f"{prefix}__{src_path.name}"
        if not candidate.exists():
            return candidate

        counter = 2
        while True:
            candidate = destination_dir / f"{prefix}_{counter}__{src_path.name}"
            if not candidate.exists():
                return candidate
            counter += 1

    for input_path in input_paths:
        input_path = Path(input_path)
        if not input_path.is_dir():
            continue

        for child in sorted(input_path.iterdir()):
            if not child.is_file():
                continue
            if child.resolve() in excluded:
                continue
            if is_pointcloud_file(child) or is_tree_sidecar_file(child):
                continue

            dest_path = _unique_dest_path(child, input_path)
            shutil.copy2(child, dest_path)
            copied += 1
            if dest_path.name == child.name:
                print(f"  Copied sidecar file: {child.name}", flush=True)
            else:
                print(
                    f"  Copied sidecar file: {child.name} -> {dest_path.name}",
                    flush=True,
                )

    return copied


def copy_mesh_sidecars_for_tiles(
    tile_name_by_path: Mapping[Path, str],
    destination_dir: Path,
) -> int:
    """
    Copy co-located mesh sidecars (filenames containing 'mesh') for each tile.

    This supports upstream tools that write per-tile `.ply` meshes next to the
    segmented pointcloud (e.g. `input_mesh.ply`, `input_trees_mesh.ply`).

    Files are flattened into `destination_dir` and prefixed with the canonical
    tile name to avoid collisions: `c00_r00__input_mesh.ply`.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pointcloud_path, tile_name in tile_name_by_path.items():
        parent = Path(pointcloud_path).parent
        if not parent.is_dir():
            continue

        for child in sorted(parent.iterdir()):
            if not child.is_file():
                continue
            if "mesh" not in child.name.lower():
                continue
            if child.suffix.lower() == ".ply":
                continue

            base_dest = destination_dir / f"{tile_name}__{child.name}"
            dest = base_dest
            counter = 2
            while dest.exists():
                dest = destination_dir / f"{tile_name}_{counter}__{child.name}"
                counter += 1

            shutil.copy2(child, dest)
            copied += 1
            if dest.name == base_dest.name:
                print(f"  Copied mesh sidecar: {child.name} -> {dest.name}", flush=True)
            else:
                print(
                    f"  Copied mesh sidecar: {child.name} -> {dest.name} (deduped)",
                    flush=True,
                )

    return copied


def resolve_filter_input_paths(params) -> List[Path]:
    """
    Resolve the single filter/remap_merge collection parameter into a list of
    file or directory paths.

    The cleaned interface uses ``segmented_folders`` for both files and
    folders. Legacy fallbacks are intentionally ignored here so task-local
    behavior stays predictable.
    """
    raw_folders = getattr(params, "segmented_folders", "") or ""
    return [Path(p.strip()) for p in raw_folders.split(",") if p.strip()]


def resolve_filter_chunk_size(params) -> int:
    """
    Resolve the single chunk-size control used by filter/remap_merge.
    """
    chunk_size = getattr(params, "chunk_size", None)
    if chunk_size is not None:
        return int(chunk_size)
    return 20_000_000


def derive_border_zone_width_from_json(tile_bounds_json: Path) -> float:
    """
    Derive the tile-border buffer width from tile_bounds_tindex.json.

    Preference order:
    1. root-level ``tile_buffer``
    2. per-tile ``bounds`` vs ``core`` difference
    """
    tile_bounds_json = Path(tile_bounds_json)
    with tile_bounds_json.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        root_buffer = data.get("tile_buffer")
        if root_buffer is not None:
            return float(root_buffer)

        tiles = data.get("tiles", [])
        for tile in tiles:
            bounds = tile.get("bounds")
            core = tile.get("core")
            if not bounds or not core:
                continue
            try:
                x_pad = (float(bounds[0][1]) - float(core[0][1]) + float(core[0][0]) - float(bounds[0][0])) / 2.0
                y_pad = (float(bounds[1][1]) - float(core[1][1]) + float(core[1][0]) - float(bounds[1][0])) / 2.0
                pad = max(x_pad, y_pad)
            except (TypeError, ValueError, IndexError):
                continue
            if pad > 0:
                return float(pad)

    raise ValueError(f"Could not derive tile buffer width from {tile_bounds_json}")


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


def _count_tree_instances(txt_path: Path) -> int:
    """Count data rows (one per tree/instance) in a tree sidecar file, skipping the 2-line header."""
    try:
        count = 0
        with open(txt_path) as fh:
            for i, line in enumerate(fh):
                if i >= 2 and line.strip():
                    count += 1
        return count
    except Exception:
        return 0


def _read_laz_point_count(laz_path: Path) -> Optional[int]:
    """Read total point count from a LAS/LAZ header without loading point data."""
    try:
        with laspy.open(str(laz_path)) as f:
            return f.header.point_count
    except Exception:
        return None


def _all_tree_xy_centroids(txt_path: Path) -> List[Tuple[float, float]]:
    """
    Return one (x, y) position per tree from a sidecar file.

    Handles optional per-tree prefix fields by finding the offset that yields
    the most complete 6-field segment groups (same heuristic as _tree_xy_bounds).
    Takes the first segment's x, y as the tree representative position.
    """
    try:
        lines = txt_path.read_text().splitlines()
    except Exception:
        return []
    points: List[Tuple[float, float]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        nums = []
        for v in line.split(","):
            v = v.strip()
            if v:
                try:
                    nums.append(float(v))
                except Exception:
                    continue
        if len(nums) < 2:
            continue
        best_offset, best_groups = 0, 0
        for offset in range(6):
            groups = max(0, (len(nums) - offset) // 6)
            if groups > best_groups:
                best_groups, best_offset = groups, offset
        if best_groups == 0:
            continue
        points.append((nums[best_offset], nums[best_offset + 1]))
    return points


def _fraction_in_bounds(
    points: List[Tuple[float, float]],
    bounds: Tuple[float, float, float, float],
) -> float:
    """Fraction of (x, y) points that fall within (xmin, xmax, ymin, ymax)."""
    if not points:
        return 0.0
    mn_x, mx_x, mn_y, mx_y = bounds
    inside = sum(1 for x, y in points if mn_x <= x <= mx_x and mn_y <= y <= mx_y)
    return inside / len(points)
