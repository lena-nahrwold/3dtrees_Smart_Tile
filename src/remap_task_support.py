from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

def _concat_laz_files(
    input_files,
    output_path,
    chunk_size=2_000_000,
    deduplicate_xyz: bool = False,
    force_point_format: Optional[int] = None,
):
    """
    Stream-concatenate multiple LAZ files that share the same point schema into
    a single output file.  All files must have been produced by the same pipeline
    step (e.g. remap_collections_to_original_files) so their extra-dimension VLRs
    are identical.

    When ``deduplicate_xyz`` is enabled, points that lie on shared upper tile/file
    boundaries are dropped according to the same west/south ownership rule used
    elsewhere in the pipeline. This is much faster than global exact-XYZ hashing
    and is intended for the final remap merge, where duplicates originate from
    overlapping file boundaries.

    ``force_point_format``: explicit LAS point format ID for the output.
    When *None* (default), the format is auto-detected:
      - format 7 when red/green/blue are present as standard dims
      - otherwise the format from the first input file
    """
    import laspy
    import numpy as np
    from merge_tiles import extra_bytes_params_from_dimension_info, has_standard_rgb_dims, point_format_with_standard_rgb

    if not input_files:
        raise ValueError("No input files provided for concatenation")

    def _schema_signature(header):
        return (
            header.point_format.id,
            str(header.version),
            tuple(str(dim) for dim in header.point_format.dimension_names),
            tuple(
                (dim.name, str(dim.dtype))
                for dim in header.point_format.extra_dimensions
            ),
        )

    def _base_schema_signature(header):
        return (
            header.point_format.id,
            str(header.version),
            tuple(str(dim) for dim in header.point_format.dimension_names),
        )

    def _extra_dim_signature(extra_dim_names, extra_dim_meta):
        return tuple(
            (name, str(extra_dim_meta[name]["dtype"]))
            for name in extra_dim_names
        )

    def _build_extra_bytes_params(dim_name, dim_meta):
        dim_info = dim_meta["dim"]
        return laspy.ExtraBytesParams(
            name=dim_name,
            type=dim_meta["dtype"],
            description=getattr(dim_info, "description", "") or "",
            offsets=getattr(dim_info, "offsets", None),
            scales=getattr(dim_info, "scales", None),
            no_data=getattr(dim_info, "no_data", None),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"  Output already exists, skipping concatenation: {output_path}")
        return

    with laspy.open(str(input_files[0])) as src0:
        src_header = src0.header
        base_signature = _schema_signature(src_header)
        base_core_signature = _base_schema_signature(src_header)
        base_std_dims = tuple(str(dim) for dim in src_header.point_format.dimension_names)
        base_vlrs = list(src_header.vlrs)
        base_offsets = src_header.offsets
        base_scales = src_header.scales
        base_extra_dim_names = tuple(
            dim.name for dim in src_header.point_format.extra_dimensions
        )
        union_extra_dims = {
            dim.name: {
                "dim": dim,
                "dtype": np.dtype(dim.dtype),
            }
            for dim in src_header.point_format.extra_dimensions
        }
        ordered_extra_dim_names = list(base_extra_dim_names)
        promoted_dims = {}

    mismatched = []
    for input_file in input_files[1:]:
        with laspy.open(str(input_file)) as reader:
            header = reader.header
            if _schema_signature(header) == base_signature:
                continue
            if _base_schema_signature(header) != base_core_signature:
                mismatched.append(input_file.name)
                continue
            for dim in header.point_format.extra_dimensions:
                existing = union_extra_dims.get(dim.name)
                if existing is None:
                    union_extra_dims[dim.name] = {
                        "dim": dim,
                        "dtype": np.dtype(dim.dtype),
                    }
                    ordered_extra_dim_names.append(dim.name)
                elif str(existing["dtype"]) != str(np.dtype(dim.dtype)):
                    promoted_dtype = np.promote_types(existing["dtype"], np.dtype(dim.dtype))
                    union_extra_dims[dim.name]["dtype"] = promoted_dtype
                    promoted_dims[dim.name] = str(promoted_dtype)

    if mismatched:
        raise ValueError(
            "Cannot concatenate files with differing base schemas. "
            f"Mismatched files: {', '.join(mismatched)}"
        )

    all_dim_names = set(base_std_dims) | set(ordered_extra_dim_names)
    if force_point_format is not None:
        out_point_format_id = force_point_format
    elif has_standard_rgb_dims(all_dim_names):
        out_point_format_id = point_format_with_standard_rgb(base_core_signature[0])
    else:
        out_point_format_id = base_core_signature[0]
    out_header = laspy.LasHeader(
        point_format=out_point_format_id,
        version=base_core_signature[1],
    )
    out_std_dims = tuple(str(dim) for dim in out_header.point_format.dimension_names)
    promote_rgb = out_point_format_id != base_core_signature[0]
    if promote_rgb:
        print(
            f"  Promoting RGB to standard LAS fields: "
            f"point_format {base_core_signature[0]} -> {out_point_format_id}",
            flush=True,
        )
    out_header.offsets = base_offsets
    out_header.scales = base_scales
    rgb_standard_dims = {"red", "green", "blue"}
    if union_extra_dims:
        extra_params = [
            _build_extra_bytes_params(name, union_extra_dims[name])
            for name in ordered_extra_dim_names
            if not (promote_rgb and name in rgb_standard_dims)
        ]
        if extra_params:
            out_header.add_extra_dims(extra_params)
    aligned_extra_dim_names = tuple(
        n for n in ordered_extra_dim_names
        if not (promote_rgb and n in rgb_standard_dims)
    )

    existing_vlr_keys = {
        (getattr(v, "user_id", ""), getattr(v, "record_id", None))
        for v in out_header.vlrs
    }
    for vlr in base_vlrs:
        user_id = getattr(vlr, "user_id", "")
        record_id = getattr(vlr, "record_id", None)
        # Skip COPC and ExtraBytes VLRs; laspy regenerates the latter from the header.
        if (record_id in (1, 2) and user_id == "copc") or (record_id == 4 and user_id == "LASF_Spec"):
            continue
        key = (user_id, record_id)
        if key not in existing_vlr_keys:
            out_header.vlrs.append(vlr)
            existing_vlr_keys.add(key)

    aligned_extra_dim_signature = _extra_dim_signature(
        aligned_extra_dim_names, union_extra_dims
    )
    if promoted_dims:
        promoted_str = ", ".join(
            f"{name}->{dtype}" for name, dtype in sorted(promoted_dims.items())
        )
        print(f"  Promoted extra-dimension dtypes for merge: {promoted_str}", flush=True)

    file_entries = []
    if deduplicate_xyz:
        abs_tol = None
        for input_file in input_files:
            with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                hdr = reader.header
                scales = np.asarray(hdr.scales, dtype=np.float64)
                scale_tol = float(np.max(scales[:2])) if scales.size >= 2 else 1e-9
                abs_tol = scale_tol if abs_tol is None else max(abs_tol, scale_tol)
                file_entries.append({
                    "name": input_file.name,
                    "path": Path(input_file),
                    "bounds": (
                        float(hdr.x_min),
                        float(hdr.x_max),
                        float(hdr.y_min),
                        float(hdr.y_max),
                    ),
                })
        _annotate_canonical_tile_edge_neighbors(file_entries, abs_tol=abs_tol or 1e-9)

    written = 0
    dropped_duplicates = 0
    effective_chunk_size = int(chunk_size)
    with laspy.open(
        str(output_path), mode="w", header=out_header,
        do_compress=output_path.suffix.lower() == ".laz",
        laz_backend=laspy.LazBackend.LazrsParallel,
    ) as writer:
        entries_iter = file_entries if deduplicate_xyz else [
            {"name": Path(f).name, "path": Path(f)} for f in input_files
        ]
        for entry in entries_iter:
            with laspy.open(str(entry["path"]), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                reader_std_dims = tuple(str(dim) for dim in reader.header.point_format.dimension_names)
                reader_extra_dims = {
                    dim.name: {
                        "dim": dim,
                        "dtype": np.dtype(dim.dtype),
                    }
                    for dim in reader.header.point_format.extra_dimensions
                }
                reader_extra_dim_names = tuple(
                    dim.name for dim in reader.header.point_format.extra_dimensions
                )
                needs_dim_alignment = (
                    reader_std_dims != base_std_dims
                    or _extra_dim_signature(reader_extra_dim_names, reader_extra_dims)
                    != aligned_extra_dim_signature
                )
                needs_format_fixup = (
                    reader.header.point_format != out_header.point_format
                )
                reader_all_dims = set(reader_std_dims) | set(reader_extra_dim_names)
                for chunk in reader.chunk_iterator(effective_chunk_size):
                    if needs_dim_alignment or needs_format_fixup:
                        out_chunk = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=out_header)
                        for dim_name in out_std_dims:
                            if dim_name in reader_all_dims:
                                out_chunk[dim_name] = np.asarray(chunk[dim_name])
                        for dim_name in aligned_extra_dim_names:
                            if dim_name in reader_extra_dims:
                                out_chunk[dim_name] = np.asarray(
                                    chunk[dim_name],
                                    dtype=union_extra_dims[dim_name]["dtype"],
                                )
                        chunk_to_write = out_chunk
                    else:
                        chunk_to_write = chunk

                    if not deduplicate_xyz:
                        writer.write_points(chunk_to_write)
                        written += len(chunk_to_write)
                        continue

                    n_chunk = len(chunk_to_write)
                    if n_chunk == 0:
                        continue

                    xmin, xmax, ymin, ymax = entry["bounds"]
                    eps = abs_tol or 1e-9
                    cx = np.asarray(chunk_to_write.x, dtype=np.float64)
                    cy = np.asarray(chunk_to_write.y, dtype=np.float64)
                    upper_x = _upper_edge_membership_mask(
                        cx,
                        xmax,
                        entry.get("east_neighbor_spans", []),
                        cy,
                        abs_tol=eps,
                    )
                    upper_y = _upper_edge_membership_mask(
                        cy,
                        ymax,
                        entry.get("north_neighbor_spans", []),
                        cx,
                        abs_tol=eps,
                    )
                    keep_mask = (
                        (cx >= xmin - eps) & upper_x
                        & (cy >= ymin - eps) & upper_y
                    )
                    kept = int(np.count_nonzero(keep_mask))
                    if kept > 0:
                        writer.write_points(chunk_to_write[keep_mask])
                        written += kept
                    dropped_duplicates += n_chunk - kept

    if deduplicate_xyz:
        print(
            f"  Concatenated {len(input_files)} files ({written:,} kept points) → {output_path.name}; "
            f"dropped {dropped_duplicates:,} overlap duplicate point(s) by boundary ownership",
            flush=True,
        )
    else:
        print(f"  Concatenated {len(input_files)} files ({written:,} points) → {output_path.name}")


def _sanitize_extra_dim_name_for_copc(name: str, used_names: Optional[Set[str]] = None) -> str:
    """Return the COPC-safe alias for an extra dimension name."""
    candidate = name
    if candidate.startswith("3DT_"):
        candidate = f"TDT_{candidate[4:]}"
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", candidate)
    if not candidate or not candidate[0].isalpha():
        candidate = f"TDT_{candidate.lstrip('_')}" if candidate else "TDT_dim"
    candidate = re.sub(r"_+", "_", candidate)
    candidate = candidate.rstrip("_") or "TDT_dim"

    if used_names is None or candidate == name:
        return candidate

    base = candidate
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


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

        for dim in extra_dims:
            sanitized = _sanitize_extra_dim_name_for_copc(dim.name, used_names=used_names)
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


def _collection_match_key(path: Path, collection_idx: int, strip_collection_suffix: bool = False) -> str:
    """
    Return the tile key used to compare files across remap collections.

    Galaxy remap inputs are symlinked as ``<original_stem>_<collection_idx>.<ext>``.
    When every file in a collection follows that pattern, the trailing wrapper
    suffix should be ignored for aligned-fusion matching.
    """
    key = _pointcloud_file_key(path)
    if strip_collection_suffix:
        suffix = f"_{collection_idx}"
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


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
    """Build a minimal output header for one fused remap source tile."""
    import laspy

    out_header = laspy.LasHeader(
        point_format=6,
        version="1.4",
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
                description=(ebp.description or "") if ebp is not None else "",
            ))
            existing_out_names.add(out_name)
    if extra_params:
        out_header.add_extra_dims(extra_params)

    base_copy_dims = [dim_name for dim_name in ("X", "Y", "Z") if dim_name in ref_header.point_format.dimension_names]
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

        collection_suffix = f"_{idx}"
        strip_collection_suffix = all(
            _pointcloud_file_key(src).endswith(collection_suffix)
            for src in files
        )
        key_map = {}
        for src in files:
            key = _collection_match_key(
                src,
                collection_idx=idx,
                strip_collection_suffix=strip_collection_suffix,
            )
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
            dim_str = "(no selected dimensions discovered)"
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
            dim_str = "(no selected dimensions discovered)"
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


def _collect_pointcloud_outputs(paths: List[Path]) -> List[Path]:
    """Expand file/directory paths into a stable list of LAS/LAZ outputs."""
    from merge_tiles import list_pointcloud_files

    collected: List[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            collected.append(path)
        elif path.is_dir():
            collected.extend(list_pointcloud_files(path))
    return sorted(collected, key=_pointcloud_file_key)


def _resolve_remap_target_dims(params):
    """
    Resolve the dimension set to transfer during remap-style operations.

    ``--remap-dims`` is a strict allowlist and may include both extra and
    standard LAS dimensions from the segmented source collections. When the flag
    is omitted, remap transfers all extra dimensions by default. The
    standardization JSON is reserved for filtering original dimensions during
    merged-file enrichment (``merged_with_originals``), not for filtering source
    prediction dimensions during remap itself.

    For COPC-normalized sources, dimensions with invalid COPC names are sanitized
    first. To keep the CLI user-facing, a requested source name also matches its
    COPC-safe alias (for example ``3DT_Foo`` also matches ``TDT_Foo``).
    """
    raw_dims = (
        {d.strip() for d in params.remap_dims.split(",") if d.strip()}
        if getattr(params, "remap_dims", None) else None
    )
    if not raw_dims:
        return None

    expanded_dims = set(raw_dims)
    alias_pairs = []
    for dim_name in sorted(raw_dims):
        sanitized = _sanitize_extra_dim_name_for_copc(dim_name)
        if sanitized != dim_name:
            expanded_dims.add(sanitized)
            alias_pairs.append((dim_name, sanitized))

    if alias_pairs:
        alias_msg = ", ".join(f"{src}->{dst}" for src, dst in alias_pairs)
        print(
            "  Remap dims alias expansion for COPC-safe names: "
            f"{alias_msg}",
            flush=True,
        )

    return expanded_dims


def _describe_laz_dimensions(laz_path: Path) -> str:
    """Return a compact string of LAS dimension names for logging."""
    import laspy

    with laspy.open(str(laz_path)) as f:
        dims = list(f.header.point_format.dimension_names)
    return ", ".join(str(d) for d in dims)
