#!/usr/bin/env python3
"""
Rename one extra-byte dimension across all LAS/LAZ files in a directory.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import laspy
import numpy as np


def _laz_backend():
    if hasattr(laspy.LazBackend, "LazrsParallel"):
        return laspy.LazBackend.LazrsParallel
    if hasattr(laspy.LazBackend, "Lazrs"):
        return laspy.LazBackend.Lazrs
    return None


def _copy_non_extra_vlrs(src_header: laspy.LasHeader, dst_header: laspy.LasHeader) -> None:
    existing = {
        (getattr(v, "user_id", ""), getattr(v, "record_id", None))
        for v in dst_header.vlrs
    }
    for vlr in src_header.vlrs:
        user_id = getattr(vlr, "user_id", "")
        record_id = getattr(vlr, "record_id", None)
        if record_id == 4 and user_id == "LASF_Spec":
            continue
        key = (user_id, record_id)
        if key not in existing:
            dst_header.vlrs.append(vlr)
            existing.add(key)


def rename_dimension_in_file(path: Path, source_dim: str, target_dim: str, chunk_size: int) -> bool:
    open_kwargs = {}
    backend = _laz_backend()
    if path.suffix.lower() == ".laz" and backend is not None:
        open_kwargs["laz_backend"] = backend

    with laspy.open(str(path), **open_kwargs) as reader:
        header = reader.header
        extra_dims = list(header.point_format.extra_dimensions)
        extra_names = [dim.name for dim in extra_dims]

        if source_dim not in extra_names:
            print(f"  Skip {path.name}: {source_dim} not present")
            return False
        if target_dim in extra_names and target_dim != source_dim:
            raise ValueError(f"{path.name} already contains target dimension {target_dim}")

        out_header = laspy.LasHeader(
            point_format=header.point_format.id,
            version=header.version,
        )
        out_header.offsets = header.offsets
        out_header.scales = header.scales
        _copy_non_extra_vlrs(header, out_header)

        for dim in extra_dims:
            dim_name = target_dim if dim.name == source_dim else dim.name
            out_header.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=dim_name,
                    type=dim.dtype,
                    description=getattr(dim, "description", "") or "",
                    offsets=getattr(dim, "offsets", None),
                    scales=getattr(dim, "scales", None),
                    no_data=getattr(dim, "no_data", None),
                )
            )

        with tempfile.NamedTemporaryFile(
            prefix=f"{path.stem}_rename_",
            suffix=path.suffix,
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with laspy.open(
                str(tmp_path),
                mode="w",
                header=out_header,
                do_compress=path.suffix.lower() == ".laz",
                laz_backend=backend,
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
                        dst_name = target_dim if src_name == source_dim else src_name
                        out_record[dst_name] = np.asarray(getattr(chunk, src_name))
                    writer.write_points(out_record)

            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    print(f"  Renamed {path.name}: {source_dim} -> {target_dim}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename one extra-byte dimension in all LAS/LAZ files in a directory.")
    parser.add_argument("--input-dir", required=True, help="Directory containing LAS/LAZ files")
    parser.add_argument("--source-dim", required=True, help="Existing extra-byte dimension name")
    parser.add_argument("--target-dim", required=True, help="New extra-byte dimension name")
    parser.add_argument("--chunk-size", type=int, default=5_000_000, help="Streaming chunk size")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    files = sorted(input_dir.glob("*.laz")) + sorted(input_dir.glob("*.las"))
    if not files:
        print(f"No LAS/LAZ files found in {input_dir}", file=sys.stderr)
        return 1

    changed = 0
    for path in files:
        if rename_dimension_in_file(path, args.source_dim, args.target_dim, args.chunk_size):
            changed += 1

    print(f"Done. Files changed: {changed}/{len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
