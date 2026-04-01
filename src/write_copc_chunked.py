#!/usr/bin/env python3
"""
Write a COPC file from one or more LAS/LAZ inputs using chunked staging.

The input data are streamed in chunks into temporary uncompressed LAS parts.
Those parts are then finalized into a single COPC output via ``untwine``.

This is useful for very large inputs where direct COPC conversion can be too
memory-hungry, while still preserving the source schema and extra dimensions.

Examples:
    python src/write_copc_chunked.py \
        --input /data/merged_with_all_dims.laz \
        --output /data/merged_with_all_dims.copc.laz

    python src/write_copc_chunked.py \
        --input /data/tile_a.laz /data/tile_b.laz \
        --output /data/merged_tiles.copc.laz \
        --chunk-size 5000000
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import laspy

# Allow direct execution from the repository root or from this src directory.
_SRC_DIR = Path(__file__).parent.resolve()
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from main_tile import _make_tile_header, get_untwine_path
from parameters import TILE_PARAMS


def _list_input_files(items: Sequence[Path]) -> List[Path]:
    """Expand input paths into a flat list of LAS/LAZ/COPC files."""
    files: List[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            dir_files = sorted(path.glob("*.las")) + sorted(path.glob("*.laz"))
            files.extend(dir_files)
        else:
            files.append(path)
    # De-duplicate while preserving order.
    seen = set()
    result = []
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _schema_signature(path: Path) -> Tuple[int, Tuple[Tuple[str, str], ...]]:
    """Return a compact schema signature for compatibility checks."""
    with laspy.open(str(path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        point_format_id = reader.header.point_format.id
        sample = reader.read_points(1)
        dim_pairs = []
        for dim in reader.header.point_format.dimension_names:
            arr = getattr(sample, dim, None)
            dtype_name = str(getattr(arr, "dtype", "unknown")) if arr is not None else "unknown"
            dim_pairs.append((str(dim), dtype_name))
    return point_format_id, tuple(dim_pairs)


def _validate_input_files(input_files: Sequence[Path]) -> None:
    """Ensure all inputs exist and share a compatible point schema."""
    missing = [str(path) for path in input_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing)}")
    if not input_files:
        raise ValueError("No LAS/LAZ input files found")

    base_sig = _schema_signature(input_files[0])
    mismatches = []
    for path in input_files[1:]:
        if _schema_signature(path) != base_sig:
            mismatches.append(path.name)
    if mismatches:
        raise ValueError(
            "All inputs must share the same schema for chunked COPC writing. "
            f"Mismatched files: {', '.join(mismatches)}"
        )


def _stream_file_into_las_parts(
    input_path: Path,
    parts_dir: Path,
    chunk_size: int,
    start_part_index: int,
) -> Tuple[List[Path], int]:
    """Stream one input file into temporary LAS parts."""
    parts: List[Path] = []
    written_points = 0

    with laspy.open(str(input_path), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        total_points = int(reader.header.point_count)
        est_chunks = max(1, math.ceil(total_points / chunk_size))
        for chunk_idx, chunk in enumerate(reader.chunk_iterator(chunk_size), start=1):
            if len(chunk) == 0:
                continue
            part_path = parts_dir / f"part_{start_part_index + len(parts):06d}.las"
            part_header = _make_tile_header(reader.header)
            with laspy.open(str(part_path), mode="w", header=part_header) as writer:
                writer.write_points(chunk)
            parts.append(part_path)
            written_points += len(chunk)
            print(
                f"    {input_path.name}: chunk {chunk_idx}/{est_chunks} "
                f"-> {part_path.name} ({len(chunk):,} pts)",
                flush=True,
            )

    return parts, written_points


def _run_untwine(parts: Iterable[Path], output_copc: Path) -> None:
    """Finalize LAS parts into a COPC file via untwine."""
    untwine_cmd = get_untwine_path(require=True)
    cmd = [untwine_cmd]
    for part in parts:
        cmd.extend(["-i", str(part)])
    cmd.extend(["-o", str(output_copc)])

    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "untwine failed").strip()
        raise RuntimeError(message[:400] or "untwine failed")
    if not output_copc.exists() or output_copc.stat().st_size == 0:
        raise RuntimeError("untwine produced no COPC output")


def write_copc_chunked(
    input_items: Sequence[Path],
    output_copc: Path,
    chunk_size: int,
    overwrite: bool = False,
    keep_temp: bool = False,
    temp_dir: Path | None = None,
) -> Path:
    """Chunk-stream inputs into temp LAS parts, then finalize to COPC."""
    input_files = _list_input_files(input_items)
    _validate_input_files(input_files)

    output_copc = Path(output_copc)
    if output_copc.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_copc}")
        output_copc.unlink()

    output_copc.parent.mkdir(parents=True, exist_ok=True)
    temp_root_parent = Path(temp_dir) if temp_dir is not None else output_copc.parent
    temp_root_parent.mkdir(parents=True, exist_ok=True)

    temp_root = Path(
        tempfile.mkdtemp(prefix=f"{output_copc.stem}_chunked_", dir=str(temp_root_parent))
    )
    parts_dir = temp_root / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Chunked COPC Writer", flush=True)
    print("=" * 60, flush=True)
    print(f"Inputs:      {len(input_files)} file(s)", flush=True)
    for path in input_files:
        print(f"  - {path}", flush=True)
    print(f"Output:      {output_copc}", flush=True)
    print(f"Chunk size:  {chunk_size:,} points", flush=True)
    print(f"Temp dir:    {temp_root}", flush=True)
    print()

    all_parts: List[Path] = []
    total_points = 0
    try:
        for file_idx, input_path in enumerate(input_files, start=1):
            print(f"[{file_idx}/{len(input_files)}] Staging {input_path.name}...", flush=True)
            parts, written_points = _stream_file_into_las_parts(
                input_path=input_path,
                parts_dir=parts_dir,
                chunk_size=chunk_size,
                start_part_index=len(all_parts),
            )
            all_parts.extend(parts)
            total_points += written_points
            print(
                f"  {input_path.name}: {len(parts)} part(s), {written_points:,} points staged",
                flush=True,
            )

        if not all_parts:
            raise RuntimeError("No temporary LAS parts were written")

        print()
        print(f"Finalizing {len(all_parts)} part(s) to COPC with untwine...", flush=True)
        _run_untwine(all_parts, output_copc)
        print(
            f"Done: wrote {output_copc} from {len(input_files)} input file(s), "
            f"{total_points:,} points staged",
            flush=True,
        )
        return output_copc
    finally:
        if keep_temp:
            print(f"Keeping temporary files in {temp_root}", flush=True)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a COPC file via chunked LAS staging and untwine.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input LAS/LAZ files or directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output COPC path (typically *.copc.laz).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(TILE_PARAMS.get("chunk_size", 20_000_000)),
        help=(
            "Points per temporary LAS part. Larger values reduce the number of "
            "part files but increase peak RAM."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output if it already exists.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary LAS parts for inspection/debugging.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Parent directory for temporary LAS parts (default: output directory).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        write_copc_chunked(
            input_items=args.input,
            output_copc=args.output,
            chunk_size=args.chunk_size,
            overwrite=bool(args.overwrite),
            keep_temp=bool(args.keep_temp),
            temp_dir=args.temp_dir,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
