#!/usr/bin/env python3
"""
Main tiling script: index building and tiling from LAZ/LAS input.

This script handles the first phase of the 3DTrees pipeline:
1. Build spatial index (tindex) from input LAZ/LAS files
2. Calculate tile bounds
3. Create overlapping tiles (laspy crop + PDAL merge to COPC)

Uses laspy and PDAL only; no untwine.

Usage:
    python main_tile.py --input_dir /path/to/input --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plot_tiles_and_copc

from parameters import TILE_PARAMS



def get_pdal_path() -> str:
    """Get the path to pdal executable."""
    import shutil
    # Use shutil.which to find pdal in PATH
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def get_pdal_wrench_path() -> str:
    """Get the path to pdal_wrench executable."""
    import shutil
    # Use shutil.which to find pdal_wrench in PATH
    wrench_path = shutil.which("pdal_wrench")
    return wrench_path if wrench_path else "pdal_wrench"


def _laspy_laz_backend():
    """Return the LAZ backend to use for laspy (Lazrs or LazrsParallel when available)."""
    try:
        import laspy
        if hasattr(laspy.LazBackend, "LazrsParallel"):
            return laspy.LazBackend.LazrsParallel
        if hasattr(laspy.LazBackend, "Lazrs"):
            return laspy.LazBackend.Lazrs
    except Exception:
        pass
    return None


def build_tindex(input_dir: Path, output_gpkg: Path) -> Path:
    """
    Build spatial index (tindex) from LAZ/LAS files.

    Uses pdal tindex to create a GeoPackage containing the spatial
    extents of all point cloud files for efficient spatial queries.

    Args:
        input_dir: Directory containing input LAZ/LAS files
        output_gpkg: Output path for tindex GeoPackage

    Returns:
        Path to created tindex file
    """
    print()
    print("=" * 60)
    print("Step 1: Building spatial index (tindex)")
    print("=" * 60)

    # Check if tindex already exists
    if output_gpkg.exists():
        print(f"  Using existing tindex: {output_gpkg}")
        return output_gpkg

    # Create output directory
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    # Find LAZ and LAS files (exclude .copc.laz)
    source_files = sorted(
        list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
    )
    source_files = [f for f in source_files if not f.name.endswith(".copc.laz")]
    if not source_files:
        raise ValueError(f"No LAZ/LAS files found in {input_dir}")

    # Try to get SRS from the first file to avoid default EPSG:4326 in tindex
    tindex_srs = None
    try:
        pdal_cmd = get_pdal_path()
        info_cmd = [pdal_cmd, "info", "--metadata", str(source_files[0])]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True, check=False)
        if info_result.returncode == 0:
            meta = json.loads(info_result.stdout)
            tindex_srs = meta.get("metadata", {}).get("srs", {}).get("compoundwkt") or \
                        meta.get("metadata", {}).get("spatialreference")
    except Exception as e:
        print(f"  Warning: Could not extract SRS for tindex: {e}")

    print(f"  Found {len(source_files)} source files")
    print(f"  Output: {output_gpkg}")

    # Create file list for pdal tindex (absolute paths, one per line).
    # Use the path as-is (do not resolve symlinks) so the path keeps .laz/.las extension;
    # Galaxy stages files as .dat and we symlink to input_dir/*.laz - resolving would give .dat and PDAL would fail.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for pf in source_files:
            f.write(f"{pf.absolute()}\n")
        file_list_path = Path(f.name)

    try:
        pdal_cmd = get_pdal_path()
        cmd = [
            pdal_cmd, "tindex", "create",
            str(output_gpkg),
            "--filelist", str(file_list_path),
            "--tindex_name=Location",
            "--ogrdriver=GPKG",
            "--fast_boundary",
            "--write_absolute_path",
        ]
        if tindex_srs:
            cmd.append(f"--t_srs={tindex_srs}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"pdal tindex failed: {result.stderr or result.stdout or 'unknown error'}"
            )

        print(f"  ✓ Tindex created: {output_gpkg}")

    finally:
        if file_list_path.exists():
            file_list_path.unlink()

    return output_gpkg


def calculate_tile_bounds(
    tindex_file: Path,
    tile_length: float,
    tile_buffer: float,
    output_dir: Path,
    grid_offset: float = 1.0
) -> Tuple[Path, Path, dict]:
    """
    Calculate tile bounds from tindex.
    
    Uses prepare_tile_jobs.py to compute tile grid based on the
    spatial extent of input files.
    
    Args:
        tindex_file: Path to tindex GeoPackage
        tile_length: Tile size in meters
        tile_buffer: Buffer overlap in meters
        grid_offset: Offset from min coordinates
        output_dir: Directory for output files
    
    Returns:
        Tuple of (tile_jobs_file, tile_bounds_json, env_dict)
    """
    print()
    print("=" * 60)
    print("Step 2: Calculating tile bounds")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    prepare_jobs_script = script_dir / "prepare_tile_jobs.py"
    
    jobs_file = output_dir / f"tile_jobs_{int(tile_length)}m.txt"
    bounds_json = output_dir / "tile_bounds_tindex.json"
    
    cmd = [
        sys.executable,
        str(prepare_jobs_script),
        str(tindex_file),
        f"--tile-length={tile_length}",
        f"--tile-buffer={tile_buffer}",
        f"--jobs-out={jobs_file}",
        f"--bounds-out={bounds_json}",
        f"--grid-offset={grid_offset}"
    ]
    
    print(f"  Tile length: {tile_length}m")
    print(f"  Tile buffer: {tile_buffer}m")
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"prepare_tile_jobs.py failed: {result.stderr}")
    
    # Parse environment variables from output
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"')
    
    tile_count = env.get('tile_count', 'unknown')
    print(f"  ✓ Calculated {tile_count} tiles")
    print(f"  Jobs file: {jobs_file}")
    print(f"  Bounds file: {bounds_json}")
    
    return jobs_file, bounds_json, env


def update_tile_bounds_json_from_files(
    tile_bounds_json: Path,
    files_dir: Path,
    file_glob: str = "*.laz",
) -> int:
    """
    Update tile_bounds_tindex.json so each tile's bounds match the actual
    file header bounds from the created tiles (e.g. subsampled LAZ).
    This keeps the JSON in sync with real data extent for remap/merge matching.

    Matches tiles by label c{col:02d}_r{row:02d} (e.g. c00_r00) to filenames
    that start with that label (e.g. c00_r00_subsampled_1cm.laz).

    Returns:
        Number of tiles whose bounds were updated.
    """
    from merge_tiles import get_tile_bounds_from_header

    if not tile_bounds_json.exists():
        return 0
    with tile_bounds_json.open() as f:
        data = json.load(f)
    tiles = data.get("tiles", [])
    if not tiles:
        return 0

    # Build label -> file path from files_dir
    label_to_path: Dict[str, Path] = {}
    for f in files_dir.glob(file_glob):
        stem = f.stem
        # Match c00_r00 (prefix before _subsampled or similar)
        for sep in ("_subsampled", "_chunk", "."):
            if sep in stem:
                stem = stem.split(sep)[0]
                break
        if stem and stem not in label_to_path:
            label_to_path[stem] = f

    updated = 0
    for tile in tiles:
        col, row = tile["col"], tile["row"]
        label = f"c{col:02d}_r{row:02d}"
        path = label_to_path.get(label)
        if path is None:
            continue
        bounds = get_tile_bounds_from_header(path)
        if bounds is None:
            continue
        minx, maxx, miny, maxy = bounds
        tile["bounds"] = [[minx, maxx], [miny, maxy]]
        updated += 1

    if updated > 0:
        with tile_bounds_json.open("w") as f:
            json.dump(data, f, indent=2)
    return updated


def get_source_files_from_tindex(tindex_file: Path) -> List[str]:
    """Get list of source point cloud files (LAZ/LAS paths) from tindex database."""
    import sqlite3
    
    conn = sqlite3.connect(str(tindex_file))
    cursor = conn.cursor()
    
    # Get table name from gpkg_contents
    cursor.execute('SELECT table_name FROM gpkg_contents WHERE data_type = "features" LIMIT 1')
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return []
    
    table_name = result[0]
    cursor.execute(f'SELECT DISTINCT Location FROM "{table_name}"')
    files = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return files


def get_source_bounds_from_tindex(tindex_file: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """Get spatial bounds for each source file from tindex GeoPackage geometry.
    
    Returns dict mapping file path -> (minx, miny, maxx, maxy).
    """
    import sqlite3
    import struct
    
    conn = sqlite3.connect(str(tindex_file))
    cursor = conn.cursor()
    
    cursor.execute('SELECT table_name, column_name FROM gpkg_geometry_columns LIMIT 1')
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {}
    table_name, geom_col = row
    
    cursor.execute(f'SELECT Location, "{geom_col}" FROM "{table_name}"')
    bounds_map = {}
    for filepath, geom_blob in cursor.fetchall():
        if not geom_blob or not filepath:
            continue
        try:
            # GeoPackage geometry binary: header (magic GP, version, flags, srs_id, envelope)
            # flags byte at offset 3 tells envelope type
            flags = geom_blob[3]
            envelope_type = (flags >> 1) & 0x07
            header_size = 8  # magic(2) + version(1) + flags(1) + srs_id(4)
            if envelope_type == 1:  # [minx, maxx, miny, maxy]
                minx, maxx, miny, maxy = struct.unpack_from('<dddd', geom_blob, header_size)
                bounds_map[filepath] = (minx, miny, maxx, maxy)
            elif envelope_type == 2:  # [minx, maxx, miny, maxy, minz, maxz]
                minx, maxx, miny, maxy = struct.unpack_from('<dddd', geom_blob, header_size)
                bounds_map[filepath] = (minx, miny, maxx, maxy)
        except (struct.error, IndexError):
            continue
    
    conn.close()
    return bounds_map


def _parse_proj_bounds(proj_bounds: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse '([xmin,xmax],[ymin,ymax])' into (xmin, ymin, xmax, ymax)."""
    try:
        s = proj_bounds.strip().strip("()")
        parts = s.split("],[")
        xpart = parts[0].strip("([])").split(",")
        ypart = parts[1].strip("([])").split(",")
        xmin, xmax = float(xpart[0]), float(xpart[1])
        ymin, ymax = float(ypart[0]), float(ypart[1])
        return (xmin, ymin, xmax, ymax)
    except (ValueError, IndexError):
        return None


def _bounds_overlap(a: Tuple[float, float, float, float],
                    b: Tuple[float, float, float, float]) -> bool:
    """Check if two (minx, miny, maxx, maxy) boxes overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _get_bounds(
    filepath: str,
    source_bounds: Dict[str, Tuple[float, float, float, float]],
    bounds_by_basename: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Look up bounds by path, with basename fallback."""
    fb = source_bounds.get(filepath)
    if fb is not None:
        return fb
    if bounds_by_basename is not None:
        return bounds_by_basename.get(Path(filepath).name)
    return None


def filter_source_files_for_tile(
    source_files: List[str],
    source_bounds: Dict[str, Tuple[float, float, float, float]],
    tile_bounds: Tuple[float, float, float, float],
    bounds_by_basename: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> List[str]:
    """Return only source files whose bounds overlap the tile bounds."""
    result = []
    for f in source_files:
        fb = _get_bounds(f, source_bounds, bounds_by_basename)
        if fb is None:
            result.append(f)  # no bounds info, keep as candidate
        elif _bounds_overlap(fb, tile_bounds):
            result.append(f)
    return result


def _crop_with_laspy(
    input_file: str,
    output_file: Path,
    bounds: Tuple[float, float, float, float],
) -> Tuple[bool, int, str]:
    """Crop a LAZ/COPC file to bounds using laspy + numpy.

    Reads the full file, applies a bounding box mask, and writes the
    cropped points as compressed LAZ.  This bypasses PDAL's readers.copc
    which hangs on large selections (>50M points).

    Args:
        input_file: Path to input LAZ/COPC file.
        output_file: Path for the cropped output LAZ file.
        bounds: (xmin, ymin, xmax, ymax) bounding box.

    Returns:
        (success, point_count, message)
    """
    import laspy
    import numpy as np

    xmin, ymin, xmax, ymax = bounds

    try:
        laz_backend = _laspy_laz_backend()
        kwargs = {}
        if input_file.lower().endswith(".laz") and laz_backend is not None:
            kwargs["laz_backend"] = laz_backend

        las = laspy.read(input_file, **kwargs)

        mask = (
            (np.asarray(las.x) >= xmin)
            & (np.asarray(las.x) <= xmax)
            & (np.asarray(las.y) >= ymin)
            & (np.asarray(las.y) <= ymax)
        )

        count = int(mask.sum())
        if count == 0:
            return (True, 0, "No points in bounds")

        cropped = las.points[mask]

        new_header = laspy.LasHeader(
            point_format=las.header.point_format,
            version=las.header.version,
        )
        new_header.offsets = las.header.offsets
        new_header.scales = las.header.scales
        # Copy non-COPC VLRs from source, avoiding duplicates
        existing_vlr_keys = {
            (getattr(v, "user_id", ""), v.record_id) for v in new_header.vlrs
        }
        for vlr in las.header.vlrs:
            if vlr.record_id in (1, 2) and getattr(vlr, "user_id", "") == "copc":
                continue
            vlr_key = (getattr(vlr, "user_id", ""), vlr.record_id)
            if vlr_key not in existing_vlr_keys:
                new_header.vlrs.append(vlr)
                existing_vlr_keys.add(vlr_key)
        # Copy extra dimensions (laspy 2.x: header.point_format.extra_dimensions)
        try:
            extra_dims_src = getattr(las.header.point_format, "extra_dimensions", None)
            if extra_dims_src:
                extra_dims_dst = getattr(new_header.point_format, "extra_dimensions", None)
                existing = {d.name for d in (extra_dims_dst or [])}
                for dim in extra_dims_src:
                    if dim.name not in existing:
                        new_header.add_extra_dim(laspy.ExtraBytesParams(
                            name=dim.name, type=dim.dtype, description=getattr(dim, "description", "") or "",
                        ))
                        existing.add(dim.name)
        except Exception:
            pass  # Header may already match; write with standard dims if needed

        new_las = laspy.LasData(new_header)
        new_las.points = cropped

        write_kwargs = {}
        if laz_backend is not None:
            write_kwargs["laz_backend"] = laz_backend
        new_las.write(str(output_file), **write_kwargs)

        return (True, count, "OK")
    except Exception as e:
        return (False, 0, str(e))


def _distribute_source_file(args: Tuple) -> List[Tuple[str, int]]:
    """
    Phase 1: Read one source file (chunked) and write cropped parts for all
    overlapping tiles.

    Each source file is read exactly once.  Points are streamed in chunks
    of CHUNK_SIZE to limit peak memory, and for each chunk the bounding-box
    mask is applied for every overlapping tile.  Matching points are
    appended to per-tile LAZ part files (one per source file).

    When the LazrsParallel backend is used, RAYON_NUM_THREADS is set from the
    threads argument so chunk decompression uses multiple threads.

    Args:
        args: (source_idx, src_file, overlapping_tiles, tiles_dir, decompress_threads, chunk_size)
              overlapping_tiles: list of (label, (xmin, ymin, xmax, ymax))
              decompress_threads: threads for LAZ decompression (LazrsParallel / Rayon)
              chunk_size: points per chunk (smaller = less peak RAM, more overhead)

    Returns:
        list of (tile_label, point_count) for tiles that received points
    """
    import laspy
    import numpy as np

    source_idx, src_file, overlapping_tiles, tiles_dir, decompress_threads, chunk_size = args

    # So LazrsParallel (Rayon) uses N threads for chunk decompression
    if decompress_threads and decompress_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(decompress_threads)

    if not os.path.isfile(src_file):
        return []

    try:
        laz_backend = _laspy_laz_backend()
        open_kwargs = {}
        if src_file.lower().endswith(".laz") and laz_backend is not None:
            open_kwargs["laz_backend"] = laz_backend

        # --- stream through the file in chunks --------------------------------
        # We accumulate per-tile arrays and flush once at the end so that
        # each tile gets exactly one part file from this source.
        tile_arrays: Dict[str, list] = {label: [] for label, _ in overlapping_tiles}
        tile_counts: Dict[str, int] = {label: 0 for label, _ in overlapping_tiles}

        # Build a compact bounds array for vectorised overlap tests
        tile_labels = [label for label, _ in overlapping_tiles]
        tile_xmin = np.array([b[0] for _, b in overlapping_tiles])
        tile_xmax = np.array([b[2] for _, b in overlapping_tiles])
        tile_ymin = np.array([b[1] for _, b in overlapping_tiles])
        tile_ymax = np.array([b[3] for _, b in overlapping_tiles])

        header_snapshot = None  # will be captured from the first chunk

        with laspy.open(src_file, **open_kwargs) as reader:
            header_snapshot = reader.header

            for chunk in reader.chunk_iterator(chunk_size):
                cx = np.asarray(chunk.x)
                cy = np.asarray(chunk.y)

                for i, label in enumerate(tile_labels):
                    mask = (
                        (cx >= tile_xmin[i])
                        & (cx <= tile_xmax[i])
                        & (cy >= tile_ymin[i])
                        & (cy <= tile_ymax[i])
                    )
                    cnt = int(mask.sum())
                    if cnt == 0:
                        continue
                    # Store the raw packed array slice (compact, avoids header copy)
                    tile_arrays[label].append(chunk.array[mask])
                    tile_counts[label] += cnt

        if header_snapshot is None:
            return []

        # --- write one part file per tile that received points -----------------
        # Keep intermediate parts uncompressed so Phase 1 avoids repeated LAZ
        # compression work before the final COPC write.
        # Build output header once (shared across all parts from this source)
        new_header = laspy.LasHeader(
            point_format=header_snapshot.point_format,
            version=header_snapshot.version,
        )
        new_header.offsets = header_snapshot.offsets
        new_header.scales = header_snapshot.scales
        # Copy non-COPC VLRs, avoiding duplicates
        existing_vlr_keys = {
            (getattr(v, "user_id", ""), v.record_id) for v in new_header.vlrs
        }
        for vlr in header_snapshot.vlrs:
            if vlr.record_id in (1, 2) and getattr(vlr, "user_id", "") == "copc":
                continue
            vlr_key = (getattr(vlr, "user_id", ""), vlr.record_id)
            if vlr_key not in existing_vlr_keys:
                new_header.vlrs.append(vlr)
                existing_vlr_keys.add(vlr_key)
        # Copy extra dimensions
        try:
            extra_dims_src = getattr(header_snapshot.point_format, "extra_dimensions", None)
            if extra_dims_src:
                extra_dims_dst = getattr(new_header.point_format, "extra_dimensions", None)
                existing = {d.name for d in (extra_dims_dst or [])}
                for dim in extra_dims_src:
                    if dim.name not in existing:
                        new_header.add_extra_dim(laspy.ExtraBytesParams(
                            name=dim.name, type=dim.dtype,
                            description=getattr(dim, "description", "") or "",
                        ))
                        existing.add(dim.name)
        except Exception:
            pass

        results: List[Tuple[str, int]] = []
        for label in tile_labels:
            if not tile_arrays[label]:
                continue
            tile_dir = tiles_dir / label
            tile_dir.mkdir(exist_ok=True)
            part_file = tile_dir / f"part_{source_idx}.las"

            combined = np.concatenate(tile_arrays[label])
            point_record = laspy.ScaleAwarePointRecord(
                combined, new_header.point_format, new_header.scales, new_header.offsets,
            )
            new_las = laspy.LasData(new_header)
            new_las.points = point_record
            new_las.write(str(part_file))

            results.append((label, tile_counts[label]))

        return results

    except Exception as e:
        print(f"    ⚠ Error processing {Path(src_file).name}: {e}")
        return []


def _finalize_tile_to_copc(args: Tuple) -> Tuple[str, bool, str]:
    """
    Phase 2: Merge a tile's LAZ part files into a single COPC tile.

    Args:
        args: (label, tiles_dir, log_dir)

    Returns:
        (label, success, message)
    """
    label, tiles_dir, log_dir, finalize_strategy = args

    final_tile = tiles_dir / f"{label}.copc.laz"

    # Skip if already finalised
    if final_tile.exists() and final_tile.stat().st_size > 0:
        return (label, True, "Already exists")

    tile_dir = tiles_dir / label
    if not tile_dir.exists():
        return (label, True, "No data in bounds")

    parts = sorted(tile_dir.glob("part_*.las"))
    if not parts:
        if not any(tile_dir.iterdir()):
            tile_dir.rmdir()
        return (label, True, "No data in bounds")

    try:
        if finalize_strategy == "laspy":
            success, message = _finalize_tile_to_copc_laspy(parts, final_tile)
        else:
            success, message = _finalize_tile_to_copc_pdal(parts, final_tile, log_dir, label)

        if not success:
            return (label, False, message)

        # Clean up parts
        for part in parts:
            if part.exists():
                part.unlink()
        if tile_dir.exists() and not any(tile_dir.iterdir()):
            tile_dir.rmdir()

        return (label, True, f"{len(parts)} parts merged via {finalize_strategy}")

    except Exception as e:
        return (label, False, str(e))


def _finalize_tile_to_copc_pdal(
    parts: List[Path],
    final_tile: Path,
    log_dir: Path,
    label: str,
) -> Tuple[bool, str]:
    """Finalize a tile with the existing PDAL merge pipeline."""
    pdal_cmd = get_pdal_path()

    if len(parts) == 1:
        pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": str(parts[0])},
                {
                    "type": "writers.copc",
                    "filename": str(final_tile),
                    "forward": "all",
                    "extra_dims": "all",
                },
            ]
        }
    else:
        readers = [{"type": "readers.las", "filename": str(p)} for p in parts]
        pipeline = {
            "pipeline": readers + [
                {"type": "filters.merge"},
                {
                    "type": "writers.copc",
                    "filename": str(final_tile),
                    "forward": "all",
                    "extra_dims": "all",
                },
            ]
        }

    pipeline_file = log_dir / f"{label}_pipeline.json"
    with open(pipeline_file, "w") as f:
        json.dump(pipeline, f)

    try:
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True, text=True, check=False,
        )
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()

    if result.returncode != 0:
        return (False, f"COPC conversion failed: {result.stderr[:200]}")
    return (True, "OK")


def _finalize_tile_to_copc_laspy(parts: List[Path], final_tile: Path) -> Tuple[bool, str]:
    """Finalize a tile by concatenating point records in laspy before COPC conversion."""
    import laspy
    import numpy as np

    laz_backend = _laspy_laz_backend()
    read_kwargs = {}
    write_kwargs = {}
    if laz_backend is not None:
        read_kwargs["laz_backend"] = laz_backend
        write_kwargs["laz_backend"] = laz_backend

    merged_temp = final_tile.with_name(f"{final_tile.stem}_merged_tmp.las")

    try:
        arrays = []
        header_snapshot = None

        for part in parts:
            las = laspy.read(str(part), **read_kwargs)
            if header_snapshot is None:
                header_snapshot = las.header
            arrays.append(las.points.array.copy())

        if header_snapshot is None:
            return (False, "No part data found")

        if len(arrays) == 1:
            combined = arrays[0]
        else:
            combined = np.concatenate(arrays)

        merged_header = laspy.LasHeader(
            point_format=header_snapshot.point_format,
            version=header_snapshot.version,
        )
        merged_header.offsets = header_snapshot.offsets
        merged_header.scales = header_snapshot.scales

        existing_vlr_keys = {
            (getattr(v, "user_id", ""), v.record_id) for v in merged_header.vlrs
        }
        for vlr in header_snapshot.vlrs:
            if vlr.record_id in (1, 2) and getattr(vlr, "user_id", "") == "copc":
                continue
            vlr_key = (getattr(vlr, "user_id", ""), vlr.record_id)
            if vlr_key not in existing_vlr_keys:
                merged_header.vlrs.append(vlr)
                existing_vlr_keys.add(vlr_key)

        try:
            extra_dims_src = getattr(header_snapshot.point_format, "extra_dimensions", None)
            if extra_dims_src:
                existing = {d.name for d in getattr(merged_header.point_format, "extra_dimensions", []) or []}
                for dim in extra_dims_src:
                    if dim.name not in existing:
                        merged_header.add_extra_dim(
                            laspy.ExtraBytesParams(
                                name=dim.name,
                                type=dim.dtype,
                                description=getattr(dim, "description", "") or "",
                            )
                        )
                        existing.add(dim.name)
        except Exception:
            pass

        point_record = laspy.ScaleAwarePointRecord(
            combined,
            merged_header.point_format,
            merged_header.scales,
            merged_header.offsets,
        )
        merged_las = laspy.LasData(merged_header)
        merged_las.points = point_record
        merged_las.write(str(merged_temp))

        if not _convert_laz_to_copc_pdal(merged_temp, final_tile):
            return (False, "LAS->COPC conversion failed after laspy merge")
        return (True, "OK")
    finally:
        if merged_temp.exists():
            merged_temp.unlink()


def create_tiles(
    tindex_file: Path,
    tile_jobs_file: Path,
    tiles_dir: Path,
    log_dir: Path,
    threads: int = 5,
    max_parallel: int = 5,
    chunk_size: int = 20_000_000,
    finalize_strategy: str = "pdal",
) -> List[Path]:
    """
    Create overlapping tiles from source LAZ/LAS files (two-phase).

    Phase 1 – Distribute: each source file is read exactly once (in chunks)
    and cropped points are written as per-tile LAZ part files.  This avoids
    the previous O(sources × tiles) read pattern.

    Phase 2 – Finalise: each tile's part files are merged and converted to
    COPC format using PDAL.  Fully parallelised across tiles.

    Args:
        tindex_file: Path to tindex GeoPackage
        tile_jobs_file: Path to tile jobs file
        tiles_dir: Output directory for tiles
        log_dir: Directory for log files
        threads: Threads used per process for LAZ chunk decompression (LazrsParallel/Rayon); also used for Phase 2 PDAL where applicable.
        max_parallel: Maximum parallel workers for each phase
        chunk_size: Points per chunk when reading source files (smaller = less peak RAM)
        finalize_strategy: Phase 2 tile finalization strategy: "pdal" or "laspy"

    Returns:
        List of created tile paths
    """
    print()
    print("=" * 60)
    print("Step 3: Creating tiles (two-phase)")
    print("=" * 60)

    # Create directories
    tiles_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse tile jobs ──────────────────────────────────────────────────
    all_tiles: Dict[str, Tuple[float, float, float, float]] = {}
    with open(tile_jobs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                label = parts[0]
                tb = _parse_proj_bounds(parts[1])
                if tb:
                    all_tiles[label] = tb

    if not all_tiles:
        raise ValueError("No tile jobs found")

    # Skip tiles whose COPC output already exists
    pending_tiles: Dict[str, Tuple[float, float, float, float]] = {}
    already_done = 0
    for label, bounds in all_tiles.items():
        final_tile = tiles_dir / f"{label}.copc.laz"
        if final_tile.exists() and final_tile.stat().st_size > 0:
            already_done += 1
        else:
            pending_tiles[label] = bounds

    # ── Source files & bounds ────────────────────────────────────────────
    source_files = get_source_files_from_tindex(tindex_file)
    if not source_files:
        raise ValueError("No source files found in tindex")

    source_bounds = get_source_bounds_from_tindex(tindex_file)
    bounds_by_basename = (
        {Path(p).name: b for p, b in source_bounds.items()} if source_bounds else {}
    )

    print(f"  Source files: {len(source_files)}")
    print(f"  Total tiles: {len(all_tiles)} ({already_done} already done, {len(pending_tiles)} pending)")
    print(f"  Workers: {max_parallel}")

    if not pending_tiles:
        print("  ✓ All tiles already exist")
        return list(tiles_dir.glob("*.copc.laz"))

    # ── Phase 1: Distribute ─────────────────────────────────────────────
    # For each source file, determine which pending tiles it overlaps.
    distribute_tasks = []
    for source_idx, src_file in enumerate(source_files):
        fb = _get_bounds(src_file, source_bounds, bounds_by_basename)
        overlapping = []
        for label, tb in pending_tiles.items():
            if fb is None or _bounds_overlap(fb, tb):
                overlapping.append((label, tb))
        if overlapping:
            distribute_tasks.append((source_idx, src_file, overlapping, tiles_dir, threads, chunk_size))

    print()
    print(f"  Phase 1: Reading {len(distribute_tasks)} source file(s), "
          f"distributing to {len(pending_tiles)} tile(s)  [chunked reads, {threads} thread(s) per decompress]")
    print()

    tile_point_counts: Dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_distribute_source_file, task): Path(task[1]).name
            for task in distribute_tasks
        }
        for future in as_completed(futures):
            src_name = futures[future]
            try:
                results = future.result()
                for label, count in results:
                    tile_point_counts[label] = tile_point_counts.get(label, 0) + count
                if results:
                    total_pts = sum(c for _, c in results)
                    print(f"    ✓ {src_name}: {total_pts:,} pts → {len(results)} tile(s)")
                else:
                    print(f"    - {src_name}: no overlapping data")
            except Exception as e:
                print(f"    ✗ {src_name}: {e}")

    # ── Phase 2: Finalise ───────────────────────────────────────────────
    finalize_tasks = [
        (label, tiles_dir, log_dir, finalize_strategy) for label in pending_tiles
    ]

    print()
    print(
        f"  Phase 2: Merging & converting {len(finalize_tasks)} tile(s) to COPC "
        f"using '{finalize_strategy}'"
    )
    print()

    successful = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_finalize_tile_to_copc, task): task[0]
            for task in finalize_tasks
        }
        for future in as_completed(futures):
            label, success, message = future.result()
            pts = tile_point_counts.get(label, 0)
            if success:
                if "Already exists" in message or "No data" in message:
                    skipped += 1
                    print(f"    - {label}: {message}")
                else:
                    successful += 1
                    print(f"    ✓ {label}: {message} ({pts:,} pts)")
            else:
                failed += 1
                print(f"    ✗ {label}: {message}")

    print()
    print(f"  Tiling complete: {successful} created, {skipped} skipped, {failed} failed")

    return list(tiles_dir.glob("*.copc.laz"))


def _convert_laz_to_copc_pdal(input_laz: Path, output_copc: Path) -> bool:
    """Convert a single LAZ file to COPC using PDAL."""
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_laz)},
            {
                "type": "writers.copc",
                "filename": str(output_copc),
                "forward": "all",
                "extra_dims": "all",
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f)
        pipeline_file = Path(f.name)
    try:
        pdal_cmd = get_pdal_path()
        r = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True, text=True, check=False,
        )
        return r.returncode == 0 and output_copc.exists() and output_copc.stat().st_size > 0
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()


def run_tiling_pipeline(
    input_dir: Path,
    output_dir: Path,
    tile_length: float = 100,
    tile_buffer: float = 5,
    grid_offset: float = 1.0,
    num_workers: int = 4,
    threads: int = 5,
    max_tile_procs: int = 5,
    dimension_reduction: bool = True,  # Ignored (kept for API compatibility)
    tiling_threshold: float = None,
    chunk_size: int = 2_000_000,
    finalize_strategy: str = "pdal",
) -> Path:
    """
    Run the complete tiling pipeline (laspy + PDAL only).

    Steps:
    1. Build spatial index (tindex) from input LAZ/LAS files
    2. Calculate tile bounds
    3. Create overlapping tiles (laspy crop, PDAL merge to COPC)

    If input folder contains a single file below tiling_threshold, converts it to
    COPC with PDAL and returns that directory for direct subsampling.

    Args:
        input_dir: Directory containing input LAZ/LAS files
        output_dir: Base output directory
        tile_length: Tile size in meters
        tile_buffer: Buffer overlap in meters
        grid_offset: Offset from min coordinates
        num_workers: Unused (kept for API compatibility)
        threads: Threads per PDAL writer
        max_tile_procs: Maximum parallel tile processes
        dimension_reduction: Ignored (kept for API compatibility)
        tiling_threshold: File size threshold in MB. If single file below this, skip tiling
        chunk_size: Points per chunk when reading LAZ/LAS in Phase 1 (smaller = less peak RAM)

    Returns:
        Path to tiles directory (or copc_single directory if tiling was skipped)
    """
    print("=" * 60)
    print("3DTrees Tiling Pipeline (laspy + PDAL)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tile size: {tile_length}m with {tile_buffer}m buffer")
    print()

    tiles_dir = output_dir / f"tiles_{int(tile_length)}m"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tindex_file = output_dir / f"tindex_{int(tile_length)}m.gpkg"

    # Check if we should skip tiling (single small file)
    should_skip_tiling = False
    if tiling_threshold is not None:
        input_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
        input_files = [f for f in input_files if not f.name.endswith(".copc.laz")]

        if len(input_files) == 1:
            original_size_mb = input_files[0].stat().st_size / (1024 * 1024)
            if original_size_mb < tiling_threshold:
                should_skip_tiling = True
                print("=" * 60)
                print("Tiling Threshold Check")
                print("=" * 60)
                print(f"  Single file detected: {input_files[0].name}")
                print(f"  Original file size: {original_size_mb:.2f} MB")
                print(f"  Threshold: {tiling_threshold} MB")
                print(f"  Decision: Will skip tiling, convert to COPC with PDAL")
                print("=" * 60)
                print()

    # Validate input
    source_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
    source_files = [f for f in source_files if not f.name.endswith(".copc.laz")]
    if not source_files:
        raise ValueError(f"No LAZ/LAS files found in {input_dir}")

    # Step 1: Build tindex from input LAZ/LAS
    tindex_file = build_tindex(input_dir, tindex_file)

    # Step 2: Calculate tile bounds
    jobs_file, bounds_json, env = calculate_tile_bounds(
        tindex_file, tile_length, tile_buffer, output_dir, grid_offset
    )

    # Symlink tindex for Galaxy if needed
    fixed_tindex = output_dir / "tindex.gpkg"
    if not fixed_tindex.exists() and tindex_file.exists():
        if fixed_tindex.is_symlink():
            fixed_tindex.unlink()
        fixed_tindex.symlink_to(tindex_file.name)

    # Plot overview
    plot_tiles_and_copc.plot_extents(
        tindex_file, bounds_json, output_dir / "overview_copc_tiles.png"
    )

    # Check if we should skip tiling (single small file)
    # Done AFTER tindex/bounds/plot so those outputs are always available for merge
    if should_skip_tiling:
        print()
        print("=" * 60)
        print("Skipping Tiling (Single Small File)")
        print("=" * 60)
        laz_file = [f for f in source_files if not f.name.endswith(".copc.laz")][0]
        copc_single_dir = output_dir / "copc_single"
        copc_single_dir.mkdir(parents=True, exist_ok=True)
        out_copc = copc_single_dir / f"{laz_file.stem}.copc.laz"
        if not out_copc.exists() or out_copc.stat().st_size == 0:
            print("  Converting LAZ to COPC (PDAL)...")
            if not _convert_laz_to_copc_pdal(laz_file, out_copc):
                raise RuntimeError(f"PDAL LAZ→COPC conversion failed: {laz_file}")
            print(f"  ✓ Created {out_copc.name}")
        else:
            print(f"  Using existing {out_copc.name}")
        print(f"  Returning COPC directory for direct subsampling")
        print("=" * 60)
        return copc_single_dir

    # Step 3: Create tiles
    tile_files = create_tiles(
        tindex_file,
        jobs_file,
        tiles_dir,
        log_dir,
        threads,
        max_tile_procs,
        chunk_size,
        finalize_strategy,
    )

    print()
    print("=" * 60)
    print("Tiling Pipeline Complete")
    print("=" * 60)
    print(f"  Source files: {len(source_files)}")
    print(f"  Tiles created: {len(tile_files)}")
    print(f"  Tiles directory: {tiles_dir}")

    return tiles_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Tiling Pipeline - laspy + PDAL tiling from LAZ/LAS input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        required=True,
        help="Input directory containing LAZ files"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="Output directory for all stages"
    )
    
    parser.add_argument(
        "--tile_length",
        type=float,
        default=TILE_PARAMS.get('tile_length', 100),
        help=f"Tile size in meters (default: {TILE_PARAMS.get('tile_length', 100)})"
    )
    
    parser.add_argument(
        "--tile_buffer",
        type=float,
        default=TILE_PARAMS.get('tile_buffer', 5),
        help=f"Buffer overlap in meters (default: {TILE_PARAMS.get('tile_buffer', 5)})"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=TILE_PARAMS.get('workers', 4),
        help=f"Number of parallel workers (default: {TILE_PARAMS.get('workers', 4)})"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=TILE_PARAMS.get('threads', 5),
        help=f"Threads per COPC writer (default: {TILE_PARAMS.get('threads', 5)})"
    )
    
    parser.add_argument(
        "--max_tile_procs",
        type=int,
        default=5,
        help="Maximum parallel tile processes (default: 5)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=TILE_PARAMS.get("chunk_size", 20_000_000),
        help="Points per chunk when reading LAZ/LAS (default: 2_000_000; smaller = less peak RAM)",
    )
    parser.add_argument(
        "--finalize_strategy",
        choices=("pdal", "laspy"),
        default="pdal",
        help="Tile finalization strategy for Phase 2 (default: pdal)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        tiles_dir = run_tiling_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tile_length=args.tile_length,
            tile_buffer=args.tile_buffer,
            num_workers=args.num_workers,
            threads=args.threads,
            max_tile_procs=args.max_tile_procs,
            chunk_size=args.chunk_size,
            finalize_strategy=args.finalize_strategy,
        )
        print(f"\nTiles ready for subsampling: {tiles_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
