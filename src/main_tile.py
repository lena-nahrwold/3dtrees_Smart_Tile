#!/usr/bin/env python3
"""
Main tiling script: COPC-first source preparation, indexing, and tiling.

This script handles the first phase of the 3DTrees pipeline:
1. Normalize source LAZ/LAS files to COPC while preserving all dimensions
2. Build a spatial index (tindex) from the COPC sources
3. Calculate tile bounds
4. Create overlapping COPC tiles

Uses COPC-aware reads where helpful during tile creation and keeps all source
dimensions intact until the later subsampling stage decides what to retain.

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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


def get_untwine_path(require: bool = False) -> Optional[str]:
    """Get the path to untwine, optionally failing if it is unavailable."""
    untwine_path = shutil.which("untwine")
    if untwine_path:
        return untwine_path
    if require:
        raise RuntimeError(
            "untwine is required for the COPC-first tiling pipeline but was not found in PATH"
        )
    return None


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


def list_point_cloud_files(input_dir: Path, include_copc: bool = True) -> List[Path]:
    """List point cloud files in a directory, optionally including COPC."""
    files = sorted(list(input_dir.glob("*.las")) + list(input_dir.glob("*.laz")))
    if include_copc:
        return files
    return [f for f in files if not f.name.endswith(".copc.laz")]


def _copc_name_for_source(source_file: Path) -> str:
    """Return the canonical COPC filename for a source point cloud."""
    if source_file.name.endswith(".copc.laz"):
        return source_file.name
    return f"{source_file.stem}.copc.laz"


def _format_pdal_bounds(bounds: Tuple[float, float, float, float]) -> str:
    """Format bounds for PDAL readers.copc."""
    xmin, ymin, xmax, ymax = bounds
    return f"([{xmin},{xmax}],[{ymin},{ymax}])"


def _union_bounds(bounds_list: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Return the union bbox for a list of bounds."""
    return (
        min(b[0] for b in bounds_list),
        min(b[1] for b in bounds_list),
        max(b[2] for b in bounds_list),
        max(b[3] for b in bounds_list),
    )


def build_tindex(input_dir: Path, output_gpkg: Path) -> Path:
    """
    Build spatial index (tindex) from point cloud files.

    Uses pdal tindex to create a GeoPackage containing the spatial
    extents of all point cloud files for efficient spatial queries.

    Args:
        input_dir: Directory containing input point cloud files
        output_gpkg: Output path for tindex GeoPackage

    Returns:
        Path to created tindex file
    """
    print()
    print("=" * 60)
    print("Building spatial index (tindex)")
    print("=" * 60)

    # Check if tindex already exists
    if output_gpkg.exists():
        print(f"  Using existing tindex: {output_gpkg}")
        return output_gpkg

    # Create output directory
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    # Find LAZ/LAS/COPC source files.
    source_files = list_point_cloud_files(input_dir, include_copc=True)
    if not source_files:
        raise ValueError(f"No point cloud files found in {input_dir}")

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
    print("Calculating tile bounds")
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
    Update tile_bounds_tindex.json with the actual file header bounds from the
    created tiles (e.g. subsampled LAZ), while preserving the planned tiling
    geometry used for ownership and neighbor logic.

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
        if "planned_bounds" not in tile and "bounds" in tile:
            tile["planned_bounds"] = tile["bounds"]
        tile["actual_bounds"] = [[minx, maxx], [miny, maxy]]
        updated += 1

    if updated > 0:
        with tile_bounds_json.open("w") as f:
            json.dump(data, f, indent=2)
    return updated


def _read_pointcloud_header_bounds(pointcloud_file: Path) -> Tuple[float, float, float, float]:
    """Read XY header bounds from a LAZ/LAS/COPC file."""
    import laspy

    if not pointcloud_file.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pointcloud_file}")

    open_kwargs = {}
    laz_backend = _laspy_laz_backend()
    if laz_backend is not None:
        open_kwargs["laz_backend"] = laz_backend

    try:
        with laspy.open(str(pointcloud_file), **open_kwargs) as las:
            return (
                float(las.header.x_min),
                float(las.header.x_max),
                float(las.header.y_min),
                float(las.header.y_max),
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read header bounds from {pointcloud_file}: {exc}"
        ) from exc


def rewrite_tile_bounds_json_for_single_file_skip(
    tile_bounds_json: Path,
    pointcloud_file: Path,
) -> Dict[str, float]:
    """
    Rewrite tile_bounds_tindex.json to one tile based on actual file bounds.

    This is used when tiling is intentionally skipped for a single small file.
    """
    if not tile_bounds_json.exists():
        raise FileNotFoundError(f"tile_bounds_tindex.json not found: {tile_bounds_json}")

    with tile_bounds_json.open() as f:
        data = json.load(f)

    minx, maxx, miny, maxy = _read_pointcloud_header_bounds(pointcloud_file)
    actual_bounds = [[minx, maxx], [miny, maxy]]
    extent = {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}

    template_tile = {}
    tiles = data.get("tiles", [])
    if isinstance(tiles, list) and tiles:
        template_tile = dict(tiles[0])

    single_tile = dict(template_tile)
    single_tile["col"] = 0
    single_tile["row"] = 0
    single_tile["core"] = actual_bounds
    single_tile["planned_bounds"] = actual_bounds
    single_tile["bounds"] = actual_bounds
    single_tile["actual_bounds"] = actual_bounds
    data["tiles"] = [single_tile]

    data["geo_extent"] = extent
    data["proj_extent"] = extent
    data["grid_bounds"] = {
        "xmin": minx,
        "xmax": maxx,
        "ymin": miny,
        "ymax": maxy,
    }

    with tile_bounds_json.open("w") as f:
        json.dump(data, f, indent=2)

    return extent


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


def _make_tile_header(header_snapshot, offsets=None, scales=None):
    """Create a LasHeader from a source header, copying VLRs and extra dimensions.

    Args:
        header_snapshot: Source laspy header to copy from.
        offsets: Optional XYZ offsets. Defaults to source header offsets.
        scales: Optional XYZ scales. Defaults to source header scales.

    Returns:
        A new laspy.LasHeader ready for writing.
    """
    import laspy

    hdr = laspy.LasHeader(
        point_format=header_snapshot.point_format,
        version=header_snapshot.version,
    )
    hdr.offsets = offsets if offsets is not None else header_snapshot.offsets
    hdr.scales = scales if scales is not None else header_snapshot.scales

    # Copy non-COPC VLRs, avoiding duplicates
    existing_vlr_keys = {
        (getattr(v, "user_id", ""), v.record_id) for v in hdr.vlrs
    }
    for vlr in header_snapshot.vlrs:
        if vlr.record_id in (1, 2) and getattr(vlr, "user_id", "") == "copc":
            continue
        vlr_key = (getattr(vlr, "user_id", ""), vlr.record_id)
        if vlr_key not in existing_vlr_keys:
            hdr.vlrs.append(vlr)
            existing_vlr_keys.add(vlr_key)

    # Copy extra dimensions
    try:
        extra_dims_src = getattr(header_snapshot.point_format, "extra_dimensions", None)
        if extra_dims_src:
            extra_dims_dst = getattr(hdr.point_format, "extra_dimensions", None)
            existing = {d.name for d in (extra_dims_dst or [])}
            for dim in extra_dims_src:
                if dim.name not in existing:
                    hdr.add_extra_dim(laspy.ExtraBytesParams(
                        name=dim.name, type=dim.dtype,
                        description=getattr(dim, "description", "") or "",
                    ))
                    existing.add(dim.name)
    except Exception:
        pass

    return hdr


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

        new_header = _make_tile_header(las.header)

        new_las = laspy.LasData(new_header)
        new_las.points = cropped

        write_kwargs = {}
        if laz_backend is not None:
            write_kwargs["laz_backend"] = laz_backend
        new_las.write(str(output_file), **write_kwargs)

        return (True, count, "OK")
    except Exception as e:
        return (False, 0, str(e))


def _materialize_copc_subset(
    src_file: str,
    bounds: Tuple[float, float, float, float],
    label: str,
) -> Tuple[Optional[Path], str]:
    """Materialize a spatial COPC subset to a temporary LAS for local chunked processing."""
    pdal_cmd = get_pdal_path()
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{label}_subset_", suffix=".las")
    os.close(tmp_fd)
    subset_path = Path(tmp_name)

    pipeline = {
        "pipeline": [
            {
                "type": "readers.copc",
                "filename": str(src_file),
                "bounds": _format_pdal_bounds(bounds),
            },
            {
                "type": "writers.las",
                "filename": str(subset_path),
                "forward": "all",
                "extra_dims": "all",
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f)
        pipeline_file = Path(f.name)

    try:
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if subset_path.exists():
                subset_path.unlink()
            stderr = (result.stderr or result.stdout or "unknown error").strip()
            return (None, stderr[:200])
        if not subset_path.exists() or subset_path.stat().st_size == 0:
            return (None, "subset pipeline produced no output")
        return (subset_path, "OK")
    finally:
        if pipeline_file.exists():
            pipeline_file.unlink()


def _distribute_source_file(args: Tuple) -> Tuple[List[Tuple[str, int]], str]:
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
        Tuple of:
        - list of (tile_label, point_count) for tiles that received points
        - description of how the source was read
    """
    import laspy
    import numpy as np

    source_idx, src_file, overlapping_tiles, tiles_dir, decompress_threads, chunk_size = args

    # Skip tiles that already have the part file for this source to avoid
    # regenerating intermediate LAS chunks during reruns/resumes.
    pending_overlaps: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for label, bounds in overlapping_tiles:
        part_file = Path(tiles_dir) / label / f"part_{source_idx}.las"
        if part_file.exists() and part_file.stat().st_size > 0:
            continue
        pending_overlaps.append((label, bounds))
    if not pending_overlaps:
        return ([], "all parts already present")
    overlapping_tiles = pending_overlaps

    # So LazrsParallel (Rayon) uses N threads for chunk decompression
    if decompress_threads and decompress_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(decompress_threads)

    if not os.path.isfile(src_file):
        return ([], "missing source file")

    try:
        stream_file = src_file
        temp_subset_path: Optional[Path] = None
        read_mode = "full scan"

        if src_file.lower().endswith(".copc.laz") and overlapping_tiles:
            query_bounds = _union_bounds([bounds for _, bounds in overlapping_tiles])
            temp_subset_path, subset_message = _materialize_copc_subset(
                src_file, query_bounds, f"src{source_idx}"
            )
            if temp_subset_path is not None:
                stream_file = str(temp_subset_path)
                read_mode = f"COPC subset {query_bounds}"
            else:
                read_mode = f"COPC fallback full scan ({subset_message})"

        laz_backend = _laspy_laz_backend()
        open_kwargs = {}
        if stream_file.lower().endswith(".laz") and laz_backend is not None:
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

        with laspy.open(stream_file, **open_kwargs) as reader:
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
            return ([], read_mode)

        # --- write one part file per tile that received points -----------------
        # Keep intermediate parts uncompressed so Phase 1 avoids repeated LAZ
        # compression work before the final COPC write.
        # Build per-tile headers with tile-centered offsets to prevent int32
        # overflow when coordinates are far from the source file's offset.
        src_scales = header_snapshot.scales
        src_offsets = header_snapshot.offsets
        tile_bounds_map = {lbl: bnds for lbl, bnds in overlapping_tiles}

        results: List[Tuple[str, int]] = []
        for label in tile_labels:
            if not tile_arrays[label]:
                continue
            tile_dir = tiles_dir / label
            tile_dir.mkdir(exist_ok=True)
            part_file = tile_dir / f"part_{source_idx}.las"

            combined = np.concatenate(tile_arrays[label])

            # Compute tile-centered offsets to keep scaled values within int32
            bxmin, bymin, bxmax, bymax = tile_bounds_map[label]
            tile_offsets = np.array([
                (bxmin + bxmax) / 2.0,
                (bymin + bymax) / 2.0,
                src_offsets[2],
            ])

            # Re-encode X/Y with tile-specific offsets
            real_x = combined['X'] * src_scales[0] + src_offsets[0]
            real_y = combined['Y'] * src_scales[1] + src_offsets[1]
            combined['X'] = np.round((real_x - tile_offsets[0]) / src_scales[0]).astype(np.int32)
            combined['Y'] = np.round((real_y - tile_offsets[1]) / src_scales[1]).astype(np.int32)

            new_header = _make_tile_header(header_snapshot, offsets=tile_offsets)
            point_record = laspy.ScaleAwarePointRecord(
                combined, new_header.point_format, new_header.scales, new_header.offsets,
            )
            new_las = laspy.LasData(new_header)
            new_las.points = point_record
            new_las.write(str(part_file))

            results.append((label, tile_counts[label]))

        return (results, read_mode)

    except Exception as e:
        print(f"    ⚠ Error processing {Path(src_file).name}: {e}")
        return ([], f"error: {e}")
    finally:
        if 'temp_subset_path' in locals() and temp_subset_path is not None and temp_subset_path.exists():
            temp_subset_path.unlink()


def _finalize_tile_to_copc(args: Tuple) -> Tuple[str, bool, str]:
    """
    Phase 2: Merge a tile's LAZ part files into a single COPC tile.

    Uses untwine for COPC generation.

    Args:
        args: (label, tiles_dir, log_dir, tile_bounds)
              tile_bounds: (xmin, ymin, xmax, ymax) or None

    Returns:
        (label, success, message)
    """
    label, tiles_dir, log_dir, tile_bounds = args

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
        success, message = _finalize_tile_to_copc_untwine(parts, final_tile, log_dir, label)

        if not success:
            return (label, False, message)

        # Clean up parts
        for part in parts:
            if part.exists():
                part.unlink()
        if tile_dir.exists() and not any(tile_dir.iterdir()):
            tile_dir.rmdir()

        return (label, True, f"{len(parts)} parts merged ({message})")

    except Exception as e:
        return (label, False, str(e))


def _read_copc_point_count(path: Path) -> Optional[int]:
    """Return the point count from a COPC header, or None when unavailable."""
    try:
        if not path.exists():
            return None
        import laspy
        with laspy.open(str(path)) as reader:
            return int(reader.header.point_count)
    except Exception:
        return None


def _finalize_tile_to_copc_pdal(
    parts: List[Path],
    final_tile: Path,
    log_dir: Path,
    label: str,
    tile_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[bool, str]:
    """Finalize a tile with the existing PDAL merge pipeline."""
    pdal_cmd = get_pdal_path()

    # Build COPC writer config with explicit offsets from tile bounds
    # to prevent int32 overflow on scaled coordinate values.
    writer_opts = {
        "type": "writers.copc",
        "filename": str(final_tile),
        "forward": "all",
        "extra_dims": "all",
    }
    if tile_bounds is not None:
        bxmin, bymin, bxmax, bymax = tile_bounds
        writer_opts["offset_x"] = (bxmin + bxmax) / 2.0
        writer_opts["offset_y"] = (bymin + bymax) / 2.0

    if len(parts) == 1:
        pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": str(parts[0])},
                writer_opts,
            ]
        }
    else:
        readers = [{"type": "readers.las", "filename": str(p)} for p in parts]
        pipeline = {
            "pipeline": readers + [
                {"type": "filters.merge"},
                writer_opts,
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


def _finalize_tile_to_copc_untwine(
    parts: List[Path],
    final_tile: Path,
    log_dir: Path,
    label: str,
) -> Tuple[bool, str]:
    """Finalize a tile using untwine for fast COPC conversion.

    Untwine is purpose-built for COPC generation and significantly faster
    than PDAL's writers.copc, especially for large point clouds.

    For multiple parts, passes all files directly to untwine (it supports
    multiple input files natively).
    """
    try:
        untwine_cmd = get_untwine_path(require=True)
    except RuntimeError as e:
        return (False, str(e))

    try:
        input_args = []
        for p in parts:
            input_args.extend(["-i", str(p)])

        result = subprocess.run(
            [untwine_cmd] + input_args + ["-o", str(final_tile)],
            capture_output=True, text=True, check=False,
        )

        if result.returncode != 0:
            return (False, f"untwine failed: {result.stderr[:200]}")

        if not final_tile.exists() or final_tile.stat().st_size == 0:
            return (False, "untwine produced no output")

        return (True, "untwine")

    except Exception as e:
        return (False, f"untwine error: {e}")


def _stream_source_into_las_parts(
    input_laz: Path,
    parts_dir: Path,
    chunk_size: int,
) -> Tuple[bool, List[Path], str]:
    """Split one source file into temporary LAS parts using chunked laspy reads."""
    import laspy

    laz_backend = _laspy_laz_backend()
    open_kwargs = {}
    if input_laz.suffix.lower() == ".laz" and laz_backend is not None:
        open_kwargs["laz_backend"] = laz_backend

    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths: List[Path] = []

    try:
        with laspy.open(str(input_laz), **open_kwargs) as reader:
            for chunk_idx, chunk in enumerate(reader.chunk_iterator(chunk_size)):
                if len(chunk) == 0:
                    continue
                part_path = parts_dir / f"part_{chunk_idx:05d}.las"
                part_header = _make_tile_header(reader.header)
                with laspy.open(str(part_path), mode="w", header=part_header) as writer:
                    writer.write_points(chunk)
                part_paths.append(part_path)
    except Exception as e:
        return (False, [], str(e))

    if not part_paths:
        return (False, [], "no points were written to temporary LAS parts")

    return (True, part_paths, f"{len(part_paths)} part(s)")


def _convert_laz_to_copc_direct(input_laz: Path, output_copc: Path) -> Tuple[bool, str]:
    """Convert a single LAZ/LAS file to COPC via a direct untwine call.

    Uses untwine so the tiling pipeline has one consistent COPC writer.
    """
    untwine_cmd = get_untwine_path(require=True)
    try:
        r = subprocess.run(
            [untwine_cmd, "-i", str(input_laz), "-o", str(output_copc)],
            capture_output=True, text=True, check=False,
        )
        success = r.returncode == 0 and output_copc.exists() and output_copc.stat().st_size > 0
        if success:
            return (True, "untwine direct")
        stderr = (r.stderr or r.stdout or "").strip()
        if r.returncode < 0:
            signal_msg = f"terminated by signal {-r.returncode}"
        elif r.returncode > 128:
            signal_msg = f"terminated by signal {r.returncode - 128}"
        else:
            signal_msg = f"rc={r.returncode}"
        detail = stderr[:200] if stderr else "unknown error"
        return (False, f"{signal_msg}: {detail}")
    except Exception as e:
        return (False, str(e))


def _convert_laz_to_copc_chunked(
    input_laz: Path,
    output_copc: Path,
    chunk_size: int,
) -> Tuple[bool, str]:
    """Convert a single LAZ/LAS file to COPC via temporary chunked LAS parts."""
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{output_copc.stem}_chunked_",
            dir=str(output_copc.parent),
        )
    )
    parts_dir = temp_dir / "parts"

    try:
        chunk_success, parts, chunk_message = _stream_source_into_las_parts(
            input_laz=input_laz,
            parts_dir=parts_dir,
            chunk_size=chunk_size,
        )
        if not chunk_success:
            return (False, f"chunk split failed: {chunk_message}")

        copc_success, copc_message = _finalize_tile_to_copc_untwine(
            parts=parts,
            final_tile=output_copc,
            log_dir=temp_dir,
            label=input_laz.stem,
        )
        if not copc_success:
            return (False, copc_message)

        return (True, f"chunked laspy -> untwine ({chunk_message})")
    except Exception as e:
        return (False, str(e))
    finally:
        if not (output_copc.exists() and output_copc.stat().st_size > 0):
            output_copc.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _convert_laz_to_copc(
    input_laz: Path,
    output_copc: Path,
    chunk_size: int,
    chunkwise_source_creation: bool = False,
) -> Tuple[bool, str]:
    """Convert a single LAZ/LAS file to COPC.

    When chunkwise_source_creation is enabled, the source is first streamed into
    temporary LAS parts before untwine builds the final COPC. This trades disk
    I/O for lower peak RAM during source normalization.
    """
    if chunkwise_source_creation:
        return _convert_laz_to_copc_chunked(
            input_laz=input_laz,
            output_copc=output_copc,
            chunk_size=chunk_size,
        )
    direct_success, direct_message = _convert_laz_to_copc_direct(input_laz, output_copc)
    if direct_success:
        return (True, direct_message)

    output_copc.unlink(missing_ok=True)
    chunked_success, chunked_message = _convert_laz_to_copc_chunked(
        input_laz=input_laz,
        output_copc=output_copc,
        chunk_size=chunk_size,
    )
    if chunked_success:
        return (
            True,
            f"chunked fallback after direct untwine failure ({direct_message})",
        )
    return (
        False,
        "direct untwine failed "
        f"({direct_message}); chunked fallback failed ({chunked_message})",
    )


def ensure_copc_sources(
    input_dir: Path,
    copc_dir: Path,
    max_workers: int = 4,
    chunk_size: int = 20_000_000,
    chunkwise_source_creation: bool = False,
) -> Tuple[Path, List[Path], List[Path]]:
    """
    Normalize source inputs to COPC, preserving all dimensions.

    Returns:
        Tuple of:
        - path to the original_copc directory
        - original non-COPC source files discovered in input_dir
        - COPC files available for downstream tiling/subsampling
    """
    original_sources = list_point_cloud_files(input_dir, include_copc=False)
    existing_copc_inputs = sorted(input_dir.glob("*.copc.laz"))

    if not original_sources and not existing_copc_inputs:
        raise ValueError(f"No point cloud files found in {input_dir}")

    copc_dir.mkdir(parents=True, exist_ok=True)

    if not original_sources:
        print()
        print("=" * 60)
        print("Step 1: Using existing COPC sources")
        print("=" * 60)
        for src in existing_copc_inputs:
            dest = copc_dir / src.name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            shutil.copy2(src, dest)
        copc_files = sorted(copc_dir.glob("*.copc.laz"))
        print(f"  Reused {len(copc_files)} COPC file(s)")
        return copc_dir, [], copc_files

    print()
    print("=" * 60)
    print("Step 1: Converting source files to COPC")
    print("=" * 60)
    print("  All source dimensions are preserved during COPC conversion.")
    if chunkwise_source_creation:
        print(
            "  COPC writer: chunked laspy staging -> untwine "
            f"({chunk_size:,} pts/chunk)"
        )
    else:
        print("  COPC writer: untwine")
    print(f"  Input files: {len(original_sources)}")
    print(f"  COPC directory: {copc_dir}")

    tasks: List[Tuple[Path, Path]] = []
    expected_outputs: List[Path] = []
    reused = 0
    for src in original_sources:
        out_copc = copc_dir / _copc_name_for_source(src)
        expected_outputs.append(out_copc)
        if out_copc.exists() and out_copc.stat().st_size > 0:
            reused += 1
            continue
        tasks.append((src, out_copc))

    if reused:
        print(f"  Reusing {reused} existing COPC file(s)")

    if tasks:
        worker_count = max(1, min(max_workers, len(tasks)))
        print(f"  Converting {len(tasks)} file(s) with {worker_count} worker(s)")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _convert_laz_to_copc,
                    src,
                    out,
                    chunk_size,
                    chunkwise_source_creation,
                ): (src, out)
                for src, out in tasks
            }
            for future in as_completed(futures):
                src, out = futures[future]
                success, message = future.result()
                if not success:
                    raise RuntimeError(f"LAZ/LAS→COPC conversion failed: {src} ({message})")
                print(f"    ✓ {src.name} -> {out.name} [{message}]")
    else:
        print("  All COPC sources already exist")

    missing_outputs = [path for path in expected_outputs if not path.exists() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(
            "Missing expected COPC file(s): "
            + ", ".join(path.name for path in missing_outputs[:5])
        )

    total_size_gb = sum(f.stat().st_size for f in expected_outputs) / (1024 ** 3)
    print(f"  ✓ COPC source preparation complete: {len(expected_outputs)} file(s), {total_size_gb:.2f} GB total")
    return copc_dir, original_sources, expected_outputs


def create_tiles(
    tindex_file: Path,
    tile_jobs_file: Path,
    tiles_dir: Path,
    log_dir: Path,
    threads: int = 5,
    max_parallel: int = 5,
    chunk_size: int = 20_000_000,
) -> List[Path]:
    """
    Create overlapping tiles from source point cloud files (two-phase).

    Phase 1 – Distribute: each source file is read exactly once (in chunks)
    and cropped points are written as per-tile LAZ part files.  This avoids
    the previous O(sources × tiles) read pattern.

    Phase 2 – Finalise: each tile's part files are merged and converted to
    COPC format with untwine. Fully parallelised across tiles.

    Args:
        tindex_file: Path to tindex GeoPackage
        tile_jobs_file: Path to tile jobs file
        tiles_dir: Output directory for tiles
        log_dir: Directory for log files
        threads: Threads used per process for LAZ chunk decompression (LazrsParallel/Rayon)
        max_parallel: Maximum parallel workers for each phase
        chunk_size: Points per chunk when reading source files (smaller = less peak RAM)

    Returns:
        List of created tile paths
    """
    print()
    print("=" * 60)
    print("Creating tiles (two-phase)")
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
                results, read_mode = future.result()
                for label, count in results:
                    tile_point_counts[label] = tile_point_counts.get(label, 0) + count
                if results:
                    total_pts = sum(c for _, c in results)
                    print(f"    ✓ {src_name}: {total_pts:,} pts → {len(results)} tile(s) [{read_mode}]")
                else:
                    print(f"    - {src_name}: no overlapping data [{read_mode}]")
            except Exception as e:
                print(f"    ✗ {src_name}: {e}")

    # ── Phase 2: Finalise ───────────────────────────────────────────────
    finalize_tasks = [
        (label, tiles_dir, log_dir, pending_tiles.get(label))
        for label in pending_tiles
    ]

    print()
    print(
        f"  Phase 2: Merging & converting {len(finalize_tasks)} tile(s) to COPC"
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
            final_tile = tiles_dir / f"{label}.copc.laz"
            pts = _read_copc_point_count(final_tile)
            if pts is None:
                pts = tile_point_counts.get(label, 0)
            if success:
                if "Already exists" in message or "No data" in message:
                    skipped += 1
                    if "No data" in message:
                        print(f"    - {label}: {message}")
                    else:
                        print(f"    - {label}: {message} ({pts:,} pts)")
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
    dimension_reduction: bool = True,
    tiling_threshold: float = None,
    chunk_size: int = 2_000_000,
    chunkwise_copc_source_creation: bool = False,
) -> Path:
    """
    Run the complete tiling pipeline.

    Steps:
    1. Convert source LAZ/LAS inputs to COPC, preserving all dimensions
    2. Build spatial index (tindex) from COPC source files
    3. Calculate tile bounds
    4. Create overlapping COPC tiles

    If input folder contains a single file below tiling_threshold, the already
    converted COPC source is returned directly for subsampling.

    Args:
        input_dir: Directory containing input LAZ/LAS files
        output_dir: Base output directory
        tile_length: Tile size in meters
        tile_buffer: Buffer overlap in meters
        grid_offset: Offset from min coordinates
        num_workers: Worker count used for source COPC conversion
        threads: Threads per PDAL writer
        max_tile_procs: Maximum parallel tile processes
        dimension_reduction:
            Kept for API compatibility with the caller. COPC conversion always
            preserves all dimensions; dimension reduction only applies later in
            the subsampling stage.
        tiling_threshold: File size threshold in MB. If single file below this, skip tiling
        chunk_size: Points per chunk when reading source data in Phase 1 (smaller = less peak RAM)

    Returns:
        Path to tiles directory (or copc_single directory if tiling was skipped)
    """
    print("=" * 60)
    print("3DTrees Tiling Pipeline (COPC-first)")
    print("=" * 60)
    untwine_cmd = get_untwine_path(require=True)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tile size: {tile_length}m with {tile_buffer}m buffer")
    print("Source normalization: COPC with all dimensions preserved")
    print(f"COPC writer: untwine ({untwine_cmd})")
    print(
        "Subsampling dimension policy: "
        f"{'standard dims only later' if dimension_reduction else 'keep all dims later'}"
    )
    print(
        "Source COPC conversion: "
        + (
            f"chunkwise staging enabled ({chunk_size:,} pts/chunk)"
            if chunkwise_copc_source_creation
            else "direct untwine"
        )
    )
    print()

    tiles_dir = output_dir / f"tiles_{int(tile_length)}m"
    source_copc_dir = output_dir / "original_copc"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tindex_file = output_dir / f"tindex_{int(tile_length)}m.gpkg"

    # Check if we should skip tiling (single small file)
    should_skip_tiling = False
    input_source_files = list_point_cloud_files(input_dir, include_copc=False)
    if not input_source_files:
        input_source_files = sorted(input_dir.glob("*.copc.laz"))
    if tiling_threshold is not None:
        if len(input_source_files) == 1:
            original_size_mb = input_source_files[0].stat().st_size / (1024 * 1024)
            if original_size_mb < tiling_threshold:
                should_skip_tiling = True
                print("=" * 60)
                print("Tiling Threshold Check")
                print("=" * 60)
                print(f"  Single file detected: {input_source_files[0].name}")
                print(f"  Original file size: {original_size_mb:.2f} MB")
                print(f"  Threshold: {tiling_threshold} MB")
                print("  Decision: Will skip tile generation after COPC normalization")
                print("=" * 60)
                print()

    # Step 1: Normalize input sources to COPC so downstream steps can use COPC-aware reads.
    source_copc_dir, original_sources, copc_sources = ensure_copc_sources(
        input_dir=input_dir,
        copc_dir=source_copc_dir,
        max_workers=num_workers,
        chunk_size=chunk_size,
        chunkwise_source_creation=chunkwise_copc_source_creation,
    )

    # Step 2: Build tindex from COPC source files
    tindex_file = build_tindex(source_copc_dir, tindex_file)

    # Step 3: Calculate tile bounds
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

    # Check if we should skip tiling.
    # Done AFTER tindex/bounds/plot so those outputs are always available for merge.
    if should_skip_tiling:
        print()
        print("=" * 60)
        print("Skipping Tiling (Single Small File)")
        print("=" * 60)
        copc_single_dir = output_dir / "copc_single"
        copc_single_dir.mkdir(parents=True, exist_ok=True)
        source_copc = copc_sources[0]
        out_copc = copc_single_dir / source_copc.name
        if not out_copc.exists() or out_copc.stat().st_size == 0:
            print("  Reusing normalized COPC source...")
            shutil.copy2(source_copc, out_copc)
            print(f"  ✓ Prepared {out_copc.name}")
        else:
            print(f"  Using existing {out_copc.name}")

        rewritten_extent = rewrite_tile_bounds_json_for_single_file_skip(bounds_json, out_copc)
        print(
            "  Rewrote tile_bounds_tindex.json for single-file skip "
            f"to bounds x=[{rewritten_extent['minx']:.3f}, {rewritten_extent['maxx']:.3f}] "
            f"y=[{rewritten_extent['miny']:.3f}, {rewritten_extent['maxy']:.3f}]"
        )
        plot_tiles_and_copc.plot_extents(
            tindex_file, bounds_json, output_dir / "overview_copc_tiles.png"
        )
        print("  Regenerated overview_copc_tiles.png with corrected single-tile bounds")
        print(f"  Returning COPC directory for direct subsampling")
        print("=" * 60)
        return copc_single_dir

    # Step 4: Create tiles from the COPC source tindex
    tile_files = create_tiles(
        tindex_file,
        jobs_file,
        tiles_dir,
        log_dir,
        threads,
        max_tile_procs,
        chunk_size,
    )

    print()
    print("=" * 60)
    print("Tiling Pipeline Complete")
    print("=" * 60)
    print(f"  Original source files: {len(original_sources) if original_sources else len(copc_sources)}")
    print(f"  COPC source files: {len(copc_sources)}")
    print(f"  Tiles created: {len(tile_files)}")
    print(f"  Tiles directory: {tiles_dir}")

    return tiles_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Tiling Pipeline - COPC-first tiling from LAZ/LAS input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        required=True,
        help="Input directory containing LAZ/LAS files"
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
        )
        print(f"\nTiles ready for subsampling: {tiles_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
