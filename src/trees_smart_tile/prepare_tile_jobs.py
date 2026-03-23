#!/usr/bin/env python3
"""
Load tindex, compute tile/grid metadata, and emit both the JSON summary
and a per-tile job list that can be consumed by retiling.sh.

This wraps get_bounds_from_tindex.py so the bash script can stay cleaner.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import argparse
from typing import List, Tuple

try:
    from pyproj import Transformer
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "pyproj is required. Install it in your environment (e.g. pip install pyproj).\n"
    )
    raise


DEFAULT_BOUNDS_JSON = Path("/home/kg281/data/output/pdal_experiments/tile_bounds_tindex.json")


def run_get_bounds(tindex_path: Path, tile_length: float, tile_buffer: float, bounds_json_path: Path, grid_offset: float = 0.0) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("get_bounds_from_tindex.py")),
        str(tindex_path),
        f"--tile-length={tile_length}",
        f"--tile-buffer={tile_buffer}",
        f"--out={bounds_json_path}",
    ]
    if grid_offset != 0.0:
        cmd.append(f"--grid-offset={grid_offset}")
    print(f"[prepare_tile_jobs] running: {' '.join(cmd)}", file=sys.stderr)
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if completed.returncode != 0:
        # Print the full error message
        if completed.stderr:
            print(f"[prepare_tile_jobs] Error output:", file=sys.stderr)
            print(completed.stderr, file=sys.stderr)
        if completed.stdout:
            print(f"[prepare_tile_jobs] Standard output:", file=sys.stderr)
            print(completed.stdout, file=sys.stderr)
        raise RuntimeError(
            f"get_bounds_from_tindex.py failed with exit code {completed.returncode}.\n"
            f"stderr: {completed.stderr}\n"
            f"stdout: {completed.stdout}"
        )
    
    env = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"')
    return env


def write_job_list(bounds_json: Path, job_file: Path) -> None:
    with bounds_json.open() as f:
        data = json.load(f)

    # Get SRS from tile bounds data
    # get_bounds_from_tindex.py uses 'proj_srs' for the working projection
    srs = data.get("proj_srs", data.get("tindex_srs", "missing"))
    
    transformer = None
    if srs != "missing":
        try:
            transformer = Transformer.from_crs(srs, "EPSG:4326", always_xy=True)
        except Exception as e:
            print(f"[prepare_tile_jobs] Warning: Could not create transformer from {srs} to EPSG:4326: {e}", file=sys.stderr)

    def to_geo_bounds(bx: List[float], by: List[float]) -> Tuple[List[float], List[float]]:
        if transformer is None:
            # Fallback: if no transformer, return original bounds (planar units)
            return bx, by
            
        corners = [
            transformer.transform(bx[0], by[0]),
            transformer.transform(bx[0], by[1]),
            transformer.transform(bx[1], by[0]),
            transformer.transform(bx[1], by[1]),
        ]
        lons, lats = zip(*corners)
        return [min(lons), max(lons)], [min(lats), max(lats)]

    lines: List[str] = []
    for tile in data["tiles"]:
        label = f"c{tile['col']:02d}_r{tile['row']:02d}"
        bx, by = tile["bounds"]
        proj_bounds = f"([{bx[0]},{bx[1]}],[{by[0]},{by[1]}])"
        geo_x, geo_y = to_geo_bounds(bx, by)
        geo_bounds = f"([{geo_x[0]},{geo_x[1]}],[{geo_y[0]},{geo_y[1]}])"
        lines.append(f"{label}|{proj_bounds}|{geo_bounds}")

    # Ensure the file ends with a newline so shell read loops don't drop the last tile.
    job_file.write_text("\n".join(lines) + "\n")
    print(f"[prepare_tile_jobs] wrote {len(lines)} jobs to {job_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Prepare tile job list and bounds env from tindex.")
    parser.add_argument("tindex_path", type=Path, help="Path to the tindex shapefile")
    parser.add_argument("--tile-length", type=float, default=40.0)
    parser.add_argument("--tile-buffer", type=float, default=5.0)
    parser.add_argument(
        "--jobs-out",
        type=Path,
        default=Path("/home/kg281/data/output/pdal_experiments/tile_jobs.txt"),
    )
    parser.add_argument(
        "--bounds-out",
        type=Path,
        default=DEFAULT_BOUNDS_JSON,
        help="Path to write the tile bounds JSON file",
    )
    parser.add_argument(
        "--grid-offset",
        type=float,
        default=0.0,
        help="Offset in meters to add to minx and miny before starting grid (default: 0.0)",
    )
    args = parser.parse_args()

    env = run_get_bounds(args.tindex_path, args.tile_length, args.tile_buffer, args.bounds_out, args.grid_offset)
    bounds_json = args.bounds_out
    write_job_list(bounds_json, args.jobs_out)

    print(f"tile_jobs_file={args.jobs_out}")
    print(f"tile_bounds_file={bounds_json}")
    print(f"tile_count={env.get('tile_count')}")
    print(f"crop_bounds=\"{env.get('crop_bounds')}\"")
    print(f"reader_bounds=\"{env.get('reader_bounds')}\"")


if __name__ == "__main__":  # pragma: no cover
    main()

