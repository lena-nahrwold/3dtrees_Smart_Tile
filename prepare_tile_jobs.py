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


def run_get_bounds(tindex_path: Path, tile_length: float, tile_buffer: float) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("get_bounds_from_tindex.py")),
        str(tindex_path),
        f"--tile-length={tile_length}",
        f"--tile-buffer={tile_buffer}",
        f"--out={DEFAULT_BOUNDS_JSON}",
    ]
    print(f"[prepare_tile_jobs] running: {' '.join(cmd)}", file=sys.stderr)
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
    srs = data.get("srs", "EPSG:32632")
    transformer = Transformer.from_crs(srs, "EPSG:4326", always_xy=True)

    def to_geo_bounds(bx: List[float], by: List[float]) -> Tuple[List[float], List[float]]:
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
    args = parser.parse_args()

    env = run_get_bounds(args.tindex_path, args.tile_length, args.tile_buffer)
    bounds_json = Path(env.get("tile_bounds_file", DEFAULT_BOUNDS_JSON))
    write_job_list(bounds_json, args.jobs_out)

    print(f"tile_jobs_file={args.jobs_out}")
    print(f"tile_bounds_file={bounds_json}")
    print(f"tile_count={env.get('tile_count')}")
    print(f"crop_bounds=\"{env.get('crop_bounds')}\"")
    print(f"reader_bounds=\"{env.get('reader_bounds')}\"")


if __name__ == "__main__":  # pragma: no cover
    main()

