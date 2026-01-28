#!/usr/bin/env python3
"""
Get extent from tindex shapefile and compute tile bounds.
Always treats coordinates as planar/metric units.
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import fiona
    from pyproj import CRS
except ImportError as e:
    print(f"ERROR: Required package missing. Install with: pip install fiona pyproj")
    print(f"Error: {e}")
    exit(1)


def load_extent_from_tindex(tindex_path: Path):
    """Load extent from tindex shapefile.
    
    Returns:
        Tuple of (minx, miny, maxx, maxy), crs_string
        
    The coordinates are returned in the native units of the data (assumed metric).
    """
    with fiona.open(tindex_path) as src:
        # Get CRS from file
        srs_info = "missing"
        if src.crs:
            try:
                crs = CRS.from_user_input(src.crs)
                srs_info = crs.to_string()
            except Exception:
                srs_info = str(src.crs)
        
        print(f"  Detected CRS: {srs_info} (Treating as Planar/Metric)", file=sys.stderr)
        
        if srs_info == "missing":
             print(f"  ⚠ Warning: CRS missing; tiling in dataset-local planar coordinates.", file=sys.stderr)
        
        # Get bounds of all features
        minx = miny = math.inf
        maxx = maxy = -math.inf
        
        feature_count = 0
        for feature in src:
            feature_count += 1
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
            elif geom['type'] == 'MultiPolygon':
                coords = [c for poly in geom['coordinates'] for c in poly[0]]
            else:
                continue
            
            xs, ys = zip(*coords)
            minx = min(minx, min(xs))
            miny = min(miny, min(ys))
            maxx = max(maxx, max(xs))
            maxy = max(maxy, max(ys))
        
        if feature_count == 0:
            bounds = src.bounds
            if bounds and bounds != (0.0, 0.0, 0.0, 0.0):
                minx, miny, maxx, maxy = bounds
            else:
                raise ValueError(f"No features found in tindex: {tindex_path}")

        return (minx, miny, maxx, maxy), srs_info


def build_tiles(minx, miny, maxx, maxy, length, buffer, align_to_grid=False, grid_offset=0.0):
    """Build tile grid.
    
    Args:
        minx, miny, maxx, maxy: Data extent bounds
        length: Tile size in units
        buffer: Buffer size in units
        align_to_grid: If True, snap to grid.
        grid_offset: Offset in units
    """
    # Validate inputs - check for infinity or NaN
    if not all(math.isfinite(v) for v in [minx, miny, maxx, maxy]):
        raise ValueError(
            f"Invalid bounds detected (infinity or NaN): "
            f"minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}."
        )
    
    if align_to_grid:
        start_x = math.floor((minx + grid_offset) / length) * length
        start_y = math.floor((miny + grid_offset) / length) * length
        end_x = math.ceil(maxx / length) * length
        end_y = math.ceil(maxy / length) * length
    else:
        start_x = minx + grid_offset
        start_y = miny + grid_offset
        x_range = maxx - minx
        y_range = maxy - miny
        if not math.isfinite(x_range) or not math.isfinite(y_range):
            raise ValueError(
                f"Invalid range calculated: x_range={x_range}, y_range={y_range}. "
                f"Bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}"
            )
        end_x = math.ceil(x_range / length) * length + start_x
        end_y = math.ceil(y_range / length) * length + start_y
    
    # Estimate number of tiles and warn if excessive
    num_tiles_x = int(math.ceil((end_x - start_x) / length))
    num_tiles_y = int(math.ceil((end_y - start_y) / length))
    total_tiles = num_tiles_x * num_tiles_y
    
    # Warn if creating too many tiles (more than 1 million)
    MAX_TILES = 1000000
    if total_tiles > MAX_TILES:
        raise ValueError(
            f"Would create {total_tiles:,} tiles ({num_tiles_x} x {num_tiles_y}), "
            f"which exceeds the maximum of {MAX_TILES:,}. "
            f"This usually indicates the data extent is too large or the tile size is too small. "
            f"Bounds: minx={minx:.2f}, miny={miny:.2f}, maxx={maxx:.2f}, maxy={maxy:.2f}, "
            f"tile_length={length}. "
            f"Consider using a larger tile size or splitting the data into smaller regions."
        )

    tiles = []
    col = 0
    x = start_x
    while x < end_x:
        row = 0
        y = start_y
        while y < end_y:
            core_x = [x, x + length]
            core_y = [y, y + length]
            buffered_x = [core_x[0] - buffer, core_x[1] + buffer]
            buffered_y = [core_y[0] - buffer, core_y[1] + buffer]
            tiles.append(
                {
                    "col": col,
                    "row": row,
                    "core": [core_x, core_y],
                    "bounds": [buffered_x, buffered_y],
                }
            )
            row += 1
            y += length
        col += 1
        x += length

    grid_bounds = (
        start_x - buffer,
        end_x + buffer,
        start_y - buffer,
        end_y + buffer,
    )
    tiles.sort(key=lambda t: (t["col"], t["row"]))
    return tiles, grid_bounds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute tile bounds for the tindex extent. "
            "Assumes everything is in planar/metric units."
        )
    )
    parser.add_argument("tindex_path", type=Path, help="Path to the tindex shapefile")
    parser.add_argument(
        "--tile-length", type=float, default=40.0, help="Tile core length (default: 40 units)"
    )
    parser.add_argument(
        "--tile-buffer", type=float, default=5.0, help="Tile buffer distance (default: 5 units)"
    )
    parser.add_argument(
        "--proj-crs",
        type=str,
        default="EPSG:32630",
        help="Projected CRS fallback (default: EPSG:32630)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/kg281/data/output/pdal_experiments/tile_bounds_tindex.json"),
        help="Where to write the tile bounds JSON summary",
    )
    parser.add_argument(
        "--grid-offset",
        type=float,
        default=0.0,
        help="Offset in units to add before starting grid (default: 0.0)",
    )
    args = parser.parse_args()

    (minx, miny, maxx, maxy), srs = load_extent_from_tindex(args.tindex_path)
    
    # Always treat as planar/metric
    proj_minx, proj_miny, proj_maxx, proj_maxy = minx, miny, maxx, maxy
    proj_crs = srs if srs != "missing" else args.proj_crs
    
    # geo_extent matches proj_extent because we assume metric
    geo_minx, geo_miny, geo_maxx, geo_maxy = minx, miny, maxx, maxy
    
    # Use data-aligned tiling
    tiles, grid_bounds = build_tiles(
        proj_minx, proj_miny, proj_maxx, proj_maxy, 
        args.tile_length, args.tile_buffer, 
        align_to_grid=False,
        grid_offset=args.grid_offset
    )

    summary = {
        "tindex": str(args.tindex_path),
        "tindex_srs": srs,
        "proj_srs": proj_crs,
        "proj_extent": {"minx": proj_minx, "miny": proj_miny, "maxx": proj_maxx, "maxy": proj_maxy},
        "geo_extent": {"minx": geo_minx, "miny": geo_miny, "maxx": geo_maxx, "maxy": geo_maxy},
        "tile_length": args.tile_length,
        "tile_buffer": args.tile_buffer,
        "grid_bounds": {
            "xmin": grid_bounds[0],
            "xmax": grid_bounds[1],
            "ymin": grid_bounds[2],
            "ymax": grid_bounds[3],
        },
        "tiles": tiles,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)

    crop_bounds = f"([{grid_bounds[0]},{grid_bounds[1]}],[{grid_bounds[2]},{grid_bounds[3]}])"
    reader_bounds = f"([{geo_minx},{geo_maxx}],[{geo_miny},{geo_maxy}])"

    print(f"tile_bounds_file={args.out}")
    print(f"tile_count={len(tiles)}")
    print(f"crop_bounds=\"{crop_bounds}\"")
    print(f"reader_bounds=\"{reader_bounds}\"")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
