#!/usr/bin/env python3
"""
Get extent from tindex shapefile and compute tile bounds.
Similar to get_bounds.py but uses tindex instead of VPC.
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import fiona
    from pyproj import Transformer
except ImportError as e:
    print(f"ERROR: Required package missing. Install with: pip install fiona pyproj")
    print(f"Error: {e}")
    exit(1)


def load_extent_from_tindex(tindex_path: Path):
    """Load extent from tindex shapefile."""
    with fiona.open(tindex_path) as src:
        # Get CRS
        src_crs_dict = src.crs if src.crs else {}
        
        # Handle different CRS formats
        if isinstance(src_crs_dict, dict):
            src_crs = src_crs_dict.get('init') or src_crs_dict.get('proj') or str(src_crs_dict)
        else:
            src_crs = str(src_crs_dict)
        
        # Default to EPSG:32632 if CRS not properly detected
        if not src_crs or src_crs == '{}' or src_crs == '':
            src_crs = 'EPSG:32632'
        elif ':' in src_crs:
            # Extract EPSG code if present
            src_crs = src_crs.split(':')[-1] if ':' in src_crs else src_crs
            if src_crs.isdigit():
                src_crs = f'EPSG:{src_crs}'
        
        # Get bounds of all features
        proj_minx = proj_miny = math.inf
        proj_maxx = proj_maxy = -math.inf
        
        # Transform to geographic if needed
        transformer = None
        if src_crs and src_crs != 'EPSG:4326':
            try:
                transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            except Exception as e:
                print(f"WARNING: Could not create transformer from {src_crs}: {e}", file=sys.stderr)
                transformer = None
        
        geo_minx = geo_miny = math.inf
        geo_maxx = geo_maxy = -math.inf
        
        feature_count = 0
        for feature in src:
            feature_count += 1
            # Get bounds from geometry
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
            elif geom['type'] == 'MultiPolygon':
                coords = [c for poly in geom['coordinates'] for c in poly[0]]
            else:
                continue
            
            xs, ys = zip(*coords)
            fx_min, fx_max = min(xs), max(xs)
            fy_min, fy_max = min(ys), max(ys)
            
            proj_minx = min(proj_minx, fx_min)
            proj_miny = min(proj_miny, fy_min)
            proj_maxx = max(proj_maxx, fx_max)
            proj_maxy = max(proj_maxy, fy_max)
            
            # Transform to geographic
            if transformer:
                corners = [
                    transformer.transform(fx_min, fy_min),
                    transformer.transform(fx_min, fy_max),
                    transformer.transform(fx_max, fy_min),
                    transformer.transform(fx_max, fy_max),
                ]
                lons, lats = zip(*corners)
                geo_minx = min(geo_minx, min(lons))
                geo_miny = min(geo_miny, min(lats))
                geo_maxx = max(geo_maxx, max(lons))
                geo_maxy = max(geo_maxy, max(lats))
            else:
                # Already in geographic, use as-is
                geo_minx = min(geo_minx, fx_min)
                geo_miny = min(geo_miny, fy_min)
                geo_maxx = max(geo_maxx, fx_max)
                geo_maxy = max(geo_maxy, fy_max)
        
        # Check if we found any features
        if feature_count == 0:
            # Try to use bounds from source metadata
            bounds = src.bounds
            if bounds and bounds != (0.0, 0.0, 0.0, 0.0):
                proj_minx, proj_miny, proj_maxx, proj_maxy = bounds
                if transformer:
                    corners = [
                        transformer.transform(proj_minx, proj_miny),
                        transformer.transform(proj_maxx, proj_maxy),
                    ]
                    geo_minx, geo_miny = corners[0]
                    geo_maxx, geo_maxy = corners[1]
                else:
                    geo_minx, geo_miny = proj_minx, proj_miny
                    geo_maxx, geo_maxy = proj_maxx, proj_maxy
            else:
                raise ValueError(f"No features found in tindex file: {tindex_path}")
        
        # Return geographic bounds and CRS (we'll transform to projected in main)
        return (geo_minx, geo_miny, geo_maxx, geo_maxy), src_crs


def build_tiles(minx, miny, maxx, maxy, length, buffer, align_to_grid=False):
    """Build tile grid.
    
    Args:
        minx, miny, maxx, maxy: Data extent bounds
        length: Tile size in meters
        buffer: Buffer size in meters
        align_to_grid: If True, snap to grid (floor to nearest tile_length multiple).
                       If False, start from actual data extent (more efficient coverage).
    """
    if align_to_grid:
        # Grid-aligned: snap to multiples of tile_length (ensures consistent grid across datasets)
        start_x = math.floor(minx / length) * length
        start_y = math.floor(miny / length) * length
        end_x = math.ceil(maxx / length) * length
        end_y = math.ceil(maxy / length) * length
    else:
        # Data-aligned: start from actual data extent (minimizes tiles, better coverage)
        start_x = minx
        start_y = miny
        end_x = math.ceil((maxx - minx) / length) * length + start_x
        end_y = math.ceil((maxy - miny) / length) * length + start_y

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
            "Compute tile bounds for the tindex extent and emit helper values "
            "for PDAL pipelines."
        )
    )
    parser.add_argument("tindex_path", type=Path, help="Path to the tindex shapefile")
    parser.add_argument(
        "--tile-length", type=float, default=40.0, help="Tile core length (default: 40 m)"
    )
    parser.add_argument(
        "--tile-buffer", type=float, default=5.0, help="Tile buffer distance (default: 5 m)"
    )
    parser.add_argument(
        "--proj-crs",
        type=str,
        default="EPSG:32632",
        help="Projected CRS for tiling (default: EPSG:32632)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/kg281/data/output/pdal_experiments/tile_bounds_tindex.json"),
        help="Where to write the tile bounds JSON summary",
    )
    args = parser.parse_args()

    (geo_minx, geo_miny, geo_maxx, geo_maxy), srs = load_extent_from_tindex(args.tindex_path)
    
    # Transform geographic bounds to projected CRS for tiling
    proj_transformer = Transformer.from_crs("EPSG:4326", args.proj_crs, always_xy=True)
    
    # Transform all four corners to get projected bounds
    corners_proj = [
        proj_transformer.transform(geo_minx, geo_miny),
        proj_transformer.transform(geo_minx, geo_maxy),
        proj_transformer.transform(geo_maxx, geo_miny),
        proj_transformer.transform(geo_maxx, geo_maxy),
    ]
    proj_xs, proj_ys = zip(*corners_proj)
    proj_minx, proj_maxx = min(proj_xs), max(proj_xs)
    proj_miny, proj_maxy = min(proj_ys), max(proj_ys)
    
    # Use data-aligned tiling (starts from actual data extent, more efficient)
    tiles, grid_bounds = build_tiles(
        proj_minx, proj_miny, proj_maxx, proj_maxy, 
        args.tile_length, args.tile_buffer, 
        align_to_grid=False
    )

    summary = {
        "tindex": str(args.tindex_path),
        "tindex_srs": srs,
        "proj_srs": args.proj_crs,
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
    main()

