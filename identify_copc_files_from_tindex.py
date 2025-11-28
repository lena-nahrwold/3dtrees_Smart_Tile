#!/usr/bin/env python3
"""
Identify which COPC files from the tindex intersect each tile's bounds.
Uses spatial queries on the tindex shapefile for efficient lookup.
"""

import json
import sys
from pathlib import Path
import argparse
from typing import List, Dict

try:
    import fiona
    from shapely.geometry import box, shape
    from pyproj import Transformer
except ImportError as e:
    print(f"ERROR: Required packages missing. Install with: pip install fiona shapely pyproj")
    print(f"Error: {e}")
    sys.exit(1)


def find_intersecting_files(
    tindex_path: Path,
    tile_bounds: List[List[float]],
    proj_crs: str,
    location_field: str = "Location",
) -> List[str]:
    """Find COPC files that intersect with tile bounds using spatial query.
    
    Args:
        tindex_path: Path to tindex shapefile
        tile_bounds: [[xmin, xmax], [ymin, ymax]] in projected CRS
        proj_crs: Projected CRS of tile bounds (e.g., 'EPSG:32632')
        location_field: Field name containing file path
    
    Returns:
        List of COPC file paths
    """
    with fiona.open(tindex_path) as src:
        # Get tindex CRS
        tindex_crs_dict = src.crs if src.crs else {}
        if isinstance(tindex_crs_dict, dict):
            tindex_crs = tindex_crs_dict.get('init') or str(tindex_crs_dict)
        else:
            tindex_crs = str(tindex_crs_dict)
        
        # Default to EPSG:4326 if not properly detected
        if not tindex_crs or tindex_crs == '{}' or tindex_crs == '':
            tindex_crs = 'EPSG:4326'
        
        # Transform projected bounds to tindex CRS (usually geographic)
        transformer = None
        if proj_crs != tindex_crs:
            try:
                transformer = Transformer.from_crs(proj_crs, tindex_crs, always_xy=True)
            except Exception as e:
                print(f"WARNING: Could not create transformer: {e}", file=sys.stderr)
        
        # Transform bounds
        proj_xmin, proj_xmax = tile_bounds[0]
        proj_ymin, proj_ymax = tile_bounds[1]
        
        if transformer:
            # Transform all four corners
            corners = [
                transformer.transform(proj_xmin, proj_ymin),
                transformer.transform(proj_xmin, proj_ymax),
                transformer.transform(proj_xmax, proj_ymin),
                transformer.transform(proj_xmax, proj_ymax),
            ]
            xs, ys = zip(*corners)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            # No transformation needed
            xmin, xmax = proj_xmin, proj_xmax
            ymin, ymax = proj_ymin, proj_ymax
        
        tile_bbox = box(xmin, ymin, xmax, ymax)
        
        intersecting = []
        
        # Filter features that intersect with tile bounds (in tindex CRS)
        for feature in src.filter(bbox=(xmin, ymin, xmax, ymax)):
            geom = shape(feature['geometry'])
            
            # Check if geometries actually intersect
            if tile_bbox.intersects(geom):
                # Get file path from feature properties
                file_path = feature['properties'].get(location_field)
                if file_path:
                    intersecting.append(file_path)
    
    return intersecting


def main():
    parser = argparse.ArgumentParser(
        description="Identify COPC files that intersect each tile using tindex spatial queries"
    )
    parser.add_argument("tindex_path", type=Path, help="Path to tindex shapefile")
    parser.add_argument("tile_bounds_json", type=Path, help="Path to tile_bounds JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tile_copc_mapping.json"),
        help="Output JSON mapping tiles to COPC files",
    )
    parser.add_argument(
        "--location-field",
        type=str,
        default="Location",
        help="Field name in tindex containing file path (default: Location)",
    )
    args = parser.parse_args()

    # Load tile bounds
    with args.tile_bounds_json.open() as f:
        tiles_data = json.load(f)

    # Get projected CRS from tile bounds data
    proj_crs = tiles_data.get("proj_srs", "EPSG:32632")

    # For each tile, find intersecting COPC files using spatial query
    mapping = {}
    total_tiles = len(tiles_data["tiles"])
    
    print(f"Processing {total_tiles} tiles...")
    
    for idx, tile in enumerate(tiles_data["tiles"], 1):
        label = f"c{tile['col']:02d}_r{tile['row']:02d}"
        bounds = tile["bounds"]

        intersecting_files = find_intersecting_files(
            args.tindex_path, bounds, proj_crs, args.location_field
        )
        
        mapping[label] = {
            "bounds": bounds,
            "copc_files": intersecting_files,
            "file_count": len(intersecting_files),
        }
        
        if idx % 10 == 0 or idx == total_tiles:
            print(f"  Processed {idx}/{total_tiles} tiles...", end='\r')
    
    print(f"\nCompleted processing {total_tiles} tiles")
    
    # Save mapping
    with args.output.open("w") as f:
        json.dump(mapping, f, indent=2)

    # Print summary statistics
    total_intersections = sum(m["file_count"] for m in mapping.values())
    avg_files_per_tile = total_intersections / len(mapping) if mapping else 0
    
    print(f"\nSummary:")
    print(f"  Total tiles: {len(mapping)}")
    print(f"  Average COPC files per tile: {avg_files_per_tile:.1f}")
    print(f"  Mapping saved to: {args.output}")


if __name__ == "__main__":
    main()

