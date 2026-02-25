#!/usr/bin/env python3
"""
Visualize COPC file extents and generated tile extents.
Shows input COPC files and output tiles on the same plot.
"""

import json
import sys
from pathlib import Path
import argparse

try:
    import fiona
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    from pyproj import Transformer
except ImportError as e:
    print(f"ERROR: Required packages missing. Install with: pip install fiona matplotlib pyproj")
    print(f"Error: {e}")
    sys.exit(1)


def load_copc_extents(tindex_path: Path, target_crs: str = "EPSG:32632"):
    """Load COPC file extents from tindex and transform to target CRS."""
    extents = []
    filenames = []
    
    with fiona.open(tindex_path) as src:
        # Get source CRS
        src_crs = str(src.crs) if src.crs else "EPSG:32632"
        
        for feature in src:
            geom = feature['geometry']
            
            # Get bounds from geometry
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
            elif geom['type'] == 'MultiPolygon':
                coords = [c for poly in geom['coordinates'] for c in poly[0]]
            else:
                continue
            
            xs, ys = zip(*coords)
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            
            # Detect if coordinates are already projected (values > 360 are clearly not lat/lon)
            is_projected = abs(xmin) > 360 or abs(xmax) > 360 or abs(ymin) > 360 or abs(ymax) > 360
            
            if is_projected and "4326" in src_crs:
                # CRS is misreported as WGS84, but coordinates are already projected
                print(f"Note: CRS reported as {src_crs} but coordinates appear projected (assuming {target_crs})")
                # Use coordinates as-is
            elif not is_projected and src_crs != target_crs:
                # Actually need to transform from geographic to projected
                try:
                    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
                    print(f"Transforming COPC extents from {src_crs} to {target_crs}")
                    corners = [
                        transformer.transform(xmin, ymin),
                        transformer.transform(xmin, ymax),
                        transformer.transform(xmax, ymin),
                        transformer.transform(xmax, ymax),
                    ]
                    proj_xs, proj_ys = zip(*corners)
                    xmin, xmax = min(proj_xs), max(proj_xs)
                    ymin, ymax = min(proj_ys), max(proj_ys)
                except Exception as e:
                    print(f"WARNING: Could not transform coordinates: {e}", file=sys.stderr)
            
            extents.append((xmin, ymin, xmax, ymax))
            
            # Get filename from properties
            file_path = feature['properties'].get('Location', '')
            if file_path:
                filename = Path(file_path).name
            else:
                filename = f"file_{len(extents)}"
            filenames.append(filename)
    
    return extents, filenames


def load_tile_extents(tile_bounds_json: Path):
    """Load tile extents from tile_bounds JSON."""
    with tile_bounds_json.open() as f:
        data = json.load(f)
    
    tiles = []
    for tile in data['tiles']:
        bounds = tile['bounds']
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        
        tiles.append({
            'label': f"c{tile['col']:02d}_r{tile['row']:02d}",
            'bounds': (xmin, ymin, xmax, ymax),
            'core': tile.get('core', None)
        })
    
    return tiles, data.get('proj_srs', 'EPSG:32632')


def plot_extents(tindex_path: Path, tile_bounds_json: Path, output_png: Path):
    """Create visualization of COPC files and tiles."""
    print("Loading tile extents...")
    tiles, proj_srs = load_tile_extents(tile_bounds_json)
    print(f"Found {len(tiles)} tiles")
    print(f"Tiles are in CRS: {proj_srs}")
    
    print("Loading COPC file extents from tindex...")
    copc_extents, copc_names = load_copc_extents(tindex_path, target_crs=proj_srs)
    print(f"Found {len(copc_extents)} COPC files")
    
    # Calculate overall extent
    all_xs = []
    all_ys = []
    
    for xmin, ymin, xmax, ymax in copc_extents:
        all_xs.extend([xmin, xmax])
        all_ys.extend([ymin, ymax])
    
    for tile in tiles:
        xmin, ymin, xmax, ymax = tile['bounds']
        all_xs.extend([xmin, xmax])
        all_ys.extend([ymin, ymax])
    
    overall_xmin, overall_xmax = min(all_xs), max(all_xs)
    overall_ymin, overall_ymax = min(all_ys), max(all_ys)
    
    # Add padding
    x_padding = (overall_xmax - overall_xmin) * 0.05
    y_padding = (overall_ymax - overall_ymin) * 0.05
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot COPC file extents
    copc_patches = []
    for xmin, ymin, xmax, ymax in copc_extents:
        width = xmax - xmin
        height = ymax - ymin
        rect = mpatches.Rectangle((xmin, ymin), width, height, 
                                  edgecolor='blue', facecolor='lightblue', 
                                  alpha=0.5, linewidth=1.5)
        copc_patches.append(rect)
    
    copc_collection = PatchCollection(copc_patches, match_original=True)
    ax.add_collection(copc_collection)
    
    # Add COPC file labels
    for (xmin, ymin, xmax, ymax), name in zip(copc_extents, copc_names):
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        ax.text(center_x, center_y, name, 
                ha='center', va='center', 
                fontsize=8, color='darkblue', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot tile extents (full bounds including buffer)
    tile_patches = []
    for tile in tiles:
        xmin, ymin, xmax, ymax = tile['bounds']
        width = xmax - xmin
        height = ymax - ymin
        rect = mpatches.Rectangle((xmin, ymin), width, height,
                                  edgecolor='indianred', facecolor='mistyrose',
                                  alpha=0.4, linewidth=2)
        tile_patches.append(rect)
        
        # Add tile label at center
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        ax.text(center_x, center_y, tile['label'],
                ha='center', va='center',
                fontsize=10, color='darkred', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    tile_collection = PatchCollection(tile_patches, match_original=True)
    ax.add_collection(tile_collection)
    
    # Set limits and labels
    ax.set_xlim(overall_xmin - x_padding, overall_xmax + x_padding)
    ax.set_ylim(overall_ymin - y_padding, overall_ymax + y_padding)
    ax.set_aspect('equal')
    ax.set_xlabel(f'X (Projected CRS: {proj_srs})', fontsize=12)
    ax.set_ylabel(f'Y (Projected CRS: {proj_srs})', fontsize=12)
    ax.set_title('COPC Files and Generated Tiles\n(Blue = COPC files, Light red = Tiles)', 
                 fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    copc_legend = mpatches.Patch(color='lightblue', alpha=0.5, label='COPC file extent')
    tile_legend = mpatches.Patch(facecolor='mistyrose', edgecolor='indianred', 
                                 alpha=0.4, linewidth=2, label='Tile extent')
    ax.legend(handles=[copc_legend, tile_legend], loc='upper right', fontsize=10)
    
    # Add statistics text box
    stats_text = f'COPC Files: {len(copc_extents)}\nTiles: {len(tiles)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_png}")
    print(f"  - {len(copc_extents)} COPC files (blue)")
    print(f"  - {len(tiles)} tiles (light red)")
    
    # Optionally show plot
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize COPC file extents and generated tile extents"
    )
    parser.add_argument(
        "tindex_path",
        type=Path,
        help="Path to tindex file (shapefile or GeoPackage)"
    )
    parser.add_argument(
        "tile_bounds_json",
        type=Path,
        help="Path to tile_bounds_tindex.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tiles_and_copc_visualization.png"),
        help="Output PNG file path"
    )
    args = parser.parse_args()
    
    if not args.tindex_path.exists():
        print(f"ERROR: Tindex file not found: {args.tindex_path}")
        sys.exit(1)
    
    if not args.tile_bounds_json.exists():
        print(f"ERROR: Tile bounds JSON not found: {args.tile_bounds_json}")
        sys.exit(1)
    
    plot_extents(args.tindex_path, args.tile_bounds_json, args.output)


if __name__ == "__main__":
    main()

