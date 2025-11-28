#!/usr/bin/env python3
"""
Compare grid-aligned vs data-aligned tiling strategies.
"""

import json
import math
from pathlib import Path

def explain_tiling_strategy():
    """Explain the difference between tiling strategies."""
    
    # Current data extent
    proj_minx = 416797.09764212347
    proj_miny = 5347274.874479232
    proj_maxx = 416868.7037103838
    proj_maxy = 5347346.9119330235
    tile_length = 100.0
    buffer = 5.0
    
    data_width = proj_maxx - proj_minx
    data_height = proj_maxy - proj_miny
    
    print("=" * 80)
    print("TILING STRATEGY EXPLANATION")
    print("=" * 80)
    print()
    
    print("Data Extent:")
    print(f"  X: [{proj_minx:.1f}, {proj_maxx:.1f}]  (width: {data_width:.1f}m)")
    print(f"  Y: [{proj_miny:.1f}, {proj_maxy:.1f}]  (height: {data_height:.1f}m)")
    print()
    
    print("-" * 80)
    print("STRATEGY 1: Grid-Aligned (Current Default)")
    print("-" * 80)
    print("  • Snaps to multiples of tile_length (100m)")
    print("  • start_x = floor(minx / 100) * 100")
    print("  • start_y = floor(miny / 100) * 100")
    print()
    
    start_x_grid = math.floor(proj_minx / tile_length) * tile_length
    start_y_grid = math.floor(proj_miny / tile_length) * tile_length
    end_x_grid = math.ceil(proj_maxx / tile_length) * tile_length
    end_y_grid = math.ceil(proj_maxy / tile_length) * tile_length
    
    tiles_x_grid = int((end_x_grid - start_x_grid) / tile_length)
    tiles_y_grid = int((end_y_grid - start_y_grid) / tile_length)
    total_tiles_grid = tiles_x_grid * tiles_y_grid
    
    print(f"  Grid start:  X[{start_x_grid:.1f}, ...], Y[{start_y_grid:.1f}, ...]")
    print(f"  Grid end:    X[..., {end_x_grid:.1f}], Y[..., {end_y_grid:.1f}]")
    print(f"  Offset from data: {proj_minx - start_x_grid:.1f}m in X, {proj_miny - start_y_grid:.1f}m in Y")
    print(f"  Tiles created: {tiles_x_grid} × {tiles_y_grid} = {total_tiles_grid}")
    print()
    print("  ✓ Pros: Consistent grid across datasets, reproducible")
    print("  ✗ Cons: May create extra empty tiles, less efficient")
    print()
    
    print("-" * 80)
    print("STRATEGY 2: Data-Aligned (More Efficient)")
    print("-" * 80)
    print("  • Starts from actual data extent")
    print("  • start_x = minx")
    print("  • start_y = miny")
    print()
    
    start_x_data = proj_minx
    start_y_data = proj_miny
    end_x_data = math.ceil((proj_maxx - proj_minx) / tile_length) * tile_length + start_x_data
    end_y_data = math.ceil((proj_maxy - proj_miny) / tile_length) * tile_length + start_y_data
    
    tiles_x_data = int((end_x_data - start_x_data) / tile_length)
    tiles_y_data = int((end_y_data - start_y_data) / tile_length)
    total_tiles_data = tiles_x_data * tiles_y_data
    
    print(f"  Grid start:  X[{start_x_data:.1f}, ...], Y[{start_y_data:.1f}, ...]")
    print(f"  Grid end:    X[..., {end_x_data:.1f}], Y[..., {end_y_data:.1f}]")
    print(f"  Offset from data: 0m (starts exactly at data boundary)")
    print(f"  Tiles created: {tiles_x_data} × {tiles_y_data} = {total_tiles_data}")
    print()
    print("  ✓ Pros: Minimizes tiles, better coverage, more efficient")
    print("  ✗ Cons: Grid position depends on data, less standardized")
    print()
    
    print("=" * 80)
    print(f"COMPARISON: Grid-aligned creates {total_tiles_grid} tiles, ")
    print(f"           Data-aligned creates {total_tiles_data} tile(s)")
    print(f"           Efficiency improvement: {total_tiles_grid / total_tiles_data:.1f}x fewer tiles!")
    print("=" * 80)

if __name__ == "__main__":
    explain_tiling_strategy()

