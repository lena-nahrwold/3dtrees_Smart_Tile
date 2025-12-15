#!/usr/bin/env python
"""
Plot each tree instance in sideview with lowest point distance to buffer edge.

For each tile, generates plots showing:
- Each instance in sideview (X-Z or Y-Z projection)
- The lowest point distance to the outer buffer edge
  (e.g., 9m = 9m from outer edge, 1m = 1m from core tile edge)

Usage:
    python plot_instances_sideview.py --input_folder <path> --output_folder <path> --buffer 10.0
"""

import os
import argparse
import numpy as np
import laspy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor


def get_tile_neighbors(tile_name: str, all_tile_names: List[str]) -> Dict[str, bool]:
    """
    Determine which edges of a tile have neighbors.
    """
    # Remove _segmented_remapped suffix if present
    clean_name = tile_name.replace('_segmented_remapped', '')
    
    parts = clean_name.split('_')
    col_str = parts[0][1:]  # Extract number after 'c'
    row_str = parts[1][1:]  # Extract number after 'r'
    col = int(col_str)
    row = int(row_str)
    
    col_padding = len(col_str)
    row_padding = len(row_str)
    
    def format_tile_name(c, r):
        return f"c{str(c).zfill(col_padding)}_r{str(r).zfill(row_padding)}"
    
    # Clean all tile names for comparison
    clean_all = [t.replace('_segmented_remapped', '') for t in all_tile_names]
    
    return {
        'east': format_tile_name(col+1, row) in clean_all,
        'west': col > 0 and format_tile_name(col-1, row) in clean_all,
        'north': format_tile_name(col, row+1) in clean_all,
        'south': row > 0 and format_tile_name(col, row-1) in clean_all,
    }


def compute_lowest_point_buffer_distance(
    lowest_point: np.ndarray,
    tile_boundary: Tuple[float, float, float, float],
    neighbors: Dict[str, bool],
    buffer: float,
) -> Tuple[float, str]:
    """
    Compute the distance from the lowest point to the outer buffer edge.
    
    Returns distance where:
    - buffer = at outer edge of buffer (far from core)
    - 0 = at inner edge of buffer (at core tile boundary)
    - negative = inside core tile
    
    Args:
        lowest_point: (x, y, z) of lowest point
        tile_boundary: (min_x, max_x, min_y, max_y)
        neighbors: dict with 'east', 'west', 'north', 'south' booleans
        buffer: buffer distance in meters
        
    Returns:
        (distance, direction) where direction is the closest edge direction
    """
    min_x, max_x, min_y, max_y = tile_boundary
    lp_x, lp_y = lowest_point[0], lowest_point[1]
    
    distances = {}
    
    # For each inner edge (edge with neighbor), compute distance to outer buffer edge
    # Distance is measured from outer edge of buffer towards core
    if neighbors['west']:
        # West buffer: from min_x to min_x + buffer
        # Distance from outer edge (min_x) = lp_x - min_x
        dist_from_outer = lp_x - min_x
        if dist_from_outer <= buffer:
            distances['west'] = dist_from_outer
    
    if neighbors['east']:
        # East buffer: from max_x - buffer to max_x
        # Distance from outer edge (max_x) = max_x - lp_x
        dist_from_outer = max_x - lp_x
        if dist_from_outer <= buffer:
            distances['east'] = dist_from_outer
    
    if neighbors['south']:
        # South buffer: from min_y to min_y + buffer
        # Distance from outer edge (min_y) = lp_y - min_y
        dist_from_outer = lp_y - min_y
        if dist_from_outer <= buffer:
            distances['south'] = dist_from_outer
    
    if neighbors['north']:
        # North buffer: from max_y - buffer to max_y
        # Distance from outer edge (max_y) = max_y - lp_y
        dist_from_outer = max_y - lp_y
        if dist_from_outer <= buffer:
            distances['north'] = dist_from_outer
    
    if not distances:
        # Not in any buffer zone - compute distance to nearest inner edge
        # (will be negative, meaning inside core)
        edge_distances = []
        if neighbors['west']:
            edge_distances.append((lp_x - (min_x + buffer), 'west'))
        if neighbors['east']:
            edge_distances.append(((max_x - buffer) - lp_x, 'east'))
        if neighbors['south']:
            edge_distances.append((lp_y - (min_y + buffer), 'south'))
        if neighbors['north']:
            edge_distances.append(((max_y - buffer) - lp_y, 'north'))
        
        if edge_distances:
            # Find minimum distance to buffer edge (positive = inside core)
            min_dist, direction = min(edge_distances, key=lambda x: x[0])
            return -min_dist, direction  # Negative to indicate inside core
        else:
            return float('inf'), 'none'
    
    # Find the closest buffer edge
    closest_edge = min(distances.items(), key=lambda x: x[1])
    return closest_edge[1], closest_edge[0]


def plot_tile_instances(
    tile_path: str,
    tile_name: str,
    all_tile_names: List[str],
    output_folder: str,
    buffer: float,
):
    """
    Generate sideview plots for all instances in a tile.
    """
    # Load LAZ file
    if os.path.isfile(tile_path):
        laz_file = tile_path
    else:
        laz_file = os.path.join(tile_path, "pc_with_species.laz")
        if not os.path.exists(laz_file):
            # Try direct file
            laz_file = f"{tile_path}.laz"
    
    if not os.path.exists(laz_file):
        print(f"  Warning: File not found: {laz_file}")
        return
    
    print(f"Processing {tile_name}...")
    
    las = laspy.read(laz_file, laz_backend=laspy.LazBackend.LazrsParallel)
    points = np.vstack((np.array(las.x), np.array(las.y), np.array(las.z))).T
    instances = np.array(las.PredInstance)
    
    # Compute tile boundary
    tile_boundary = (
        np.min(points[:, 0]),
        np.max(points[:, 0]),
        np.min(points[:, 1]),
        np.max(points[:, 1]),
    )
    min_x, max_x, min_y, max_y = tile_boundary
    
    # Get neighbor info
    neighbors = get_tile_neighbors(tile_name, all_tile_names)
    
    # Get unique instances
    unique_instances = np.unique(instances)
    unique_instances = unique_instances[unique_instances > 0]
    
    if len(unique_instances) == 0:
        print(f"  No instances found in {tile_name}")
        return
    
    # Create output folder for this tile
    tile_output_folder = os.path.join(output_folder, tile_name.replace('_segmented_remapped', ''))
    os.makedirs(tile_output_folder, exist_ok=True)
    
    # First pass: collect instance info and filter to only buffer + 5m into core
    instance_info = []
    instances_to_plot = []
    
    for inst_id in unique_instances:
        mask = instances == inst_id
        inst_points = points[mask]
        
        # Find lowest point
        lowest_z_idx = np.argmin(inst_points[:, 2])
        lowest_point = inst_points[lowest_z_idx]
        
        # Compute buffer distance
        dist, direction = compute_lowest_point_buffer_distance(
            lowest_point, tile_boundary, neighbors, buffer
        )
        
        # Determine if in buffer
        in_buffer = dist >= 0 and dist <= buffer
        
        info = {
            'id': inst_id,
            'lowest_point': lowest_point,
            'buffer_distance': dist,
            'direction': direction,
            'in_buffer': in_buffer,
            'num_points': len(inst_points),
            'height': np.max(inst_points[:, 2]) - np.min(inst_points[:, 2]),
            'points': inst_points,
        }
        instance_info.append(info)
        
        # Only plot if in buffer OR within 5m of buffer edge (i.e., dist > buffer but dist <= buffer + 5)
        # For instances in buffer: dist is 0 to buffer
        # For instances in core: dist is negative (distance inside core)
        # We want: in buffer (dist >= 0 and dist <= buffer) OR just inside core (dist < 0 and dist >= -5)
        if in_buffer or (dist < 0 and dist >= -5):
            instances_to_plot.append(info)
    
    print(f"  {tile_name}: {len(unique_instances)} total instances, {len(instances_to_plot)} in buffer/edge zone to plot")
    
    # Second pass: create plots only for filtered instances
    for info in instances_to_plot:
        inst_id = info['id']
        inst_points = info['points']
        lowest_point = info['lowest_point']
        dist = info['buffer_distance']
        direction = info['direction']
        in_buffer = info['in_buffer']
        
        # Create individual sideview plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # X-Z sideview (looking from south)
        ax1 = axes[0]
        ax1.scatter(inst_points[:, 0], inst_points[:, 2], s=0.5, alpha=0.5, c='green')
        ax1.scatter([lowest_point[0]], [lowest_point[2]], s=100, c='red', marker='*', 
                   label=f'Lowest point', zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title(f'X-Z Sideview (looking from South)')
        ax1.set_aspect('equal')
        ax1.legend()
        
        # Add buffer zone lines
        if neighbors['west']:
            ax1.axvline(min_x, color='red', linestyle='--', alpha=0.7, label='Outer edge')
            ax1.axvline(min_x + buffer, color='orange', linestyle='--', alpha=0.7, label='Buffer edge')
        if neighbors['east']:
            ax1.axvline(max_x, color='red', linestyle='--', alpha=0.7)
            ax1.axvline(max_x - buffer, color='orange', linestyle='--', alpha=0.7)
        
        # Y-Z sideview (looking from west)
        ax2 = axes[1]
        ax2.scatter(inst_points[:, 1], inst_points[:, 2], s=0.5, alpha=0.5, c='green')
        ax2.scatter([lowest_point[1]], [lowest_point[2]], s=100, c='red', marker='*', 
                   label=f'Lowest point', zorder=5)
        ax2.set_xlabel('Y (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title(f'Y-Z Sideview (looking from West)')
        ax2.set_aspect('equal')
        ax2.legend()
        
        # Add buffer zone lines
        if neighbors['south']:
            ax2.axvline(min_y, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(min_y + buffer, color='orange', linestyle='--', alpha=0.7)
        if neighbors['north']:
            ax2.axvline(max_y, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(max_y - buffer, color='orange', linestyle='--', alpha=0.7)
        
        # Add title with buffer distance info
        status = "IN BUFFER" if in_buffer else "IN CORE (edge)"
        if dist == float('inf'):
            dist_str = "N/A (no inner edges)"
        elif in_buffer:
            dist_str = f"{dist:.1f}m from outer edge ({direction})"
        else:
            dist_str = f"{-dist:.1f}m inside core (nearest: {direction})"
        
        fig.suptitle(f'Instance {inst_id} - {status}\n'
                    f'Lowest point distance: {dist_str}\n'
                    f'Points: {len(inst_points)}, Height: {np.max(inst_points[:, 2]) - np.min(inst_points[:, 2]):.1f}m',
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(tile_output_folder, f'instance_{inst_id:04d}.png'), dpi=150)
        plt.close()
    
    # Create summary plot showing only instances in buffer + 5m into core
    # Filter to instances_to_plot for summary
    fig, ax = plt.subplots(figsize=(12, max(6, len(instances_to_plot) * 0.15)))
    
    # Sort by buffer distance
    instances_to_plot.sort(key=lambda x: x['buffer_distance'] if x['buffer_distance'] != float('inf') else 1000)
    
    ids = [info['id'] for info in instances_to_plot]
    distances = [info['buffer_distance'] if info['buffer_distance'] != float('inf') else buffer + 5 for info in instances_to_plot]
    colors = ['red' if info['in_buffer'] else 'green' for info in instances_to_plot]
    
    bars = ax.barh(range(len(ids)), distances, color=colors, alpha=0.7)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"Inst {id}" for id in ids], fontsize=8)
    ax.set_xlabel('Distance from outer buffer edge (m)')
    ax.set_title(f'{tile_name}\nLowest Point Distance to Buffer Edge\n'
                f'Red = in buffer (will be filtered), Green = in core (kept)\n'
                f'Buffer size: {buffer}m, Showing: buffer + 5m into core')
    
    # Add vertical line at buffer boundary
    ax.axvline(buffer, color='orange', linestyle='--', linewidth=2, label=f'Buffer edge ({buffer}m)')
    ax.axvline(0, color='red', linestyle='-', linewidth=2, label='Outer tile edge')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(tile_output_folder, 'summary_buffer_distances.png'), dpi=150)
    plt.close()
    
    # Print statistics
    in_buffer_count = sum(1 for info in instance_info if info['in_buffer'])
    in_core_count = len(instance_info) - in_buffer_count
    print(f"    Total: {len(instance_info)} instances ({in_buffer_count} in buffer, {in_core_count} in core)")
    
    return instance_info


def main():
    parser = argparse.ArgumentParser(
        description="Plot tree instances in sideview with buffer distance info"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing LAZ files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder for plots"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=10.0,
        help="Buffer distance in meters (default: 10.0)"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel processing (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Find all tile files
    input_folder = args.input_folder
    
    # Check for subfolders or direct files
    tile_items = []
    
    # Check for subfolders ending with _segmented_remapped
    subfolders = sorted([
        d for d in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, d)) and d.endswith('_segmented_remapped')
    ])
    
    if subfolders:
        tile_items = [(os.path.join(input_folder, d), d) for d in subfolders]
    else:
        # Check for direct LAZ files
        laz_files = sorted([
            f for f in os.listdir(input_folder)
            if f.endswith('_segmented_remapped.laz')
        ])
        if laz_files:
            tile_items = [(os.path.join(input_folder, f), f.replace('.laz', '')) for f in laz_files]
    
    if not tile_items:
        print(f"No tiles found in {input_folder}")
        return
    
    print(f"Found {len(tile_items)} tiles")
    
    # Extract tile names for neighbor detection
    all_tile_names = [name for _, name in tile_items]
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Process each tile
    for tile_path, tile_name in tile_items:
        plot_tile_instances(
            tile_path=tile_path,
            tile_name=tile_name,
            all_tile_names=all_tile_names,
            output_folder=args.output_folder,
            buffer=args.buffer,
        )
    
    print(f"\nPlots saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
