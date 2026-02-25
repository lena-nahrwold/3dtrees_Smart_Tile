#!/usr/bin/env python3
"""
Buffer Filter Preprocessing Script

Removes instances (and all their points) whose centroids are in buffer zones
facing neighboring tiles. This is a preprocessing step that should run before merging.

Usage:
    python filter_buffer_instances.py \
        --input-dir /path/to/segmented_remapped \
        --output-dir /path/to/filtered \
        --buffer 10.0
"""

import argparse
import numpy as np
import laspy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


def get_tile_neighbors(tile_name: str, all_tile_names: List[str]) -> Dict[str, bool]:
    """
    Determine which edges of a tile have neighbors.
    Returns dict with 'east', 'west', 'north', 'south' boolean values.
    """
    # Parse tile name format: c{col}_r{row}
    parts = tile_name.split('_')
    col_str = parts[0][1:]  # Extract number after 'c'
    row_str = parts[1][1:]  # Extract number after 'r'
    col = int(col_str)
    row = int(row_str)
    
    col_padding = len(col_str)
    row_padding = len(row_str)
    
    def format_tile_name(c, r):
        return f"c{str(c).zfill(col_padding)}_r{str(r).zfill(row_padding)}"
    
    return {
        'east': format_tile_name(col+1, row) in all_tile_names,
        'west': col > 0 and format_tile_name(col-1, row) in all_tile_names,
        'north': format_tile_name(col, row+1) in all_tile_names,
        'south': row > 0 and format_tile_name(col, row-1) in all_tile_names,
    }


def compute_tile_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Get the XY bounding box of a point cloud."""
    return (
        points[:, 0].min(),
        points[:, 0].max(),
        points[:, 1].min(),
        points[:, 1].max()
    )


def get_instances_to_remove(
    points: np.ndarray,
    instances: np.ndarray,
    boundary: Tuple[float, float, float, float],
    tile_name: str,
    all_tile_names: List[str],
    buffer: float = 10.0,
) -> Set[int]:
    """
    Find instances whose centroid is in the buffer zone on inner edges.
    
    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        boundary: (min_x, max_x, min_y, max_y) of the tile
        tile_name: Name of the tile (e.g., "c00_r00")
        all_tile_names: List of all tile names to determine neighbors
        buffer: Buffer distance from inner edges
    
    Returns:
        Set of instance IDs to REMOVE (centroid in buffer zone)
    """
    min_x, max_x, min_y, max_y = boundary
    
    # Determine which edges have neighbors
    neighbors = get_tile_neighbors(tile_name, all_tile_names)
    
    # Calculate tile dimensions and cap buffer
    tile_width = max_x - min_x
    tile_height = max_y - min_y
    min_dimension = min(tile_width, tile_height)
    actual_buffer = min(buffer, min_dimension * 0.4)
    actual_buffer = max(actual_buffer, 2.0)
    
    # Define buffer zone boundaries (only on inner edges)
    buf_min_x = min_x + (actual_buffer if neighbors['west'] else 0)
    buf_max_x = max_x - (actual_buffer if neighbors['east'] else 0)
    buf_min_y = min_y + (actual_buffer if neighbors['south'] else 0)
    buf_max_y = max_y - (actual_buffer if neighbors['north'] else 0)
    
    # Find instances to remove
    instances_to_remove = set()
    unique_ids = np.unique(instances)
    
    for inst_id in unique_ids:
        if inst_id <= 0:
            continue
        
        mask = instances == inst_id
        inst_points = points[mask]
        
        # Calculate centroid (XYZ)
        centroid = np.mean(inst_points, axis=0)
        cx, cy = centroid[0], centroid[1]
        
        # Check if centroid is in buffer zone
        in_west_buffer = neighbors['west'] and cx < buf_min_x
        in_east_buffer = neighbors['east'] and cx > buf_max_x
        in_south_buffer = neighbors['south'] and cy < buf_min_y
        in_north_buffer = neighbors['north'] and cy > buf_max_y
        
        if in_west_buffer or in_east_buffer or in_south_buffer or in_north_buffer:
            instances_to_remove.add(inst_id)
    
    return instances_to_remove


def process_tile(
    input_file: Path,
    output_file: Path,
    all_tile_names: List[str],
    buffer: float = 10.0,
) -> Tuple[int, int, int]:
    """
    Process a single tile: load, filter buffer instances, save.
    
    Returns:
        Tuple of (original_points, removed_points, removed_instances)
    """
    print(f"Processing {input_file.name}...")
    
    # Load LAZ file
    try:
        las = laspy.read(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel)
    except Exception as e:
        print(f"  Error loading {input_file}: {e}")
        return 0, 0, 0
    
    # Extract points and instances
    points = np.vstack((
        np.array(las.x),
        np.array(las.y),
        np.array(las.z)
    )).T
    
    # Get instance IDs
    if hasattr(las, 'PredInstance'):
        instances = np.array(las.PredInstance)
    elif hasattr(las, 'treeID'):
        instances = np.array(las.treeID)
    else:
        print(f"  Warning: No instance attribute found in {input_file}")
        # No instances to filter, just copy file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        las.write(str(output_file), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
        return len(points), 0, 0
    
    original_point_count = len(points)
    
    # Extract tile name from filename
    tile_name = input_file.stem
    # Remove common suffixes
    for suffix in ['_segmented_remapped', '_segmented', '_remapped', '_filtered']:
        tile_name = tile_name.replace(suffix, '')
    
    # Compute tile boundary
    boundary = compute_tile_bounds(points)
    
    # Find instances to remove
    instances_to_remove = get_instances_to_remove(
        points, instances, boundary, tile_name, all_tile_names, buffer
    )
    
    # Create boolean mask: True = keep point, False = remove point
    keep_mask = np.ones(len(points), dtype=bool)
    for inst_id in instances_to_remove:
        keep_mask[instances == inst_id] = False
    
    # Filter points
    filtered_points = points[keep_mask]
    removed_point_count = original_point_count - len(filtered_points)
    removed_instance_count = len(instances_to_remove)
    
    if removed_instance_count == 0:
        print(f"  {original_point_count:,} points, 0 instances removed")
        # No filtering needed, just copy file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        las.write(str(output_file), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
        return original_point_count, 0, 0
    
    # Create new LAS file with filtered points
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create new header (copy from original)
    header = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
    header.offsets = las.header.offsets
    header.scales = las.header.scales
    
    # Copy header metadata
    for prop in ['system_identifier', 'generating_software', 'file_creation']:
        if hasattr(las.header, prop):
            setattr(header, prop, getattr(las.header, prop))
    
    # Create new LAS data
    output_las = laspy.LasData(header)
    
    # Copy all point dimensions (filtered)
    for dim_name in las.point_format.dimension_names:
        try:
            dim_data = getattr(las, dim_name)
            if hasattr(dim_data, '__len__') and len(dim_data) == len(points):
                filtered_data = dim_data[keep_mask]
                setattr(output_las, dim_name, filtered_data)
        except Exception as e:
            # Skip dimensions that can't be copied
            pass
    
    # Write output
    output_las.write(str(output_file), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
    
    print(f"  {original_point_count:,} → {len(filtered_points):,} points "
          f"({removed_point_count:,} removed, {removed_instance_count} instances)")
    
    return original_point_count, removed_point_count, removed_instance_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter buffer zone instances - Remove instances whose centroids are in buffer zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing input LAZ tiles"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for filtered LAZ files"
    )
    
    parser.add_argument(
        "--buffer",
        type=float,
        default=10.0,
        help="Buffer zone distance in meters (default: 10.0)"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        default="_filtered",
        help="Suffix to add to output filenames (default: '_filtered')"
    )
    
    args = parser.parse_args()
    
    # Find all input LAZ files
    laz_files = sorted(args.input_dir.glob("*.laz"))
    if not laz_files:
        laz_files = sorted(args.input_dir.glob("*.las"))
    
    if len(laz_files) == 0:
        print(f"No LAZ/LAS files found in {args.input_dir}")
        return
    
    print("=" * 60)
    print("Buffer Filter Preprocessing")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Buffer: {args.buffer}m")
    print(f"Found {len(laz_files)} tiles to process")
    print("=" * 60)
    
    # Extract tile names for neighbor detection
    all_tile_names = []
    for f in laz_files:
        name = f.stem
        for suffix in ['_segmented_remapped', '_segmented', '_remapped', '_filtered']:
            name = name.replace(suffix, '')
        all_tile_names.append(name)
    
    # Process each tile
    total_original = 0
    total_removed = 0
    total_instances_removed = 0
    
    for input_file in laz_files:
        # Generate output filename
        output_file = args.output_dir / f"{input_file.stem}{args.suffix}{input_file.suffix}"
        
        orig, removed, inst_removed = process_tile(
            input_file, output_file, all_tile_names, args.buffer
        )
        
        total_original += orig
        total_removed += removed
        total_instances_removed += inst_removed
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total points: {total_original:,} → {total_original - total_removed:,} "
          f"({total_removed:,} removed, {100*total_removed/max(total_original,1):.1f}%)")
    print(f"Total instances removed: {total_instances_removed}")
    print("=" * 60)


if __name__ == "__main__":
    main()

