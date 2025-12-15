#!/usr/bin/env python3
"""
Tile Merger - Merge Segmented Point Cloud Tiles with Species ID Preservation

Merges overlapping segmented point cloud tiles using:
1. Centroid-based buffer zone filtering (remove instances in overlap zones)
2. Deduplication keeping higher instance ID
3. Overlap ratio matching for cross-tile instance merging
4. Small cluster reassignment to nearest centroid
5. Retiling merged results back to original point cloud files

Key feature: Species ID is always preserved from the LARGER instance (by point count)
during all merge and reassignment operations.

Usage:
    python merge_tiles.py \
        --input-dir /path/to/segmented_remapped \
        --original-tiles-dir /path/to/original_tiles \
        --output-merged /path/to/merged.laz \
        --output-tiles-dir /path/to/output_tiles
"""

import argparse
import os
import numpy as np
import laspy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from scipy.spatial import cKDTree, KDTree
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TileData:
    """Container for tile point cloud data."""
    name: str
    points: np.ndarray
    instances: np.ndarray
    species_ids: np.ndarray
    boundary: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    las: laspy.LasData


# =============================================================================
# Union-Find Data Structure
# =============================================================================

class UnionFind:
    """
    Union-Find (Disjoint Set) data structure for grouping matched instances.
    Tracks instance sizes for species ID preservation.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.size = {}  # Track size for species preservation

    def make_set(self, x, size: int = 0):
        """Create a new set containing only x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = size

    def find(self, x) -> int:
        """Find the root of the set containing x with path compression."""
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y) -> int:
        """
        Merge the sets containing x and y.
        Returns the root of the merged set (the larger one by size).
        """
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return root_x
        
        # Union by size (larger becomes root for species preservation)
        if self.size.get(root_x, 0) >= self.size.get(root_y, 0):
            self.parent[root_y] = root_x
            self.size[root_x] = self.size.get(root_x, 0) + self.size.get(root_y, 0)
            return root_x
        else:
            self.parent[root_x] = root_y
            self.size[root_y] = self.size.get(root_x, 0) + self.size.get(root_y, 0)
            return root_y

    def get_components(self) -> Dict[int, List[int]]:
        """Get all connected components as {root: [members]}."""
        components = defaultdict(list)
        for x in self.parent:
            root = self.find(x)
            components[root].append(x)
        return dict(components)


# =============================================================================
# Stage 1: Load and Filter
# =============================================================================

def compute_tile_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Get the XY bounding box of a point cloud."""
    return (
        points[:, 0].min(),
        points[:, 0].max(),
        points[:, 1].min(),
        points[:, 1].max()
    )


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


def filter_by_centroid_in_buffer(
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
        
        # Calculate centroid
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


def load_tile(filepath: Path, all_tile_names: List[str], buffer: float) -> Optional[TileData]:
    """
    Load a LAZ tile and filter instances in buffer zones.
    
    Returns:
        TileData object or None if loading fails
    """
    print(f"Loading {filepath.name}...")
    
    try:
        las = laspy.read(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel)
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None
    
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
        print(f"  Warning: No instance attribute found in {filepath}")
        instances = np.zeros(len(points), dtype=np.int32)
    
    # Get species IDs if available
    if hasattr(las, 'species_id'):
        species_ids = np.array(las.species_id)
    else:
        species_ids = np.zeros(len(points), dtype=np.int32)
    
    boundary = compute_tile_bounds(points)
    
    # Extract tile name from filename
    tile_name = filepath.stem
    # Remove common suffixes
    for suffix in ['_segmented_remapped', '_segmented', '_remapped']:
        tile_name = tile_name.replace(suffix, '')
    
    # Filter instances with centroid in buffer zone
    instances_to_remove = filter_by_centroid_in_buffer(
        points, instances, boundary, tile_name, all_tile_names, buffer
    )
    
    # Keep track of which instances survived filtering
    kept_instances = set(np.unique(instances)) - instances_to_remove - {0, -1}
    
    print(f"  {len(points):,} points, {len(kept_instances)} instances kept, {len(instances_to_remove)} filtered")
    
    return TileData(
        name=tile_name,
        points=points,
        instances=instances,
        species_ids=species_ids,
        boundary=boundary,
        las=las
    ), instances_to_remove, kept_instances


# =============================================================================
# Stage 2: Deduplicate
# =============================================================================

def deduplicate_points(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    tolerance: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove duplicate points from overlapping tiles.
    When duplicates exist, keep the one with higher instance ID.
    
    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        species_ids: Array of species IDs
        tolerance: Distance tolerance (default 1mm)
    
    Returns:
        Tuple of (unique_points, unique_instances, unique_species_ids)
    """
    # Round coordinates to tolerance for grouping
    scale = 1.0 / tolerance
    rounded = np.round(points * scale).astype(np.int64)
    
    # Create unique key for each point
    dtype = [('x', np.int64), ('y', np.int64), ('z', np.int64)]
    keys = np.empty(len(rounded), dtype=dtype)
    keys['x'] = rounded[:, 0]
    keys['y'] = rounded[:, 1]
    keys['z'] = rounded[:, 2]
    
    # Sort by key, then by instance (descending) so higher instance comes first
    sort_order = np.lexsort((-instances, keys['z'], keys['y'], keys['x']))
    
    sorted_keys = keys[sort_order]
    sorted_points = points[sort_order]
    sorted_instances = instances[sort_order]
    sorted_species = species_ids[sort_order]
    
    # Find first occurrence of each unique key
    unique_mask = np.ones(len(sorted_keys), dtype=bool)
    unique_mask[1:] = (sorted_keys[1:] != sorted_keys[:-1])
    
    unique_points = sorted_points[unique_mask]
    unique_instances = sorted_instances[unique_mask]
    unique_species = sorted_species[unique_mask]
    
    removed = len(points) - len(unique_points)
    print(f"  Removed {removed:,} duplicate points ({100*removed/len(points):.1f}%)")
    
    return unique_points, unique_instances, unique_species


# =============================================================================
# Stage 3: FF3D Instance Matching
# =============================================================================

def find_overlap_region(
    bounds_a: Tuple[float, float, float, float],
    bounds_b: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float, float, float]]:
    """Find the overlap region between two bounding boxes."""
    minx_a, maxx_a, miny_a, maxy_a = bounds_a
    minx_b, maxx_b, miny_b, maxy_b = bounds_b
    
    overlap_minx = max(minx_a, minx_b)
    overlap_maxx = min(maxx_a, maxx_b)
    overlap_miny = max(miny_a, miny_b)
    overlap_maxy = min(maxy_a, maxy_b)
    
    if overlap_minx < overlap_maxx and overlap_miny < overlap_maxy:
        return (overlap_minx, overlap_maxx, overlap_miny, overlap_maxy)
    return None


def get_points_in_region(
    points: np.ndarray,
    instances: np.ndarray,
    region: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract points within a spatial region."""
    minx, maxx, miny, maxy = region
    mask = (
        (points[:, 0] >= minx) & (points[:, 0] <= maxx) &
        (points[:, 1] >= miny) & (points[:, 1] <= maxy)
    )
    return points[mask], instances[mask], mask


def compute_ff3d_overlap_ratios(
    instances_a: np.ndarray,
    instances_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    correspondence_tolerance: float = 0.05
) -> Dict[Tuple[int, int], float]:
    """
    Compute FF3D-style overlap ratios between all instance pairs.
    
    Only counts points as "same point" if they're within correspondence_tolerance
    (should be small, ~5cm, to only match actual duplicate points from overlapping tiles).
    
    FF3D metric: max(intersection/size_a, intersection/size_b)
    More lenient than IoU for asymmetric overlaps.
    
    Returns:
        Dictionary mapping (inst_a, inst_b) pairs to their overlap ratio
    """
    # Build KD-tree for nearest neighbor matching
    # IMPORTANT: tolerance should be small (5cm) to only match actual duplicate points,
    # not nearby points from different trees!
    tree_b = cKDTree(points_b)
    distances, correspondence = tree_b.query(points_a, k=1)
    valid_correspondence = distances < correspondence_tolerance
    
    # Count points per instance in overlap region
    size_a = defaultdict(int)
    size_b = defaultdict(int)
    for inst in instances_a:
        if inst > 0:
            size_a[inst] += 1
    for inst in instances_b:
        if inst > 0:
            size_b[inst] += 1
    
    # Count intersections - only where points are truly the same point (within tolerance)
    intersection_counts = defaultdict(int)
    for i, (inst_a, corr_idx, valid) in enumerate(zip(instances_a, correspondence, valid_correspondence)):
        if not valid or inst_a <= 0:
            continue
        inst_b = instances_b[corr_idx]
        if inst_b <= 0:
            continue
        intersection_counts[(inst_a, inst_b)] += 1
    
    # Compute FF3D overlap ratio for each pair
    overlap_ratios = {}
    for (inst_a, inst_b), intersection in intersection_counts.items():
        ratio_a = intersection / size_a[inst_a] if size_a[inst_a] > 0 else 0
        ratio_b = intersection / size_b[inst_b] if size_b[inst_b] > 0 else 0
        # FF3D uses max of the two ratios
        overlap_ratios[(inst_a, inst_b)] = max(ratio_a, ratio_b)
    
    return overlap_ratios, dict(size_a), dict(size_b)


# =============================================================================
# Stage 4: Small Cluster Reassignment
# =============================================================================

def reassign_small_clusters(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    instance_species_map: Dict[int, int],
    instance_sizes: Dict[int, int],
    min_cluster_size: int = 300,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Reassign small clusters to nearest larger instance by centroid distance.
    Species ID is taken from the target (larger) instance.
    
    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs (modified in-place)
        species_ids: Array of species IDs (modified in-place)
        instance_species_map: Mapping of instance ID to species ID
        instance_sizes: Mapping of instance ID to point count
        min_cluster_size: Minimum size for a cluster to be kept
    
    Returns:
        Updated (instances, species_ids, instance_species_map)
    """
    unique_instances = np.unique(instances)
    unique_instances = unique_instances[unique_instances > 0]
    
    # Separate small and large instances
    small_instances = []
    large_instances = []
    
    for inst_id in unique_instances:
        size = instance_sizes.get(inst_id, np.sum(instances == inst_id))
        if size < min_cluster_size:
            small_instances.append(inst_id)
        else:
            large_instances.append(inst_id)
    
    if len(small_instances) == 0 or len(large_instances) == 0:
        print(f"  No small clusters to reassign")
        return instances, species_ids, instance_species_map
    
    print(f"  Found {len(small_instances)} small clusters (<{min_cluster_size} points)")
    
    # Compute centroids for large instances
    large_centroids = {}
    for inst_id in large_instances:
        mask = instances == inst_id
        large_centroids[inst_id] = np.mean(points[mask], axis=0)
    
    # Build KD-tree from large instance centroids
    large_ids = list(large_centroids.keys())
    large_coords = np.array([large_centroids[i] for i in large_ids])
    tree = cKDTree(large_coords)
    
    # Reassign each small cluster
    total_reassigned = 0
    for small_inst in small_instances:
        mask = instances == small_inst
        small_centroid = np.mean(points[mask], axis=0)
        
        # Find nearest large instance
        distance, idx = tree.query(small_centroid)
        target_inst = large_ids[idx]
        
        # Reassign points
        instances[mask] = target_inst
        
        # Use species ID from target (larger) instance
        target_species = instance_species_map.get(target_inst, 0)
        species_ids[mask] = target_species
        
        total_reassigned += np.sum(mask)
        print(f"    Instance {small_inst} ({np.sum(mask)} pts) → Instance {target_inst} (dist: {distance:.2f}m)")
    
    print(f"  Reassigned {total_reassigned:,} points from {len(small_instances)} small clusters")
    
    return instances, species_ids, instance_species_map


def merge_small_volume_instances(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    instance_species_map: Dict[int, int],
    max_points_for_check: int = 10000,
    max_volume_for_merge: float = 4.0,
    max_search_radius: float = 5.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Merge small-volume instances to nearest large instance by centroid distance.
    
    For instances with < max_points_for_check points:
    1. Calculate convex hull volume
    2. If volume < max_volume_for_merge, merge to nearest large instance centroid
    
    Species ID is taken from the target (larger) instance.
    
    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs (modified in-place)
        species_ids: Array of species IDs (modified in-place)
        instance_species_map: Mapping of instance ID to species ID
        max_points_for_check: Only check instances with fewer points than this
        max_volume_for_merge: Merge instances with convex hull volume below this (m³)
        max_search_radius: Max distance to search for target instance (m)
        verbose: Print detailed decisions
    
    Returns:
        Updated (instances, species_ids, instance_species_map)
    """
    from scipy.spatial import ConvexHull
    
    unique_instances = np.unique(instances)
    unique_instances = unique_instances[unique_instances > 0]
    
    # Categorize instances by size and volume
    small_volume_instances = []  # (inst_id, point_count, volume)
    large_instances = []  # inst_id
    
    for inst_id in unique_instances:
        mask = instances == inst_id
        count = np.sum(mask)
        
        # Skip large instances (by point count) - they're kept
        if count >= max_points_for_check:
            large_instances.append(inst_id)
            continue
        
        # Skip too-small instances (can't compute hull)
        if count < 4:
            if verbose:
                print(f"    Instance {inst_id}: {count} pts - too few for hull, keeping")
            large_instances.append(inst_id)
            continue
        
        # Calculate convex hull volume
        pts = points[mask]
        try:
            hull = ConvexHull(pts)
            volume = hull.volume
        except Exception:
            # If hull fails, keep the instance
            if verbose:
                print(f"    Instance {inst_id}: {count} pts - hull failed, keeping")
            large_instances.append(inst_id)
            continue
        
        # Check volume threshold
        if volume < max_volume_for_merge:
            small_volume_instances.append((inst_id, count, volume))
            if verbose:
                print(f"    Instance {inst_id}: {count} pts, {volume:.2f} m³ - SMALL (< {max_volume_for_merge} m³)")
        else:
            large_instances.append(inst_id)
            if verbose:
                print(f"    Instance {inst_id}: {count} pts, {volume:.2f} m³ - keeping (volume ok)")
    
    if len(small_volume_instances) == 0:
        print(f"  No small-volume instances to merge")
        return instances, species_ids, instance_species_map
    
    if len(large_instances) == 0:
        print(f"  No large instances to merge into")
        return instances, species_ids, instance_species_map
    
    print(f"  Found {len(small_volume_instances)} small-volume instances (< {max_volume_for_merge} m³)")
    
    # Compute centroids for large instances
    large_centroids = {}
    large_sizes = {}
    for inst_id in large_instances:
        mask = instances == inst_id
        large_centroids[inst_id] = np.mean(points[mask], axis=0)
        large_sizes[inst_id] = np.sum(mask)
    
    # Build KD-tree from large instance centroids
    large_ids = list(large_centroids.keys())
    large_coords = np.array([large_centroids[i] for i in large_ids])
    tree = cKDTree(large_coords)
    
    # Merge each small-volume instance
    total_merged = 0
    for inst_id, count, volume in small_volume_instances:
        mask = instances == inst_id
        small_centroid = np.mean(points[mask], axis=0)
        
        # Find nearest large instance
        distance, idx = tree.query(small_centroid)
        
        if distance > max_search_radius:
            print(f"    Instance {inst_id}: {count} pts, {volume:.2f} m³ → no target within {max_search_radius}m (nearest: {distance:.1f}m)")
            continue
        
        target_inst = large_ids[idx]
        
        # Reassign points
        instances[mask] = target_inst
        
        # Use species ID from target (larger) instance
        target_species = instance_species_map.get(target_inst, 0)
        species_ids[mask] = target_species
        
        total_merged += count
        print(f"    Instance {inst_id}: {count} pts, {volume:.2f} m³ → Instance {target_inst} ({large_sizes[target_inst]} pts, dist: {distance:.1f}m)")
    
    print(f"  Merged {total_merged:,} points from {len(small_volume_instances)} small-volume instances")
    
    return instances, species_ids, instance_species_map


# =============================================================================
# Stage 5: Retile to Original Files
# =============================================================================

def retile_to_original_files(
    merged_points: np.ndarray,
    merged_instances: np.ndarray,
    merged_species_ids: np.ndarray,
    original_tiles_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
):
    """
    Map merged instance IDs back to original tile point clouds.
    
    For each original tile:
    1. Load original points
    2. Build KDTree from merged points
    3. For each original point, find nearest merged point
    4. If distance < tolerance, copy PredInstance and species_id
    5. Save updated tile
    """
    print(f"\n{'='*60}")
    print("Retiling merged results to original files")
    print(f"{'='*60}")
    
    # Find all LAZ files in original directory
    original_files = sorted(original_tiles_dir.glob("*.laz"))
    if not original_files:
        original_files = sorted(original_tiles_dir.glob("*.las"))
    
    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_tiles_dir}")
        return
    
    print(f"  Found {len(original_files)} original tile files")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build KDTree from merged points
    print("  Building KDTree from merged points...")
    merged_tree = cKDTree(merged_points)
    
    # Process each original tile
    for orig_file in original_files:
        print(f"\n  Processing {orig_file.name}...")
        
        # Load original tile
        orig_las = laspy.read(str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel)
        orig_points = np.vstack((
            np.array(orig_las.x),
            np.array(orig_las.y),
            np.array(orig_las.z)
        )).T
        
        # Query nearest merged point for each original point
        distances, indices = merged_tree.query(orig_points, workers=num_threads)
        
        # Create new instance and species arrays
        new_instances = np.zeros(len(orig_points), dtype=np.int32)
        new_species = np.zeros(len(orig_points), dtype=np.int32)
        
        # Copy from merged where distance is within tolerance
        valid_mask = distances < tolerance
        new_instances[valid_mask] = merged_instances[indices[valid_mask]]
        new_species[valid_mask] = merged_species_ids[indices[valid_mask]]
        
        # Add/update extra dimensions
        extra_dims = {dim.name for dim in orig_las.point_format.extra_dimensions}
        
        if "PredInstance" not in extra_dims:
            orig_las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.int32))
        if "species_id" not in extra_dims:
            orig_las.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.int32))
        
        orig_las.PredInstance = new_instances
        orig_las.species_id = new_species
        
        # Save to output directory
        output_file = output_dir / orig_file.name
        orig_las.write(str(output_file), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
        
        matched = np.sum(valid_mask)
        unique_inst = len(np.unique(new_instances[new_instances > 0]))
        print(f"    {matched:,}/{len(orig_points):,} points matched, {unique_inst} instances → {output_file.name}")


# =============================================================================
# Main Merge Function
# =============================================================================

def merge_tiles(
    input_dir: Path,
    original_tiles_dir: Optional[Path],
    output_merged: Path,
    output_tiles_dir: Optional[Path],
    buffer: float = 10.0,
    overlap_threshold: float = 0.3,
    max_centroid_distance: float = 3.0,
    correspondence_tolerance: float = 0.05,
    max_volume_for_merge: float = 4.0,
    num_threads: int = 8,
    enable_matching: bool = True,
    require_overlap: bool = True,
    enable_volume_merge: bool = True,
    verbose: bool = False,
):
    """
    Main merge function implementing the tile merging pipeline.
    """
    print("=" * 60)
    print("Tile Merger")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Original tiles: {original_tiles_dir}")
    print(f"Output merged: {output_merged}")
    print(f"Output tiles: {output_tiles_dir}")
    print(f"Buffer: {buffer}m")
    print(f"Instance matching: {'ENABLED' if enable_matching else 'DISABLED'}")
    if enable_matching:
        if require_overlap:
            print(f"  Overlap threshold: {overlap_threshold}")
        else:
            print(f"  Overlap check: DISABLED (centroid distance only)")
        print(f"  Max centroid distance: {max_centroid_distance}m")
    print(f"Volume merge: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    if enable_volume_merge:
        print(f"  Max volume for merge: {max_volume_for_merge} m³")
    print(f"Verbose: {verbose}")
    print("=" * 60)
    
    # Find all input LAZ files
    laz_files = sorted(input_dir.glob("*.laz"))
    if not laz_files:
        laz_files = sorted(input_dir.glob("*.las"))
    
    if len(laz_files) == 0:
        print(f"No LAZ/LAS files found in {input_dir}")
        return
    
    print(f"\nFound {len(laz_files)} tiles to merge")
    
    # Extract tile names for neighbor detection
    all_tile_names = []
    for f in laz_files:
        name = f.stem
        for suffix in ['_segmented_remapped', '_segmented', '_remapped']:
            name = name.replace(suffix, '')
        all_tile_names.append(name)
    
    # =========================================================================
    # Stage 1: Load and Filter
    # =========================================================================
    print(f"\n{'='*60}")
    print("Stage 1: Loading tiles and filtering buffer zone instances")
    print(f"{'='*60}")
    print(f"  Loading {len(laz_files)} files using {num_threads} threads...")
    
    tiles = []
    filtered_instances_per_tile = {}
    kept_instances_per_tile = {}
    
    # Load tiles in parallel using ThreadPoolExecutor
    def load_tile_wrapper(f):
        return load_tile(f, all_tile_names, buffer)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(load_tile_wrapper, laz_files))
    
    # Process results
    for result in results:
        if result is not None:
            tile_data, filtered, kept = result
            tiles.append(tile_data)
            filtered_instances_per_tile[tile_data.name] = filtered
            kept_instances_per_tile[tile_data.name] = kept
    
    if len(tiles) == 0:
        print("No tiles loaded successfully")
        return
    
    # =========================================================================
    # Assign global instance IDs and track species
    # =========================================================================
    print(f"\n{'='*60}")
    print("Assigning global instance IDs")
    print(f"{'='*60}")
    
    TILE_OFFSET = 100000  # Unique global ID: tile_idx * OFFSET + local_id
    
    def global_id(tile_idx: int, local_id: int) -> int:
        return tile_idx * TILE_OFFSET + local_id
    
    def local_id(gid: int) -> Tuple[int, int]:
        return gid // TILE_OFFSET, gid % TILE_OFFSET
    
    # Initialize Union-Find and track species per global instance
    uf = UnionFind()
    instance_species_map = {}  # global_id -> species_id
    instance_sizes = {}  # global_id -> point count
    
    for tile_idx, tile in enumerate(tiles):
        kept_instances = kept_instances_per_tile[tile.name]
        for local_inst in kept_instances:
            mask = tile.instances == local_inst
            size = np.sum(mask)
            gid = global_id(tile_idx, local_inst)
            uf.make_set(gid, size)
            instance_sizes[gid] = size
            
            # Get most common species_id for this instance
            inst_species = tile.species_ids[mask]
            if len(inst_species) > 0:
                unique, counts = np.unique(inst_species, return_counts=True)
                instance_species_map[gid] = unique[np.argmax(counts)]
    
    print(f"  Total global instances: {len(instance_sizes)}")
    
    # =========================================================================
    # Stage 3: Cross-Tile Instance Matching (Optional)
    # =========================================================================
    if enable_matching:
        print(f"\n{'='*60}")
        print("Stage 3: Cross-Tile Instance Matching")
        print(f"{'='*60}")
        
        total_matches = 0
        
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                tile_a = tiles[i]
                tile_b = tiles[j]
                
                # Check for overlap
                overlap = find_overlap_region(tile_a.boundary, tile_b.boundary)
                if overlap is None:
                    continue
                
                print(f"\n  Tile {tile_a.name} <-> {tile_b.name}")
                
                # Get points in overlap region (only kept instances)
                kept_mask_a = np.isin(tile_a.instances, list(kept_instances_per_tile[tile_a.name]))
                kept_mask_b = np.isin(tile_b.instances, list(kept_instances_per_tile[tile_b.name]))
                
                points_a, inst_a, _ = get_points_in_region(
                    tile_a.points[kept_mask_a], tile_a.instances[kept_mask_a], overlap
                )
                points_b, inst_b, _ = get_points_in_region(
                    tile_b.points[kept_mask_b], tile_b.instances[kept_mask_b], overlap
                )
                
                if len(points_a) == 0 or len(points_b) == 0:
                    continue
                
                # Compute overlap ratios
                computed_overlap_ratios, size_a, size_b = compute_ff3d_overlap_ratios(
                    inst_a, inst_b, points_a, points_b, correspondence_tolerance
                )
                
                # Pre-compute centroids for all instances in overlap region
                centroids_a = {}
                centroids_b = {}
                for inst_id in np.unique(inst_a):
                    if inst_id > 0:
                        centroids_a[inst_id] = np.mean(points_a[inst_a == inst_id], axis=0)
                for inst_id in np.unique(inst_b):
                    if inst_id > 0:
                        centroids_b[inst_id] = np.mean(points_b[inst_b == inst_id], axis=0)
                
                # Find matching pairs above threshold AND within centroid distance
                matches_in_pair = 0
                rejected_low_overlap = 0
                rejected_far_centroid = 0
                
                for (inst_a_local, inst_b_local), ratio in computed_overlap_ratios.items():
                    # Compute centroid distance
                    if inst_a_local in centroids_a and inst_b_local in centroids_b:
                        centroid_dist = np.linalg.norm(centroids_a[inst_a_local] - centroids_b[inst_b_local])
                    else:
                        centroid_dist = float('inf')
                    
                    # Check conditions (overlap is optional)
                    passes_overlap = not require_overlap or (ratio >= overlap_threshold)
                    passes_distance = centroid_dist < max_centroid_distance
                    
                    if verbose:
                        status = "MERGED" if (passes_overlap and passes_distance) else "REJECTED"
                        reason = ""
                        if require_overlap and ratio < overlap_threshold:
                            reason = f"low overlap ({ratio:.2f} < {overlap_threshold})"
                        elif not passes_distance:
                            reason = f"too far ({centroid_dist:.1f}m > {max_centroid_distance}m)"
                        print(f"      Inst {inst_a_local} <-> {inst_b_local}: ratio={ratio:.2f}, dist={centroid_dist:.1f}m → {status} {reason}")
                    
                    if passes_overlap and passes_distance:
                        gid_a = global_id(i, inst_a_local)
                        gid_b = global_id(j, inst_b_local)
                        
                        # Union with species from larger instance
                        root = uf.union(gid_a, gid_b)
                        
                        # Keep species from larger instance
                        size_gid_a = instance_sizes.get(gid_a, 0)
                        size_gid_b = instance_sizes.get(gid_b, 0)
                        if size_gid_a >= size_gid_b:
                            winner_species = instance_species_map.get(gid_a, 0)
                        else:
                            winner_species = instance_species_map.get(gid_b, 0)
                        instance_species_map[root] = winner_species
                        
                        matches_in_pair += 1
                    elif passes_overlap and not passes_distance:
                        rejected_far_centroid += 1
                    elif require_overlap:
                        rejected_low_overlap += 1
                
                if matches_in_pair > 0 or verbose:
                    if require_overlap:
                        print(f"    {matches_in_pair} pairs merged, {rejected_far_centroid} rejected (centroid too far), {rejected_low_overlap} rejected (low overlap)")
                    else:
                        print(f"    {matches_in_pair} pairs merged, {rejected_far_centroid} rejected (centroid too far)")
                total_matches += matches_in_pair
        
        print(f"\n  Total matching pairs: {total_matches}")
    else:
        print(f"\n{'='*60}")
        print("Stage 3: Cross-Tile Instance Matching (DISABLED)")
        print(f"{'='*60}")
        print("  Skipping instance matching - instances will not be merged across tiles")
    
    # Get connected components
    components = uf.get_components()
    print(f"  Connected components: {len(components)}")
    
    # Create mapping from global ID to final merged ID
    global_to_merged = {}
    merged_species = {}  # merged_id -> species_id
    
    for merged_id, (root, members) in enumerate(components.items(), start=1):
        # Find largest member for species
        largest_member = max(members, key=lambda m: instance_sizes.get(m, 0))
        merged_species[merged_id] = instance_species_map.get(largest_member, 0)
        
        for gid in members:
            global_to_merged[gid] = merged_id
    
    # =========================================================================
    # Stage 2: Merge and Deduplicate
    # =========================================================================
    print(f"\n{'='*60}")
    print("Stage 2: Merging tiles and deduplicating")
    print(f"{'='*60}")
    
    all_points = []
    all_instances = []
    all_species = []
    
    for tile_idx, tile in enumerate(tiles):
        # Create remapped instances
        remapped_instances = np.zeros_like(tile.instances)
        remapped_species = np.zeros_like(tile.species_ids)
        
        kept_instances = kept_instances_per_tile[tile.name]
        
        for local_inst in np.unique(tile.instances):
            if local_inst <= 0 or local_inst not in kept_instances:
                continue
            
            gid = global_id(tile_idx, local_inst)
            merged_id = global_to_merged.get(gid, 0)
            
            mask = tile.instances == local_inst
            remapped_instances[mask] = merged_id
            remapped_species[mask] = merged_species.get(merged_id, 0)
        
        all_points.append(tile.points)
        all_instances.append(remapped_instances)
        all_species.append(remapped_species)
    
    # Concatenate
    merged_points = np.vstack(all_points)
    merged_instances = np.concatenate(all_instances)
    merged_species_ids = np.concatenate(all_species)
    
    print(f"  Total points before dedup: {len(merged_points):,}")
    
    # Deduplicate
    print("\n  Deduplicating...")
    merged_points, merged_instances, merged_species_ids = deduplicate_points(
        merged_points, merged_instances, merged_species_ids
    )
    
    print(f"  Total points after dedup: {len(merged_points):,}")
    print(f"  Unique instances: {len(np.unique(merged_instances[merged_instances > 0]))}")
    
    # =========================================================================
    # Stage 4: Small Volume Instance Merging
    # =========================================================================
    if enable_volume_merge:
        print(f"\n{'='*60}")
        print("Stage 4: Small Volume Instance Merging")
        print(f"{'='*60}")
        
        # Build species map for final instances
        final_species_map = {}
        for inst_id in np.unique(merged_instances):
            if inst_id > 0:
                mask = merged_instances == inst_id
                species = merged_species_ids[mask]
                if len(species) > 0:
                    unique, counts = np.unique(species, return_counts=True)
                    final_species_map[inst_id] = unique[np.argmax(counts)]
        
        merged_instances, merged_species_ids, final_species_map = merge_small_volume_instances(
            merged_points,
            merged_instances,
            merged_species_ids,
            final_species_map,
            max_points_for_check=10000,
            max_volume_for_merge=max_volume_for_merge,
            max_search_radius=5.0,
            verbose=verbose,
        )
    else:
        print(f"\n{'='*60}")
        print("Stage 4: Small Volume Instance Merging (DISABLED)")
        print(f"{'='*60}")
    
    # =========================================================================
    # Renumber instances to continuous IDs
    # =========================================================================
    print(f"\n{'='*60}")
    print("Renumbering instances")
    print(f"{'='*60}")
    
    unique_instances = sorted(set(merged_instances) - {0, -1})
    old_to_new = {0: 0, -1: -1}
    new_species_map = {}
    
    for new_id, old_id in enumerate(unique_instances, start=1):
        old_to_new[old_id] = new_id
        new_species_map[new_id] = final_species_map.get(old_id, 0)
    
    merged_instances = np.array([old_to_new.get(x, 0) for x in merged_instances], dtype=np.int32)
    
    # Update species IDs based on new instance numbering
    for new_id, species in new_species_map.items():
        mask = merged_instances == new_id
        merged_species_ids[mask] = species
    
    print(f"  Final instance count: {len(unique_instances)}")
    
    # =========================================================================
    # Save merged output
    # =========================================================================
    print(f"\n{'='*60}")
    print("Saving merged output")
    print(f"{'='*60}")
    
    output_merged.parent.mkdir(parents=True, exist_ok=True)
    
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.min(merged_points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    
    output_las = laspy.LasData(header)
    output_las.x = merged_points[:, 0]
    output_las.y = merged_points[:, 1]
    output_las.z = merged_points[:, 2]
    
    output_las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.int32))
    output_las.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.int32))
    
    output_las.PredInstance = merged_instances
    output_las.species_id = merged_species_ids
    
    output_las.write(str(output_merged), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
    
    print(f"  Saved merged output: {output_merged}")
    print(f"  Total points: {len(merged_points):,}")
    print(f"  Total instances: {len(unique_instances)}")
    
    # =========================================================================
    # Stage 5: Retile to Original Files
    # =========================================================================
    if original_tiles_dir and output_tiles_dir:
        retile_to_original_files(
            merged_points,
            merged_instances,
            merged_species_ids,
            original_tiles_dir,
            output_tiles_dir,
            tolerance=0.1,
            num_threads=num_threads,
        )
    
    print(f"\n{'='*60}")
    print("Merge complete!")
    print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tile Merger - Merge segmented point cloud tiles with species ID preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing segmented LAZ tiles"
    )
    
    parser.add_argument(
        "--original-tiles-dir",
        type=Path,
        default=None,
        help="Directory containing original tile files for retiling"
    )
    
    parser.add_argument(
        "--output-merged", "-o",
        type=Path,
        required=True,
        help="Output path for merged LAZ file"
    )
    
    parser.add_argument(
        "--output-tiles-dir",
        type=Path,
        default=None,
        help="Output directory for retiled files"
    )
    
    parser.add_argument(
        "--buffer",
        type=float,
        default=10.0,
        help="Buffer zone distance in meters (default: 10.0)"
    )
    
    parser.add_argument(
        "--overlap-threshold", "--ff3d-threshold",
        type=float,
        default=0.3,
        dest="overlap_threshold",
        help="Overlap ratio threshold for instance matching (default: 0.3 = 30%%)"
    )
    
    parser.add_argument(
        "--max-centroid-distance",
        type=float,
        default=3.0,
        help="Max distance between centroids to merge instances (default: 3.0m)"
    )
    
    parser.add_argument(
        "--correspondence-tolerance",
        type=float,
        default=0.05,
        help="Max distance for point correspondence in meters (default: 0.05). "
             "Should be small (~5cm) to only match actual duplicate points from overlapping tiles."
    )
    
    parser.add_argument(
        "--max-volume-for-merge",
        type=float,
        default=4.0,
        help="Max convex hull volume (m³) for small instance merging (default: 4.0)"
    )
    
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Number of threads for parallel processing (default: 8)"
    )
    
    parser.add_argument(
        "--disable-matching", "--disable-ff3d",
        action="store_true",
        dest="disable_matching",
        help="Disable cross-tile instance matching"
    )
    
    parser.add_argument(
        "--disable-volume-merge",
        action="store_true",
        help="Disable small volume instance merging"
    )
    
    parser.add_argument(
        "--disable-overlap-check",
        action="store_true",
        help="Disable overlap ratio check - merge based on centroid distance only"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed merge decisions"
    )
    
    args = parser.parse_args()
    
    merge_tiles(
        input_dir=args.input_dir,
        original_tiles_dir=args.original_tiles_dir,
        output_merged=args.output_merged,
        output_tiles_dir=args.output_tiles_dir,
        buffer=args.buffer,
        overlap_threshold=args.overlap_threshold,
        max_centroid_distance=args.max_centroid_distance,
        correspondence_tolerance=args.correspondence_tolerance,
        max_volume_for_merge=args.max_volume_for_merge,
        num_threads=args.num_threads,
        enable_matching=not args.disable_matching,
        require_overlap=not args.disable_overlap_check,
        enable_volume_merge=not args.disable_volume_merge,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

