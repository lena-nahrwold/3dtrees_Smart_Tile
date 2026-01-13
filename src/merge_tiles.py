#!/usr/bin/env python3
"""
Tile Merger - Merge Segmented Point Cloud Tiles with Species ID Preservation

Merges overlapping segmented point cloud tiles using:
1. Load and filter: Centroid-based buffer zone filtering (remove instances in overlap zones)
2. Assign global IDs: Create unique instance IDs across all tiles
3. Cross-tile matching: Overlap ratio matching for cross-tile instance merging
4. Merge and deduplicate: Combine tiles and remove duplicate points
5. Small volume merging: Reassign small clusters to nearest large instance
6. Retiling: Map merged results back to original point cloud files

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
import gc
import os
import sys
import numpy as np
import laspy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, TextIO
from scipy.spatial import cKDTree, KDTree
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

# Force unbuffered output for real-time progress feedback
# (especially important when running in Docker/containers)
sys.stdout.reconfigure(line_buffering=True)


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
    species_prob: Optional[np.ndarray] = None  # Species probability (if available)
    has_species_id: bool = False  # Whether species_id was present in input file
    # Note: las field removed to save memory - was never used after loading


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
        points[:, 1].max(),
    )


def get_tile_bounds_from_header(filepath: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get spatial bounds of a tile from its header (without loading points).
    
    Args:
        filepath: Path to LAZ file
    
    Returns:
        Tuple of (minx, maxx, miny, maxy) or None on error
    """
    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as las:
            return (las.header.x_min, las.header.x_max, las.header.y_min, las.header.y_max)
    except Exception:
        return None


def find_spatial_neighbors(
    tile_boundary: Tuple[float, float, float, float],
    tile_name: str,
    all_tiles: Dict[str, Tuple[float, float, float, float]],  # name -> boundary
    tolerance: float = 1.0
) -> Dict[str, Optional[str]]:
    """
    Find neighboring tiles based on actual spatial overlaps.
    
    Detects neighbors by checking if tiles overlap spatially (not just aligned edges).
    This handles cases where tiles extend into each other by buffer meters.
    
    Args:
        tile_boundary: (minx, maxx, miny, maxy) of the current tile
        tile_name: Name of the current tile
        all_tiles: Dictionary mapping tile names to their boundaries
        tolerance: Minimum overlap distance to consider tiles as neighbors (default: 1.0m)
    
    Returns:
        Dictionary with 'east', 'west', 'north', 'south' -> neighbor_name or None
    """
    minx_a, maxx_a, miny_a, maxy_a = tile_boundary
    
    neighbors = {
        "east": None,
        "west": None,
        "north": None,
        "south": None
    }
    
    # Track overlaps for each direction to pick the best neighbor
    east_overlaps = []  # (overlap_area, other_name)
    west_overlaps = []
    north_overlaps = []
    south_overlaps = []
    
    for other_name, (minx_b, maxx_b, miny_b, maxy_b) in all_tiles.items():
        if other_name == tile_name:
            continue
        
        # Check for actual spatial overlap (not just edge alignment)
        overlap = find_overlap_region(tile_boundary, (minx_b, maxx_b, miny_b, maxy_b))
        if overlap is None:
            continue
        
        overlap_minx, overlap_maxx, overlap_miny, overlap_maxy = overlap
        overlap_width = overlap_maxx - overlap_minx
        overlap_height = overlap_maxy - overlap_miny
        overlap_area = overlap_width * overlap_height
        
        # Determine which edge(s) the overlap is on
        # Check if overlap is significant enough (at least tolerance meters)
        
        # East neighbor: other tile extends to the right (east) of this tile
        # The overlap should be on the right side of this tile
        if minx_b > minx_a and overlap_width >= tolerance:
            # Check if there's vertical overlap
            if not (maxy_b < miny_a or miny_b > maxy_a):
                east_overlaps.append((overlap_area, other_name))
        
        # West neighbor: other tile extends to the left (west) of this tile
        # The overlap should be on the left side of this tile
        if maxx_b < maxx_a and overlap_width >= tolerance:
            # Check if there's vertical overlap
            if not (maxy_b < miny_a or miny_b > maxy_a):
                west_overlaps.append((overlap_area, other_name))
        
        # North neighbor: other tile extends above (north) of this tile
        # The overlap should be on the top side of this tile
        if miny_b > miny_a and overlap_height >= tolerance:
            # Check if there's horizontal overlap
            if not (maxx_b < minx_a or minx_b > maxx_a):
                north_overlaps.append((overlap_area, other_name))
        
        # South neighbor: other tile extends below (south) of this tile
        # The overlap should be on the bottom side of this tile
        if maxy_b < maxy_a and overlap_height >= tolerance:
            # Check if there's horizontal overlap
            if not (maxx_b < minx_a or minx_b > maxx_a):
                south_overlaps.append((overlap_area, other_name))
    
    # Pick the neighbor with the largest overlap for each direction
    if east_overlaps:
        neighbors["east"] = max(east_overlaps, key=lambda x: x[0])[1]
    if west_overlaps:
        neighbors["west"] = max(west_overlaps, key=lambda x: x[0])[1]
    if north_overlaps:
        neighbors["north"] = max(north_overlaps, key=lambda x: x[0])[1]
    if south_overlaps:
        neighbors["south"] = max(south_overlaps, key=lambda x: x[0])[1]
    
    return neighbors


def filter_by_centroid_in_buffer(
    points: np.ndarray,
    instances: np.ndarray,
    boundary: Tuple[float, float, float, float],
    tile_name: str,
    all_tiles: Dict[str, Tuple[float, float, float, float]],
    buffer: float = 10.0,
) -> Tuple[Set[int], Dict[int, str]]:
    """
    Find instances whose centroid is in the buffer zone on overlapping edges.
    Uses vectorized centroid computation for efficiency.

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        boundary: (min_x, max_x, min_y, max_y) of the tile
        tile_name: Name of the tile (e.g., "c00_r00")
        all_tiles: Dictionary mapping tile names to their boundaries for neighbor detection
        buffer: Buffer distance from inner edges (also used as minimum overlap to consider)

    Returns:
        Tuple of:
        - Set of instance IDs to REMOVE (centroid in buffer zone)
        - Dict mapping instance ID to buffer direction ('east', 'west', 'north', 'south')
            For instances in multiple buffers (corners), uses priority: west > south > east > north
    """
    min_x, max_x, min_y, max_y = boundary

    # Determine which edges have neighbors using spatial bounds
    neighbors = find_spatial_neighbors(boundary, tile_name, all_tiles, tolerance=buffer)

    # Calculate tile dimensions and cap buffer
    tile_width = max_x - min_x
    tile_height = max_y - min_y
    min_dimension = min(tile_width, tile_height)
    actual_buffer = min(buffer, min_dimension * 0.4)
    actual_buffer = max(actual_buffer, 2.0)

    # Define buffer zone boundaries (only on edges with neighbors)
    # Simple approach: buffer meters from each edge that has a neighbor
    buf_min_x = min_x + (actual_buffer if neighbors["west"] is not None else 0)
    buf_max_x = max_x - (actual_buffer if neighbors["east"] is not None else 0)
    buf_min_y = min_y + (actual_buffer if neighbors["south"] is not None else 0)
    buf_max_y = max_y - (actual_buffer if neighbors["north"] is not None else 0)

    # Vectorized centroid computation: O(n log n) instead of O(n * k)
    # Where n = number of points, k = number of instances
    # This provides ~100-250x speedup for typical tiles with many instances
    centroids = compute_centroids_vectorized(points, instances)

    # Find instances to remove and track their buffer direction
    instances_to_remove = set()
    instance_buffer_direction = {}  # inst_id -> direction

    for inst_id, centroid in centroids.items():
        if inst_id <= 0:
            continue

        cx, cy = centroid[0], centroid[1]

        # Check if centroid is in buffer zone (any direction with a neighbor)
        in_west_buffer = neighbors["west"] is not None and cx < buf_min_x
        in_east_buffer = neighbors["east"] is not None and cx > buf_max_x
        in_south_buffer = neighbors["south"] is not None and cy < buf_min_y
        in_north_buffer = neighbors["north"] is not None and cy > buf_max_y

        if in_west_buffer or in_east_buffer or in_south_buffer or in_north_buffer:
            instances_to_remove.add(inst_id)
            # Priority for corner cases: west/south (don't recover) > east/north (recover)
            # If in west or south buffer, neighbor with lower col/row should have it
            if in_west_buffer:
                instance_buffer_direction[inst_id] = "west"
            elif in_south_buffer:
                instance_buffer_direction[inst_id] = "south"
            elif in_east_buffer:
                instance_buffer_direction[inst_id] = "east"
            else:  # in_north_buffer
                instance_buffer_direction[inst_id] = "north"

    return instances_to_remove, instance_buffer_direction


def load_tile(
    filepath: Path,
    all_tiles: Dict[str, Tuple[float, float, float, float]],
    buffer: float,
    chunk_size: int = 1_000_000,
) -> Optional[Tuple[TileData, Set[int], Set[int], Dict[int, str]]]:
    """
    Load a LAZ tile using chunked reading for memory efficiency.

    Args:
        filepath: Path to the LAZ file
        all_tiles: Dictionary mapping tile names to their boundaries for neighbor detection
        buffer: Buffer distance for filtering
        chunk_size: Number of points to read per chunk (default 1M)

    Returns:
        Tuple of (TileData, instances_to_remove, kept_instances, instance_buffer_direction) or None if loading fails
    """
    print(f"Loading {filepath.name}...")

    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            # Get total point count from header
            n_points = f.header.point_count

            # Check which extra dimensions are available
            extra_dims = {dim.name for dim in f.header.point_format.extra_dimensions}
            has_pred_instance = "PredInstance" in extra_dims
            has_tree_id = "treeID" in extra_dims
            has_species_id = "species_id" in extra_dims
            has_species_prob = "species_prob" in extra_dims

            # Pre-allocate arrays - avoids multiple copies
            points = np.empty((n_points, 3), dtype=np.float64)
            instances = np.zeros(n_points, dtype=np.int32)
            species_ids = np.zeros(n_points, dtype=np.int32)
            species_prob = None
            if has_species_prob:
                species_prob = np.zeros(n_points, dtype=np.float32)

            # Read in chunks to reduce peak memory
            offset = 0
            for chunk in f.chunk_iterator(chunk_size):
                chunk_len = len(chunk)
                end = offset + chunk_len

                # Direct assignment - no intermediate arrays
                points[offset:end, 0] = chunk.x
                points[offset:end, 1] = chunk.y
                points[offset:end, 2] = chunk.z

                # Get instance IDs
                if has_pred_instance:
                    instances[offset:end] = chunk.PredInstance
                elif has_tree_id:
                    instances[offset:end] = chunk.treeID

                # Get species IDs
                if has_species_id:
                    species_ids[offset:end] = chunk.species_id

                # Get species probability
                if has_species_prob:
                    species_prob[offset:end] = chunk.species_prob

                offset = end

    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None

    if not has_pred_instance and not has_tree_id:
        print(f"  Warning: No instance attribute found in {filepath}")

    boundary = compute_tile_bounds(points)

    # Extract tile name from filename
    tile_name = filepath.stem
    # Remove common suffixes
    for suffix in ["_segmented_remapped", "_segmented", "_remapped"]:
        tile_name = tile_name.replace(suffix, "")

    # Filter instances with centroid in buffer zone
    instances_to_remove, instance_buffer_direction = filter_by_centroid_in_buffer(
        points, instances, boundary, tile_name, all_tiles, buffer
    )

    # Keep track of which instances survived filtering
    kept_instances = set(np.unique(instances)) - instances_to_remove - {0}

    print(
        f"  {len(points):,} points, {len(kept_instances)} instances kept, {len(instances_to_remove)} filtered"
    )

    return (
        TileData(
            name=tile_name,
            points=points,
            instances=instances,
            species_ids=species_ids,
            species_prob=species_prob,
            has_species_id=has_species_id,
            boundary=boundary,
        ),
        instances_to_remove,
        kept_instances,
        instance_buffer_direction,
    )


# =============================================================================
# Helper function for multiprocessing (must be at module level for pickling)
# =============================================================================


def _load_tile_wrapper(args):
    """
    Wrapper function for load_tile to make it pickleable for ProcessPoolExecutor.
    
    Args:
        args: Tuple of (filepath, tile_boundaries, buffer)
    
    Returns:
        Result from load_tile()
    """
    filepath, tile_boundaries, buffer = args
    return load_tile(filepath, tile_boundaries, buffer)


def _compute_hull_wrapper(args):
    """
    Wrapper function for convex hull computation to make it pickleable for ProcessPoolExecutor.
    
    Args:
        args: Tuple of (points, bbox_volume)
    
    Returns:
        Tuple of (volume, success) where success is True if hull computation succeeded
    """
    from scipy.spatial import ConvexHull
    
    points, bbox_volume = args
    try:
        hull = ConvexHull(points)
        return (hull.volume, True)
    except Exception:
        # If hull fails, use bounding box volume as fallback
        return (bbox_volume, False)


# =============================================================================
# Stage 4: Deduplicate (used in Stage 4: Merge and Deduplicate)
# =============================================================================


def deduplicate_points(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    species_prob: Optional[np.ndarray] = None,
    tolerance: float = 0.001,
    grid_size: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Remove duplicate points from overlapping tiles using grid-based processing.
    When duplicates exist, keep the one with higher instance ID.

    Uses spatial grid cells to reduce memory usage - instead of sorting billions
    of points at once, we process smaller cells independently.

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        species_ids: Array of species IDs
        species_prob: Array of species probabilities (optional)
        tolerance: Distance tolerance (default 1mm)
        grid_size: Size of spatial grid cells in meters (default 50m)

    Returns:
        Tuple of (unique_points, unique_instances, unique_species_ids, unique_species_prob)
    """
    n_points = len(points)
    scale = 1.0 / tolerance

    # Compute grid cell indices for each point
    min_coords = points.min(axis=0)
    grid_indices = ((points[:, :2] - min_coords[:2]) / grid_size).astype(np.int32)

    # Create cell keys (combine x,y grid indices into single key)
    max_grid_y = grid_indices[:, 1].max() + 1
    cell_keys = grid_indices[:, 0] * max_grid_y + grid_indices[:, 1]

    # Round coordinates to tolerance for duplicate detection
    rounded = np.floor(points * scale).astype(np.int64)

    # Create point hash: combine cell key, rounded coords, and negative instance for sorting
    # We want higher instance IDs to come first within same position
    point_hash = rounded[:, 0] + rounded[:, 1] * 73856093 + rounded[:, 2] * 19349669

    # Sort by: cell_key, point_hash, then -instance (so higher instance comes first)
    sort_order = np.lexsort((-instances, point_hash, cell_keys))

    sorted_cell_keys = cell_keys[sort_order]
    sorted_point_hash = point_hash[sort_order]

    # Find duplicates: same cell AND same point hash
    is_duplicate = np.zeros(n_points, dtype=bool)
    is_duplicate[1:] = (sorted_cell_keys[1:] == sorted_cell_keys[:-1]) & (
        sorted_point_hash[1:] == sorted_point_hash[:-1]
    )

    # Map back to original indices
    keep_mask = np.ones(n_points, dtype=bool)
    keep_mask[sort_order[is_duplicate]] = False

    # Extract kept points
    unique_points = points[keep_mask]
    unique_instances = instances[keep_mask]
    unique_species = species_ids[keep_mask]
    unique_species_prob = None
    if species_prob is not None:
        unique_species_prob = species_prob[keep_mask]

    removed = n_points - len(unique_points)
    print(f"  Removed {removed:,} duplicate points ({100 * removed / n_points:.1f}%)")

    return unique_points, unique_instances, unique_species, unique_species_prob


# =============================================================================
# Stage 3: FF3D Instance Matching
# =============================================================================


def find_overlap_region(
    bounds_a: Tuple[float, float, float, float],
    bounds_b: Tuple[float, float, float, float],
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
    points: np.ndarray, instances: np.ndarray, region: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract points within a spatial region."""
    minx, maxx, miny, maxy = region
    mask = (
        (points[:, 0] >= minx)
        & (points[:, 0] <= maxx)
        & (points[:, 1] >= miny)
        & (points[:, 1] <= maxy)
    )
    return points[mask], instances[mask], mask


def compute_ff3d_overlap_ratios(
    instances_a: np.ndarray,
    instances_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    correspondence_tolerance: float = 0.05,
) -> Dict[Tuple[int, int], float]:
    """
    Compute FF3D-style overlap ratios between all instance pairs.

    Only counts points as "same point" if they're within correspondence_tolerance
    (should be small, ~5cm, to only match actual duplicate points from overlapping tiles).

    FF3D metric: max(intersection/size_a, intersection/size_b)
    More lenient than IoU for asymmetric overlaps.

    Returns:
        Dictionary mapping (inst_a, inst_b) pairs to their overlap ratio

    Note: Uses hash-based grid matching instead of KDTree for O(n) instead of O(n log n).
    """
    # Vectorized: Count points per instance using np.unique
    unique_a, counts_a = np.unique(instances_a[instances_a > 0], return_counts=True)
    unique_b, counts_b = np.unique(instances_b[instances_b > 0], return_counts=True)
    size_a = dict(zip(unique_a, counts_a))
    size_b = dict(zip(unique_b, counts_b))

    # Grid-based matching: O(n) instead of KDTree O(n log n)
    # Round points to grid cells based on tolerance
    scale = 1.0 / correspondence_tolerance

    # Create grid keys for points_b (the lookup table)
    grid_b = np.floor(points_b * scale).astype(np.int64)
    # Combine x, y, z into single hash key using large primes
    hash_b = grid_b[:, 0] + grid_b[:, 1] * 73856093 + grid_b[:, 2] * 19349669

    # Create grid keys for points_a
    grid_a = np.floor(points_a * scale).astype(np.int64)
    hash_a = grid_a[:, 0] + grid_a[:, 1] * 73856093 + grid_a[:, 2] * 19349669

    # Fully vectorized: Get unique grid cells from B with their instance IDs
    # Sort by hash to enable searchsorted lookup
    sort_idx_b = np.argsort(hash_b)
    sorted_hash_b = hash_b[sort_idx_b]
    sorted_inst_b = instances_b[sort_idx_b]

    # Get first occurrence of each unique hash (deduplicate grid cells)
    unique_hash_b, first_idx = np.unique(sorted_hash_b, return_index=True)
    unique_inst_b = sorted_inst_b[first_idx]

    # Find matches: where hash_a exists in unique_hash_b
    insert_pos = np.searchsorted(unique_hash_b, hash_a)

    # Clamp to valid range and check for actual matches
    insert_pos_clamped = np.clip(insert_pos, 0, len(unique_hash_b) - 1)
    matches_mask = unique_hash_b[insert_pos_clamped] == hash_a

    # Get matched instance IDs
    matched_inst_b = np.zeros(len(hash_a), dtype=instances_b.dtype)
    matched_inst_b[matches_mask] = unique_inst_b[insert_pos_clamped[matches_mask]]

    # Filter to valid matches with positive instance IDs
    valid_mask = matches_mask & (instances_a > 0) & (matched_inst_b > 0)
    valid_inst_a = instances_a[valid_mask]
    valid_inst_b = matched_inst_b[valid_mask]

    # Create unique pair keys and count occurrences
    if len(valid_inst_a) > 0:
        max_inst = max(instances_a.max(), instances_b.max()) + 1
        pair_keys = valid_inst_a.astype(np.int64) * max_inst + valid_inst_b.astype(
            np.int64
        )
        unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)

        # Decode back to instance pairs
        intersection_counts = {}
        for key, count in zip(unique_pairs, pair_counts):
            inst_a_val = int(key // max_inst)
            inst_b_val = int(key % max_inst)
            intersection_counts[(inst_a_val, inst_b_val)] = count
    else:
        intersection_counts = {}

    # Compute FF3D overlap ratio for each pair
    overlap_ratios = {}
    for (inst_a, inst_b), intersection in intersection_counts.items():
        ratio_a = intersection / size_a[inst_a] if size_a.get(inst_a, 0) > 0 else 0
        ratio_b = intersection / size_b[inst_b] if size_b.get(inst_b, 0) > 0 else 0
        # FF3D uses max of the two ratios
        overlap_ratios[(inst_a, inst_b)] = max(ratio_a, ratio_b)

    return overlap_ratios, size_a, size_b


def compute_centroids_vectorized(
    points: np.ndarray, instances: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Compute centroids for all instances using vectorized operations.

    Instead of looping over each instance and scanning the full array,
    this uses sorting and cumulative sums for O(n log n) instead of O(n * k).

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs

    Returns:
        Dictionary mapping instance_id -> centroid (3D array)
    """
    # Filter to positive instances
    valid_mask = instances > 0
    valid_points = points[valid_mask]
    valid_instances = instances[valid_mask]

    if len(valid_instances) == 0:
        return {}

    # Sort by instance ID
    sort_idx = np.argsort(valid_instances)
    sorted_instances = valid_instances[sort_idx]
    sorted_points = valid_points[sort_idx]

    # Find boundaries between different instances
    unique_instances, first_indices, counts = np.unique(
        sorted_instances, return_index=True, return_counts=True
    )

    # Compute cumulative sums for efficient mean calculation
    cumsum = np.zeros((len(sorted_points) + 1, 3), dtype=np.float64)
    cumsum[1:] = np.cumsum(sorted_points, axis=0)

    # Calculate centroids using cumulative sum differences
    centroids = {}
    for i, (inst_id, start_idx, count) in enumerate(
        zip(unique_instances, first_indices, counts)
    ):
        end_idx = start_idx + count
        centroid = (cumsum[end_idx] - cumsum[start_idx]) / count
        centroids[int(inst_id)] = centroid

    return centroids


def get_border_region_mask(
    points: np.ndarray,
    boundary: Tuple[float, float, float, float],
    inner_dist: float,
    outer_dist: float,
    neighbors: Dict[str, Optional[str]],
) -> np.ndarray:
    """
    Get mask for points in the donut-shaped border region.
    
    Only considers edges that have neighbors (no point checking for edges without neighbors).
    
    Args:
        points: Nx3 array of point coordinates
        boundary: (min_x, max_x, min_y, max_y) tile bounds
        inner_dist: Inner distance from edge (buffer zone boundary)
        outer_dist: Outer distance from edge (end of border region)
        neighbors: Dict with 'east', 'west', 'north', 'south' -> neighbor name or None
    
    Returns:
        Boolean mask for points in border region
    """
    min_x, max_x, min_y, max_y = boundary
    x, y = points[:, 0], points[:, 1]
    
    # Start with no points selected
    in_border = np.zeros(len(points), dtype=bool)
    
    # West border region: inner_dist <= (x - min_x) < outer_dist
    if neighbors.get("west") is not None:
        west_mask = (x >= min_x + inner_dist) & (x < min_x + outer_dist)
        in_border |= west_mask
    
    # East border region: inner_dist <= (max_x - x) < outer_dist
    if neighbors.get("east") is not None:
        east_mask = (x > max_x - outer_dist) & (x <= max_x - inner_dist)
        in_border |= east_mask
    
    # South border region: inner_dist <= (y - min_y) < outer_dist
    if neighbors.get("south") is not None:
        south_mask = (y >= min_y + inner_dist) & (y < min_y + outer_dist)
        in_border |= south_mask
    
    # North border region: inner_dist <= (max_y - y) < outer_dist
    if neighbors.get("north") is not None:
        north_mask = (y > max_y - outer_dist) & (y <= max_y - inner_dist)
        in_border |= north_mask
    
    return in_border


# =============================================================================
# Stage 5: Small Cluster Reassignment / Volume Merging
# =============================================================================


def reassign_small_clusters(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    species_prob: Optional[np.ndarray],
    instance_species_map: Dict[int, int],
    instance_species_prob_map: Dict[int, float],
    instance_sizes: Dict[int, int],
    min_cluster_size: int = 300,
) -> Tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Dict[int, int], Dict[int, float], int
]:
    """
    Reassign small clusters to nearest larger instance by centroid distance.
    Species ID and species_prob are taken from the target (larger) instance.

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs (modified in-place)
        species_ids: Array of species IDs (modified in-place)
        species_prob: Array of species probabilities (modified in-place, optional)
        instance_species_map: Mapping of instance ID to species ID
        instance_species_prob_map: Mapping of instance ID to species probability
        instance_sizes: Mapping of instance ID to point count
        min_cluster_size: Minimum size for a cluster to be kept

    Returns:
        Updated (instances, species_ids, species_prob, instance_species_map, instance_species_prob_map)
    """
    # Get unique instances and their sizes in one pass
    unique_inst, counts = np.unique(instances[instances > 0], return_counts=True)
    inst_counts = dict(zip(unique_inst, counts))

    # Separate small and large instances using pre-computed sizes
    small_instances = []
    large_instances = []

    for inst_id in unique_inst:
        size = instance_sizes.get(inst_id, inst_counts.get(inst_id, 0))
        if size < min_cluster_size:
            small_instances.append(inst_id)
        else:
            large_instances.append(inst_id)

    if len(small_instances) == 0 or len(large_instances) == 0:
        print(f"  No small clusters to reassign")
        return (
            instances,
            species_ids,
            species_prob,
            instance_species_map,
            instance_species_prob_map,
            0,
        )

    print(f"  Found {len(small_instances)} small clusters (<{min_cluster_size} points)")

    # Compute ALL centroids in one vectorized pass
    all_centroids = compute_centroids_vectorized(points, instances)

    # Extract large instance centroids
    large_ids = list(large_instances)
    large_coords = np.array([all_centroids[i] for i in large_ids if i in all_centroids])

    if len(large_coords) == 0:
        print(f"  No large instance centroids found")
        return (
            instances,
            species_ids,
            species_prob,
            instance_species_map,
            instance_species_prob_map,
            0,
        )

    # Build KD-tree from large instance centroids (small tree, fast)
    tree = cKDTree(large_coords)

    # Build lookup table for vectorized reassignment
    max_inst = instances.max() + 1
    inst_to_target = np.arange(max_inst, dtype=np.int32)  # Default: map to self
    inst_to_species = np.zeros(max_inst, dtype=np.int32)
    inst_to_species_prob = np.zeros(max_inst, dtype=np.float32)

    # Fill in species and species_prob for all instances
    for inst_id, species in instance_species_map.items():
        if inst_id < max_inst:
            inst_to_species[inst_id] = species
    for inst_id, prob in instance_species_prob_map.items():
        if inst_id < max_inst:
            inst_to_species_prob[inst_id] = prob

    # Find targets for small instances
    # Optimization: Batch KD-tree queries for better cache efficiency
    if len(small_instances) > 0:
        # Collect valid small instance centroids
        valid_small_instances = []
        valid_centroids = []
        for small_inst in small_instances:
            if small_inst in all_centroids:
                valid_small_instances.append(small_inst)
                valid_centroids.append(all_centroids[small_inst])

        if len(valid_centroids) > 0:
            # Batch query all centroids at once (much faster than individual queries)
            valid_centroids = np.array(valid_centroids)
            distances, indices = tree.query(valid_centroids)

            total_reassigned = 0
            for i, small_inst in enumerate(valid_small_instances):
                distance = distances[i]
                idx = indices[i]
                target_inst = large_ids[idx]
                count = inst_counts.get(small_inst, 0)

                # Store mapping - take species and species_prob from target (larger) instance
                inst_to_target[small_inst] = target_inst
                target_species = instance_species_map.get(target_inst, 0)
                target_species_prob = instance_species_prob_map.get(target_inst, 0.0)
                inst_to_species[small_inst] = target_species
                inst_to_species_prob[small_inst] = target_species_prob

                total_reassigned += count
                print(
                    f"    Cluster {small_inst} ({count} pts) → Instance {target_inst} (dist: {distance:.2f}m)"
                )
        else:
            total_reassigned = 0
    else:
        total_reassigned = 0

    # Vectorized reassignment - single pass over all points!
    valid_mask = (instances > 0) & (instances < max_inst)
    instances[valid_mask] = inst_to_target[instances[valid_mask]]
    species_ids[valid_mask] = inst_to_species[instances[valid_mask]]
    if species_prob is not None:
        species_prob[valid_mask] = inst_to_species_prob[instances[valid_mask]]

    print(
        f"  Reassigned {total_reassigned:,} points from {len(small_instances)} small clusters"
    )

    return (
        instances,
        species_ids,
        species_prob,
        instance_species_map,
        instance_species_prob_map,
        0,
    )


def merge_small_volume_instances(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    species_prob: Optional[np.ndarray],
    instance_species_map: Dict[int, int],
    instance_species_prob_map: Dict[int, float],
    min_points_for_hull_check: int = 1000,
    min_cluster_size: int = 300,
    max_volume_for_merge: float = 4.0,
    max_search_radius: float = 5.0,
    num_threads: int = 1,
    verbose: bool = True,
    # Pre-sorted data (optional) - if provided, skip sorting step
    presorted_points: Optional[np.ndarray] = None,
    presorted_instances: Optional[np.ndarray] = None,
    presorted_unique_inst: Optional[np.ndarray] = None,
    presorted_first_idx: Optional[np.ndarray] = None,
    presorted_inst_counts: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Dict[int, int], Dict[int, float], int
]:
    """
    Merge small-volume instances to nearest large instance by centroid distance.

    Logic:
    - For instances with >= min_points_for_hull_check (1000) points: Keep instance (skip hull computation)
    - For instances with < min_points_for_hull_check (1000) points:
      1. Calculate convex hull volume
      2. If volume < max_volume_for_merge (4.0 m³): Merge to nearest large instance
      3. Else if point_count < min_cluster_size: Redistribute to nearest instance
      4. Else: Keep instance

    Species ID and species_prob are taken from the target (larger) instance.

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs (modified in-place)
        species_ids: Array of species IDs (modified in-place)
        species_prob: Array of species probabilities (modified in-place, optional)
        instance_species_map: Mapping of instance ID to species ID
        instance_species_prob_map: Mapping of instance ID to species probability
        min_points_for_hull_check: Only compute convex hull for instances with fewer points than this (default: 1000)
        min_cluster_size: Redistribute instances with fewer points than this if volume >= threshold (default: 300)
        max_volume_for_merge: Merge instances with convex hull volume below this (m³) (default: 4.0)
        max_search_radius: Max distance to search for target instance (m) (default: 5.0)
        num_threads: Number of workers for parallel hull computation (default: 1)
        verbose: Print detailed decisions (default: True)
        presorted_points: Pre-sorted points array (optional, to avoid redundant sorting)
        presorted_instances: Pre-sorted instances array (optional)
        presorted_unique_inst: Pre-computed unique instances (optional)
        presorted_first_idx: Pre-computed first indices (optional)
        presorted_inst_counts: Pre-computed instance counts (optional)

    Returns:
        Updated (instances, species_ids, species_prob, instance_species_map, instance_species_prob_map, bbox_skipped_count)
    """
    import sys
    
    # Check if pre-sorted data is provided
    use_presorted = (
        presorted_points is not None and
        presorted_instances is not None and
        presorted_unique_inst is not None and
        presorted_first_idx is not None and
        presorted_inst_counts is not None
    )
    
    if use_presorted:
        print(f"  Using pre-sorted data (skipping sort step)...", flush=True)
        sys.stdout.flush()
        sorted_points = presorted_points
        sorted_instances = presorted_instances
        unique_inst = presorted_unique_inst
        first_idx = presorted_first_idx
        inst_counts = presorted_inst_counts
        print(f"  {len(unique_inst):,} unique instances from pre-sorted data.", flush=True)
        sys.stdout.flush()
    else:
        # OPTIMIZATION: Filter out instance 0 BEFORE sorting
        # Instance 0 is unlabeled/ground and never processed, but often 50-80% of points
        total_points = len(instances)
        print(f"  Total points: {total_points:,}", flush=True)
        sys.stdout.flush()
        
        nonzero_mask = instances > 0
        nonzero_count = nonzero_mask.sum()
        zero_count = total_points - nonzero_count
        
        print(f"  Filtering out {zero_count:,} unlabeled points (instance 0, {100*zero_count/total_points:.1f}%)...", flush=True)
        print(f"  Points to process: {nonzero_count:,} ({100*nonzero_count/total_points:.1f}%)", flush=True)
        sys.stdout.flush()
        
        # Extract only non-zero instance points for sorting (much faster than sorting all)
        instances_to_sort = instances[nonzero_mask]
        points_to_sort = points[nonzero_mask]
        
        # Sort only the non-zero instance points - O(n log n) on much smaller n
        print(f"  Sorting {len(instances_to_sort):,} points by instance ID...", flush=True)
        sys.stdout.flush()
        sort_idx = np.argsort(instances_to_sort)
        sorted_instances = instances_to_sort[sort_idx]
        sorted_points = points_to_sort[sort_idx]
        print(f"  Sorting complete.", flush=True)
        sys.stdout.flush()

        # Get unique instances with boundaries (already filtered, no need for pos_mask)
        print(f"  Finding unique instances...", flush=True)
        sys.stdout.flush()
        unique_inst, first_idx, inst_counts = np.unique(
            sorted_instances, return_index=True, return_counts=True
        )
        print(f"  Found {len(unique_inst):,} unique instances.", flush=True)
        sys.stdout.flush()

    # Categorize instances and compute volumes using sorted slices
    # Optimization: Only compute convex hull for instances < min_points_for_hull_check (1000 points)
    # Optimization: Only compute centroids for instances < 1000 points after bbox filtering
    # First pass: collect candidates that need hull computation (after bbox filter)
    hull_candidates = []  # (inst_id, count, start, end, bbox_volume) - no centroid/points yet
    small_volume_instances = []  # (inst_id, point_count, volume, centroid)
    small_point_count_instances = []  # (inst_id, point_count, centroid) - for redistribution
    large_instances = []  # (inst_id, count, centroid)
    bbox_skipped_count = 0  # Track instances skipped by bounding box filter

    total_instances = len(unique_inst)
    print(f"  Categorizing {total_instances:,} instances (checking bbox volumes)...", flush=True)

    for idx, (inst_id, start, count) in enumerate(zip(unique_inst, first_idx, inst_counts)):
        count = int(count)

        # Progress print every 500 instances for better feedback
        if idx % 500 == 0 or idx == total_instances - 1:
            progress = (idx + 1) * 100.0 / total_instances
            print(f"    Categorizing: {idx + 1:,}/{total_instances:,} instances ({progress:.1f}%)...", flush=True)

        # Skip large instances (>= min_points_for_hull_check) - they're kept (no hull computation)
        if count >= min_points_for_hull_check:
            end = start + count
            centroid = sorted_points[start:end].mean(axis=0)
            large_instances.append((inst_id, count, centroid))
            continue

        # For instances < 1000 points: compute bbox volume without extracting points array
        # This avoids memory copies for instances that will be filtered out
        end = start + count
        # Compute bbox volume directly from slice (no need to copy points array)
        bbox_volume = np.prod(
            sorted_points[start:end].max(axis=0) - sorted_points[start:end].min(axis=0)
        )

        # If bounding box volume is already too large, skip convex hull computation
        # But still check point count - sparse large-bbox instances should be redistributed
        if (
            bbox_volume >= max_volume_for_merge * 4.0
        ):  # Conservative threshold (bbox >= 4x target)
            centroid = sorted_points[start:end].mean(axis=0)
            
            # Check if instance has too few points - redistribute sparse noise
            if count < min_cluster_size:
                small_point_count_instances.append((inst_id, count, centroid))
                bbox_skipped_count += 1
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, bbox {bbox_volume:.2f} m³ - REDISTRIBUTE (sparse, < {min_cluster_size} pts)"
                    )
            else:
                # Large bbox with enough points - keep as large instance
                large_instances.append((inst_id, count, centroid))
                bbox_skipped_count += 1
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, bbox {bbox_volume:.2f} m³ - keeping (bbox too large, enough points)"
                    )
            continue

        # Collect candidates for hull computation (< min_points_for_hull_check, passed bbox filter)
        # Store indices instead of points/centroid to avoid memory overhead
        # We'll compute centroids only for instances that pass hull computation
        hull_candidates.append((inst_id, count, start, end, bbox_volume))

    if verbose:
        print(f"  Categorized instances: {len(large_instances):,} large, {bbox_skipped_count:,} skipped (large bbox), {len(hull_candidates):,} need hull computation", flush=True)

    # Parallel convex hull computation for candidates using --workers (num_threads)
    # Now compute centroids and extract points only for hull candidates
    if len(hull_candidates) > 0:
        if verbose:
            print(f"  Computing centroids and convex hulls for {len(hull_candidates):,} instances (< {min_points_for_hull_check} points)...", flush=True)
            if num_threads > 1:
                print(f"    Using {num_threads} workers (--workers={num_threads}) for parallel processing...", flush=True)
        
        # Extract points and compute centroids only for hull candidates
        # This avoids memory overhead for instances filtered by bbox
        hull_args = []
        hull_centroids = []
        for inst_id, count, start, end, bbox_volume in hull_candidates:
            pts = sorted_points[start:end]  # Only extract points for hull candidates
            centroid = pts.mean(axis=0)  # Only compute centroids for hull candidates
            hull_args.append((pts, bbox_volume))
            hull_centroids.append(centroid)
        
        # Use --workers (num_threads) for parallelization
        # Parallelize if num_threads > 1 and we have enough candidates
        use_parallel = num_threads > 1 and len(hull_candidates) > 10
        if use_parallel:
            # Parallel computation using ProcessPoolExecutor with progress updates
            batch_size = max(100, len(hull_args) // 20)  # ~20 progress updates
            hull_results = []
            
            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                for batch_idx in range(0, len(hull_args), batch_size):
                    batch = hull_args[batch_idx:batch_idx + batch_size]
                    batch_results = list(executor.map(_compute_hull_wrapper, batch))
                    hull_results.extend(batch_results)
                    
                    if verbose:
                        progress = min(100.0, (len(hull_results) * 100.0 / len(hull_candidates)))
                        print(f"    Hull progress: {len(hull_results):,}/{len(hull_candidates):,} ({progress:.1f}%)...", flush=True)
        else:
            # Sequential computation for small batches or single thread with progress updates
            if verbose and num_threads == 1:
                print(f"    Using sequential computation (--workers=1 or <10 candidates)...", flush=True)
            hull_results = []
            for idx, args in enumerate(hull_args):
                result = _compute_hull_wrapper(args)
                hull_results.append(result)
                
                if verbose and (idx % 100 == 0 or idx == len(hull_args) - 1):
                    progress = (idx + 1) * 100.0 / len(hull_candidates)
                    print(f"    Hull progress: {idx + 1:,}/{len(hull_candidates):,} ({progress:.1f}%)...", flush=True)
        
        # Process hull computation results
        if verbose:
            print(f"  Processing hull results and categorizing instances...", flush=True)
            
        for (inst_id, count, start, end, bbox_volume), (volume, hull_success), centroid in zip(hull_candidates, hull_results, hull_centroids):
            if verbose and not hull_success:
                print(f"    Instance {inst_id}: hull computation failed, using bbox volume")
            
            # Check volume threshold
            if volume < max_volume_for_merge:
                # Small volume - merge to nearest instance
                small_volume_instances.append((inst_id, count, volume, centroid))
                if verbose:
                    print(
                        f"    Instance {inst_id}: {count} pts, {volume:.2f} m³ - SMALL (< {max_volume_for_merge} m³) - merge"
                    )
            else:
                # Volume >= threshold - check point count for redistribution
                if count < min_cluster_size:
                    # Small point count - redistribute to nearest instance
                    small_point_count_instances.append((inst_id, count, centroid))
                    if verbose:
                        print(
                            f"    Instance {inst_id}: {count} pts, {volume:.2f} m³ - REDISTRIBUTE (< {min_cluster_size} pts)"
                        )
                else:
                    # Keep instance
                    large_instances.append((inst_id, count, centroid))
                    if verbose:
                        print(
                            f"    Instance {inst_id}: {count} pts, {volume:.2f} m³ - keeping (volume ok, enough points)"
                        )

    # Combine small volume and small point count instances for merging/redistribution
    all_small_instances = small_volume_instances + [(inst_id, count, 0.0, centroid) 
                                                     for inst_id, count, centroid in small_point_count_instances]
    
    if len(all_small_instances) == 0:
        print(f"  No small instances to merge/redistribute", flush=True)
        if bbox_skipped_count > 0:
            print(
                f"  Skipped {bbox_skipped_count} instances using bounding box filter (bbox >= {max_volume_for_merge * 4.0:.1f} m³)", flush=True
            )
        return (
            instances,
            species_ids,
            species_prob,
            instance_species_map,
            instance_species_prob_map,
            bbox_skipped_count,
        )

    if len(large_instances) == 0:
        print(f"  No large instances to merge into", flush=True)
        if bbox_skipped_count > 0:
            print(
                f"  Skipped {bbox_skipped_count} instances using bounding box filter (bbox >= {max_volume_for_merge * 4.0:.1f} m³)", flush=True
            )
        return (
            instances,
            species_ids,
            species_prob,
            instance_species_map,
            instance_species_prob_map,
            bbox_skipped_count,
        )

    print(
        f"  Found {len(small_volume_instances)} small-volume instances (< {max_volume_for_merge} m³) to merge", flush=True
    )
    if len(small_point_count_instances) > 0:
        print(
            f"  Found {len(small_point_count_instances)} small point-count instances (< {min_cluster_size} pts, volume >= {max_volume_for_merge} m³) to redistribute", flush=True
        )
    if bbox_skipped_count > 0:
        print(
            f"  Skipped {bbox_skipped_count} instances using bounding box filter (bbox >= {max_volume_for_merge * 4.0:.1f} m³) - saved convex hull computation", flush=True
        )

    # Build KD-tree from large instance centroids (already computed)
    large_ids = [x[0] for x in large_instances]
    large_sizes = {x[0]: x[1] for x in large_instances}
    large_coords = np.array([x[2] for x in large_instances])
    tree = cKDTree(large_coords)

    # Build lookup table for vectorized reassignment
    max_inst = instances.max() + 1
    inst_to_target = np.arange(max_inst, dtype=np.int32)  # Default: map to self
    inst_to_species = np.zeros(max_inst, dtype=np.int32)
    inst_to_species_prob = np.zeros(max_inst, dtype=np.float32)

    # Fill in species and species_prob for all instances
    for inst_id, species in instance_species_map.items():
        if inst_id < max_inst:
            inst_to_species[inst_id] = species
    for inst_id, prob in instance_species_prob_map.items():
        if inst_id < max_inst:
            inst_to_species_prob[inst_id] = prob

    # Determine targets for small instances (volume-based merge and point-count-based redistribute)
    # Optimization: Batch KD-tree queries for better cache efficiency
    if len(all_small_instances) > 0:
        small_centroids = np.array(
            [centroid for _, _, _, centroid in all_small_instances]
        )
        distances, indices = tree.query(small_centroids)

        # Process results
        total_merged = 0
        for i, (inst_id, count, volume, centroid) in enumerate(all_small_instances):
            distance = distances[i]
            idx = indices[i]

            if distance > max_search_radius:
                if verbose:
                    if volume > 0:  # Small volume instance
                        print(
                            f"    Cluster {inst_id}: {count} pts, {volume:.2f} m³ → no target within {max_search_radius}m (nearest: {distance:.1f}m)"
                        )
                    else:  # Small point count instance
                        print(
                            f"    Cluster {inst_id}: {count} pts → no target within {max_search_radius}m (nearest: {distance:.1f}m)"
                        )
                continue

            target_inst = large_ids[idx]

            # Store in lookup table - take species and species_prob from target (larger) instance
            inst_to_target[inst_id] = target_inst
            target_species = instance_species_map.get(target_inst, 0)
            target_species_prob = instance_species_prob_map.get(target_inst, 0.0)
            inst_to_species[inst_id] = target_species
            inst_to_species_prob[inst_id] = target_species_prob

            total_merged += count
            if volume > 0:  # Small volume instance
                print(
                    f"    Cluster {inst_id} ({count} pts, {volume:.2f} m³) → Instance {target_inst} ({large_sizes[target_inst]} pts, dist: {distance:.1f}m)"
                )
            else:  # Small point count instance
                print(
                    f"    Cluster {inst_id} ({count} pts) → Instance {target_inst} ({large_sizes[target_inst]} pts, dist: {distance:.1f}m) - redistributed"
                )
    else:
        total_merged = 0

    # Vectorized reassignment - single pass over all points!
    valid_mask = (instances > 0) & (instances < max_inst)
    instances[valid_mask] = inst_to_target[instances[valid_mask]]
    species_ids[valid_mask] = inst_to_species[instances[valid_mask]]
    if species_prob is not None:
        species_prob[valid_mask] = inst_to_species_prob[instances[valid_mask]]

    print(
        f"  Merged/redistributed {total_merged:,} points from {len(all_small_instances)} small instances "
        f"({len(small_volume_instances)} small-volume + {len(small_point_count_instances)} small point-count)", flush=True
    )

    return (
        instances,
        species_ids,
        species_prob,
        instance_species_map,
        instance_species_prob_map,
        bbox_skipped_count,
    )


# =============================================================================
# Load Existing Merged File
# =============================================================================


def load_merged_file(
    merged_file: Path, chunk_size: int = 1_000_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Load merged point cloud data from an existing LAZ file.

    Args:
        merged_file: Path to the merged LAZ file
        chunk_size: Number of points to read per chunk (default 1M)

    Returns:
        Tuple of (points, instances, species_ids, has_species_id) as numpy arrays and bool
    """
    print(f"Loading existing merged file: {merged_file}")

    try:
        with laspy.open(
            str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel
        ) as f:
            n_points = f.header.point_count

            # Check which extra dimensions are available
            extra_dims = {dim.name for dim in f.header.point_format.extra_dimensions}
            has_pred_instance = "PredInstance" in extra_dims
            has_tree_id = "treeID" in extra_dims
            has_species_id = "species_id" in extra_dims

            if not has_pred_instance and not has_tree_id:
                raise ValueError(f"No instance attribute found in {merged_file}")

            # Pre-allocate arrays
            points = np.empty((n_points, 3), dtype=np.float64)
            instances = np.zeros(n_points, dtype=np.int32)
            species_ids = np.zeros(n_points, dtype=np.int32)

            # Read in chunks to reduce peak memory
            offset = 0
            for chunk in f.chunk_iterator(chunk_size):
                chunk_len = len(chunk)
                end = offset + chunk_len

                points[offset:end, 0] = chunk.x
                points[offset:end, 1] = chunk.y
                points[offset:end, 2] = chunk.z

                # Get instance IDs
                if has_pred_instance:
                    instances[offset:end] = chunk.PredInstance
                elif has_tree_id:
                    instances[offset:end] = chunk.treeID

                # Get species IDs
                if has_species_id:
                    species_ids[offset:end] = chunk.species_id

                offset = end

        unique_instances = len(np.unique(instances[instances > 0]))
        print(f"  Loaded {len(points):,} points, {unique_instances} unique instances")
        if has_species_id:
            print(f"  Species ID: Found in merged file")
        else:
            print(f"  Species ID: Not found in merged file")

        return points, instances, species_ids, has_species_id

    except Exception as e:
        raise ValueError(f"Error loading merged file {merged_file}: {e}")


# =============================================================================
# Stage 6: Retile to Original Files
# =============================================================================


def _process_single_tile(args):
    """
    Process a single tile for retiling. Designed to be called in parallel.
    
    Args:
        args: Tuple of (orig_file, output_file, merged_points, merged_instances, 
              merged_species_ids, tolerance, spatial_buffer, num_threads, all_have_species_id)
    
    Returns:
        Tuple of (filename, matched_count, total_count, unique_instances, success, message)
    """
    (orig_file, output_file, merged_points, merged_instances, merged_species_ids,
     tolerance, spatial_buffer, kdtree_workers, all_have_species_id) = args
    
    try:
        # Get tile bounds from header without loading all points
        with laspy.open(
            str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel
        ) as f:
            bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
            n_orig_points = f.header.point_count

        # Filter merged points to this tile's region + buffer
        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )

        local_merged_points = merged_points[mask]
        local_merged_instances = merged_instances[mask]
        local_merged_species = merged_species_ids[mask]

        if len(local_merged_points) == 0:
            return (orig_file.name, 0, n_orig_points, 0, False, "No merged points in tile region")

        # Build small local KDTree
        local_tree = cKDTree(local_merged_points)

        # Load and process original tile
        orig_las = laspy.read(
            str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel
        )
        orig_points = np.empty((n_orig_points, 3), dtype=np.float64)
        orig_points[:, 0] = orig_las.x
        orig_points[:, 1] = orig_las.y
        orig_points[:, 2] = orig_las.z

        # Query local tree with specified number of KDTree workers
        distances, indices = local_tree.query(orig_points, workers=kdtree_workers)

        # Create new instance and species arrays
        new_instances = np.zeros(n_orig_points, dtype=np.int32)
        new_species = np.zeros(n_orig_points, dtype=np.int32)

        # Copy from merged where distance is within tolerance
        valid_mask = distances < tolerance
        new_instances[valid_mask] = local_merged_instances[indices[valid_mask]]
        new_species[valid_mask] = local_merged_species[indices[valid_mask]]

        # Add/update extra dimensions
        extra_dims = {dim.name for dim in orig_las.point_format.extra_dimensions}

        if "PredInstance" not in extra_dims:
            orig_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
            )
        orig_las.PredInstance = new_instances

        # Only add species_id if it was present in all input files
        if all_have_species_id:
            if "species_id" not in extra_dims:
                orig_las.add_extra_dim(
                    laspy.ExtraBytesParams(name="species_id", type=np.int32)
                )
            orig_las.species_id = new_species

        # Save to output directory
        orig_las.write(
            str(output_file),
            do_compress=True,
            laz_backend=laspy.LazBackend.LazrsParallel,
        )

        matched = int(np.sum(valid_mask))
        unique_inst = len(np.unique(new_instances[new_instances > 0]))
        
        return (orig_file.name, matched, n_orig_points, unique_inst, True, "OK")
    
    except Exception as e:
        return (orig_file.name, 0, 0, 0, False, str(e))


def retile_to_original_files(
    merged_points: np.ndarray,
    merged_instances: np.ndarray,
    merged_species_ids: np.ndarray,
    original_tiles_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
    chunk_size: int = 1_000_000,
    all_have_species_id: bool = True,
    parallel_tiles: int = 1,
):
    """
    Map merged instance IDs back to original tile point clouds.

    Uses spatial partitioning for memory efficiency:
    - For each tile, filter merged points to tile bounds + buffer
    - Build small KDTree for just that region instead of global tree
    - Use chunked reading/writing for large tiles

    For each original tile:
    1. Get tile bounds from header
    2. Filter merged points to tile region
    3. Build local KDTree
    4. Load original points in chunks
    5. Query and update with instances
    6. Save updated tile
    
    Args:
        merged_points: Merged point cloud coordinates
        merged_instances: Merged instance IDs
        merged_species_ids: Merged species IDs
        original_tiles_dir: Directory containing original tile files
        output_dir: Directory to write output files
        tolerance: Distance tolerance for point matching
        num_threads: Number of threads for KDTree queries (per tile)
        chunk_size: Chunk size for reading large files
        all_have_species_id: Whether to include species_id in output
        parallel_tiles: Number of tiles to process in parallel (default: 1 = sequential)
    """
    import gc
    from concurrent.futures import ThreadPoolExecutor
    import sys

    print(f"\n{'=' * 60}", flush=True)
    print("Retiling merged results to original files (spatial partitioning)", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.stdout.flush()

    # Find all LAZ files in original directory
    original_files = sorted(original_tiles_dir.glob("*.laz"))
    if not original_files:
        original_files = sorted(original_tiles_dir.glob("*.las"))

    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_tiles_dir}", flush=True)
        return

    print(f"  Found {len(original_files)} original tile files", flush=True)
    print(f"  Parallel tiles: {parallel_tiles}, KDTree workers per tile: {num_threads}", flush=True)
    sys.stdout.flush()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spatial buffer for filtering merged points (slightly larger than tolerance)
    spatial_buffer = max(tolerance * 2, 1.0)

    # Filter out already-processed files
    tiles_to_process = []
    for orig_file in original_files:
        output_file = output_dir / orig_file.name
        if output_file.exists():
            print(f"  Skipping {orig_file.name} (output already exists)", flush=True)
        else:
            tiles_to_process.append((orig_file, output_file))
    
    if len(tiles_to_process) == 0:
        print(f"  All tiles already processed!", flush=True)
        return
    
    print(f"  Processing {len(tiles_to_process)} tiles...", flush=True)
    sys.stdout.flush()

    # Prepare arguments for parallel processing
    # Use -1 for KDTree workers = use all available CPUs
    kdtree_workers = -1
    print(f"  KDTree workers: {kdtree_workers} (all available CPUs)", flush=True)
    
    process_args = [
        (orig_file, output_file, merged_points, merged_instances, merged_species_ids,
         tolerance, spatial_buffer, kdtree_workers, all_have_species_id)
        for orig_file, output_file in tiles_to_process
    ]

    # Process tiles (parallel or sequential)
    if parallel_tiles > 1:
        print(f"\n  Starting parallel processing ({parallel_tiles} tiles at a time)...", flush=True)
        sys.stdout.flush()
        
        completed = 0
        with ThreadPoolExecutor(max_workers=parallel_tiles) as executor:
            for result in executor.map(_process_single_tile, process_args):
                filename, matched, total, unique_inst, success, message = result
                completed += 1
                
                if success:
                    print(f"  [{completed}/{len(tiles_to_process)}] {filename}: {matched:,}/{total:,} matched, {unique_inst} instances", flush=True)
                else:
                    print(f"  [{completed}/{len(tiles_to_process)}] {filename}: FAILED - {message}", flush=True)
                sys.stdout.flush()
    else:
        # Sequential processing with detailed output
        for i, args in enumerate(process_args):
            orig_file = args[0]
            print(f"\n  [{i+1}/{len(tiles_to_process)}] Processing {orig_file.name}...", flush=True)
            sys.stdout.flush()
            
            result = _process_single_tile(args)
            filename, matched, total, unique_inst, success, message = result
            
            if success:
                print(f"    {matched:,}/{total:,} points matched, {unique_inst} instances → {filename}", flush=True)
            else:
                print(f"    FAILED: {message}", flush=True)
            sys.stdout.flush()
            
            # Clean up after each tile in sequential mode
            gc.collect()

    print(f"\n  Retiling complete: {len(tiles_to_process)} tiles processed", flush=True)
    gc.collect()


# =============================================================================
# Stage 7: Remap to Original Input Files
# =============================================================================


def _process_single_original_input_file(args):
    """
    Process a single original input LAZ file for remapping. Designed to be called in parallel.
    
    Args:
        args: Tuple of (input_file, output_file, merged_points, merged_instances, 
              merged_species_ids, tolerance, spatial_buffer, kdtree_workers, all_have_species_id)
    
    Returns:
        Tuple of (filename, matched_count, total_count, unique_instances, success, message)
    """
    (input_file, output_file, merged_points, merged_instances, merged_species_ids,
     tolerance, spatial_buffer, kdtree_workers, all_have_species_id) = args
    
    try:
        # Get file bounds from header without loading all points
        with laspy.open(
            str(input_file), laz_backend=laspy.LazBackend.LazrsParallel
        ) as f:
            bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
            n_input_points = f.header.point_count

        # Filter merged points to this file's region + buffer
        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )

        local_merged_points = merged_points[mask]
        local_merged_instances = merged_instances[mask]
        local_merged_species = merged_species_ids[mask]

        if len(local_merged_points) == 0:
            return (input_file.name, 0, n_input_points, 0, False, "No merged points in file region")

        # Build small local KDTree from merged points
        local_tree = cKDTree(local_merged_points)

        # Load original input file
        input_las = laspy.read(
            str(input_file), laz_backend=laspy.LazBackend.LazrsParallel
        )
        input_points = np.empty((n_input_points, 3), dtype=np.float64)
        input_points[:, 0] = input_las.x
        input_points[:, 1] = input_las.y
        input_points[:, 2] = input_las.z

        # Query local tree with specified number of KDTree workers
        distances, indices = local_tree.query(input_points, workers=kdtree_workers)

        # Create new instance and species arrays
        new_instances = np.zeros(n_input_points, dtype=np.int32)
        new_species = np.zeros(n_input_points, dtype=np.int32)

        # Copy from merged where distance is within tolerance
        valid_mask = distances < tolerance
        matched_count = np.sum(valid_mask)
        new_instances[valid_mask] = local_merged_instances[indices[valid_mask]]
        new_species[valid_mask] = local_merged_species[indices[valid_mask]]

        # Count unique non-zero instances
        unique_instances = len(np.unique(new_instances[new_instances > 0]))

        # Add/update extra dimensions
        extra_dims = {dim.name for dim in input_las.point_format.extra_dimensions}

        if "PredInstance" not in extra_dims:
            input_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
            )
        input_las.PredInstance = new_instances

        # Only add species_id if it was present in all input files
        if all_have_species_id:
            if "species_id" not in extra_dims:
                input_las.add_extra_dim(
                    laspy.ExtraBytesParams(name="species_id", type=np.int32)
                )
            input_las.species_id = new_species

        # Save to output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        input_las.write(
            str(output_file),
            do_compress=True,
            laz_backend=laspy.LazBackend.LazrsParallel,
        )

        return (input_file.name, matched_count, n_input_points, unique_instances, True, "Success")

    except Exception as e:
        return (input_file.name, 0, 0, 0, False, str(e))


def remap_to_original_input_files(
    merged_points: np.ndarray,
    merged_instances: np.ndarray,
    merged_species_ids: np.ndarray,
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
    all_have_species_id: bool = True,
):
    """
    Map merged instance IDs back to original input LAZ files (pre-tiling).
    
    This is the final stage of the pipeline that transfers PredInstance labels
    from the merged/processed point cloud back to the original input files,
    ensuring ALL original points have the new dimension.

    Uses cKDTree for efficient nearest-neighbor lookup:
    - For each original input file, filter merged points to file bounds + buffer
    - Build local cKDTree from filtered merged points
    - Query nearest neighbor for all original points
    - Add PredInstance dimension with queried values
    - Points with distance > tolerance get PredInstance=0 (unmatched)

    Args:
        merged_points: Merged point cloud coordinates (N, 3)
        merged_instances: Merged instance IDs (N,)
        merged_species_ids: Merged species IDs (N,)
        original_input_dir: Directory containing original input LAZ files (pre-tiling)
        output_dir: Directory to write output files with PredInstance
        tolerance: Distance tolerance for point matching (default: 0.1m)
        num_threads: Number of threads for KDTree queries (default: 8)
        all_have_species_id: Whether to include species_id in output
    """
    from concurrent.futures import ThreadPoolExecutor
    
    print(f"\n{'=' * 60}", flush=True)
    print("Stage 7: Remapping to Original Input Files", flush=True)
    print(f"{'=' * 60}", flush=True)
    sys.stdout.flush()

    # Find all LAZ files in original input directory
    original_files = sorted(original_input_dir.glob("*.laz"))
    if not original_files:
        original_files = sorted(original_input_dir.glob("*.las"))

    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_input_dir}", flush=True)
        return
    
    # Skip COPC files (these are intermediate, not original inputs)
    original_files = [f for f in original_files if not f.name.endswith('.copc.laz')]
    
    if len(original_files) == 0:
        print(f"  No original input files found (only COPC files present)", flush=True)
        return

    print(f"  Found {len(original_files)} original input files", flush=True)
    print(f"  Output directory: {output_dir}", flush=True)
    print(f"  Tolerance: {tolerance}m", flush=True)
    sys.stdout.flush()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spatial buffer for filtering merged points (slightly larger than tolerance)
    spatial_buffer = max(tolerance * 2, 1.0)

    # Filter out already-processed files
    files_to_process = []
    for input_file in original_files:
        output_file = output_dir / input_file.name
        if output_file.exists():
            print(f"  Skipping {input_file.name} (output already exists)", flush=True)
        else:
            files_to_process.append((input_file, output_file))
    
    if len(files_to_process) == 0:
        print(f"  All files already processed!", flush=True)
        return
    
    print(f"  Processing {len(files_to_process)} files...", flush=True)
    sys.stdout.flush()

    # Use -1 for KDTree workers = use all available CPUs
    kdtree_workers = -1
    
    process_args = [
        (input_file, output_file, merged_points, merged_instances, merged_species_ids,
         tolerance, spatial_buffer, kdtree_workers, all_have_species_id)
        for input_file, output_file in files_to_process
    ]

    # Process files sequentially for stability with large files
    total_matched = 0
    total_points = 0
    
    for i, args in enumerate(process_args):
        input_file = args[0]
        print(f"\n  [{i+1}/{len(files_to_process)}] Processing {input_file.name}...", flush=True)
        sys.stdout.flush()
        
        result = _process_single_original_input_file(args)
        filename, matched, total, unique_inst, success, message = result
        
        if success:
            match_pct = (matched / total * 100) if total > 0 else 0
            print(f"    {matched:,}/{total:,} points matched ({match_pct:.1f}%), {unique_inst} instances", flush=True)
            total_matched += matched
            total_points += total
        else:
            print(f"    FAILED: {message}", flush=True)
        sys.stdout.flush()
        
        # Clean up after each file
        gc.collect()

    # Summary
    overall_match_pct = (total_matched / total_points * 100) if total_points > 0 else 0
    print(f"\n  Remap to original files complete:", flush=True)
    print(f"    Files processed: {len(files_to_process)}", flush=True)
    print(f"    Total points: {total_points:,}", flush=True)
    print(f"    Total matched: {total_matched:,} ({overall_match_pct:.1f}%)", flush=True)
    print(f"    Output: {output_dir}", flush=True)
    gc.collect()


# =============================================================================
# Main Merge Function
# =============================================================================


def merge_tiles(
    input_dir: Path,
    original_tiles_dir: Path,
    output_merged: Path,
    output_tiles_dir: Path,
    original_input_dir: Optional[Path] = None,
    buffer: float = 10.0,
    overlap_threshold: float = 0.3,
    max_centroid_distance: float = 3.0,
    correspondence_tolerance: float = 0.05,
    max_volume_for_merge: float = 4.0,
    min_cluster_size: int = 300,
    num_threads: int = 8,
    enable_matching: bool = True,
    require_overlap: bool = True,
    enable_volume_merge: bool = True,
    skip_merged_file: bool = False,
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
    print(f"Output merged: {output_merged}" + (" (SKIPPED)" if skip_merged_file else ""))
    print(f"Output tiles: {output_tiles_dir}")
    print(f"Buffer: {buffer}m")
    print(f"Instance matching: {'ENABLED' if enable_matching else 'DISABLED'}")
    if enable_matching:
        if require_overlap:
            print(f"  Overlap threshold: {overlap_threshold}")
        else:
            print(f"  Overlap check: DISABLED (centroid distance only)")
            print(f"  Max centroid distance: {max_centroid_distance}m")
        print(f"Small cluster reassignment: ENABLED")
    print(f"  Min cluster size: {min_cluster_size} points")
    print(f"Volume merge: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    if enable_volume_merge:
        print(f"  Max volume for merge: {max_volume_for_merge} m³")
        print(f"Verbose: {verbose}")
    print("=" * 60)

    # Check if merged output file already exists
    if output_merged.exists():
        print(f"\n{'=' * 60}")
        print(f"Merged file already exists: {output_merged}")
        print(f"{'=' * 60}")
        print("  Loading merged file and proceeding to retiling stage...")
        
        merged_points, merged_instances, merged_species_ids, all_have_species_id = load_merged_file(
            output_merged
        )

        # Proceed directly to retiling (required)
        retile_to_original_files(
            merged_points,
            merged_instances,
            merged_species_ids,
            original_tiles_dir,
            output_tiles_dir,
            tolerance=0.1,
            num_threads=num_threads,
            all_have_species_id=all_have_species_id,
        )

        print(f"\n{'=' * 60}")
        print("Retiling complete!")
        print(f"{'=' * 60}")
        return

    # Find all input LAZ files
    laz_files = sorted(input_dir.glob("*.laz"))
    if not laz_files:
        laz_files = sorted(input_dir.glob("*.las"))

    if len(laz_files) == 0:
        print(f"No LAZ/LAS files found in {input_dir}")
        return

    print(f"\nFound {len(laz_files)} tiles to merge")

    # Extract tile names and get bounds from headers for neighbor detection
    print("  Extracting tile bounds from headers...")
    tile_boundaries = {}
    tile_names = []
    
    for f in laz_files:
        name = f.stem
        for suffix in ["_segmented_remapped", "_segmented", "_remapped"]:
            name = name.replace(suffix, "")
        tile_names.append(name)
        
        bounds = get_tile_bounds_from_header(f)
        if bounds:
            tile_boundaries[name] = bounds
        else:
            print(f"    Warning: Could not extract bounds from {f.name}")
    
    if len(tile_boundaries) == 0:
        print("Error: Could not extract bounds from any tile files")
        return
    
    print(f"  Extracted bounds from {len(tile_boundaries)} tiles")

    # =========================================================================
    # Stage 1: Load and Filter
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 1: Loading tiles and filtering buffer zone instances")
    print(f"{'=' * 60}")
    print(f"  Loading {len(laz_files)} files using {num_threads} processes...")

    tiles = []
    filtered_instances_per_tile = {}
    kept_instances_per_tile = {}

    # Load tiles in parallel using ProcessPoolExecutor for true CPU parallelism
    # Prepare arguments for multiprocessing (must be pickleable)
    load_args = [(f, tile_boundaries, buffer) for f in laz_files]

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(_load_tile_wrapper, load_args))

    # Process results
    buffer_direction_per_tile = {}  # tile_name -> {inst_id -> direction}
    for result in results:
        if result is not None:
            tile_data, filtered, kept, buffer_dirs = result
            tiles.append(tile_data)
            filtered_instances_per_tile[tile_data.name] = filtered
            kept_instances_per_tile[tile_data.name] = kept
            buffer_direction_per_tile[tile_data.name] = buffer_dirs

    # Clean up loading intermediates
    del results
    gc.collect()

    if len(tiles) == 0:
        print("No tiles loaded successfully")
        return
    
    # Check if ALL tiles have species_id (only include if present in all input files)
    all_have_species_id = all(tile.has_species_id for tile in tiles)
    if all_have_species_id:
        print(f"  Species ID: Found in all input files (will be included in output)")
    else:
        tiles_with = sum(1 for tile in tiles if tile.has_species_id)
        print(f"  Species ID: Only found in {tiles_with}/{len(tiles)} input files (will be omitted from output)")
    
    total_points = sum(len(tile.points) for tile in tiles)
    total_kept = sum(len(kept) for kept in kept_instances_per_tile.values())
    total_filtered = sum(len(filtered) for filtered in filtered_instances_per_tile.values())
    print(f"  ✓ Stage 1 completed: {len(tiles)} tiles loaded, {total_points:,} total points")
    print(f"    Kept {total_kept} instances, filtered {total_filtered} buffer zone instances")

    # =========================================================================
    # Assign global instance IDs and track species
    # =========================================================================
    # =========================================================================
    # Stage 2: Assign Global Instance IDs
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 2: Assigning global instance IDs")
    print(f"{'=' * 60}")

    TILE_OFFSET = 100000  # Unique global ID: tile_idx * OFFSET + local_id

    def global_id(tile_idx: int, local_id: int) -> int:
        return tile_idx * TILE_OFFSET + local_id

    def local_id(gid: int) -> Tuple[int, int]:
        return gid // TILE_OFFSET, gid % TILE_OFFSET

    # Initialize Union-Find and track species per global instance
    uf = UnionFind()
    instance_species_map = {}  # global_id -> species_id
    instance_species_prob_map = {}  # global_id -> species_prob (mean value)
    instance_sizes = {}  # global_id -> point count

    for tile_idx, tile in enumerate(tiles):
        print(f"  Processing tile {tile_idx + 1}/{len(tiles)}: {tile.name} ({len(tile.points):,} points)...")
        kept_instances = kept_instances_per_tile[tile.name]

        # Single pass: sort by instance for efficient grouping
        sort_idx = np.argsort(tile.instances)
        sorted_inst = tile.instances[sort_idx]
        sorted_species = tile.species_ids[sort_idx]
        sorted_species_prob = None
        if tile.species_prob is not None:
            sorted_species_prob = tile.species_prob[sort_idx]

        # Get unique instances with their positions and counts
        unique_inst, first_idx, inst_counts = np.unique(
            sorted_inst, return_index=True, return_counts=True
        )

        # Process each unique instance
        for i, local_inst in enumerate(unique_inst):
            if local_inst <= 0 or local_inst not in kept_instances:
                continue

            gid = global_id(tile_idx, local_inst)
            size = int(inst_counts[i])
            uf.make_set(gid, size)
            instance_sizes[gid] = size

            # Get most common species_id for this instance (from sorted slice)
            start = first_idx[i]
            end = start + inst_counts[i]
            species_slice = sorted_species[start:end]
            if len(species_slice) > 0:
                species_unique, species_counts = np.unique(
                    species_slice, return_counts=True
                )
                instance_species_map[gid] = species_unique[np.argmax(species_counts)]

            # Get mean species_prob for this instance (if available)
            if sorted_species_prob is not None:
                species_prob_slice = sorted_species_prob[start:end]
                if len(species_prob_slice) > 0:
                    # Use mean of species_prob values for this instance
                    instance_species_prob_map[gid] = float(np.mean(species_prob_slice))

        # Clean up sorted arrays
        del sort_idx, sorted_inst, sorted_species
        if sorted_species_prob is not None:
            del sorted_species_prob

    print(f"  Total global instances: {len(instance_sizes)}")
    print(f"  ✓ Stage 2 completed: Assigned global IDs to {len(instance_sizes)} instances")

    # Helper functions for border matching
    def get_opposite_direction(direction: str) -> str:
        """Get opposite direction."""
        opposites = {"east": "west", "west": "east", "north": "south", "south": "north"}
        return opposites.get(direction, direction)
    
    def bboxes_overlap(bbox_a: Tuple[float, float, float, float], bbox_b: Tuple[float, float, float, float], tolerance: float = 0.1) -> bool:
        """
        Check if two bounding boxes overlap or are within tolerance distance.
        
        Args:
            bbox_a: (minx, maxx, miny, maxy) of first bounding box
            bbox_b: (minx, maxx, miny, maxy) of second bounding box
            tolerance: Maximum distance between boxes to still consider them (default: 0.1m = 10cm)
        
        Returns:
            True if boxes overlap or are within tolerance distance
        """
        minx_a, maxx_a, miny_a, maxy_a = bbox_a
        minx_b, maxx_b, miny_b, maxy_b = bbox_b
        
        # Check if boxes overlap (original check)
        if not (maxx_a < minx_b or minx_a > maxx_b or maxy_a < miny_b or miny_a > maxy_b):
            return True
        
        # Check if boxes are within tolerance distance (almost touching)
        # Compute gaps in X and Y dimensions
        # If boxes don't overlap, find the minimum separation
        x_gap = 0.0
        if maxx_a < minx_b:
            x_gap = minx_b - maxx_a  # A is to the left of B
        elif maxx_b < minx_a:
            x_gap = minx_a - maxx_b  # B is to the left of A
        # else: they overlap in X, x_gap = 0
        
        y_gap = 0.0
        if maxy_a < miny_b:
            y_gap = miny_b - maxy_a  # A is below B
        elif maxy_b < miny_a:
            y_gap = miny_a - maxy_b  # B is below A
        # else: they overlap in Y, y_gap = 0
        
        # Minimum separation is the diagonal distance between closest corners
        # For non-overlapping boxes: √(x_gap² + y_gap²)
        # But if boxes overlap in one dimension, we use the gap in the other dimension
        separation = np.sqrt(x_gap * x_gap + y_gap * y_gap)
        
        return separation <= tolerance

    # =========================================================================
    # Stage 2.5: Border Region Instance Matching
    # =========================================================================
    # Note: Cross-tile matching is optimized - each tile pair is checked exactly once
    # using `for j in range(i + 1, len(tiles))`, avoiding duplicate A->B and B->A checks.
    print(f"\n{'=' * 60}")
    print("Stage 2.5: Border Region Instance Matching")
    print(f"{'=' * 60}")
    print(f"  Finding border region instances (centroids in buffer to buffer+10m zone)...")
    
    # Find border region instances (centroids in buffer to buffer+10m zone)
    border_instances = {}  # tile_name -> {instance_id: {'centroid': [...], 'points': [...], 'boundary': [...]}}
    
    # Build tile name to index mapping
    tile_name_to_idx = {tile.name: idx for idx, tile in enumerate(tiles)}
    
    for tile_idx, tile in enumerate(tiles):
        print(f"    Processing tile {tile_idx + 1}/{len(tiles)}: {tile.name} ({len(tile.points):,} points)...")
        tile_name = tile.name
        neighbors = find_spatial_neighbors(tile.boundary, tile_name, tile_boundaries)
        kept_instances = kept_instances_per_tile[tile_name]
        
        min_x, max_x, min_y, max_y = tile.boundary
        tile_width = max_x - min_x
        tile_height = max_y - min_y
        min_dimension = min(tile_width, tile_height)
        actual_buffer = min(buffer, min_dimension * 0.4)
        actual_buffer = max(actual_buffer, 2.0)
        border_zone_end = actual_buffer + 10.0  # 10m beyond buffer
        
        # Define border region boundaries (buffer to buffer+10m from edges with neighbors)
        # Inner edge of border region (end of buffer zone)
        border_inner_min_x = min_x + (actual_buffer if neighbors["west"] is not None else 0)
        border_inner_max_x = max_x - (actual_buffer if neighbors["east"] is not None else 0)
        border_inner_min_y = min_y + (actual_buffer if neighbors["south"] is not None else 0)
        border_inner_max_y = max_y - (actual_buffer if neighbors["north"] is not None else 0)
        
        # Outer edge of border region (buffer+10m from tile edge)
        border_outer_min_x = min_x + (border_zone_end if neighbors["west"] is not None else 0)
        border_outer_max_x = max_x - (border_zone_end if neighbors["east"] is not None else 0)
        border_outer_min_y = min_y + (border_zone_end if neighbors["south"] is not None else 0)
        border_outer_max_y = max_y - (border_zone_end if neighbors["north"] is not None else 0)
        
        border_instances[tile_name] = {}
        
        # OPTIMIZATION: Filter points to donut-shaped border region FIRST
        # This dramatically reduces the number of centroids we need to compute
        print(f"      Filtering points to border region (donut)...")
        border_mask = get_border_region_mask(
            tile.points, tile.boundary, actual_buffer, border_zone_end, neighbors
        )
        border_points = tile.points[border_mask]
        border_inst_ids = tile.instances[border_mask]
        n_border_points = np.sum(border_mask)
        print(f"      Border region: {n_border_points:,}/{len(tile.points):,} points ({100*n_border_points/len(tile.points):.1f}%)")
        
        # Get unique instances in border region (much smaller set than all instances)
        border_unique_insts = set(np.unique(border_inst_ids)) - {0}
        border_unique_insts &= kept_instances  # Only kept instances
        
        if len(border_unique_insts) == 0:
            print(f"      No border region instances in {tile.name}")
            continue
        
        # Compute centroids ONLY for instances in border region
        print(f"      Computing centroids for {len(border_unique_insts)} border instances (vs {len(kept_instances)} total)...")
        border_centroids = compute_centroids_vectorized(border_points, border_inst_ids)
        
        border_count = 0
        
        # For each border instance, determine direction and extract full points
        for inst_id in border_unique_insts:
            if inst_id not in border_centroids:
                continue
            
            centroid = border_centroids[inst_id]
            cx, cy = centroid[0], centroid[1]
            
            # Determine border direction based on centroid position
            border_direction = None
            if neighbors["west"] is not None and cx < min_x + border_zone_end:
                border_direction = "west"
            elif neighbors["east"] is not None and cx > max_x - border_zone_end:
                border_direction = "east"
            elif neighbors["south"] is not None and cy < min_y + border_zone_end:
                border_direction = "south"
            elif neighbors["north"] is not None and cy > max_y - border_zone_end:
                border_direction = "north"
            
            if border_direction is None:
                continue
            
            # Extract full instance points (from original tile, not just border region)
            inst_mask = tile.instances == inst_id
            inst_points = tile.points[inst_mask]
            
            # Compute instance bounding box
            inst_minx = inst_points[:, 0].min()
            inst_maxx = inst_points[:, 0].max()
            inst_miny = inst_points[:, 1].min()
            inst_maxy = inst_points[:, 1].max()
            
            border_instances[tile_name][inst_id] = {
                'centroid': centroid,
                'points': inst_points,
                'boundary': (inst_minx, inst_maxx, inst_miny, inst_maxy),
                'direction': border_direction,
                'tile_idx': tile_idx
            }
            border_count += 1
        
        print(f"      Found {border_count} border region instances in {tile.name}")
    
    # Match border region instances between neighbor tiles
    total_border_insts = sum(len(insts) for insts in border_instances.values())
    tiles_with_border = len([t for t in border_instances if border_instances[t]])
    print(f"  Found {total_border_insts} border region instances across {tiles_with_border} tiles")
    print(f"  Processing tile pairs...")
    
    # Track which global IDs have already been matched to avoid duplicate checks
    matched_gids = set()
    
    border_matches = 0
    total_bbox_checks = 0
    total_centroid_checks = 0
    total_ff3d_computations = 0
    tiles_processed = 0
    
    for i in range(len(tiles)):
        tile_a = tiles[i]
        neighbors_a = find_spatial_neighbors(tile_a.boundary, tile_a.name, tile_boundaries)
        
        for direction, neighbor_name in neighbors_a.items():
            if neighbor_name is None:
                continue
            
            # Find neighbor tile index
            tile_b_idx = tile_name_to_idx.get(neighbor_name)
            if tile_b_idx is None:
                continue
            
            tile_b = tiles[tile_b_idx]
            
            # Get border instances from both tiles in this direction
            border_insts_a = {
                inst_id: data for inst_id, data in border_instances.get(tile_a.name, {}).items()
                if data['direction'] == direction
            }
            border_insts_b = {
                inst_id: data for inst_id, data in border_instances.get(tile_b.name, {}).items()
                if data['direction'] == get_opposite_direction(direction)
            }
            
            if not border_insts_a or not border_insts_b:
                continue
            
            # Progress: Show which tile pair is being processed
            matches_before = border_matches
            print(f"    Checking {tile_a.name} ({direction}) <-> {tile_b.name} ({get_opposite_direction(direction)}): "
                  f"{len(border_insts_a)} vs {len(border_insts_b)} border instances", end=" ... ")
            
            # Build KDTree from tile B centroids for fast spatial lookup
            # Only for instances not already matched
            centroids_b_list = []
            inst_ids_b_list = []
            gids_b_list = []
            
            for inst_id_b, data_b in border_insts_b.items():
                gid_b = global_id(tile_b_idx, inst_id_b)
                # Skip if already matched (avoid redundant checks)
                if gid_b not in matched_gids:
                    centroids_b_list.append(data_b['centroid'])
                    inst_ids_b_list.append(inst_id_b)
                    gids_b_list.append(gid_b)
            
            if not centroids_b_list:
                continue
            
            centroids_b_array = np.array(centroids_b_list)
            tree_b = cKDTree(centroids_b_array)
            
            # For each instance in tile A, find nearby instances in tile B using KDTree
            for inst_id_a, data_a in border_insts_a.items():
                gid_a = global_id(i, inst_id_a)
                
                # Skip if already matched
                if gid_a in matched_gids:
                    continue
                
                centroid_a = data_a['centroid']
                bbox_a = data_a['boundary']
                
                # Query KDTree to find instances within max_centroid_distance
                # This dramatically reduces the number of pairs to check
                k = min(len(centroids_b_list), 10)
                if k == 1:
                    # Single result: query returns scalar
                    distance, idx = tree_b.query(centroid_a, k=1, distance_upper_bound=max_centroid_distance)
                    if np.isfinite(distance):
                        candidate_indices = [idx]
                        candidate_distances = [distance]
                    else:
                        candidate_indices = []
                        candidate_distances = []
                else:
                    # Multiple results: query returns arrays
                    distances, indices = tree_b.query(centroid_a, k=k, distance_upper_bound=max_centroid_distance)
                    # Filter to valid results (within distance, not inf)
                    valid_mask = np.isfinite(distances)
                    candidate_indices = indices[valid_mask].tolist()
                    candidate_distances = distances[valid_mask].tolist()
                
                total_centroid_checks += len(candidate_indices)
                
                # Check each candidate pair
                for idx, dist in zip(candidate_indices, candidate_distances):
                    inst_id_b = inst_ids_b_list[idx]
                    gid_b = gids_b_list[idx]
                    
                    # Skip if already matched
                    if gid_b in matched_gids:
                        continue
                    
                    data_b = border_insts_b[inst_id_b]
                    bbox_b = data_b['boundary']
                    
                    # Quick bounding box overlap/nearby check (within 10cm tolerance)
                    total_bbox_checks += 1
                    if not bboxes_overlap(bbox_a, bbox_b, tolerance=0.1):
                        continue
                    
                    # Now compute expensive FF3D overlap ratio
                    total_ff3d_computations += 1
                    points_a = data_a['points']
                    points_b = data_b['points']
                    instances_a = np.full(len(points_a), inst_id_a, dtype=np.int32)
                    instances_b = np.full(len(points_b), inst_id_b, dtype=np.int32)
                    
                    overlap_ratios_dict, size_a, size_b = compute_ff3d_overlap_ratios(
                        instances_a, instances_b, points_a, points_b, correspondence_tolerance
                    )
                    
                    overlap_ratio = overlap_ratios_dict.get((inst_id_a, inst_id_b), 0.0)
                    
                    if overlap_ratio >= overlap_threshold:
                        # Merge via Union-Find
                        root = uf.union(gid_a, gid_b)
                        matched_gids.add(gid_a)
                        matched_gids.add(gid_b)
                        border_matches += 1
                        if verbose:
                            print(f"      ✓ Match: {tile_a.name}:{inst_id_a} <-> {tile_b.name}:{inst_id_b} (overlap: {overlap_ratio:.3f}, dist: {dist:.2f}m)")
            
            # Progress: Show results for this tile pair
            matches_this_pair = border_matches - matches_before
            if matches_this_pair > 0:
                print(f"{matches_this_pair} match(es) found")
            else:
                print("no matches")
            tiles_processed += 1
            
            # Periodic progress update every 10 tile pairs
            if tiles_processed % 10 == 0:
                print(f"  Progress: {tiles_processed} tile pairs processed, {border_matches} total matches so far...")
    
    print(f"  Matched {border_matches} border region instance pairs")
    print(f"  Performance: {total_bbox_checks} bbox checks, {total_centroid_checks} centroid checks, {total_ff3d_computations} FF3D computations")
    print(f"  ✓ Stage 2.5 completed: Border region matching done")

    # =========================================================================
    # Stage 3: Cross-Tile Instance Matching (Optional)
    # =========================================================================
    if enable_matching:
        print(f"\n{'=' * 60}")
        print("Stage 3: Cross-Tile Instance Matching")
        print(f"{'=' * 60}")

        # Pre-compute kept masks for all tiles (cache to avoid recomputing per pair)
        print("  Pre-computing kept instance masks...")
        kept_masks = {}
        for tile in tiles:
            kept_set = kept_instances_per_tile[tile.name]
            # Convert set to sorted array for faster np.isin
            kept_array = np.array(sorted(kept_set), dtype=tile.instances.dtype)
            kept_masks[tile.name] = np.isin(tile.instances, kept_array)

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

                # Use cached kept masks
                kept_mask_a = kept_masks[tile_a.name]
                kept_mask_b = kept_masks[tile_b.name]

                points_a, inst_a, _ = get_points_in_region(
                    tile_a.points[kept_mask_a], tile_a.instances[kept_mask_a], overlap
                )
                points_b, inst_b, _ = get_points_in_region(
                    tile_b.points[kept_mask_b], tile_b.instances[kept_mask_b], overlap
                )

                if len(points_a) == 0 or len(points_b) == 0:
                    continue

                # Compute overlap ratios (vectorized)
                computed_overlap_ratios, size_a, size_b = compute_ff3d_overlap_ratios(
                    inst_a, inst_b, points_a, points_b, correspondence_tolerance
                )

                # Pre-compute centroids using vectorized function
                centroids_a = compute_centroids_vectorized(points_a, inst_a)
                centroids_b = compute_centroids_vectorized(points_b, inst_b)

                # Find matching pairs above threshold AND within centroid distance
                matches_in_pair = 0
                rejected_low_overlap = 0
                rejected_far_centroid = 0

                for (
                    inst_a_local,
                    inst_b_local,
                ), ratio in computed_overlap_ratios.items():
                    # Compute centroid distance
                    if inst_a_local in centroids_a and inst_b_local in centroids_b:
                        centroid_dist = np.linalg.norm(
                            centroids_a[inst_a_local] - centroids_b[inst_b_local]
                        )
                    else:
                        centroid_dist = float("inf")

                    # Check conditions (overlap is optional)
                    passes_overlap = not require_overlap or (ratio >= overlap_threshold)
                    passes_distance = centroid_dist < max_centroid_distance

                    if verbose:
                        status = (
                            "MERGED"
                            if (passes_overlap and passes_distance)
                            else "REJECTED"
                        )
                        reason = ""
                        if require_overlap and ratio < overlap_threshold:
                            reason = f"low overlap ({ratio:.2f} < {overlap_threshold})"
                        elif not passes_distance:
                            reason = f"too far ({centroid_dist:.1f}m > {max_centroid_distance}m)"
                        print(
                            f"      Inst {inst_a_local} <-> {inst_b_local}: ratio={ratio:.2f}, dist={centroid_dist:.1f}m → {status} {reason}"
                        )

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
                        print(
                            f"{matches_in_pair} pairs merged, {rejected_far_centroid} rejected (centroid too far), {rejected_low_overlap} rejected (low overlap)"
                        )
                    else:
                        print(
                            f"{matches_in_pair} pairs merged, {rejected_far_centroid} rejected (centroid too far)"
                        )
                total_matches += matches_in_pair

        print(f"\n  Total matching pairs: {total_matches}")
        print(f"  ✓ Stage 3 completed: Matched {total_matches} cross-tile instance pairs")
    else:
        print(f"\n{'=' * 60}")
        print("Stage 3: Cross-Tile Instance Matching (DISABLED)")
        print(f"{'=' * 60}")
        print(
            "  Skipping instance matching - instances will not be merged across tiles"
        )
        print(f"  ✓ Stage 3 skipped (disabled)")

    # Get connected components
    components = uf.get_components()
    print(f"  Connected components: {len(components)}")
    print(f"  ✓ Instance matching completed: {len(components)} merged instance groups")

    # Create mapping from global ID to final merged ID
    global_to_merged = {}
    merged_species = {}  # merged_id -> species_id
    merged_species_prob = {}  # merged_id -> species_prob
    merged_instance_sources = {}  # merged_id -> list of source global IDs (for CSV tracking)

    for merged_id, (root, members) in enumerate(components.items(), start=1):
        # Find largest member for species and species_prob
        largest_member = max(members, key=lambda m: instance_sizes.get(m, 0))
        merged_species[merged_id] = instance_species_map.get(largest_member, 0)
        merged_species_prob[merged_id] = instance_species_prob_map.get(largest_member, 0.0)

        # Track which global IDs contributed to this merged instance (for CSV)
        merged_instance_sources[merged_id] = list(members)

        for gid in members:
            global_to_merged[gid] = merged_id

    # =========================================================================
    # Orphan Recovery: Recover filtered instances that would otherwise be lost
    # =========================================================================
    # Problem: A tree can be filtered from BOTH tiles if segmented slightly
    # differently (centroids in buffer zones of both tiles).
    #
    # Solution: Recover filtered instances from east/north buffer ONLY if
    # the neighboring tile doesn't have the tree covered (checked via bbox overlap).
    print("\n  Checking for orphaned filtered instances...")

    # Build tile name to index map
    tile_name_to_idx = {tile.name: idx for idx, tile in enumerate(tiles)}

    # Helper to get neighbor tile name using spatial bounds
    def get_neighbor_tile_name(tile_name: str, direction: str) -> Optional[str]:
        tile_idx = tile_name_to_idx.get(tile_name)
        if tile_idx is None:
            return None
        
        tile = tiles[tile_idx]
        neighbors = find_spatial_neighbors(tile.boundary, tile_name, tile_boundaries)
        neighbor_name = neighbors.get(direction)
        
        return neighbor_name if neighbor_name in tile_name_to_idx else None

    # Pre-compute bounding boxes for ALL instances in ALL tiles (O(N) single pass per tile)
    instance_bboxes = {}  # tile_name -> {inst_id -> (min_xyz, max_xyz)}
    instance_species_cache = {}  # tile_name -> {inst_id -> species_id}
    instance_species_prob_cache = {}  # tile_name -> {inst_id -> species_prob (mean)}

    for tile in tiles:
        bboxes = {}
        species_cache = {}
        species_prob_cache = {}

        # Single pass: sort by instance for efficient grouping
        sort_idx = np.argsort(tile.instances)
        sorted_inst = tile.instances[sort_idx]
        sorted_points = tile.points[sort_idx]
        sorted_species = tile.species_ids[sort_idx]
        sorted_species_prob = None
        if tile.species_prob is not None:
            sorted_species_prob = tile.species_prob[sort_idx]

        unique_inst, first_idx, counts = np.unique(
            sorted_inst, return_index=True, return_counts=True
        )

        for i, inst_id in enumerate(unique_inst):
            if inst_id <= 0:
                continue
            start = first_idx[i]
            end = start + counts[i]
            pts = sorted_points[start:end]
            bboxes[inst_id] = (pts.min(axis=0), pts.max(axis=0))

            # Cache most common species
            sp_slice = sorted_species[start:end]
            if len(sp_slice) > 0:
                unique_sp, sp_counts = np.unique(sp_slice, return_counts=True)
                species_cache[inst_id] = unique_sp[np.argmax(sp_counts)]
            else:
                species_cache[inst_id] = 0

            # Cache mean species_prob (if available)
            if sorted_species_prob is not None:
                sp_prob_slice = sorted_species_prob[start:end]
                if len(sp_prob_slice) > 0:
                    species_prob_cache[inst_id] = float(np.mean(sp_prob_slice))

        instance_bboxes[tile.name] = bboxes
        instance_species_cache[tile.name] = species_cache
        instance_species_prob_cache[tile.name] = species_prob_cache

    next_merged_id = max(global_to_merged.values()) + 1 if global_to_merged else 1
    recovered_count = 0
    skipped_covered = 0

    for tile_idx, tile in enumerate(tiles):
        filtered_instances = filtered_instances_per_tile[tile.name]
        buffer_directions = buffer_direction_per_tile[tile.name]
        tile_bboxes = instance_bboxes[tile.name]
        tile_species = instance_species_cache[tile.name]
        tile_species_prob = instance_species_prob_cache[tile.name]

        for local_inst in filtered_instances:
            if local_inst <= 0 or local_inst not in tile_bboxes:
                continue

            direction = buffer_directions.get(local_inst, None)
            if direction not in ("east", "north"):
                continue

            # Get pre-computed bounding box
            fmin, fmax = tile_bboxes[local_inst]

            # Check if neighbor tile has this tree covered
            neighbor_name = get_neighbor_tile_name(tile.name, direction)
            neighbor_has_tree = False

            if neighbor_name and neighbor_name in instance_bboxes:
                neighbor_bboxes = instance_bboxes[neighbor_name]
                neighbor_kept = kept_instances_per_tile[neighbor_name]

                # Check overlap with neighbor's kept instances using pre-computed bboxes
                for neighbor_inst in neighbor_kept:
                    if neighbor_inst not in neighbor_bboxes:
                        continue
                    nmin, nmax = neighbor_bboxes[neighbor_inst]

                    # XY overlap check with 2m tolerance
                    if (
                        fmax[0] < nmin[0] - 2
                        or fmin[0] > nmax[0] + 2
                        or fmax[1] < nmin[1] - 2
                        or fmin[1] > nmax[1] + 2
                    ):
                        continue

                    neighbor_has_tree = True
                    break

            if neighbor_has_tree:
                skipped_covered += 1
                continue

            # Truly orphaned - recover this instance
            gid = global_id(tile_idx, local_inst)
            global_to_merged[gid] = next_merged_id
            merged_species[next_merged_id] = tile_species.get(local_inst, 0)
            merged_species_prob[next_merged_id] = tile_species_prob.get(local_inst, 0.0)
            merged_instance_sources[next_merged_id] = [gid]  # Single source for recovered instances

            kept_instances_per_tile[tile.name].add(local_inst)
            next_merged_id += 1
            recovered_count += 1

    # Clean up bbox cache
    del instance_bboxes
    del instance_species_cache
    del instance_species_prob_cache

    if recovered_count > 0 or skipped_covered > 0:
        print(f"  Recovered {recovered_count} orphaned instances")
        print(f"  Skipped {skipped_covered} instances (neighbor has overlapping tree)")
    else:
        print(f"  No orphaned instances found")

    # =========================================================================
    # Stage 4: Merge and Deduplicate
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 4: Merging tiles and deduplicating")
    print(f"{'=' * 60}")

    all_points = []
    all_instances = []
    all_species = []
    all_species_prob = []

    for tile_idx, tile in enumerate(tiles):
        kept_instances = kept_instances_per_tile[tile.name]

        # Build lookup arrays for vectorized remapping
        # Find max local instance ID to size the lookup table
        max_local_inst = tile.instances.max() + 1

        # Create lookup tables: local_inst -> merged_id, local_inst -> species, local_inst -> species_prob
        # Default is 0 (filtered), ground points remain as 0
        inst_to_merged = np.zeros(max_local_inst, dtype=np.int32)
        inst_to_species = np.zeros(max_local_inst, dtype=np.int32)
        inst_to_species_prob = np.zeros(max_local_inst, dtype=np.float32)

        for local_inst in kept_instances:
            if local_inst <= 0:
                continue
            gid = global_id(tile_idx, local_inst)
            merged_id = global_to_merged.get(gid, 0)
            inst_to_merged[local_inst] = merged_id
            inst_to_species[local_inst] = merged_species.get(merged_id, 0)
            inst_to_species_prob[local_inst] = merged_species_prob.get(merged_id, 0.0)

        # Vectorized remapping using advanced indexing - single pass!
        # Clamp negative indices to 0 (they map to 0 anyway)
        safe_instances = np.clip(tile.instances, 0, max_local_inst - 1)
        remapped_instances = inst_to_merged[safe_instances]
        remapped_species = inst_to_species[safe_instances]
        remapped_species_prob = inst_to_species_prob[safe_instances]

        # Ground points (instance_id <= 0) remain as 0

        all_points.append(tile.points)
        all_instances.append(remapped_instances)
        all_species.append(remapped_species)
        # Store remapped species_prob (from merged_species_prob map, not original tile values)
        all_species_prob.append(remapped_species_prob)

    # Free tile data - no longer needed after extracting points
    del tiles
    del filtered_instances_per_tile
    del kept_instances_per_tile
    gc.collect()

    # Concatenate (species_prob is now always present as remapped values from merged_species_prob map)
    merged_points = np.vstack(all_points)
    merged_instances = np.concatenate(all_instances)
    merged_species_ids = np.concatenate(all_species)
    merged_species_prob = np.concatenate(all_species_prob)  # All values are now numpy arrays (remapped)

    # Free the lists - data now in merged arrays
    del all_points
    del all_instances
    del all_species
    del all_species_prob
    gc.collect()

    # Remove points from filtered instances only (instance_id = 0 for filtered)
    # Keep: instance_id > 0 (trees) OR instance_id = 0 (ground)
    # After removing filtered instances, remaining 0s are ground points
    valid_points_mask = merged_instances != 0
    n_before_filter = len(merged_points)
    n_filtered_removed = np.sum(merged_instances == 0)

    merged_points = merged_points[valid_points_mask]
    merged_instances = merged_instances[valid_points_mask]
    merged_species_ids = merged_species_ids[valid_points_mask]
    merged_species_prob = merged_species_prob[valid_points_mask]

    print(f"  Removed {n_filtered_removed:,} points from filtered buffer instances")
    gc.collect()

    print(f"  Total points before dedup: {len(merged_points):,}")

    # Deduplicate
    # Note: ground points have instance_id=0, tree points have positive IDs
    # Dedup prefers higher IDs, so tree points are kept when overlapping ground
    print("\n  Deduplicating...")
    merged_points, merged_instances, merged_species_ids, merged_species_prob = deduplicate_points(
        merged_points, merged_instances, merged_species_ids, merged_species_prob
    )
    gc.collect()  # Free memory from deduplication intermediates

    # Count ground points after deduplication
    ground_count_after_dedup = np.sum(merged_instances == 0)

    print(f"  Total points after dedup: {len(merged_points):,}")
    print(f"  Ground points after dedup: {ground_count_after_dedup:,}")
    print(
        f"  Unique tree instances: {len(np.unique(merged_instances[merged_instances > 0]))}"
    )
    print(f"  ✓ Stage 4 completed: Merged and deduplicated {len(merged_points):,} points")

    # Clean up matching data structures no longer needed
    # Note: merged_species, merged_species_prob, and merged_instance_sources kept for CSV output
    del uf
    del instance_species_map
    del instance_sizes
    del global_to_merged
    gc.collect()

    # =========================================================================
    # Stage 5: Small Volume Instance Merging
    # =========================================================================
    if enable_volume_merge:
        import sys
        print(f"\n{'=' * 60}", flush=True)
        print("Stage 5: Small Volume Instance Merging", flush=True)
        print(f"{'=' * 60}", flush=True)
        sys.stdout.flush()

        # Filter to non-zero instances ONCE (used for both species maps and volume merge)
        print(f"  Filtering {len(merged_instances):,} points...", flush=True)
        sys.stdout.flush()
        pos_mask = merged_instances > 0
        nonzero_count = pos_mask.sum()
        zero_count = len(merged_instances) - nonzero_count
        print(f"  Non-zero: {nonzero_count:,}, Zero/ground: {zero_count:,} ({100*zero_count/len(merged_instances):.1f}%)", flush=True)
        sys.stdout.flush()
        
        # Extract non-zero points/instances for sorting
        pos_points = merged_points[pos_mask]
        pos_instances = merged_instances[pos_mask]
        
        # Sort ONCE - this sorted data will be reused for both species maps and volume merge
        print(f"  Sorting {len(pos_instances):,} points by instance (single sort for all operations)...", flush=True)
        sys.stdout.flush()
        sort_idx = np.argsort(pos_instances)
        sorted_points = pos_points[sort_idx]
        sorted_instances = pos_instances[sort_idx]
        print(f"  Sort complete.", flush=True)
        sys.stdout.flush()
        
        # Get unique instances with boundaries (used for both species maps and volume merge)
        unique_inst, first_idx, inst_counts = np.unique(
            sorted_instances, return_index=True, return_counts=True
        )
        print(f"  Found {len(unique_inst):,} unique instances.", flush=True)
        sys.stdout.flush()
        
        # Build species maps only if species data is available
        final_species_map = {}
        final_species_prob_map = {}
        
        if all_have_species_id:
            print(f"  Building species maps (species data available)...", flush=True)
            sys.stdout.flush()
            
            # Extract and sort species data using the same sort index
            pos_species = merged_species_ids[pos_mask]
            sorted_species = pos_species[sort_idx]
            
            pos_species_prob = merged_species_prob[pos_mask] if merged_species_prob is not None else None
            sorted_species_prob = pos_species_prob[sort_idx] if pos_species_prob is not None else None
            
            for i, (inst_id, start, count) in enumerate(zip(unique_inst, first_idx, inst_counts)):
                if i % 1000 == 0:
                    print(f"    Species progress: {i:,}/{len(unique_inst):,} ({100*i/len(unique_inst):.1f}%)...", flush=True)
                    sys.stdout.flush()
                
                end = start + count
                species_slice = sorted_species[start:end]
                
                if len(species_slice) > 0:
                    unique, counts = np.unique(species_slice, return_counts=True)
                    final_species_map[inst_id] = unique[np.argmax(counts)]
                
                if sorted_species_prob is not None:
                    prob_slice = sorted_species_prob[start:end]
                    if len(prob_slice) > 0:
                        final_species_prob_map[inst_id] = float(np.mean(prob_slice))
                    else:
                        final_species_prob_map[inst_id] = 0.0
                else:
                    final_species_prob_map[inst_id] = 0.0
            
            print(f"  Species maps built for {len(unique_inst):,} instances.", flush=True)
            sys.stdout.flush()
        else:
            print(f"  Skipping species map building (no species data in input files).", flush=True)
            sys.stdout.flush()

        # Call volume merge with pre-sorted data (avoids redundant sort!)
        print(f"  Starting volume-based instance merging...", flush=True)
        sys.stdout.flush()
        
        merged_instances, merged_species_ids, merged_species_prob, final_species_map, final_species_prob_map, _ = (
            merge_small_volume_instances(
                merged_points,
                merged_instances,
                merged_species_ids,
                merged_species_prob,
                final_species_map,
                final_species_prob_map,
                min_points_for_hull_check=1000,
                min_cluster_size=min_cluster_size,
                max_volume_for_merge=max_volume_for_merge,
                max_search_radius=5.0,
                num_threads=num_threads,
                verbose=verbose,
                # Pass pre-sorted data to avoid sorting again!
                presorted_points=sorted_points,
                presorted_instances=sorted_instances,
                presorted_unique_inst=unique_inst,
                presorted_first_idx=first_idx,
                presorted_inst_counts=inst_counts,
            )
        )
        print(f"  ✓ Stage 5 completed: Small volume instance merging done", flush=True)
    else:
        print(f"\n{'=' * 60}", flush=True)
        print("Stage 5: Small Volume Instance Merging (DISABLED)", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"  ✓ Stage 5 skipped (disabled)", flush=True)

    # =========================================================================
    # Renumber instances to continuous IDs
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Renumbering instances")
    print(f"{'=' * 60}")

    unique_instances = sorted(set(merged_instances) - {0})
    old_to_new = {0: 0}
    new_species_map = {}

    for new_id, old_id in enumerate(unique_instances, start=1):
        old_to_new[old_id] = new_id
        new_species_map[new_id] = final_species_map.get(old_id, 0)

    # Vectorized remapping using numpy lookup tables - O(n) instead of O(k*n) loop
    max_old_id = int(merged_instances.max()) + 1
    max_new_id = len(unique_instances) + 1
    
    # Lookup table: old_id → new_id
    instance_lookup = np.zeros(max_old_id, dtype=np.int32)
    for old_id, new_id in old_to_new.items():
        if old_id < max_old_id:
            instance_lookup[old_id] = new_id
    
    # Lookup table: new_id → species_id
    species_lookup = np.zeros(max_new_id, dtype=np.int32)
    for new_id, species in new_species_map.items():
        if new_id < max_new_id:
            species_lookup[new_id] = species
    
    # Vectorized remapping: O(n) single pass
    merged_instances = instance_lookup[merged_instances]
    merged_species_ids = species_lookup[merged_instances]

    print(f"  Final instance count: {len(unique_instances)}")

    # =========================================================================
    # Save merged output (optional - can be skipped with skip_merged_file=True)
    # =========================================================================
    if skip_merged_file:
        print(f"\n{'=' * 60}")
        print("Saving merged output (SKIPPED)")
        print(f"{'=' * 60}")
        print(f"  Skipped merged LAZ file creation (--skip_merged_file)")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(unique_instances)}")
    else:
        print(f"\n{'=' * 60}")
        print("Saving merged output")
        print(f"{'=' * 60}")

        output_merged.parent.mkdir(parents=True, exist_ok=True)

        header = laspy.LasHeader(point_format=6, version="1.4")
        header.offsets = np.min(merged_points, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])

        output_las = laspy.LasData(header)
        output_las.x = merged_points[:, 0]
        output_las.y = merged_points[:, 1]
        output_las.z = merged_points[:, 2]

        output_las.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.int32))
        output_las.PredInstance = merged_instances

        # Only add species_id if it was present in all input files
        if all_have_species_id:
            output_las.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.int32))
            output_las.species_id = merged_species_ids

        # Note: species_prob is NOT saved to LAZ file, but to a separate CSV instead

        output_las.write(
            str(output_merged), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel
        )

        # Clean up output LasData object
        del output_las
        gc.collect()

        print(f"  Saved merged output: {output_merged}")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(unique_instances)}")

    # =========================================================================
    # Create CSV with instance metadata (PredInstance, species_id, has_added_clusters)
    # =========================================================================
    import csv
    
    csv_output_path = output_merged.parent / f"{output_merged.stem}_instance_metadata.csv"
    
    # Build reverse mapping: final_id -> old_merged_id (for looking up species_prob)
    final_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}
    
    # Build set of final instance IDs that have added clusters (multiple source instances)
    instances_with_clusters = set()
    for old_merged_id, sources in merged_instance_sources.items():
        if len(sources) > 1:
            # Map to final instance ID if it still exists
            final_id = old_to_new.get(old_merged_id, None)
            if final_id is not None and final_id > 0:  # Skip ground (0) and invalid
                instances_with_clusters.add(final_id)
    
    print(f"\n  Writing instance metadata CSV: {csv_output_path}")
    print(f"  Found {len(instances_with_clusters)} final instances with added clusters from cross-tile merging")
    
    with open(csv_output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header - conditionally include species_id
        if all_have_species_id:
            writer.writerow(["PredInstance", "species_id", "has_added_clusters"])
        else:
            writer.writerow(["PredInstance", "has_added_clusters"])
        
        # Write one row per unique instance ID
        for final_id in sorted(unique_instances):
            has_clusters = final_id in instances_with_clusters
            
            if all_have_species_id:
                writer.writerow([
                    final_id,
                    new_species_map.get(final_id, 0),
                    1 if has_clusters else 0
                ])
            else:
                writer.writerow([
                    final_id,
                    1 if has_clusters else 0
                ])
    
    # Clean up dictionaries used for CSV (no longer needed)
    del merged_species
    del merged_species_prob
    del merged_instance_sources
    gc.collect()

    # =========================================================================
    # Stage 6: Retile to Original Files (Required)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 6: Retiling to Original Files")
    print(f"{'=' * 60}")
    
    retile_to_original_files(
        merged_points,
        merged_instances,
        merged_species_ids,
        original_tiles_dir,
        output_tiles_dir,
        tolerance=0.1,
        num_threads=num_threads,
        all_have_species_id=all_have_species_id,
    )
    print(f"  ✓ Stage 6 completed: Retiled to original files")

    # =========================================================================
    # Stage 7: Remap to Original Input Files (if provided)
    # =========================================================================
    if original_input_dir is not None:
        print(f"\n{'=' * 60}")
        print("Stage 7: Remapping to Original Input Files")
        print(f"{'=' * 60}")
        
        # Output directory for original files with predictions
        original_output_dir = output_tiles_dir.parent / "original_with_predictions"
        
        remap_to_original_input_files(
            merged_points,
            merged_instances,
            merged_species_ids,
            original_input_dir,
            original_output_dir,
            tolerance=0.1,
            num_threads=num_threads,
            all_have_species_id=all_have_species_id,
        )
        print(f"  ✓ Stage 7 completed: Remapped to original input files")
    else:
        print(f"\n  Note: --original-input-dir not provided, skipping Stage 7 (remap to original input files)")

    print(f"\n{'=' * 60}")
    print("Merge complete!")
    print(f"{'=' * 60}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Tile Merger - Merge segmented point cloud tiles with species ID preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing segmented LAZ tiles",
    )

    parser.add_argument(
        "--original-tiles-dir",
        type=Path,
        default=None,
        help="Directory containing original tile files for retiling",
    )

    parser.add_argument(
        "--output-merged",
        "-o",
        type=Path,
        required=True,
        help="Output path for merged LAZ file",
    )

    parser.add_argument(
        "--output-tiles-dir",
        type=Path,
        default=None,
        help="Output directory for retiled files",
    )

    parser.add_argument(
        "--original-input-dir",
        type=Path,
        default=None,
        help="Directory with original input LAZ files for final remap (optional, enables Stage 7)",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=10.0,
        help="Buffer zone distance in meters (default: 10.0)",
    )

    parser.add_argument(
        "--overlap-threshold",
        "--ff3d-threshold",
        type=float,
        default=0.3,
        dest="overlap_threshold",
        help="Overlap ratio threshold for instance matching (default: 0.3 = 30%%)",
    )

    parser.add_argument(
        "--max-centroid-distance",
        type=float,
        default=3.0,
        help="Max distance between centroids to merge instances (default: 3.0m)",
    )

    parser.add_argument(
        "--correspondence-tolerance",
        type=float,
        default=0.05,
        help="Max distance for point correspondence in meters (default: 0.05). "
        "Should be small (~5cm) to only match actual duplicate points from overlapping tiles.",
    )

    parser.add_argument(
        "--max-volume-for-merge",
        type=float,
        default=4.0,
        help="Max convex hull volume (m³) for small instance merging (default: 4.0)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        dest="num_threads",
        help="Number of workers for parallel processing (default: 4)",
    )

    parser.add_argument(
        "--disable-matching",
        "--disable-ff3d",
        action="store_true",
        dest="disable_matching",
        help="Disable cross-tile instance matching",
    )

    parser.add_argument(
        "--disable-volume-merge",
        action="store_true",
        help="Disable small volume instance merging",
    )

    parser.add_argument(
        "--disable-overlap-check",
        action="store_true",
        help="Disable overlap ratio check - merge based on centroid distance only",
    )

    parser.add_argument(
        "--skip-merged-file",
        action="store_true",
        help="Skip creating merged LAZ file (only create retiled outputs)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed merge decisions"
    )

    args = parser.parse_args()

    merge_tiles(
        input_dir=args.input_dir,
        original_tiles_dir=args.original_tiles_dir,
        output_merged=args.output_merged,
        output_tiles_dir=args.output_tiles_dir,
        original_input_dir=args.original_input_dir,
        buffer=args.buffer,
        overlap_threshold=args.overlap_threshold,
        max_centroid_distance=args.max_centroid_distance,
        correspondence_tolerance=args.correspondence_tolerance,
        max_volume_for_merge=args.max_volume_for_merge,
        num_threads=args.num_threads,
        enable_matching=not args.disable_matching,
        require_overlap=not args.disable_overlap_check,
        enable_volume_merge=not args.disable_volume_merge,
        skip_merged_file=args.skip_merged_file,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
