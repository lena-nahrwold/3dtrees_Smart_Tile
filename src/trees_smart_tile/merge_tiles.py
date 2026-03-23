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
import sys
import json
import math
import numpy as np
import laspy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from scipy.spatial import cKDTree
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field

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
    boundary: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    extra_dims: Dict[str, np.ndarray] = field(default_factory=dict)


def extra_bytes_params_from_dimension_info(
    dim_info,
    name: Optional[str] = None,
) -> laspy.ExtraBytesParams:
    """Build ExtraBytesParams from laspy DimensionInfo while preserving metadata."""
    return laspy.ExtraBytesParams(
        name=name or dim_info.name,
        type=dim_info.dtype,
        description=getattr(dim_info, "description", "") or "",
        offsets=getattr(dim_info, "offsets", None),
        scales=getattr(dim_info, "scales", None),
        no_data=getattr(dim_info, "no_data", None),
    )


def _next_available_suffix(base: str, used: set) -> str:
    """Return base_1, base_2, ... first not in used. Used to avoid losing dimensions on name collision."""
    for i in range(1, 10000):
        cand = f"{base}_{i}"
        if cand not in used:
            return cand
    return f"{base}_9999"


def _suffixes_for_collision(base: str, used: set) -> tuple[str, str]:
    """Return (name_1, name_2) for original vs merged, per dimension. Suffix is 1/2 per base name, not global."""
    cand_1 = f"{base}_1"
    cand_2 = f"{base}_2"
    out_1 = cand_1 if cand_1 not in used else _next_available_suffix(base, used)
    used.add(out_1)
    out_2 = cand_2 if cand_2 not in used else _next_available_suffix(base, used)
    used.add(out_2)
    return (out_1, out_2)


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


# =============================================================================
# Neighbor graph based on tile_bounds_tindex.json
# =============================================================================


def build_neighbor_graph_from_bounds_json(
    tile_bounds_json: Path,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float]], List[Dict[str, Optional[int]]]]:
    """
    Build a neighbor graph from tile_bounds_tindex.json.

    Returns:
        json_bounds: list of (minx, maxx, miny, maxy) for each JSON tile
        centers: list of (cx, cy) centers for each JSON tile
        neighbors_idx: list of dicts {dir -> neighbor_index or None} for each JSON tile
    """
    if not tile_bounds_json.exists():
        raise FileNotFoundError(f"tile_bounds_tindex.json not found: {tile_bounds_json}")

    with tile_bounds_json.open() as f:
        data = json.load(f)

    tiles = data.get("tiles", [])
    if not tiles:
        raise ValueError(f"No tiles found in tile bounds JSON: {tile_bounds_json}")

    json_bounds: List[Tuple[float, float, float, float]] = []
    centers: List[Tuple[float, float]] = []

    for tile in tiles:
        bx, by = tile["bounds"]
        minx, maxx = float(bx[0]), float(bx[1])
        miny, maxy = float(by[0]), float(by[1])
        json_bounds.append((minx, maxx, miny, maxy))
        cx = (minx + maxx) * 0.5
        cy = (miny + maxy) * 0.5
        centers.append((cx, cy))

    n = len(json_bounds)
    neighbors_idx: List[Dict[str, Optional[int]]] = [
        {"east": None, "west": None, "north": None, "south": None} for _ in range(n)
    ]

    # Prefer grid-based neighbor detection when col/row are present (deterministic, correct
    # even when bounds are cropped so center_x/center_y differ only slightly between tiles).
    col_row_to_idx: Dict[Tuple[int, int], int] = {}
    for i, tile in enumerate(tiles):
        if "col" in tile and "row" in tile:
            col_row_to_idx[(int(tile["col"]), int(tile["row"]))] = i

    if col_row_to_idx:
        for i, tile in enumerate(tiles):
            if "col" not in tile or "row" not in tile:
                continue
            c, r = int(tile["col"]), int(tile["row"])
            neighbors_idx[i]["east"] = col_row_to_idx.get((c + 1, r))
            neighbors_idx[i]["west"] = col_row_to_idx.get((c - 1, r))
            neighbors_idx[i]["north"] = col_row_to_idx.get((c, r + 1))
            neighbors_idx[i]["south"] = col_row_to_idx.get((c, r - 1))
    else:
        # Fallback: geometry-based neighbor detection (center + overlap)
        for i in range(n):
            minx_i, maxx_i, miny_i, maxy_i = json_bounds[i]
            cx_i, cy_i = centers[i]

            best_east = None  # (distance, j)
            best_west = None
            best_north = None
            best_south = None

            for j in range(n):
                if i == j:
                    continue

                minx_j, maxx_j, miny_j, maxy_j = json_bounds[j]
                cx_j, cy_j = centers[j]

                overlap_y = not (maxy_i <= miny_j or maxy_j <= miny_i)
                overlap_x = not (maxx_i <= minx_j or maxx_j <= minx_i)

                if cx_j > cx_i and overlap_y:
                    dx = cx_j - cx_i
                    if best_east is None or dx < best_east[0]:
                        best_east = (dx, j)

                if cx_j < cx_i and overlap_y:
                    dx = cx_i - cx_j
                    if best_west is None or dx < best_west[0]:
                        best_west = (dx, j)

                if cy_j > cy_i and overlap_x:
                    dy = cy_j - cy_i
                    if best_north is None or dy < best_north[0]:
                        best_north = (dy, j)

                if cy_j < cy_i and overlap_x:
                    dy = cy_i - cy_j
                    if best_south is None or dy < best_south[0]:
                        best_south = (dy, j)

            if best_east is not None:
                neighbors_idx[i]["east"] = best_east[1]
            if best_west is not None:
                neighbors_idx[i]["west"] = best_west[1]
            if best_north is not None:
                neighbors_idx[i]["north"] = best_north[1]
            if best_south is not None:
                neighbors_idx[i]["south"] = best_south[1]

    return json_bounds, centers, neighbors_idx


def _match_tiles_to_json_bounds(
    tile_boundaries: Dict[str, Tuple[float, float, float, float]],
    json_bounds: List[Tuple[float, float, float, float]],
    centers: List[Tuple[float, float]],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Match loaded tiles (from LAS headers) to JSON tiles using a stepwise strategy:
    1) bounds-based matching (L1 distance) within tolerance
    2) centroid-based matching (Euclidean distance) within tolerance

    Tolerance is increased stepwise until all tiles are matched or max tolerance reached.
    Raises ValueError if any tile remains unmatched.
    """
    tile_items = list(tile_boundaries.items())
    tile_to_json: Dict[str, int] = {}
    json_to_tile: Dict[int, str] = {}
    used_json: Set[int] = set()

    # Single-tile shortcut: 1 file and 1 JSON entry -> pair directly
    if len(tile_items) == 1 and len(json_bounds) == 1:
        name = tile_items[0][0]
        tile_to_json[name] = 0
        json_to_tile[0] = name
        return tile_to_json, json_to_tile

    # Tolerance schedule in meters
    tolerance_steps = [0.1, 0.5, 1.0, 2.0, 5.0]

    for tol in tolerance_steps:
        # Phase 1: bounds-based matching
        for name, bounds in tile_items:
            if name in tile_to_json:
                continue
            minx_a, maxx_a, miny_a, maxy_a = bounds

            best_j = None
            best_l1 = None
            for j, (minx_b, maxx_b, miny_b, maxy_b) in enumerate(json_bounds):
                if j in used_json:
                    continue
                if (
                    abs(minx_a - minx_b) <= tol
                    and abs(maxx_a - maxx_b) <= tol
                    and abs(miny_a - miny_b) <= tol
                    and abs(maxy_a - maxy_b) <= tol
                ):
                    l1 = (
                        abs(minx_a - minx_b)
                        + abs(maxx_a - maxx_b)
                        + abs(miny_a - miny_b)
                        + abs(maxy_a - maxy_b)
                    )
                    if best_l1 is None or l1 < best_l1:
                        best_l1 = l1
                        best_j = j

            if best_j is not None:
                tile_to_json[name] = best_j
                json_to_tile[best_j] = name
                used_json.add(best_j)

        # Phase 2: centroid-based matching for remaining tiles
        for name, bounds in tile_items:
            if name in tile_to_json:
                continue
            minx_a, maxx_a, miny_a, maxy_a = bounds
            cx_a = (minx_a + maxx_a) * 0.5
            cy_a = (miny_a + maxy_a) * 0.5

            best_j = None
            best_dist = None
            for j, (cx_b, cy_b) in enumerate(centers):
                if j in used_json:
                    continue
                dx = cx_b - cx_a
                dy = cy_b - cy_a
                dist = math.hypot(dx, dy)
                if dist <= tol and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                tile_to_json[name] = best_j
                json_to_tile[best_j] = name
                used_json.add(best_j)

        # If all tiles matched, we can stop early
        if len(tile_to_json) == len(tile_boundaries):
            break

    if len(tile_to_json) != len(tile_boundaries):
        unmatched = sorted(set(tile_boundaries.keys()) - set(tile_to_json.keys()))
        raise ValueError(
            "Failed to match all tiles to entries in tile_bounds_tindex.json. "
            f"Unmatched tiles: {', '.join(unmatched)}"
        )

    return tile_to_json, json_to_tile


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
    tile_width_a = maxx_a - minx_a
    tile_height_a = maxy_a - miny_a
    
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
        # Neighbor must be mostly to the east (right) AND its left edge must overlap/near this tile's right edge
        # This ensures we only match true edge neighbors, not tiles that are far away
        if minx_b > minx_a and minx_b <= maxx_a + tolerance and overlap_width >= tolerance:
            # Check if there's vertical overlap
            if not (maxy_b < miny_a or miny_b > maxy_a):
                # Calculate alignment: same row = high alignment
                # For same row, the vertical overlap should span most of BOTH tiles' heights
                # OR if overlap spans most of the smaller tile (indicates same row with size mismatch)
                tile_height_b = maxy_b - miny_b
                overlap_height_ratio_a = overlap_height / tile_height_a if tile_height_a > 0 else 0.0
                overlap_height_ratio_b = overlap_height / tile_height_b if tile_height_b > 0 else 0.0
                # High alignment if:
                # 1. Both tiles have >80% overlap (perfect alignment)
                # 2. Exact Y bounds match (perfect alignment)
                # 3. Smaller tile has >80% overlap (indicates same row with size mismatch)
                max_ratio = max(overlap_height_ratio_a, overlap_height_ratio_b)
                y_alignment = 1.0 if (overlap_height_ratio_a > 0.8 and overlap_height_ratio_b > 0.8) or (miny_b == miny_a and maxy_b == maxy_a) or (max_ratio > 0.8) else 0.5
                # Edge alignment: check if Y edges align (for same-row detection)
                # Tiles in same row should have miny or maxy very close (within 0.1m tolerance)
                edge_tolerance = 0.1
                bottom_edge_align = abs(miny_a - miny_b) < edge_tolerance
                top_edge_align = abs(maxy_a - maxy_b) < edge_tolerance
                # Higher score if BOTH edges align (perfect same row), medium if one aligns
                edge_alignment = 2.0 if (bottom_edge_align and top_edge_align) else (1.0 if (bottom_edge_align or top_edge_align) else 0.0)
                # Store (overlap_area, y_alignment, minx_b - minx_a, edge_alignment, other_name)
                # Priority: alignment (same row), then overlap area, then distance, then edge alignment
                east_overlaps.append((overlap_area, y_alignment, minx_b - minx_a, edge_alignment, other_name))
        
        # West neighbor: other tile extends to the left (west) of this tile
        # The overlap should be on the left side of this tile
        # Neighbor must be mostly to the west (left) AND its right edge must overlap/near this tile's left edge
        # This ensures we only match true edge neighbors, not tiles contained within or diagonal overlaps
        if minx_b < minx_a and maxx_b >= minx_a - tolerance and overlap_width >= tolerance:
            # Check if there's vertical overlap
            if not (maxy_b < miny_a or miny_b > maxy_a):
                # Calculate alignment: same row = high alignment
                # For same row, the vertical overlap should span most of BOTH tiles' heights
                # OR if overlap spans most of the smaller tile (indicates same row with size mismatch)
                tile_height_b = maxy_b - miny_b
                overlap_height_ratio_a = overlap_height / tile_height_a if tile_height_a > 0 else 0.0
                overlap_height_ratio_b = overlap_height / tile_height_b if tile_height_b > 0 else 0.0
                # High alignment if:
                # 1. Both tiles have >80% overlap (perfect alignment)
                # 2. Exact Y bounds match (perfect alignment)
                # 3. Smaller tile has >80% overlap (indicates same row with size mismatch)
                max_ratio = max(overlap_height_ratio_a, overlap_height_ratio_b)
                y_alignment = 1.0 if (overlap_height_ratio_a > 0.8 and overlap_height_ratio_b > 0.8) or (miny_b == miny_a and maxy_b == maxy_a) or (max_ratio > 0.8) else 0.5
                # Edge alignment: check if Y edges align (for same-row detection)
                # Tiles in same row should have miny or maxy very close (within 0.1m tolerance)
                edge_tolerance = 0.1
                bottom_edge_align = abs(miny_a - miny_b) < edge_tolerance
                top_edge_align = abs(maxy_a - maxy_b) < edge_tolerance
                # Higher score if BOTH edges align (perfect same row), medium if one aligns
                edge_alignment = 2.0 if (bottom_edge_align and top_edge_align) else (1.0 if (bottom_edge_align or top_edge_align) else 0.0)
                # Store (overlap_area, y_alignment, minx_a - minx_b, edge_alignment, other_name)
                # Priority: alignment (same row), then overlap area, then distance, then edge alignment
                west_overlaps.append((overlap_area, y_alignment, minx_a - minx_b, edge_alignment, other_name))
        
        # North neighbor: other tile extends above (north) of this tile
        # The overlap should be on the top side of this tile
        # Neighbor must be mostly to the north (above) AND its bottom edge must overlap/near this tile's top edge
        # This ensures we only match true edge neighbors, not tiles that are far away
        if miny_b > miny_a and miny_b <= maxy_a + tolerance and overlap_height >= tolerance:
            # Check if there's horizontal overlap
            if not (maxx_b < minx_a or minx_b > maxx_a):
                # Calculate alignment: same column = high alignment
                # For same column, the horizontal overlap should span most of BOTH tiles' widths
                # OR if overlap spans most of the smaller tile (indicates same column with size mismatch)
                tile_width_b = maxx_b - minx_b
                overlap_width_ratio_a = overlap_width / tile_width_a if tile_width_a > 0 else 0.0
                overlap_width_ratio_b = overlap_width / tile_width_b if tile_width_b > 0 else 0.0
                # High alignment if:
                # 1. Both tiles have >80% overlap (perfect alignment)
                # 2. Exact X bounds match (perfect alignment)
                # 3. Smaller tile has >80% overlap (indicates same column with size mismatch)
                max_ratio = max(overlap_width_ratio_a, overlap_width_ratio_b)
                x_alignment = 1.0 if (overlap_width_ratio_a > 0.8 and overlap_width_ratio_b > 0.8) or (minx_b == minx_a and maxx_b == maxx_a) or (max_ratio > 0.8) else 0.5
                # Edge alignment: check if X edges align (for same-column detection)
                # Tiles in same column should have minx or maxx very close (within 0.1m tolerance)
                edge_tolerance = 0.1
                left_edge_align = abs(minx_a - minx_b) < edge_tolerance
                right_edge_align = abs(maxx_a - maxx_b) < edge_tolerance
                # Higher score if BOTH edges align (perfect same column), medium if one aligns
                edge_alignment = 2.0 if (left_edge_align and right_edge_align) else (1.0 if (left_edge_align or right_edge_align) else 0.0)
                # Store (overlap_area, x_alignment, miny_b - miny_a, edge_alignment, other_name)
                # Priority: alignment (same column), then overlap area, then distance, then edge alignment
                north_overlaps.append((overlap_area, x_alignment, miny_b - miny_a, edge_alignment, other_name))
        
        # South neighbor: other tile extends below (south) of this tile
        # The overlap should be on the bottom side of this tile
        # Neighbor must be mostly to the south (below) AND its top edge must overlap/near this tile's bottom edge
        # This ensures we only match true edge neighbors, not tiles contained within or diagonal overlaps
        if miny_b < miny_a and maxy_b >= miny_a - tolerance and overlap_height >= tolerance:
            # Check if there's horizontal overlap
            if not (maxx_b < minx_a or minx_b > maxx_a):
                # Calculate alignment: same column = high alignment
                # For same column, the horizontal overlap should span most of BOTH tiles' widths
                # OR if overlap spans most of the smaller tile (indicates same column with size mismatch)
                tile_width_b = maxx_b - minx_b
                overlap_width_ratio_a = overlap_width / tile_width_a if tile_width_a > 0 else 0.0
                overlap_width_ratio_b = overlap_width / tile_width_b if tile_width_b > 0 else 0.0
                # High alignment if:
                # 1. Both tiles have >80% overlap (perfect alignment)
                # 2. Exact X bounds match (perfect alignment)
                # 3. Smaller tile has >80% overlap (indicates same column with size mismatch)
                max_ratio = max(overlap_width_ratio_a, overlap_width_ratio_b)
                x_alignment = 1.0 if (overlap_width_ratio_a > 0.8 and overlap_width_ratio_b > 0.8) or (minx_b == minx_a and maxx_b == maxx_a) or (max_ratio > 0.8) else 0.5
                # Edge alignment: check if X edges align (for same-column detection)
                # Tiles in same column should have minx or maxx very close (within 0.1m tolerance)
                edge_tolerance = 0.1
                left_edge_align = abs(minx_a - minx_b) < edge_tolerance
                right_edge_align = abs(maxx_a - maxx_b) < edge_tolerance
                # Higher score if BOTH edges align (perfect same column), medium if one aligns
                edge_alignment = 2.0 if (left_edge_align and right_edge_align) else (1.0 if (left_edge_align or right_edge_align) else 0.0)
                # Store (overlap_area, x_alignment, miny_a - miny_b, edge_alignment, other_name)
                # Priority: alignment (same column), then overlap area, then distance, then edge alignment
                south_overlaps.append((overlap_area, x_alignment, miny_a - miny_b, edge_alignment, other_name))
    
    # Pick the neighbor with the best alignment first, then largest overlap, then closest distance, then best edge alignment
    # Alignment (same row/column) is prioritized over overlap area to avoid diagonal neighbors
    if east_overlaps:
        best_east = max(east_overlaps, key=lambda x: (x[1], x[3], x[0], -x[2]))  # Alignment, edge alignment, overlap, min distance
        neighbors["east"] = best_east[4]  # other_name is now at index 4
        # Debug logging
        if len(east_overlaps) > 1:
            print(f"      DEBUG {tile_name} east neighbor: selected {best_east[4]} from {len(east_overlaps)} candidates (overlap: {best_east[0]:.2f} m², alignment: {best_east[1]:.1f}, distance: {best_east[2]:.2f}m)")
    if west_overlaps:
        best_west = max(west_overlaps, key=lambda x: (x[1], x[3], x[0], -x[2]))  # Alignment, edge alignment, overlap, min distance
        neighbors["west"] = best_west[4]  # other_name is now at index 4
        # Debug logging
        if len(west_overlaps) > 1:
            print(f"      DEBUG {tile_name} west neighbor: selected {best_west[4]} from {len(west_overlaps)} candidates (overlap: {best_west[0]:.2f} m², alignment: {best_west[1]:.1f}, distance: {best_west[2]:.2f}m)")
    if north_overlaps:
        best_north = max(north_overlaps, key=lambda x: (x[1], x[3], x[0], -x[2]))  # Alignment, edge alignment, overlap, min distance
        neighbors["north"] = best_north[4]  # other_name is now at index 4
        # Debug logging
        if len(north_overlaps) > 1:
            print(f"      DEBUG {tile_name} north neighbor: selected {best_north[4]} from {len(north_overlaps)} candidates (overlap: {best_north[0]:.2f} m², alignment: {best_north[1]:.1f}, distance: {best_north[2]:.2f}m)")
    if south_overlaps:
        best_south = max(south_overlaps, key=lambda x: (x[1], x[3], x[0], -x[2]))  # Alignment, edge alignment, overlap, min distance
        neighbors["south"] = best_south[4]  # other_name is now at index 4
        # Debug logging
        if len(south_overlaps) > 1:
            print(f"      DEBUG {tile_name} south neighbor: selected {best_south[4]} from {len(south_overlaps)} candidates (overlap: {best_south[0]:.2f} m², alignment: {best_south[1]:.1f}, distance: {best_south[2]:.2f}m)")
    
    return neighbors


def filter_by_centroid_in_buffer(
    points: np.ndarray,
    instances: np.ndarray,
    boundary: Tuple[float, float, float, float],
    tile_name: str,
    all_tiles: Dict[str, Tuple[float, float, float, float]],
    buffer: float = 10.0,
    precomputed_neighbors: Optional[Dict[str, Optional[str]]] = None,
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

    # Determine which edges have neighbors using either precomputed neighbors
    # from tile_bounds_tindex.json or spatial bounds as a fallback.
    if precomputed_neighbors is not None:
        neighbors = {
            "east": precomputed_neighbors.get("east"),
            "west": precomputed_neighbors.get("west"),
            "north": precomputed_neighbors.get("north"),
            "south": precomputed_neighbors.get("south"),
        }
    else:
        neighbors = find_spatial_neighbors(boundary, tile_name, all_tiles, tolerance=buffer)

    # Define buffer zone boundaries (only on edges with neighbors)
    # Simple approach: buffer meters from each edge that has a neighbor
    buf_min_x = min_x + (buffer if neighbors["west"] is not None else 0)
    buf_max_x = max_x - (buffer if neighbors["east"] is not None else 0)
    buf_min_y = min_y + (buffer if neighbors["south"] is not None else 0)
    buf_max_y = max_y - (buffer if neighbors["north"] is not None else 0)

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
    neighbors_by_tile: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    chunk_size: int = 1_000_000,
    instance_dimension: str = "PredInstance",
) -> Optional[Tuple[TileData, Set[int], Set[int], Dict[int, str]]]:
    """
    Load a LAZ tile using chunked reading for memory efficiency.

    Args:
        filepath: Path to the LAZ file
        all_tiles: Dictionary mapping tile names to their boundaries for neighbor detection
        buffer: Buffer distance for filtering
        chunk_size: Number of points to read per chunk (default 1M)
        instance_dimension: Name of the instance dimension (default: PredInstance, fallback: treeID)

    Returns:
        Tuple of (TileData, instances_to_remove, kept_instances, instance_buffer_direction) or None if loading fails
    """
    print(f"Loading {filepath.name}...")

    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.Lazrs) as f:
            n_points = f.header.point_count

            header_extra_dims = {dim.name: dim for dim in f.header.point_format.extra_dimensions}
            has_instance_dim = instance_dimension in header_extra_dims
            has_tree_id = "treeID" in header_extra_dims

            # Pre-allocate arrays
            points = np.empty((n_points, 3), dtype=np.float64)
            instances = np.zeros(n_points, dtype=np.int32)

            # Pre-allocate generic extra dims (all except the instance dimension)
            extra_dims: Dict[str, np.ndarray] = {}
            for dim in f.header.point_format.extra_dimensions:
                if dim.name == instance_dimension or (not has_instance_dim and dim.name == "treeID"):
                    continue
                extra_dims[dim.name] = np.zeros(n_points, dtype=dim.dtype)

            # Read in chunks to reduce peak memory
            offset = 0
            for chunk in f.chunk_iterator(chunk_size):
                chunk_len = len(chunk)
                end = offset + chunk_len

                points[offset:end, 0] = chunk.x
                points[offset:end, 1] = chunk.y
                points[offset:end, 2] = chunk.z

                if has_instance_dim:
                    instances[offset:end] = getattr(chunk, instance_dimension)
                elif has_tree_id:
                    instances[offset:end] = chunk.treeID

                for dim_name in extra_dims:
                    extra_dims[dim_name][offset:end] = getattr(chunk, dim_name)

                offset = end

    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None

    if not has_instance_dim and not has_tree_id:
        print(f"  Warning: No instance attribute ({instance_dimension}/treeID) found in {filepath}")

    boundary = compute_tile_bounds(points)

    tile_name = filepath.stem
    for suffix in ["_segmented_remapped", "_segmented", "_remapped"]:
        tile_name = tile_name.replace(suffix, "")

    neighbors_for_tile = None
    if neighbors_by_tile is not None:
        neighbors_for_tile = neighbors_by_tile.get(tile_name)

    instances_to_remove, instance_buffer_direction = filter_by_centroid_in_buffer(
        points, instances, boundary, tile_name, all_tiles, buffer, precomputed_neighbors=neighbors_for_tile
    )

    kept_instances = set(np.unique(instances)) - instances_to_remove - {0}

    print(
        f"  {len(points):,} points, {len(kept_instances)} instances kept, {len(instances_to_remove)} filtered"
    )

    return (
        TileData(
            name=tile_name,
            points=points,
            instances=instances,
            boundary=boundary,
            extra_dims=extra_dims,
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
        args: Tuple of (filepath, tile_boundaries, buffer, neighbors_by_tile, instance_dimension)
    
    Returns:
        Result from load_tile()
    """
    filepath, tile_boundaries, buffer, neighbors_by_tile, instance_dimension = args
    return load_tile(filepath, tile_boundaries, buffer, neighbors_by_tile, instance_dimension=instance_dimension)


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
    extra_dims: Dict[str, np.ndarray],
    tolerance: float = 0.01,
    grid_size: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Remove duplicate points from overlapping tiles using grid-based processing.
    When duplicates exist, keep the one with higher instance ID.

    Uses spatial grid cells to reduce memory usage - instead of sorting billions
    of points at once, we process smaller cells independently.

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        extra_dims: Dict of extra dimension name -> array (passenger data)
        tolerance: Distance tolerance (default 1cm)
        grid_size: Size of spatial grid cells in meters (default 50m)

    Returns:
        Tuple of (unique_points, unique_instances, unique_extra_dims)
    """
    n_points = len(points)
    scale = 1.0 / tolerance

    min_coords = points.min(axis=0)
    grid_indices = ((points[:, :2] - min_coords[:2]) / grid_size).astype(np.int32)

    max_grid_y = grid_indices[:, 1].max() + 1
    cell_keys = grid_indices[:, 0] * max_grid_y + grid_indices[:, 1]

    rounded = np.floor(points * scale).astype(np.int64)

    point_hash = rounded[:, 0] + rounded[:, 1] * 73856093 + rounded[:, 2] * 19349669

    sort_order = np.lexsort((-instances, point_hash, cell_keys))

    sorted_cell_keys = cell_keys[sort_order]
    sorted_point_hash = point_hash[sort_order]

    is_duplicate = np.zeros(n_points, dtype=bool)
    is_duplicate[1:] = (sorted_cell_keys[1:] == sorted_cell_keys[:-1]) & (
        sorted_point_hash[1:] == sorted_point_hash[:-1]
    )

    keep_mask = np.ones(n_points, dtype=bool)
    keep_mask[sort_order[is_duplicate]] = False

    unique_points = points[keep_mask]
    unique_instances = instances[keep_mask]
    unique_extras = {name: arr[keep_mask] for name, arr in extra_dims.items()}

    return unique_points, unique_instances, unique_extras


# =============================================================================
# Stage 3: FF3D Instance Matching (Border Region Instance Matching)
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


def compute_ff3d_overlap_ratios(
    instances_a: np.ndarray,
    instances_b: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    correspondence_tolerance: float = 0.1,
) -> Dict[Tuple[int, int], float]:
    """
    Compute FF3D-style overlap ratios between all instance pairs.

    Only counts points as "same point" if they're within correspondence_tolerance
    (should be small, ~10cm, to only match actual duplicate points from overlapping tiles).

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


def merge_small_volume_instances(
    points: np.ndarray,
    instances: np.ndarray,
    min_points_for_hull_check: int = 1000,
    min_cluster_size: int = 300,
    max_volume_for_merge: float = 4.0,
    max_search_radius: float = 5.0,
    num_threads: int = 1,
    verbose: bool = True,
    presorted_points: Optional[np.ndarray] = None,
    presorted_instances: Optional[np.ndarray] = None,
    presorted_unique_inst: Optional[np.ndarray] = None,
    presorted_first_idx: Optional[np.ndarray] = None,
    presorted_inst_counts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Merge small-volume instances to nearest large instance by centroid distance.

    Only reassigns instance IDs. Extra dims are untouched (values stay with their points).

    Logic:
    - For instances with >= min_points_for_hull_check (1000) points: Keep instance
    - For instances with < min_points_for_hull_check (1000) points:
      1. Calculate convex hull volume
      2. If volume < max_volume_for_merge (4.0 m³): Merge to nearest large instance
      3. Else if point_count < min_cluster_size: Redistribute to nearest instance
      4. Else: Keep instance

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs (modified in-place)
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
        Updated (instances, bbox_skipped_count)
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
        sorted_points = presorted_points
        sorted_instances = presorted_instances
        unique_inst = presorted_unique_inst
        first_idx = presorted_first_idx
        inst_counts = presorted_inst_counts
        print(f"  {len(unique_inst):,} unique instances (pre-sorted).", flush=True)
    else:
        total_points = len(instances)
        nonzero_mask = instances > 0
        nonzero_count = nonzero_mask.sum()
        
        instances_to_sort = instances[nonzero_mask]
        points_to_sort = points[nonzero_mask]
        
        print(f"  Sorting {nonzero_count:,} instance points (of {total_points:,} total)...", flush=True)
        sort_idx = np.argsort(instances_to_sort)
        sorted_instances = instances_to_sort[sort_idx]
        sorted_points = points_to_sort[sort_idx]

        unique_inst, first_idx, inst_counts = np.unique(
            sorted_instances, return_index=True, return_counts=True
        )
        print(f"  Found {len(unique_inst):,} unique instances.", flush=True)

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
    print(f"  Categorizing {total_instances:,} instances...", flush=True)

    for idx, (inst_id, start, count) in enumerate(zip(unique_inst, first_idx, inst_counts)):
        count = int(count)

        if total_instances >= 1000 and idx % 1000 == 0 and idx > 0:
            print(f"    {idx:,}/{total_instances:,} instances processed...", flush=True)

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
        return (instances, bbox_skipped_count)

    if len(large_instances) == 0:
        print(f"  No large instances to merge into", flush=True)
        if bbox_skipped_count > 0:
            print(
                f"  Skipped {bbox_skipped_count} instances using bounding box filter (bbox >= {max_volume_for_merge * 4.0:.1f} m³)", flush=True
            )
        return (instances, bbox_skipped_count)

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

    # Build lookup table for vectorized reassignment (instance IDs only)
    max_inst = instances.max() + 1
    inst_to_target = np.arange(max_inst, dtype=np.int32)  # Default: map to self

    # Determine targets for small instances
    if len(all_small_instances) > 0:
        small_centroids = np.array(
            [centroid for _, _, _, centroid in all_small_instances]
        )
        distances, indices = tree.query(small_centroids)

        total_merged = 0
        for i, (inst_id, count, volume, centroid) in enumerate(all_small_instances):
            distance = distances[i]
            idx = indices[i]

            if distance > max_search_radius:
                if verbose:
                    print(f"    ✗ Cluster {inst_id} ({count} pts) - no target within {max_search_radius}m")
                continue

            target_inst = large_ids[idx]
            inst_to_target[inst_id] = target_inst

            total_merged += count
            if verbose:
                print(f"    ✓ Cluster {inst_id} ({count} pts) → Instance {target_inst} (dist: {distance:.1f}m)")
    else:
        total_merged = 0

    # Vectorized reassignment - only instance IDs; extra dims stay with their points
    valid_mask = (instances > 0) & (instances < max_inst)
    instances[valid_mask] = inst_to_target[instances[valid_mask]]

    print(
        f"  Merged/redistributed {total_merged:,} points from {len(all_small_instances)} small instances "
        f"({len(small_volume_instances)} small-volume + {len(small_point_count_instances)} small point-count)", flush=True
    )

    return (instances, bbox_skipped_count)


# =============================================================================
# Load Existing Merged File
# =============================================================================


def load_merged_file(
    merged_file: Path,
    chunk_size: int = 1_000_000,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, laspy.ExtraBytesParams]]:
    """
    Load merged point cloud data from an existing LAZ file.
    All dimensions except X,Y,Z are returned in one dict (standard + extra; no special "instance" dimension).
    Used by remap_to_original_input_files so that standard dims (intensity, etc.) can be transferred too.

    Args:
        merged_file: Path to the merged LAZ file
        chunk_size: Number of points to read per chunk (default 1M)

    Returns:
        Tuple of (points, all_dims, extra_dim_params) where all_dims contains every
        dimension in the file except X,Y,Z and extra_dim_params preserves metadata
        for extra dimensions present in the merged file.
    """
    print(f"Loading existing merged file: {merged_file}")

    try:
        with laspy.open(
            str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel
        ) as f:
            n_points = f.header.point_count
            points = np.empty((n_points, 3), dtype=np.float64)
            all_dims: Dict[str, np.ndarray] = {}
            extra_dim_params: Dict[str, laspy.ExtraBytesParams] = {}
            offset = 0
            for chunk in f.chunk_iterator(chunk_size):
                chunk_len = len(chunk)
                end = offset + chunk_len
                points[offset:end, 0] = chunk.x
                points[offset:end, 1] = chunk.y
                points[offset:end, 2] = chunk.z
                # Standard dimensions (except X,Y,Z)
                for dim_name in f.header.point_format.dimension_names:
                    if dim_name in ("X", "Y", "Z"):
                        continue
                    arr = getattr(chunk, dim_name, None)
                    if arr is not None:
                        if dim_name not in all_dims:
                            all_dims[dim_name] = np.zeros(n_points, dtype=arr.dtype)
                        all_dims[dim_name][offset:end] = arr
                # Extra dimensions
                for dim in f.header.point_format.extra_dimensions:
                    if dim.name not in extra_dim_params:
                        extra_dim_params[dim.name] = extra_bytes_params_from_dimension_info(dim)
                    if dim.name not in all_dims:
                        all_dims[dim.name] = np.zeros(n_points, dtype=dim.dtype)
                    all_dims[dim.name][offset:end] = getattr(chunk, dim.name)
                offset = end

        print(f"  Loaded {len(points):,} points")
        if all_dims:
            print(f"  Dimensions from merged: {', '.join(sorted(all_dims.keys()))}")

        return points, all_dims, extra_dim_params

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
              merged_extra_dims, tolerance, spatial_buffer, kdtree_workers, instance_dimension)

    Returns:
        Tuple of (filename, matched_count, total_count, unique_instances, success, message)
    """
    (orig_file, output_file, merged_points, merged_instances, merged_extra_dims,
     merged_extra_dim_params, tolerance, spatial_buffer, kdtree_workers,
     instance_dimension) = args
    
    try:
        with laspy.open(
            str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel
        ) as f:
            bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
            n_orig_points = f.header.point_count

        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )

        local_merged_points = merged_points[mask]
        local_merged_instances = merged_instances[mask]
        local_merged_extras = {name: arr[mask] for name, arr in merged_extra_dims.items()}

        if len(local_merged_points) == 0:
            return (orig_file.name, 0, n_orig_points, 0, False, "No merged points in tile region")

        local_tree = cKDTree(local_merged_points)

        orig_las = laspy.read(
            str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel
        )
        orig_points = np.empty((n_orig_points, 3), dtype=np.float64)
        orig_points[:, 0] = orig_las.x
        orig_points[:, 1] = orig_las.y
        orig_points[:, 2] = orig_las.z

        distances, indices = local_tree.query(orig_points, workers=kdtree_workers)

        new_instances = local_merged_instances[indices]
        new_extras = {name: arr[indices] for name, arr in local_merged_extras.items()}

        new_header = laspy.LasHeader(
            point_format=orig_las.header.point_format,
            version=orig_las.header.version
        )
        new_header.offsets = orig_las.header.offsets
        new_header.scales = orig_las.header.scales
        
        output_las = laspy.LasData(new_header)
        
        # Collect all extra dims to add in one batch (avoids repeated data copies)
        orig_standard_names = set(orig_las.point_format.dimension_names)
        orig_extra_dim_names = {dim.name for dim in orig_las.point_format.extra_dimensions}
        orig_dim_names = orig_standard_names | orig_extra_dim_names
        merged_dim_names = {instance_dimension} | set(new_extras.keys())
        collision = orig_dim_names & merged_dim_names
        
        # When a dimension exists in both original and merged, rename to base_1 (original) and base_2 (merged)
        # Per dimension: use _1 and _2 for this base name only, so we get PredInstance_1/2, PredSemantic_1/2, not global _3/_4
        used_names = set(orig_dim_names)
        orig_rename = {}
        merged_rename = {}
        for name in sorted(collision):
            o1, o2 = _suffixes_for_collision(name, used_names)
            orig_rename[name] = o1
            merged_rename[name] = o2
        
        output_extra_dim_names = {dim.name for dim in output_las.point_format.extra_dimensions}
        extra_dims_to_add = []
        for dim in orig_las.point_format.extra_dimensions:
            out_name = orig_rename.get(dim.name, dim.name)
            if out_name not in output_extra_dim_names:
                extra_dims_to_add.append(extra_bytes_params_from_dimension_info(dim, name=out_name))
                output_extra_dim_names.add(out_name)
        
        inst_out_name = merged_rename.get(instance_dimension, instance_dimension)
        if inst_out_name not in output_extra_dim_names:
            extra_dims_to_add.append(laspy.ExtraBytesParams(name=inst_out_name, type=np.int32))
            output_extra_dim_names.add(inst_out_name)
        
        for dim_name, values in new_extras.items():
            out_name = merged_rename.get(dim_name, dim_name)
            if out_name not in output_extra_dim_names:
                if merged_extra_dim_params and dim_name in merged_extra_dim_params:
                    params = merged_extra_dim_params[dim_name]
                    extra_dims_to_add.append(laspy.ExtraBytesParams(name=out_name, type=params.type))
                else:
                    extra_dims_to_add.append(laspy.ExtraBytesParams(name=out_name, type=values.dtype))
                output_extra_dim_names.add(out_name)
        
        if extra_dims_to_add:
            output_las.add_extra_dims(extra_dims_to_add)
        
        # Copy original dimensions (standard + extra) so we preserve full precision
        for dim_name in orig_las.point_format.dimension_names:
            try:
                if hasattr(orig_las, dim_name):
                    setattr(output_las, dim_name, getattr(orig_las, dim_name))
            except Exception:
                pass
        for dim in orig_las.point_format.extra_dimensions:
            name = dim.name
            out_name = orig_rename.get(name, name)
            if hasattr(orig_las, name):
                try:
                    setattr(output_las, out_name, getattr(orig_las, name))
                except Exception:
                    pass
        
        # Write merged dimensions (under original name or renamed to base_2 when collision)
        setattr(output_las, merged_rename.get(instance_dimension, instance_dimension), new_instances)
        for dim_name, values in new_extras.items():
            setattr(output_las, merged_rename.get(dim_name, dim_name), values)

        output_las.write(
            str(output_file),
            do_compress=True,
            laz_backend=laspy.LazBackend.LazrsParallel,
        )
        
        del orig_las
        del output_las

        matched = n_orig_points
        unique_inst = len(np.unique(new_instances[new_instances > 0]))

        return (orig_file.name, matched, n_orig_points, unique_inst, True, "OK")
    
    except Exception as e:
        return (orig_file.name, 0, 0, 0, False, str(e))


def retile_to_original_files(
    merged_points: np.ndarray,
    merged_instances: np.ndarray,
    merged_extra_dims: Dict[str, np.ndarray],
    merged_extra_dim_params: Optional[Dict[str, laspy.ExtraBytesParams]],
    original_tiles_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
    chunk_size: int = 1_000_000,
    parallel_tiles: int = 1,
    retile_buffer: float = 2.0,
    instance_dimension: str = "PredInstance",
):
    """
    Map merged instance IDs back to original tile point clouds.

    Args:
        merged_points: Merged point cloud coordinates
        merged_instances: Merged instance IDs
        merged_extra_dims: Dict of extra dimension name -> array (passenger data from merge)
        original_tiles_dir: Directory containing original tile files
        output_dir: Directory to write output files
        tolerance: Distance tolerance for point matching
        num_threads: Number of threads for KDTree queries (per tile)
        chunk_size: Chunk size for reading large files
        parallel_tiles: Number of tiles to process in parallel (default: 1 = sequential)
        retile_buffer: Additional spatial buffer in meters (fixed: 2.0m)
        instance_dimension: Name of the instance dimension
    """
    import gc
    from concurrent.futures import ThreadPoolExecutor

    print(f"\n{'=' * 60}", flush=True)
    print("Retiling merged results to original tile files", flush=True)
    print(f"{'=' * 60}", flush=True)

    original_files = sorted(original_tiles_dir.glob("*.laz"))
    if not original_files:
        original_files = sorted(original_tiles_dir.glob("*.las"))

    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_tiles_dir}", flush=True)
        return

    print(f"  Found {len(original_files)} original tile files", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer

    tiles_to_process = []
    skipped = 0
    for orig_file in original_files:
        output_name = orig_file.name.replace('.copc.laz', '.laz')
        output_file = output_dir / output_name
        if output_file.exists():
            skipped += 1
        else:
            tiles_to_process.append((orig_file, output_file))
    
    if skipped > 0:
        print(f"  Skipping {skipped} already processed tiles", flush=True)
    
    if len(tiles_to_process) == 0:
        print(f"  All tiles already processed!", flush=True)
        return
    
    print(f"  Processing {len(tiles_to_process)} tiles...", flush=True)

    kdtree_workers = -1
    
    process_args = [
        (orig_file, output_file, merged_points, merged_instances, merged_extra_dims,
         merged_extra_dim_params, tolerance, spatial_buffer, kdtree_workers,
         instance_dimension)
        for orig_file, output_file in tiles_to_process
    ]

    if parallel_tiles > 1:
        completed = 0
        with ThreadPoolExecutor(max_workers=parallel_tiles) as executor:
            for result in executor.map(_process_single_tile, process_args):
                filename, matched, total, unique_inst, success, message = result
                completed += 1
                match_pct = (matched / total * 100) if total > 0 else 0
                
                if success:
                    print(f"  [{completed}/{len(tiles_to_process)}] {filename}: {matched:,}/{total:,} matched ({match_pct:.1f}%), {unique_inst} instances", flush=True)
                else:
                    print(f"  [{completed}/{len(tiles_to_process)}] {filename}: FAILED - {message}", flush=True)
    else:
        for i, args in enumerate(process_args):
            orig_file = args[0]
            
            result = _process_single_tile(args)
            filename, matched, total, unique_inst, success, message = result
            
            if success:
                match_pct = (matched / total * 100) if total > 0 else 0
                print(f"  [{i+1}/{len(tiles_to_process)}] {matched:,}/{total:,} matched ({match_pct:.1f}%), {unique_inst} instances → {filename}", flush=True)
            else:
                print(f"  [{i+1}/{len(tiles_to_process)}] FAILED: {message} → {filename}", flush=True)
            
            gc.collect()

    print(f"\n  ✓ Retiling complete: {len(tiles_to_process)} tiles processed", flush=True)
    gc.collect()


# =============================================================================
# Stage 7: Remap to Original Input Files
# =============================================================================


def _process_single_original_input_file(args):
    """
    Process a single original input LAZ file for remapping. Designed to be called in parallel.
    Transfers only 3DTrees-specific dimensions from merged (by nearest-neighbor) to the original points,
    renaming them to 3DT_{name}_{suffix} format (e.g. 3DT_PredInstance_SAT).

    Args:
        args: Tuple of (input_file, output_file, merged_points, merged_extra_dims,
              merged_extra_dim_params, tolerance, spatial_buffer, kdtree_workers,
              threedtrees_dims, threedtrees_suffix)

    Returns:
        Tuple of (filename, matched_count, total_count, unique_instances, success, message)
    """
    (input_file, output_file, merged_points, merged_extra_dims,
     merged_extra_dim_params, tolerance, spatial_buffer, kdtree_workers,
     threedtrees_dims, threedtrees_suffix) = args
    
    try:
        with laspy.open(
            str(input_file), laz_backend=laspy.LazBackend.LazrsParallel
        ) as f:
            bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
            n_input_points = f.header.point_count

        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )

        local_merged_points = merged_points[mask]
        local_merged_extras = {name: arr[mask] for name, arr in merged_extra_dims.items()}

        if len(local_merged_points) == 0:
            return (input_file.name, 0, n_input_points, 0, False, "No merged points in file region")

        local_tree = cKDTree(local_merged_points)

        input_las = laspy.read(
            str(input_file), laz_backend=laspy.LazBackend.LazrsParallel
        )
        input_points = np.empty((n_input_points, 3), dtype=np.float64)
        input_points[:, 0] = input_las.x
        input_points[:, 1] = input_las.y
        input_points[:, 2] = input_las.z

        distances, indices = local_tree.query(input_points, workers=kdtree_workers)

        # Filter merged dims to only 3DTrees-specific dimensions
        if threedtrees_dims:
            filtered_extras = {name: arr[indices] for name, arr in local_merged_extras.items()
                               if name in threedtrees_dims}
        else:
            # No filtering — transfer all (fallback, should not normally happen)
            filtered_extras = {name: arr[indices] for name, arr in local_merged_extras.items()}

        matched_count = n_input_points
        # Heuristic: count unique non-zero values from first int-like dimension for reporting
        unique_instances = 0
        for arr in filtered_extras.values():
            if np.issubdtype(arr.dtype, np.integer) and len(arr) > 0:
                unique_instances = max(unique_instances, len(np.unique(arr[arr > 0])))
                break

        # Build branded output names: 3DT_{name}_{suffix} (e.g. 3DT_PredInstance_SAT)
        branded_names = {}
        for dim_name in filtered_extras:
            if threedtrees_suffix:
                branded_names[dim_name] = f"3DT_{dim_name}_{threedtrees_suffix}"
            else:
                branded_names[dim_name] = f"3DT_{dim_name}"

        # Use original file's point format so output preserves original structure
        fmt_id = getattr(input_las.header.point_format, "id", input_las.header.point_format)
        if hasattr(fmt_id, "id"):
            fmt_id = fmt_id.id
        new_header = laspy.LasHeader(
            point_format=int(fmt_id),
            version=input_las.header.version
        )
        new_header.offsets = input_las.header.offsets
        new_header.scales = input_las.header.scales

        output_las = laspy.LasData(new_header)
        output_standard_dim_names = set(output_las.point_format.dimension_names)

        extra_dims_to_add = []
        added_extra_names = set(output_standard_dim_names)  # start with standard names to avoid duplicates
        # Original extra dimensions — keep as-is (skip if name clashes with standard dims)
        for dim in input_las.point_format.extra_dimensions:
            if dim.name not in added_extra_names:
                extra_dims_to_add.append(extra_bytes_params_from_dimension_info(dim))
                added_extra_names.add(dim.name)
        # Original "standard" dims not in the chosen standard format → add as extra
        for dim_name in input_las.point_format.dimension_names:
            if dim_name not in added_extra_names:
                arr = getattr(input_las, dim_name, None)
                dtype = arr.dtype if arr is not None else np.int32
                extra_dims_to_add.append(laspy.ExtraBytesParams(name=dim_name, type=dtype))
                added_extra_names.add(dim_name)
        # 3DTrees branded dimensions from merged file
        for dim_name, values in filtered_extras.items():
            out_name = branded_names[dim_name]
            if out_name not in added_extra_names:
                if merged_extra_dim_params and dim_name in merged_extra_dim_params:
                    params = merged_extra_dim_params[dim_name]
                    extra_dims_to_add.append(laspy.ExtraBytesParams(name=out_name, type=params.type))
                else:
                    extra_dims_to_add.append(laspy.ExtraBytesParams(name=out_name, type=values.dtype))
                added_extra_names.add(out_name)

        if extra_dims_to_add:
            output_las.add_extra_dims(extra_dims_to_add)

        # Copy all original dimensions (standard + extra) as-is
        for dim_name in output_las.point_format.dimension_names:
            try:
                if hasattr(input_las, dim_name):
                    setattr(output_las, dim_name, getattr(input_las, dim_name))
            except Exception:
                pass
        for dim in input_las.point_format.extra_dimensions:
            if hasattr(input_las, dim.name):
                try:
                    setattr(output_las, dim.name, getattr(input_las, dim.name))
                except Exception:
                    pass
        # Write 3DTrees branded dimensions
        for dim_name, values in filtered_extras.items():
            setattr(output_las, branded_names[dim_name], values)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_las.write(
            str(output_file),
            do_compress=True,
            laz_backend=laspy.LazBackend.LazrsParallel,
        )

        del input_las
        del output_las

        return (input_file.name, matched_count, n_input_points, unique_instances, True, "Success")

    except Exception as e:
        return (input_file.name, 0, 0, 0, False, str(e))


def _validate_common_dimensions_minmax(original_path: Path, output_path: Path, rel_tol: float = 1e-5) -> None:
    """Compare min/max of common dimensions between original and output; warn if they differ (rounding/loss)."""
    try:
        orig = laspy.read(str(original_path), laz_backend=laspy.LazBackend.LazrsParallel)
        out = laspy.read(str(output_path), laz_backend=laspy.LazBackend.LazrsParallel)
    except Exception as e:
        print(f"  Validation skip: could not read files ({e})", flush=True)
        return
    try:
        orig_names = set(orig.point_format.dimension_names) | {d.name for d in orig.point_format.extra_dimensions}
        out_names = set(out.point_format.dimension_names) | {d.name for d in out.point_format.extra_dimensions}
        common = orig_names & out_names - {"X", "Y", "Z"}
        if not common:
            return
        diffs = []
        for name in sorted(common):
            oa = getattr(orig, name, None)
            oo = getattr(out, name, None)
            if oa is None or oo is None or len(oa) != len(oo):
                continue
            oa, oo = np.asarray(oa), np.asarray(oo)
            omin, omax = float(np.min(oa)), float(np.max(oa))
            wmin, wmax = float(np.min(oo)), float(np.max(oo))
            if np.issubdtype(oa.dtype, np.integer) and np.issubdtype(oo.dtype, np.integer):
                if (omin != wmin or omax != wmax) and (int(omin) != int(wmin) or int(omax) != int(wmax)):
                    diffs.append((name, omin, omax, wmin, wmax))
            else:
                span = max(omax - omin, 1e-12)
                if abs(omin - wmin) > rel_tol * span or abs(omax - wmax) > rel_tol * span:
                    diffs.append((name, omin, omax, wmin, wmax))
        if diffs:
            print(f"  Warning: common dimensions min/max differ (output may have rounding/loss):", flush=True)
            for name, omin, omax, wmin, wmax in diffs:
                print(f"    {name}: original [{omin}, {omax}] vs output [{wmin}, {wmax}]", flush=True)
            print(f"  Tip: dimensions above were not overwritten by merged where range would be lost.", flush=True)
        del orig
        del out
    except Exception as e:
        print(f"  Validation skip: {e}", flush=True)


def remap_to_original_input_files(
    merged_points: np.ndarray,
    merged_extra_dims: Dict[str, np.ndarray],
    merged_extra_dim_params: Optional[Dict[str, laspy.ExtraBytesParams]],
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    num_threads: int = 8,
    retile_buffer: float = 2.0,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
):
    """
    Transfer 3DTrees dimensions from merged to original input LAZ files by nearest-neighbor.
    Only dimensions listed in threedtrees_dims are transferred, renamed to 3DT_{name}_{suffix}.

    Args:
        merged_points: Merged point cloud coordinates (N, 3)
        merged_extra_dims: Dict of all extra dimension name -> array (from merged file)
        original_input_dir: Directory containing original input LAZ files (pre-tiling)
        output_dir: Directory to write output files
        tolerance: Distance tolerance for point matching (used for spatial buffer calculation)
        num_threads: Number of threads for KDTree queries (default: 8)
        retile_buffer: Additional spatial buffer in meters (fixed: 2.0m)
        threedtrees_dims: List of dimension names to transfer (default: ["PredInstance", "PredSemantic"])
        threedtrees_suffix: Suffix for branding (e.g. "SAT" → 3DT_PredInstance_SAT)
    """
    from concurrent.futures import ThreadPoolExecutor
    
    print(f"\n{'=' * 60}", flush=True)
    print("Remapping to original input files", flush=True)
    print(f"{'=' * 60}", flush=True)

    original_files = sorted(original_input_dir.glob("*.laz"))
    if not original_files:
        original_files = sorted(original_input_dir.glob("*.las"))

    if len(original_files) == 0:
        print(f"  No LAZ/LAS files found in {original_input_dir}", flush=True)
        return
    
    original_files = [f for f in original_files if not f.name.endswith('.copc.laz')]
    
    if len(original_files) == 0:
        print(f"  No original input files found (only COPC files present)", flush=True)
        return

    print(f"  Found {len(original_files)} original input files", flush=True)
    print(f"  Output: {output_dir}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer

    files_to_process = []
    skipped = 0
    for input_file in original_files:
        output_name = input_file.name.replace('.copc.laz', '.laz')
        output_file = output_dir / output_name
        if output_file.exists():
            skipped += 1
        else:
            files_to_process.append((input_file, output_file))
    
    if skipped > 0:
        print(f"  Skipping {skipped} already processed files", flush=True)
    
    if len(files_to_process) == 0:
        print(f"  All files already processed!", flush=True)
        return
    
    print(f"  Processing {len(files_to_process)} files...", flush=True)

    kdtree_workers = -1
    
    # Default threedtrees_dims if not provided
    if threedtrees_dims is None:
        threedtrees_dims = ["PredInstance", "PredSemantic"]
    threedtrees_dims_set = set(threedtrees_dims)

    # Log which dims will be transferred
    available_3dt = sorted(threedtrees_dims_set & set(merged_extra_dims.keys()))
    if available_3dt:
        branded = [f"3DT_{d}_{threedtrees_suffix}" if threedtrees_suffix else f"3DT_{d}" for d in available_3dt]
        print(f"  3DTrees dimensions to transfer: {', '.join(available_3dt)} → {', '.join(branded)}", flush=True)
    else:
        print(f"  Warning: No 3DTrees dimensions found in merged file (looked for: {', '.join(sorted(threedtrees_dims_set))})", flush=True)

    process_args = [
        (input_file, output_file, merged_points, merged_extra_dims,
         merged_extra_dim_params, tolerance, spatial_buffer, kdtree_workers,
         threedtrees_dims_set, threedtrees_suffix)
        for input_file, output_file in files_to_process
    ]

    total_matched = 0
    total_points = 0

    # Use ThreadPoolExecutor: KDTree.query releases GIL so threads give real speedup
    parallel_workers = min(num_threads, len(files_to_process)) if num_threads > 1 else 1
    if parallel_workers > 1:
        print(f"  Processing with {parallel_workers} parallel workers...", flush=True)
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for i, result in enumerate(executor.map(_process_single_original_input_file, process_args)):
                filename, matched, total, unique_inst, success, message = result
                if success:
                    match_pct = (matched / total * 100) if total > 0 else 0
                    print(f"  [{i+1}/{len(files_to_process)}] {matched:,}/{total:,} matched ({match_pct:.1f}%), {unique_inst} instances → {filename}", flush=True)
                    total_matched += matched
                    total_points += total
                else:
                    print(f"  [{i+1}/{len(files_to_process)}] FAILED: {message} → {filename}", flush=True)
    else:
        for i, args in enumerate(process_args):
            result = _process_single_original_input_file(args)
            filename, matched, total, unique_inst, success, message = result
            if success:
                match_pct = (matched / total * 100) if total > 0 else 0
                print(f"  [{i+1}/{len(files_to_process)}] {matched:,}/{total:,} matched ({match_pct:.1f}%), {unique_inst} instances → {filename}", flush=True)
                total_matched += matched
                total_points += total
            else:
                print(f"  [{i+1}/{len(files_to_process)}] FAILED: {message} → {filename}", flush=True)
            gc.collect()

    overall_match_pct = (total_matched / total_points * 100) if total_points > 0 else 0
    print(f"\n  ✓ Remap complete: {len(files_to_process)} files, {total_matched:,}/{total_points:,} matched ({overall_match_pct:.1f}%)", flush=True)

    # Validate: compare min/max of common dimensions between first output and original
    if files_to_process and total_matched > 0:
        first_input, first_output = files_to_process[0]
        if first_output.exists():
            _validate_common_dimensions_minmax(first_input, first_output)

    gc.collect()


def _dims_to_fill_from_source(
    source_dims: Dict[str, np.dtype],
    target_dim_names: Set[str],
    get_target_array,
    skip: Optional[Set[str]] = None,
) -> Tuple[Dict[str, np.dtype], Dict[str, np.dtype]]:
    """
    Shared logic for "which dimensions to fill from source": add new (in source not in target)
    and overwrite (in target but empty/constant, use source). Used by add_original_dimensions_to_merged
    and by remap_to_original_input_files.
    Returns (dims_to_add_new, dims_to_overwrite), both name -> dtype.
    """
    skip = skip or {"X", "Y", "Z"}
    dims_to_add_new = {k: v for k, v in source_dims.items() if k not in target_dim_names and k not in skip}
    dims_to_overwrite: Dict[str, np.dtype] = {}
    for dim_name in (target_dim_names & set(source_dims.keys())) - skip:
        arr = get_target_array(dim_name)
        if arr is None:
            continue
        a = np.asarray(arr)
        if np.min(a) == np.max(a) or np.count_nonzero(a) == 0:
            dims_to_overwrite[dim_name] = source_dims[dim_name]
    return dims_to_add_new, dims_to_overwrite


def add_original_dimensions_to_merged(
    merged_laz: Path,
    original_input_dir: Path,
    output_path: Path,
    tolerance: float = 0.1,
    retile_buffer: float = 2.0,
    distance_threshold: Optional[float] = None,
    num_threads: int = 4,
) -> None:
    """
    Add dimensions from original input files to the merged point cloud by
    nearest-neighbor matching, and write an enriched merged LAZ file.

    Only dimensions that exist in originals but NOT in the merged file are added
    (merged dimensions are never overwritten). X, Y, Z are never copied from originals.

    Args:
        merged_laz: Path to the merged LAZ file.
        original_input_dir: Directory containing original LAZ/LAS files (pre-tiling).
        output_path: Path for the output merged LAZ with original dimensions added.
        tolerance: Distance tolerance for spatial buffer calculation.
        retile_buffer: Additional spatial buffer in meters.
        distance_threshold: Max distance for accepting a match (default: 2 * tolerance + retile_buffer).
        num_threads: Number of original files to process in parallel (default 4). Use 1 for sequential.
    """
    if output_path.resolve() == Path(merged_laz).resolve():
        raise ValueError("output_path must differ from merged_laz to avoid overwriting input")

    original_files = sorted(original_input_dir.glob("*.laz"))
    if not original_files:
        original_files = sorted(original_input_dir.glob("*.las"))
    original_files = [f for f in original_files if not f.name.endswith(".copc.laz")]
    if not original_files:
        print("  No original input files found; skipping merged-with-originals output.", flush=True)
        return

    print(f"\n{'=' * 60}", flush=True)
    print("Adding original-file dimensions to merged point cloud", flush=True)
    print(f"{'=' * 60}", flush=True)

    merged = laspy.read(str(merged_laz), laz_backend=laspy.LazBackend.LazrsParallel)
    n_merged = len(merged.points)
    merged_points = np.column_stack([merged.x, merged.y, merged.z])
    merged_dim_names = set(merged.point_format.dimension_names)
    for dim in merged.point_format.extra_dimensions:
        merged_dim_names.add(dim.name)

    skip_core = {"X", "Y", "Z"}
    # Collect all dimensions present in originals with their real dtypes so we
    # preserve packed flag fields and scaled extra dimensions.
    orig_dims: Dict[str, np.dtype] = {}
    orig_extra_dim_info: Dict[str, object] = {}
    for orig_path in original_files:
        with laspy.open(str(orig_path), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            pf = f.header.point_format
            # Read one point so we can inspect laspy views for packed subfields like
            # return_number / withheld as well as any scaled extra bytes.
            pt_dtype = None
            try:
                one = f.read_points(1)
                if one is not None and one.size > 0:
                    arr = getattr(one, "array", one)
                    dt = getattr(arr, "dtype", None)
                    if dt is not None and getattr(dt, "names", None) is not None:
                        pt_dtype = dt
                if one is not None and one.size > 0:
                    for dim_name in pf.dimension_names:
                        if dim_name in skip_core or dim_name in orig_dims:
                            continue
                        dim_view = getattr(one, dim_name, None)
                        if dim_view is not None and hasattr(dim_view, "dtype"):
                            orig_dims[dim_name] = np.dtype(dim_view.dtype)
                        elif pt_dtype is not None and dim_name in pt_dtype.names:
                            orig_dims[dim_name] = pt_dtype.fields[dim_name][0]
                else:
                    one = None
            except Exception:
                one = None
            if one is None or pt_dtype is None:
                for dim_name in pf.dimension_names:
                    if dim_name in skip_core or dim_name in orig_dims:
                        continue
                    orig_dims[dim_name] = np.float64
            for dim in pf.extra_dimensions:
                if dim.name in skip_core:
                    continue
                if dim.name not in orig_dims:
                    orig_dims[dim.name] = dim.dtype
                if dim.name not in orig_extra_dim_info:
                    orig_extra_dim_info[dim.name] = dim

    # Reuse shared logic: which dims to add new and which to overwrite (empty/constant in target)
    dims_to_add_new, dims_to_overwrite = _dims_to_fill_from_source(
        orig_dims, merged_dim_names, lambda n: getattr(merged, n, None), skip=skip_core
    )
    # When a dimension exists in both merged and originals and we're not overwriting (merged has real data),
    # add the original dimension under a new name (base_1) so we don't lose it. Per dimension: use _1.
    used_names = set(merged_dim_names) | set(dims_to_add_new.keys()) | set(dims_to_overwrite.keys())
    collision = (set(orig_dims.keys()) & merged_dim_names) - set(dims_to_overwrite.keys()) - skip_core
    orig_rename_for_merged: Dict[str, str] = {}
    for name in sorted(collision):
        cand = f"{name}_1"
        out_name = cand if cand not in used_names else _next_available_suffix(name, used_names)
        orig_rename_for_merged[name] = out_name
        used_names.add(out_name)
    dims_to_add_renamed = {orig_rename_for_merged[n]: orig_dims[n] for n in collision}
    dims_to_add = {**dims_to_add_new, **dims_to_overwrite, **dims_to_add_renamed}
    # Map each output dimension name -> original dimension name to read from originals
    orig_dim_to_read = {k: k for k in dims_to_add_new} | {k: k for k in dims_to_overwrite}
    orig_dim_to_read.update({out_name: name for name, out_name in orig_rename_for_merged.items()})

    if not dims_to_add:
        print("  No dimensions to add or replace from originals; writing copy of merged file.", flush=True)
        merged.write(str(output_path), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
        del merged
        gc.collect()
        return

    for dim_name in sorted(dims_to_add_new.keys()):
        print(f"  Adding dimension: {dim_name}", flush=True)
    for dim_name in sorted(dims_to_overwrite.keys()):
        print(f"  Replacing dimension (was empty/constant): {dim_name}", flush=True)
    for orig_name, out_name in sorted(orig_rename_for_merged.items()):
        print(f"  Adding dimension from originals (collision with merged): {orig_name} -> {out_name}", flush=True)

    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
    max_dist = distance_threshold if distance_threshold is not None else spatial_buffer

    best_dist = np.full(n_merged, np.inf, dtype=np.float64)
    new_arrays: Dict[str, np.ndarray] = {}
    for name, dtype in dims_to_add.items():
        new_arrays[name] = np.zeros(n_merged, dtype=dtype)

    def _process_one_original(orig_path: Path):
        """Process one original file; returns (merged_idx, distances, orig_idx, orig_dim_arrays) or None."""
        try:
            orig_las = laspy.read(str(orig_path), laz_backend=laspy.LazBackend.LazrsParallel)
        except Exception:
            return None
        bounds = (
            orig_las.header.x_min,
            orig_las.header.x_max,
            orig_las.header.y_min,
            orig_las.header.y_max,
        )
        orig_points = np.column_stack([orig_las.x, orig_las.y, orig_las.z])
        mask = (
            (merged_points[:, 0] >= bounds[0] - spatial_buffer)
            & (merged_points[:, 0] <= bounds[1] + spatial_buffer)
            & (merged_points[:, 1] >= bounds[2] - spatial_buffer)
            & (merged_points[:, 1] <= bounds[3] + spatial_buffer)
        )
        merged_idx = np.where(mask)[0]
        if len(merged_idx) == 0:
            del orig_las
            return None
        tree = cKDTree(orig_points)
        distances, orig_idx = tree.query(merged_points[merged_idx], k=1, workers=1)
        if distances.ndim == 2:
            distances = distances[:, 0]
            orig_idx = orig_idx[:, 0]
        orig_dim_arrays = {}
        for out_name, orig_name in orig_dim_to_read.items():
            arr = getattr(orig_las, orig_name, None)
            if arr is not None:
                orig_dim_arrays[out_name] = np.asarray(arr)
        del orig_las
        return (merged_idx, distances, orig_idx, orig_dim_arrays)

    if num_threads > 1:
        print(f"  Processing {len(original_files)} original files with {num_threads} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(_process_one_original, original_files))
    else:
        results = [_process_one_original(p) for p in original_files]

    for result in results:
        if result is None:
            continue
        merged_idx, distances, orig_idx, orig_dim_arrays = result
        # Vectorised best-distance update (replaces slow Python-level loop)
        within_max = distances <= max_dist
        better = distances < best_dist[merged_idx]
        accept = within_max & better
        if np.any(accept):
            acc_merged = merged_idx[accept]
            acc_orig = orig_idx[accept]
            best_dist[acc_merged] = distances[accept]
            for dim_name, arr_np in orig_dim_arrays.items():
                new_arrays[dim_name][acc_merged] = arr_np[acc_orig]
    gc.collect()

    # Check min/max and warn if dimension is empty or constant
    for name, arr in new_arrays.items():
        arr = np.asarray(arr)
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        n_nonzero = int(np.count_nonzero(arr))
        if vmin == vmax or n_nonzero == 0:
            print(f"  Warning: {name} has no variation (min=max={vmin}, non-zero={n_nonzero})", flush=True)
        else:
            print(f"  {name}: min={vmin}, max={vmax}, non-zero={n_nonzero}", flush=True)

    # Only add new extra dims for dimensions not already in merged; overwrite existing in place
    extra_params = []
    for name, dtype in dims_to_add_new.items():
        if name in orig_extra_dim_info:
            extra_params.append(
                extra_bytes_params_from_dimension_info(orig_extra_dim_info[name], name=name)
            )
        else:
            extra_params.append(laspy.ExtraBytesParams(name=name, type=dtype))
    # Collision dimensions from originals (added under base_1, etc.)
    orig_name_by_out = {out_name: name for name, out_name in orig_rename_for_merged.items()}
    for out_name, dtype in dims_to_add_renamed.items():
        orig_name = orig_name_by_out.get(out_name)
        if orig_name is not None and orig_name in orig_extra_dim_info:
            extra_params.append(
                extra_bytes_params_from_dimension_info(orig_extra_dim_info[orig_name], name=out_name)
            )
        else:
            extra_params.append(laspy.ExtraBytesParams(name=out_name, type=dtype))
    if extra_params:
        merged.add_extra_dims(extra_params)
    # Set all dimensions; cast to merged point record dtype to avoid laspy left_shift TypeError.
    for name, arr in new_arrays.items():
        arr = np.asarray(arr)
        try:
            target_dtype = getattr(merged.points, name).dtype
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
        except (AttributeError, KeyError, TypeError):
            pass
        setattr(merged, name, arr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(output_path), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
    del merged
    gc.collect()
    print(f"  Saved merged with original dimensions: {output_path}", flush=True)


# =============================================================================
# Main Merge Function
# =============================================================================


def merge_tiles(
    input_dir: Path,
    original_tiles_dir: Path,
    output_merged: Path,
    output_tiles_dir: Path,
    tile_bounds_json: Path,
    original_input_dir: Optional[Path] = None,
    buffer: float = 10.0,
    overlap_threshold: float = 0.3,
    correspondence_tolerance: float = 0.1,
    max_volume_for_merge: float = 4.0,
    border_zone_width: float = 10.0,
    min_cluster_size: int = 300,
    num_threads: int = 8,
    enable_matching: bool = True,
    enable_volume_merge: bool = True,
    skip_merged_file: bool = False,
    verbose: bool = False,
    retile_buffer: float = 2.0,
    retile_max_radius: float = 2.0,
    debug_instance_ids: Optional[Set[int]] = None,
    match_all_instances: bool = False,
    instance_dimension: str = "PredInstance",
    transfer_original_dims_to_merged: bool = True,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
):
    """
    Main merge function implementing the tile merging pipeline.
    """
    if tile_bounds_json is None:
        raise ValueError("tile_bounds_json is required but was not provided.")
    if not tile_bounds_json.exists():
        raise FileNotFoundError(
            f"tile_bounds_tindex.json not found: {tile_bounds_json}. "
            "Merge requires this file and will not run without it."
        )

    print("=" * 60)
    print("Tile Merger")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Original tiles: {original_tiles_dir}")
    print(f"Output merged: {output_merged}" + (" (SKIPPED)" if skip_merged_file else ""))
    print(f"Output tiles: {output_tiles_dir}")
    print(f"Tile bounds JSON: {tile_bounds_json}")
    print(f"Instance dimension: {instance_dimension}")
    print(f"Buffer: {buffer}m")
    print(f"Workers: {num_threads}")
    print(f"Instance matching: {'ENABLED' if enable_matching else 'DISABLED'}")
    if enable_matching:
        print(f"  Overlap threshold: {overlap_threshold}")
        print(f"  Match all instances: {'YES' if match_all_instances else 'NO (border region only)'}")
    print(f"Small cluster reassignment: ENABLED")
    print(f"  Min cluster size: {min_cluster_size} points")
    print(f"Volume merge: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    if enable_volume_merge:
        print(f"  Max volume for merge: {max_volume_for_merge} m³")
    if original_input_dir:
        print(f"Original input dir: {original_input_dir} (Stage 7 enabled)")
    print(f"Verbose: {verbose}")
    print("=" * 60)

    # Check if merged output file already exists
    if output_merged.exists():
        print(f"\n{'=' * 60}")
        print(f"Merged file already exists: {output_merged}")
        print(f"{'=' * 60}")
        print("  Loading merged file and proceeding to retiling stage...")
        
        merged_points, all_merged_dims, merged_extra_dim_params = load_merged_file(output_merged)
        # For retile we need (instances, extra_dims) split; for remap we pass all_merged_dims as-is
        if instance_dimension in all_merged_dims:
            merged_instances = all_merged_dims[instance_dimension]
            merged_extra_dims = {k: v for k, v in all_merged_dims.items() if k != instance_dimension}
        elif "treeID" in all_merged_dims:
            merged_instances = all_merged_dims["treeID"]
            merged_extra_dims = {k: v for k, v in all_merged_dims.items() if k != "treeID"}
        else:
            merged_instances = None
            merged_extra_dims = {}
            for k, v in all_merged_dims.items():
                if merged_instances is None and np.issubdtype(v.dtype, np.integer):
                    merged_instances = v
                else:
                    merged_extra_dims[k] = v
            if merged_instances is None:
                merged_instances = np.zeros(len(merged_points), dtype=np.int32)

        retile_to_original_files(
            merged_points,
            merged_instances,
            merged_extra_dims,
            merged_extra_dim_params,
            original_tiles_dir,
            output_tiles_dir,
            tolerance=0.1,
            num_threads=num_threads,
            retile_buffer=retile_buffer,
            instance_dimension=instance_dimension,
        )
        print(f"  ✓ Stage 6 completed: Retiled to original files")

        if original_input_dir is not None:
            print(f"\n{'=' * 60}")
            print("Stage 7: Remapping to Original Input Files")
            print(f"{'=' * 60}")
            
            original_output_dir = output_tiles_dir.parent / "original_with_predictions"
            remap_to_original_input_files(
                merged_points,
                all_merged_dims,
                merged_extra_dim_params,
                original_input_dir,
                original_output_dir,
                tolerance=retile_max_radius,
                num_threads=num_threads,
                retile_buffer=retile_buffer,
                threedtrees_dims=threedtrees_dims,
                threedtrees_suffix=threedtrees_suffix,
            )
            print(f"  ✓ Stage 7 completed: Remapped to original input files")
        else:
            print(f"\n  Note: --original-input-dir not provided, skipping Stage 7 (remap to original input files)")

        print(f"\n{'=' * 60}")
        print("Merge complete!")
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
    tile_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    tile_names: List[str] = []
    
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
        raise ValueError("Could not extract bounds from any tile files")
    
    print(f"  Extracted bounds from {len(tile_boundaries)} tiles")

    # Build neighbor graph from tile_bounds_tindex.json and match to loaded tiles
    print("  Loading neighbor graph from tile_bounds_tindex.json...")
    json_bounds, centers, neighbors_idx = build_neighbor_graph_from_bounds_json(tile_bounds_json)
    print(f"  JSON tiles in bounds file: {len(json_bounds)}")

    tile_to_json, json_to_tile = _match_tiles_to_json_bounds(tile_boundaries, json_bounds, centers)
    print("  Matched tiles to JSON bounds successfully")

    # Build neighbors per tile name using the JSON neighbor graph.
    # JSON can contain tiles that were never created (no points in bounds); json_to_tile only
    # has entries for JSON indices that matched an actual LAZ file. So neighbor_name =
    # json_to_tile.get(n_idx) is None when the neighbor exists only in JSON (no point cloud).
    # Border detection and cross-tile matching then ignore that edge (no points filtered).
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]] = {}
    for tile_name, json_idx in tile_to_json.items():
        neighbors_for_tile: Dict[str, Optional[str]] = {"east": None, "west": None, "north": None, "south": None}
        for direction in ("east", "west", "north", "south"):
            n_idx = neighbors_idx[json_idx].get(direction)
            if n_idx is None:
                neighbors_for_tile[direction] = None
            else:
                neighbor_name = json_to_tile.get(n_idx)  # None if no LAZ for that JSON tile
                neighbors_for_tile[direction] = neighbor_name
        neighbors_by_tile[tile_name] = neighbors_for_tile

    print("  Built neighbor mapping per tile from JSON graph")

    # =========================================================================
    # Stage 1: Load and Filter
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Stage 1: Loading tiles and filtering buffer zone instances")
    print(f"{'=' * 60}")
    print(f"  Loading {len(laz_files)} files using {num_threads} workers (--workers={num_threads})...")

    tiles = []
    filtered_instances_per_tile = {}
    kept_instances_per_tile = {}

    # Load tiles in parallel using ProcessPoolExecutor for true CPU parallelism
    # Prepare arguments for multiprocessing (must be pickleable)
    load_args = [(f, tile_boundaries, buffer, neighbors_by_tile, instance_dimension) for f in laz_files]

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
    
    # Log extra dims found across tiles
    all_extra_dim_names = set()
    for tile in tiles:
        all_extra_dim_names.update(tile.extra_dims.keys())
    if all_extra_dim_names:
        print(f"  Extra dimensions (passenger data): {', '.join(sorted(all_extra_dim_names))}")
    
    total_points = sum(len(tile.points) for tile in tiles)
    total_kept = sum(len(kept) for kept in kept_instances_per_tile.values())
    total_filtered = sum(len(filtered) for filtered in filtered_instances_per_tile.values())
    print(f"  ✓ Stage 1 completed: {len(tiles)} tiles loaded, {total_points:,} total points")
    print(f"    Kept {total_kept} instances, filtered {total_filtered} buffer zone instances")
    
    # Save filtered tiles (with filtered instances removed)
    filtered_tiles_dir = output_tiles_dir / "filtered_tiles"
    filtered_tiles_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving filtered tiles to {filtered_tiles_dir}...")
    for tile in tiles:
        kept_instances = kept_instances_per_tile[tile.name]
        keep_mask = np.isin(tile.instances, list(kept_instances) + [0])
        
        filtered_points = tile.points[keep_mask]
        filtered_instances = tile.instances[keep_mask]
        filtered_extra_dims = {name: arr[keep_mask] for name, arr in tile.extra_dims.items()}
        
        if len(filtered_points) == 0:
            print(f"    Warning: {tile.name} has no points after filtering, skipping")
            continue
        
        filtered_output_path = filtered_tiles_dir / f"{tile.name}.laz"
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.offsets = [filtered_points[:, 0].min(), filtered_points[:, 1].min(), filtered_points[:, 2].min()]
        header.scales = [0.01, 0.01, 0.01]
        
        output_las = laspy.LasData(header)
        output_las.x = filtered_points[:, 0]
        output_las.y = filtered_points[:, 1]
        output_las.z = filtered_points[:, 2]
        
        extra_dims_params = [laspy.ExtraBytesParams(name=instance_dimension, type=np.int32)]
        for dim_name, dim_arr in filtered_extra_dims.items():
            extra_dims_params.append(laspy.ExtraBytesParams(name=dim_name, type=dim_arr.dtype))
        output_las.add_extra_dims(extra_dims_params)
        
        setattr(output_las, instance_dimension, filtered_instances)
        for dim_name, dim_arr in filtered_extra_dims.items():
            setattr(output_las, dim_name, dim_arr)
        
        output_las.write(str(filtered_output_path))
    print(f"  ✓ Saved {len(tiles)} filtered tiles as .laz files (filtered instances removed)")

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

    # Initialize Union-Find and track instance sizes
    uf = UnionFind()
    instance_sizes = {}  # global_id -> point count

    for tile_idx, tile in enumerate(tiles):
        print(f"  Processing tile {tile_idx + 1}/{len(tiles)}: {tile.name} ({len(tile.points):,} points)...")
        kept_instances = kept_instances_per_tile[tile.name]

        unique_inst, inst_counts = np.unique(tile.instances, return_counts=True)

        for i, local_inst in enumerate(unique_inst):
            if local_inst <= 0 or local_inst not in kept_instances:
                continue

            gid = global_id(tile_idx, local_inst)
            size = int(inst_counts[i])
            uf.make_set(gid, size)
            instance_sizes[gid] = size

    print(f"  Total global instances: {len(instance_sizes)}")
    print(f"  ✓ Stage 2 completed: Assigned global IDs to {len(instance_sizes)} instances")

    # Helper functions for border matching
    def get_opposite_direction(direction: str) -> str:
        """Get opposite direction."""
        opposites = {"east": "west", "west": "east", "north": "south", "south": "north"}
        return opposites.get(direction, direction)
    
    def log_instance_pair_analysis(
        inst_id_a: int,
        inst_id_b: int,
        tile_a_name: str,
        tile_b_name: str,
        direction: str,
        bbox_a: Tuple[float, float, float, float],
        bbox_b: Tuple[float, float, float, float],
        overlap_ratio: float,
        overlap_threshold: float,
        bbox_overlaps: bool,
        centroid_a: np.ndarray,
        centroid_b: np.ndarray,
        size_a: int,
        size_b: int,
        matched: bool,
    ):
        """Log detailed analysis of an instance pair for debugging."""
        print(f"\n{'='*60}")
        print(f"DEBUG: Instance Pair Analysis")
        print(f"{'='*60}")
        print(f"Instance {inst_id_a} ({tile_a_name}) <-> Instance {inst_id_b} ({tile_b_name})")
        print(f"Direction: {tile_a_name} ({direction}) <-> {tile_b_name} ({get_opposite_direction(direction)})")
        print(f"\nInstance {inst_id_a}:")
        print(f"  Tile: {tile_a_name}")
        print(f"  Point count: {size_a:,}")
        print(f"  Centroid: ({centroid_a[0]:.2f}, {centroid_a[1]:.2f}, {centroid_a[2]:.2f})")
        print(f"  BBox: ({bbox_a[0]:.2f}, {bbox_a[1]:.2f}) x ({bbox_a[2]:.2f}, {bbox_a[3]:.2f})")
        print(f"\nInstance {inst_id_b}:")
        print(f"  Tile: {tile_b_name}")
        print(f"  Point count: {size_b:,}")
        print(f"  Centroid: ({centroid_b[0]:.2f}, {centroid_b[1]:.2f}, {centroid_b[2]:.2f})")
        print(f"  BBox: ({bbox_b[0]:.2f}, {bbox_b[1]:.2f}) x ({bbox_b[2]:.2f}, {bbox_b[3]:.2f})")
        centroid_dist = np.linalg.norm(centroid_a - centroid_b)
        print(f"\nCentroid distance: {centroid_dist:.2f}m")
        print(f"BBox overlaps (10cm tolerance): {'YES' if bbox_overlaps else 'NO'}")
        print(f"FF3D overlap ratio: {overlap_ratio:.4f}")
        print(f"Overlap threshold: {overlap_threshold:.4f}")
        print(f"Match result: {'MATCHED' if matched else 'NOT MATCHED'}")
        if not matched:
            reasons = []
            if not bbox_overlaps:
                reasons.append("BBox doesn't overlap (within 10cm)")
            if overlap_ratio < overlap_threshold:
                reasons.append(f"Overlap ratio {overlap_ratio:.4f} < threshold {overlap_threshold:.4f}")
            if reasons:
                print(f"  Reasons: {', '.join(reasons)}")
        print(f"{'='*60}\n")
    
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
    # Stage 3: Border Region Instance Matching (or All Instance Matching)
    # =========================================================================
    # Note: Cross-tile matching is optimized - each tile pair is checked exactly once
    # using `for j in range(i + 1, len(tiles))`, avoiding duplicate A->B and B->A checks.
    stage_name = "All Instance Matching" if match_all_instances else "Border Region Instance Matching"
    print(f"\n{'=' * 60}")
    print(f"Stage 3: {stage_name}")
    print(f"{'=' * 60}")
    
    # Instance tracking for debugging
    instance_tracking = {}  # (tile_name, local_inst_id) -> tracking info
    if debug_instance_ids:
        print(f"  Debug mode enabled for instances: {sorted(debug_instance_ids)}")
        # Initialize tracking for all instances in all tiles
        for tile_idx, tile in enumerate(tiles):
            unique_instances = np.unique(tile.instances[tile.instances > 0])
            for local_inst in unique_instances:
                gid = global_id(tile_idx, local_inst)
                if local_inst in debug_instance_ids:
                    instance_tracking[(tile.name, local_inst)] = {
                        "tile_name": tile.name,
                        "local_id": local_inst,
                        "global_id": gid,
                        "filtered_in_stage1": local_inst not in kept_instances_per_tile[tile.name],
                        "in_border_region": False,
                        "border_direction": None,
                        "compared_with": [],
                        "matched_with": None
                    }
    
    if match_all_instances:
        print(f"  Finding all instances (matching all instances, not just border region)...")
    else:
        print(f"  Finding border region instances (centroids in buffer to buffer+{border_zone_width}m zone)...")
    
    # Find instances to match (border region or all instances)
    border_instances = {}  # tile_name -> {instance_id: {'centroid': [...], 'points': [...], 'boundary': [...]}}
    
    # Build tile name to index mapping
    tile_name_to_idx = {tile.name: idx for idx, tile in enumerate(tiles)}
    
    for tile_idx, tile in enumerate(tiles):
        print(f"    Processing tile {tile_idx + 1}/{len(tiles)}: {tile.name} ({len(tile.points):,} points)...")
        tile_name = tile.name
        # Use neighbors from JSON graph when available; fall back to spatial neighbors only
        # if this tile was somehow not present in the JSON mapping (should not happen).
        if tile_name in neighbors_by_tile:
            neighbors = neighbors_by_tile[tile_name]
        else:
            neighbors = find_spatial_neighbors(tile.boundary, tile_name, tile_boundaries, tolerance=buffer)
        kept_instances = kept_instances_per_tile[tile_name]
        
        neighbor_names = [n for n in neighbors.values() if n is not None]
        print(f"      Neighbors: {', '.join(neighbor_names) if neighbor_names else 'none'}")
        if verbose:
            for direction, neighbor_name in neighbors.items():
                if neighbor_name is not None:
                    neighbor_boundary = tile_boundaries.get(neighbor_name)
                    if neighbor_boundary:
                        overlap = find_overlap_region(tile.boundary, neighbor_boundary)
                        if overlap:
                            ov_minx, ov_maxx, ov_miny, ov_maxy = overlap
                            ov_width = ov_maxx - ov_minx
                            ov_height = ov_maxy - ov_miny
                            print(f"        {direction.upper()} {neighbor_name}: overlap {ov_width:.1f}m x {ov_height:.1f}m")
        
        min_x, max_x, min_y, max_y = tile.boundary
        border_zone_end = buffer + border_zone_width  # border_zone_width beyond buffer
        
        # Define border region boundaries (buffer to buffer+border_zone_width from edges with neighbors)
        # Inner edge of border region (end of buffer zone)
        border_inner_min_x = min_x + (buffer if neighbors["west"] is not None else 0)
        border_inner_max_x = max_x - (buffer if neighbors["east"] is not None else 0)
        border_inner_min_y = min_y + (buffer if neighbors["south"] is not None else 0)
        border_inner_max_y = max_y - (buffer if neighbors["north"] is not None else 0)
        
        # Outer edge of border region (buffer+border_zone_width from tile edge)
        border_outer_min_x = min_x + (border_zone_end if neighbors["west"] is not None else 0)
        border_outer_max_x = max_x - (border_zone_end if neighbors["east"] is not None else 0)
        border_outer_min_y = min_y + (border_zone_end if neighbors["south"] is not None else 0)
        border_outer_max_y = max_y - (border_zone_end if neighbors["north"] is not None else 0)
        
        border_instances[tile_name] = {}
        
        if match_all_instances:
            # Collect ALL kept instances (not just border region)
            all_unique_insts = kept_instances - {0}  # All kept instances except ground
            
            if len(all_unique_insts) == 0:
                print(f"      No instances to match in {tile.name}")
                continue
            
            # Compute centroids for all instances
            print(f"      Computing centroids for {len(all_unique_insts)} instances (all instances)...")
            all_centroids = compute_centroids_vectorized(tile.points, tile.instances)
            instance_centroids = {
                inst_id: all_centroids[inst_id]
                for inst_id in all_unique_insts
                if inst_id in all_centroids
            }
            
            instance_count = 0
            
            # For each instance, extract full points (no direction filtering)
            for inst_id in all_unique_insts:
                if inst_id not in instance_centroids:
                    continue
                
                centroid = instance_centroids[inst_id]
                
                # Extract full instance points
                inst_mask = tile.instances == inst_id
                inst_points = tile.points[inst_mask]
                
                # Compute instance bounding box
                inst_minx = inst_points[:, 0].min()
                inst_maxx = inst_points[:, 0].max()
                inst_miny = inst_points[:, 1].min()
                inst_maxy = inst_points[:, 1].max()
                
                # Use "all" as direction to indicate this is not border-specific
                border_instances[tile_name][inst_id] = {
                    'centroid': centroid,
                    'points': inst_points,
                    'boundary': (inst_minx, inst_maxx, inst_miny, inst_maxy),
                    'direction': 'all',  # Special direction for all-instance matching
                    'tile_idx': tile_idx
                }
                instance_count += 1
                
                # Update tracking for debug instances
                if debug_instance_ids and inst_id in debug_instance_ids:
                    key = (tile_name, inst_id)
                    if key in instance_tracking:
                        instance_tracking[key]["in_border_region"] = True
                        instance_tracking[key]["border_direction"] = 'all'
                        print(f"      DEBUG: Instance {inst_id} included in all-instance matching")
            
            print(f"      Found {instance_count} instances in {tile.name} (all instances)")
        else:
            pass

        border_mask = get_border_region_mask(
                tile.points, tile.boundary, buffer, border_zone_end, neighbors
        )
        border_points = tile.points[border_mask]
        border_inst_ids = tile.instances[border_mask]
        
        # Get unique instances in border region (much smaller set than all instances)
        border_unique_insts = set(np.unique(border_inst_ids)) - {0}
        border_unique_insts &= kept_instances  # Only kept instances
        
        if len(border_unique_insts) == 0:
            continue
        
        if verbose:
            print(f"      Computing centroids for {len(border_unique_insts)} border instances...")
        all_centroids = compute_centroids_vectorized(tile.points, tile.instances)
        border_centroids = {
            inst_id: all_centroids[inst_id]
            for inst_id in border_unique_insts
            if inst_id in all_centroids
        }
        
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
            
            # Update tracking for debug instances
            if debug_instance_ids and inst_id in debug_instance_ids:
                    key = (tile_name, inst_id)
                    if key in instance_tracking:
                        instance_tracking[key]["in_border_region"] = True
                        instance_tracking[key]["border_direction"] = border_direction
                        print(f"      DEBUG: Instance {inst_id} in border region ({border_direction})")
        
        if border_count > 0:
            print(f"      {tile.name}: {border_count} border instances")
    
    # Match instances between neighbor tiles
    total_border_insts = sum(len(insts) for insts in border_instances.values())
    tiles_with_border = len([t for t in border_instances if border_instances[t]])
    if match_all_instances:
        print(f"  Found {total_border_insts} instances across {tiles_with_border} tiles (all instances)")
    else:
        print(f"  Found {total_border_insts} border region instances across {tiles_with_border} tiles")
    print(f"  Processing tile pairs...")
    
    # Track which global IDs have already been matched to avoid duplicate checks
    matched_gids = set()
    
    border_matches = 0
    total_bbox_checks = 0
    total_ff3d_computations = 0
    tiles_processed = 0
    
    for i in range(len(tiles)):
        tile_a = tiles[i]
        # Use JSON-based neighbors when available
        if tile_a.name in neighbors_by_tile:
            neighbors_a = neighbors_by_tile[tile_a.name]
        else:
            neighbors_a = find_spatial_neighbors(tile_a.boundary, tile_a.name, tile_boundaries)
        
        for direction, neighbor_name in neighbors_a.items():
            if neighbor_name is None:
                continue
            
            # Find neighbor tile index
            tile_b_idx = tile_name_to_idx.get(neighbor_name)
            if tile_b_idx is None:
                continue
            
            tile_b = tiles[tile_b_idx]
            
            # Get instances from both tiles
            if match_all_instances:
                # Match ALL instances between neighbor tiles (no direction filtering)
                border_insts_a = border_instances.get(tile_a.name, {})
                border_insts_b = border_instances.get(tile_b.name, {})
            else:
                # Original logic: only match border instances in specific directions
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
            if match_all_instances:
                print(f"    Checking {tile_a.name} <-> {tile_b.name} ({direction} neighbors): "
                      f"{len(border_insts_a)} vs {len(border_insts_b)} instances", end=" ... ")
            else:
                print(f"    Checking {tile_a.name} ({direction}) <-> {tile_b.name} ({get_opposite_direction(direction)}): "
                      f"{len(border_insts_a)} vs {len(border_insts_b)} border instances", end=" ... ")
            
            # Build list of candidate instances from tile B (not already matched)
            candidates_b = []
            for inst_id_b, data_b in border_insts_b.items():
                gid_b = global_id(tile_b_idx, inst_id_b)
                if gid_b not in matched_gids:
                    candidates_b.append((inst_id_b, gid_b, data_b))
            
            if not candidates_b:
                continue
            
            # For each instance in tile A, check overlap with all candidate instances in tile B
            for inst_id_a, data_a in border_insts_a.items():
                gid_a = global_id(i, inst_id_a)
                
                # Skip if already matched
                if gid_a in matched_gids:
                    continue
                
                bbox_a = data_a['boundary']
                
                # Check each candidate in tile B
                for inst_id_b, gid_b, data_b in candidates_b:
                    # Skip if already matched
                    if gid_b in matched_gids:
                        continue
                    
                    bbox_b = data_b['boundary']
                    
                    # Quick bounding box overlap/nearby check (within 10cm tolerance)
                    total_bbox_checks += 1
                    bbox_overlaps = bboxes_overlap(bbox_a, bbox_b, tolerance=0.1)
                    
                    # Check if we should debug this pair
                    should_debug = (
                        debug_instance_ids is not None and
                        (inst_id_a in debug_instance_ids or inst_id_b in debug_instance_ids)
                    )
                    
                    if should_debug:
                        print(f"\n  DEBUG: Checking pair {inst_id_a} <-> {inst_id_b}")
                        print(f"    BBox overlap check: {'PASS' if bbox_overlaps else 'FAIL'}")
                    
                    if not bbox_overlaps:
                        if should_debug:
                            print(f"    Skipping: BBox doesn't overlap (within 10cm tolerance)")
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
                    
                    # Debug logging for instance pairs
                    if should_debug:
                        centroid_a = data_a['centroid']
                        centroid_b = data_b['centroid']
                        size_a_val = size_a.get(inst_id_a, 0)
                        size_b_val = size_b.get(inst_id_b, 0)
                        
                        # Update tracking
                        key_a = (tile_a.name, inst_id_a)
                        key_b = (tile_b.name, inst_id_b)
                        if key_a in instance_tracking:
                            instance_tracking[key_a]["compared_with"].append({
                                "tile": tile_b.name,
                                "instance": inst_id_b,
                                "overlap_ratio": overlap_ratio,
                                "matched": overlap_ratio >= overlap_threshold
                            })
                        if key_b in instance_tracking:
                            instance_tracking[key_b]["compared_with"].append({
                                "tile": tile_a.name,
                                "instance": inst_id_a,
                                "overlap_ratio": overlap_ratio,
                                "matched": overlap_ratio >= overlap_threshold
                            })
                        
                        log_instance_pair_analysis(
                            inst_id_a, inst_id_b,
                            tile_a.name, tile_b.name, direction,
                            bbox_a, bbox_b,
                            overlap_ratio, overlap_threshold,
                            bbox_overlaps,
                            centroid_a, centroid_b,
                            size_a_val, size_b_val,
                            overlap_ratio >= overlap_threshold
                        )
                    
                    if overlap_ratio >= overlap_threshold:
                        # Merge via Union-Find
                        root = uf.union(gid_a, gid_b)
                        matched_gids.add(gid_a)
                        matched_gids.add(gid_b)
                        border_matches += 1
                        
                        # Update tracking for matched instances
                        if debug_instance_ids:
                            key_a = (tile_a.name, inst_id_a)
                            key_b = (tile_b.name, inst_id_b)
                            if key_a in instance_tracking:
                                instance_tracking[key_a]["matched_with"] = (tile_b.name, inst_id_b)
                            if key_b in instance_tracking:
                                instance_tracking[key_b]["matched_with"] = (tile_a.name, inst_id_a)
                        
                        if verbose:
                            print(f"      ✓ Match: {tile_a.name}:{inst_id_a} <-> {tile_b.name}:{inst_id_b} (overlap: {overlap_ratio:.3f})")
            
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
    
    if match_all_instances:
        print(f"  Matched {border_matches} instance pairs (all instances)")
    else:
        print(f"  Matched {border_matches} border region instance pairs")
    print(f"  Performance: {total_bbox_checks} bbox checks, {total_ff3d_computations} FF3D computations")
    print(f"  ✓ Stage 3 completed: {stage_name} done")

    # Print instance tracking summary for debug instances
    if debug_instance_ids and instance_tracking:
        print(f"\n{'='*60}")
        print("Instance Tracking Summary")
        print(f"{'='*60}")
        for (tile_name, local_id), info in sorted(instance_tracking.items()):
            print(f"\nInstance {local_id} (Tile: {tile_name}):")
            print(f"  Global ID: {info['global_id']}")
            print(f"  Filtered in Stage 1: {'YES' if info['filtered_in_stage1'] else 'NO'}")
            print(f"  In border region: {'YES' if info['in_border_region'] else 'NO'}")
            if info['in_border_region']:
                print(f"  Border direction: {info['border_direction']}")
            print(f"  Compared with {len(info['compared_with'])} instance(s):")
            for comp in info['compared_with']:
                print(f"    - {comp['tile']}:{comp['instance']} (overlap: {comp['overlap_ratio']:.4f}, matched: {comp['matched']})")
            if info['matched_with']:
                print(f"  Matched with: {info['matched_with'][0]}:{info['matched_with'][1]}")
            else:
                print(f"  Matched with: NONE")
        print(f"{'='*60}\n")

    # Get connected components
    components = uf.get_components()
    print(f"  Connected components: {len(components)}")
    print(f"  ✓ Instance matching completed: {len(components)} merged instance groups")

    # Create mapping from global ID to final merged ID
    global_to_merged = {}
    merged_instance_sources = {}  # merged_id -> list of source global IDs (for CSV tracking)
    
    for merged_id, (root, members) in enumerate(components.items(), start=1):
        merged_instance_sources[merged_id] = list(members)
        
        if len(members) > 1 and verbose:
            print(f"  Merged ID {merged_id} created from {len(members)} global IDs: {sorted(members)}")

        for gid in members:
            global_to_merged[gid] = merged_id

    # =========================================================================
    # Orphan Recovery: Recover filtered instances that would otherwise be lost
    # =========================================================================
    # Problem: A tree can be filtered from BOTH tiles if segmented slightly
    # differently (centroids in buffer zones of both tiles).
    #
    # Solution: Recover filtered instances from ANY buffer direction if
    # the neighboring tile doesn't have the tree covered (checked via bbox overlap).
    # This ensures no instances are lost, even if filtered from both tiles.
    print("\n  Checking for orphaned filtered instances...")

    # Build tile name to index map
    tile_name_to_idx = {tile.name: idx for idx, tile in enumerate(tiles)}
    
    # Store tile index to name mapping for diagnostic logging later (used in renumbering)
    tile_idx_to_name = {idx: tile.name for idx, tile in enumerate(tiles)}

    # Bounding boxes only for instances we need: orphans (filtered) + kept instances in border region.
    # Use centroids to decide which kept instances are in border, then compute bbox only for those.
    border_zone_end = buffer + border_zone_width
    instance_bboxes = {}  # tile_name -> {inst_id -> (min_xyz, max_xyz)}

    for tile in tiles:
        centroids = compute_centroids_vectorized(tile.points, tile.instances)
        kept_instances = kept_instances_per_tile[tile.name]
        neighbors = neighbors_by_tile.get(tile.name) or {"east": None, "west": None, "north": None, "south": None}
        min_x, max_x, min_y, max_y = tile.boundary
        bi_min_x = min_x + (buffer if neighbors.get("west") is not None else 0)
        bi_max_x = max_x - (buffer if neighbors.get("east") is not None else 0)
        bi_min_y = min_y + (buffer if neighbors.get("south") is not None else 0)
        bi_max_y = max_y - (buffer if neighbors.get("north") is not None else 0)
        bo_min_x = min_x + (border_zone_end if neighbors.get("west") is not None else 0)
        bo_max_x = max_x - (border_zone_end if neighbors.get("east") is not None else 0)
        bo_min_y = min_y + (border_zone_end if neighbors.get("south") is not None else 0)
        bo_max_y = max_y - (border_zone_end if neighbors.get("north") is not None else 0)

        def in_border(cx: float, cy: float) -> bool:
            return (
                (neighbors.get("west") is not None and bi_min_x <= cx <= bo_min_x)
                or (neighbors.get("east") is not None and bo_max_x <= cx <= bi_max_x)
                or (neighbors.get("south") is not None and bi_min_y <= cy <= bo_min_y)
                or (neighbors.get("north") is not None and bo_max_y <= cy <= bi_max_y)
            )

        kept_in_border = {
            inst_id for inst_id in kept_instances
            if inst_id in centroids and in_border(centroids[inst_id][0], centroids[inst_id][1])
        }
        need_bbox = filtered_instances_per_tile[tile.name] | kept_in_border

        sort_idx = np.argsort(tile.instances)
        sorted_inst = tile.instances[sort_idx]
        sorted_points = tile.points[sort_idx]
        unique_inst, first_idx, counts = np.unique(
            sorted_inst, return_index=True, return_counts=True
        )
        bboxes = {}
        for i, inst_id in enumerate(unique_inst):
            if inst_id <= 0 or inst_id not in need_bbox:
                continue
            start = first_idx[i]
            end = start + counts[i]
            pts = sorted_points[start:end]
            bboxes[inst_id] = (pts.min(axis=0), pts.max(axis=0))
        instance_bboxes[tile.name] = bboxes

    # Build spatial index of kept instance centers in border region only
    # (tile_bboxes only contains border kept + filtered, so every kept in tile_bboxes is border)
    print("  Building spatial index of kept instances (border region only)...")
    kept_instance_data = []  # List of (center_x, center_y, tile_idx, inst_id, bbox_min, bbox_max)

    for tile_idx, tile in enumerate(tiles):
        tile_bboxes = instance_bboxes[tile.name]
        kept_instances = kept_instances_per_tile[tile.name]
        for inst_id in kept_instances:
            if inst_id not in tile_bboxes:
                continue
            bbox_min, bbox_max = tile_bboxes[inst_id]
            center_x = (bbox_min[0] + bbox_max[0]) / 2.0
            center_y = (bbox_min[1] + bbox_max[1]) / 2.0
            kept_instance_data.append((center_x, center_y, tile_idx, inst_id, bbox_min, bbox_max))

    # Build cKDTree for spatial queries (5m search radius)
    search_radius = 5.0  # 5m radius
    if kept_instance_data:
        centers = np.array([(x, y) for x, y, _, _, _, _ in kept_instance_data])
        kept_tree = cKDTree(centers)
    else:
        kept_tree = None
        print("  Warning: No kept instances found for spatial indexing")

    # Pre-build KDTrees for all kept instances in border (used by parallel orphan check)
    neighbor_trees: Dict[Tuple[int, int], cKDTree] = {}
    for _, _, check_tile_idx, neighbor_inst, _, _ in kept_instance_data:
        cache_key = (check_tile_idx, neighbor_inst)
        if cache_key in neighbor_trees:
            continue
        check_tile = tiles[check_tile_idx]
        neighbor_mask = check_tile.instances == neighbor_inst
        neighbor_points = check_tile.points[neighbor_mask]
        if len(neighbor_points) > 0:
            neighbor_trees[cache_key] = cKDTree(neighbor_points[:, :2])

    overlap_tolerance = 1.0  # 1m tolerance for tree instances

    def _check_one_orphan_covered(item: Tuple[int, int]) -> Tuple[int, int, bool]:
        """Returns (tile_idx, local_inst, covered). True if a neighbor covers this orphan."""
        tile_idx, local_inst = item
        tile = tiles[tile_idx]
        tile_name = tile.name
        tile_bboxes = instance_bboxes[tile_name]
        if local_inst <= 0 or local_inst not in tile_bboxes:
            return (tile_idx, local_inst, True)
        if buffer_direction_per_tile[tile_name].get(local_inst) is None:
            return (tile_idx, local_inst, True)
        fmin, fmax = tile_bboxes[local_inst]
        orphan_center = ((fmin[0] + fmax[0]) / 2.0, (fmin[1] + fmax[1]) / 2.0)
        inst_mask = tile.instances == local_inst
        filtered_points = tile.points[inst_mask]
        if len(filtered_points) == 0:
            return (tile_idx, local_inst, True)
        neighbor_has_tree = False
        if kept_tree is not None:
            nearby_indices = kept_tree.query_ball_point(orphan_center, r=search_radius)
            for idx in nearby_indices:
                _, _, check_tile_idx, neighbor_inst, nmin, nmax = kept_instance_data[idx]
                if check_tile_idx == tile_idx:
                    continue
                if (fmax[0] < nmin[0] - overlap_tolerance or fmin[0] > nmax[0] + overlap_tolerance or
                    fmax[1] < nmin[1] - overlap_tolerance or fmin[1] > nmax[1] + overlap_tolerance):
                    continue
                cache_key = (check_tile_idx, neighbor_inst)
                neighbor_tree = neighbor_trees.get(cache_key)
                if neighbor_tree is None:
                    continue
                distances, _ = neighbor_tree.query(filtered_points[:, :2], k=1)
                n_within = np.sum(distances <= overlap_tolerance)
                fraction_within = n_within / len(filtered_points)
                if fraction_within > 0.50:
                    neighbor_gid = global_id(check_tile_idx, neighbor_inst)
                    if neighbor_gid in global_to_merged:
                        neighbor_has_tree = True
                        break
                    else:
                        neighbor_has_tree = True
                        break
        return (tile_idx, local_inst, neighbor_has_tree)

    # Collect orphan candidates
    orphan_candidates: List[Tuple[int, int]] = []
    for tile_idx, tile in enumerate(tiles):
        filtered_instances = filtered_instances_per_tile[tile.name]
        buffer_directions = buffer_direction_per_tile[tile.name]
        tile_bboxes = instance_bboxes[tile.name]
        for local_inst in filtered_instances:
            if local_inst <= 0 or local_inst not in tile_bboxes:
                continue
            if buffer_directions.get(local_inst) is None:
                continue
            inst_mask = tile.instances == local_inst
            if np.sum(inst_mask) == 0:
                continue
            orphan_candidates.append((tile_idx, local_inst))

    next_merged_id = max(global_to_merged.values()) + 1 if global_to_merged else 1
    recovered_count = 0
    skipped_covered = 0
    orphan_parallel_workers = 10

    if orphan_candidates:
        print(f"  Checking {len(orphan_candidates)} orphan candidates with {orphan_parallel_workers} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=orphan_parallel_workers) as executor:
            orphan_results = list(executor.map(_check_one_orphan_covered, orphan_candidates))
        for (tile_idx, local_inst, covered) in orphan_results:
            if covered:
                skipped_covered += 1
                continue
            tile = tiles[tile_idx]
            gid = global_id(tile_idx, local_inst)
            global_to_merged[gid] = next_merged_id
            merged_instance_sources[next_merged_id] = [gid]
            if verbose:
                print(f"  Recovered orphan - global_id={gid} (tile={tile.name}, local={local_inst}) -> merged_id={next_merged_id}")
            kept_instances_per_tile[tile.name].add(local_inst)
            next_merged_id += 1
            recovered_count += 1

    del instance_bboxes

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
    all_extra_dims_lists: Dict[str, list] = {name: [] for name in all_extra_dim_names}

    for tile_idx, tile in enumerate(tiles):
        kept_instances = kept_instances_per_tile[tile.name]

        max_local_inst = tile.instances.max() + 1

        inst_to_merged = np.full(max_local_inst, -1, dtype=np.int32)
        
        if max_local_inst > 0:
            inst_to_merged[0] = 0

        for local_inst in kept_instances:
            if local_inst <= 0:
                continue
            gid = global_id(tile_idx, local_inst)
            merged_id = global_to_merged.get(gid, -1)
            inst_to_merged[local_inst] = merged_id

        safe_instances = np.clip(tile.instances, 0, max_local_inst - 1)
        remapped_instances = inst_to_merged[safe_instances]

        all_points.append(tile.points)
        all_instances.append(remapped_instances)
        
        # Collect extra dims (passenger data) - use zeros for dims missing from this tile
        for dim_name in all_extra_dim_names:
            if dim_name in tile.extra_dims:
                all_extra_dims_lists[dim_name].append(tile.extra_dims[dim_name])
            else:
                all_extra_dims_lists[dim_name].append(
                    np.zeros(len(tile.points), dtype=np.int32)
                )

    del tiles
    del filtered_instances_per_tile
    del kept_instances_per_tile
    gc.collect()

    merged_points = np.vstack(all_points)
    merged_instances = np.concatenate(all_instances)
    merged_extra_dims: Dict[str, np.ndarray] = {
        name: np.concatenate(arrays) for name, arrays in all_extra_dims_lists.items()
    }

    del all_points
    del all_instances
    del all_extra_dims_lists
    gc.collect()

    # Remove points from filtered buffer instances (instance_id = -1)
    valid_points_mask = merged_instances != -1
    n_filtered_removed = np.sum(merged_instances == -1)
    n_ground_points = np.sum(merged_instances == 0)

    merged_points = merged_points[valid_points_mask]
    merged_instances = merged_instances[valid_points_mask]
    merged_extra_dims = {name: arr[valid_points_mask] for name, arr in merged_extra_dims.items()}

    if n_filtered_removed > 0:
        print(f"  Removed {n_filtered_removed:,} points from filtered buffer instances")
    gc.collect()

    total_before = len(merged_points)
    print(f"  Total points: {total_before:,}")
    print(f"  Deduplicating...", flush=True)
    merged_points, merged_instances, merged_extra_dims = deduplicate_points(
        merged_points, merged_instances, merged_extra_dims
    )
    gc.collect()

    n_removed = total_before - len(merged_points)
    n_tree_instances = len(np.unique(merged_instances[merged_instances > 0]))
    print(f"  Removed {n_removed:,} duplicate points ({100*n_removed/total_before:.1f}%)")
    print(f"  ✓ Stage 4 completed: {len(merged_points):,} points, {n_tree_instances} tree instances")

    del uf
    del instance_sizes
    del global_to_merged
    gc.collect()

    # =========================================================================
    # Stage 5: Small Volume Instance Merging
    # =========================================================================
    if enable_volume_merge:
        print(f"\n{'=' * 60}")
        print("Stage 5: Small Volume Instance Merging")
        print(f"{'=' * 60}")

        pos_mask = merged_instances > 0
        nonzero_count = pos_mask.sum()
        zero_count = len(merged_instances) - nonzero_count
        print(f"  Instance points: {nonzero_count:,}, ground points: {zero_count:,}")
        
        pos_points = merged_points[pos_mask]
        pos_instances = merged_instances[pos_mask]
        
        sort_idx = np.argsort(pos_instances)
        sorted_points = pos_points[sort_idx]
        sorted_instances = pos_instances[sort_idx]
        
        unique_inst, first_idx, inst_counts = np.unique(
            sorted_instances, return_index=True, return_counts=True
        )
        print(f"  Processing {len(unique_inst):,} unique instances...", flush=True)
        
        merged_instances, _ = merge_small_volume_instances(
            merged_points,
            merged_instances,
            min_points_for_hull_check=1000,
            min_cluster_size=min_cluster_size,
            max_volume_for_merge=max_volume_for_merge,
            max_search_radius=5.0,
            num_threads=num_threads,
            verbose=verbose,
            presorted_points=sorted_points,
            presorted_instances=sorted_instances,
            presorted_unique_inst=unique_inst,
            presorted_first_idx=first_idx,
            presorted_inst_counts=inst_counts,
        )
        print(f"  ✓ Stage 5 completed")
    else:
        print(f"\n{'=' * 60}")
        print("Stage 5: Small Volume Instance Merging (SKIPPED)")
        print(f"{'=' * 60}")

    # =========================================================================
    # Renumber instances to continuous IDs (north-to-south ordering)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Renumbering instances (north-to-south ordering)")
    print(f"{'=' * 60}")

    # Compute bounding box centers for sorting
    print("  Computing bounding box centers...")
    pos_mask = merged_instances > 0
    pos_points = merged_points[pos_mask]
    pos_instances = merged_instances[pos_mask]
    
    # Sort and compute unique instances
    sort_idx = np.argsort(pos_instances)
    sorted_points = pos_points[sort_idx]
    sorted_instances = pos_instances[sort_idx]
    unique_inst, first_idx, inst_counts = np.unique(
        sorted_instances, return_index=True, return_counts=True
    )

    # Compute bounding box centers (Y, X) for each instance
    bbox_centers = {}
    for inst_id, start, count in zip(unique_inst, first_idx, inst_counts):
        pts = sorted_points[start:start + count]
        min_pt, max_pt = pts.min(axis=0), pts.max(axis=0)
        bbox_centers[inst_id] = ((min_pt[1] + max_pt[1]) / 2.0, (min_pt[0] + max_pt[0]) / 2.0)

    # Sort by Y descending (north to south), then X ascending (west to east)
    sorted_by_location = sorted(
        unique_inst,
        key=lambda inst_id: (-bbox_centers[inst_id][0], bbox_centers[inst_id][1])
    )

    old_to_new = {0: 0}

    for new_id, old_id in enumerate(sorted_by_location, start=1):
        old_to_new[old_id] = new_id

    print(f"  Instances renumbered from north to south")

    max_old_id = int(merged_instances.max()) + 1
    
    instance_lookup = np.zeros(max_old_id, dtype=np.int32)
    for old_id, new_id in old_to_new.items():
        if old_id < max_old_id:
            instance_lookup[old_id] = new_id
    
    merged_instances = instance_lookup[merged_instances]

    print(f"  Final instance count: {len(sorted_by_location)}")

    # =========================================================================
    # Save merged output (optional - can be skipped with skip_merged_file=True)
    # =========================================================================
    if skip_merged_file:
        print(f"\n{'=' * 60}")
        print("Saving merged output (SKIPPED)")
        print(f"{'=' * 60}")
        print(f"  Skipped merged LAZ file creation (--skip_merged_file)")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(sorted_by_location)}")
    else:
        print(f"\n{'=' * 60}")
        print("Saving merged output")
        print(f"{'=' * 60}")

        output_merged.parent.mkdir(parents=True, exist_ok=True)

        # Write initial merged to a temp path so we can either enrich it or use as final
        merged_init = output_merged.parent / (output_merged.stem + "_init.laz")

        header = laspy.LasHeader(point_format=6, version="1.4")
        header.offsets = np.min(merged_points, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])

        output_las = laspy.LasData(header)
        output_las.x = merged_points[:, 0]
        output_las.y = merged_points[:, 1]
        output_las.z = merged_points[:, 2]

        # Add instance dimension and passenger extra dimensions from the merge.
        extra_dims_params = [laspy.ExtraBytesParams(name=instance_dimension, type=np.int32)]
        for dim_name, dim_arr in merged_extra_dims.items():
            extra_dims_params.append(laspy.ExtraBytesParams(name=dim_name, type=dim_arr.dtype))
        output_las.add_extra_dims(extra_dims_params)

        setattr(output_las, instance_dimension, merged_instances)
        for dim_name, dim_arr in merged_extra_dims.items():
            setattr(output_las, dim_name, dim_arr)

        output_las.write(
            str(merged_init), do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel
        )

        del output_las
        gc.collect()

        print(f"  Saved merged (initial): {merged_init}")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(sorted_by_location)}")

        import shutil
        # Add original-file dimensions and write directly to final output (so merged file is always enriched when requested)
        if original_input_dir is not None and transfer_original_dims_to_merged:
            try:
                add_original_dimensions_to_merged(
                    merged_init,
                    original_input_dir,
                    output_merged,
                    tolerance=0.1,
                    retile_buffer=retile_buffer,
                    num_threads=num_threads,
                )
                print(f"  Enriched merged file with original-file dimensions: {output_merged}")
            except Exception as e:
                print(f"  Warning: Could not add original dimensions to merged file: {e}")
                shutil.copy2(str(merged_init), str(output_merged))
                print(f"  Wrote un-enriched merged to {output_merged}")
        else:
            shutil.copy2(str(merged_init), str(output_merged))
            print(f"  Saved merged output: {output_merged}")

        try:
            merged_init.unlink()
        except OSError:
            pass

        # Also save a copy to output_tiles_dir for convenience
        merged_copy_path = output_tiles_dir / output_merged.name
        if merged_copy_path != output_merged:
            shutil.copy2(str(output_merged), str(merged_copy_path))
            print(f"  Copied to output tiles folder: {merged_copy_path}")

    # =========================================================================
    # Create CSV with instance metadata
    # =========================================================================
    import csv
    
    csv_output_path = output_merged.parent / f"{output_merged.stem}_instance_metadata.csv"
    
    instances_with_clusters = set()
    for old_merged_id, sources in merged_instance_sources.items():
        if len(sources) > 1:
            final_id = old_to_new.get(old_merged_id, None)
            if final_id is not None and final_id > 0:
                instances_with_clusters.add(final_id)
    
    print(f"\n  Writing instance metadata CSV: {csv_output_path}")
    print(f"  Found {len(instances_with_clusters)} final instances with added clusters from cross-tile merging")
    
    # Collect all final instance IDs
    final_instance_ids = sorted(set(old_to_new.values()) - {0})
    
    with open(csv_output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([instance_dimension, "has_added_clusters"])
        
        for final_id in final_instance_ids:
            has_clusters = final_id in instances_with_clusters
            writer.writerow([final_id, 1 if has_clusters else 0])
    
    csv_copy_path = output_tiles_dir / csv_output_path.name
    if csv_copy_path != csv_output_path:
        import shutil
        shutil.copy2(str(csv_output_path), str(csv_copy_path))
        print(f"  Copied CSV to output tiles folder: {csv_copy_path}")
    
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
        merged_extra_dims,
        None,
        original_tiles_dir,
        output_tiles_dir,
        tolerance=0.1,
        num_threads=num_threads,
        retile_buffer=retile_buffer,
        instance_dimension=instance_dimension,
    )
    print(f"  ✓ Stage 6 completed: Retiled to original files")

    # =========================================================================
    # Stage 7: Remap to Original Input Files (if provided)
    # =========================================================================
    if original_input_dir is not None:
        print(f"\n{'=' * 60}")
        print("Stage 7: Remapping to Original Input Files")
        print(f"{'=' * 60}")
        
        original_output_dir = output_tiles_dir.parent / "original_with_predictions"
        all_merged_dims = {instance_dimension: merged_instances, **merged_extra_dims}
        remap_to_original_input_files(
            merged_points,
            all_merged_dims,
            None,
            original_input_dir,
            original_output_dir,
            tolerance=0.1,
            num_threads=num_threads,
            retile_buffer=retile_buffer,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
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
        "--correspondence-tolerance",
        type=float,
        default=0.1,
        help="Max distance for point correspondence in meters (default: 0.1). "
        "Should be small (~10cm) to only match actual duplicate points from overlapping tiles.",
    )

    parser.add_argument(
        "--max-volume-for-merge",
        type=float,
        default=4.0,
        help="Max convex hull volume (m³) for small instance merging (default: 4.0)",
    )

    parser.add_argument(
        "--border-zone-width",
        type=float,
        default=10.0,
        help="Width of border zone beyond buffer for instance matching (default: 10.0m)",
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
        "--skip-merged-file",
        action="store_true",
        help="Skip creating merged LAZ file (only create retiled outputs)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed merge decisions"
    )

    parser.add_argument(
        "--debug-instances",
        type=str,
        default=None,
        help="Comma-separated list of instance IDs to debug (e.g., '485,73'). Enables detailed logging for these instances in Stage 3.",
    )

    parser.add_argument(
        "--match-all-instances",
        action="store_true",
        dest="match_all_instances",
        help="Match all instances between neighbor tiles, not just border region instances. "
        "When enabled, Stage 3 will check all instances in overlapping tiles for matching, "
        "not just those in border regions. Default: False (only border region instances are matched).",
    )

    parser.add_argument(
        "--retile-buffer",
        type=float,
        default=2.0,
        help="Spatial buffer expansion in meters for filtering merged points during retiling (fixed: 2.0m)",
    )

    parser.add_argument(
        "--retile-max-radius",
        type=float,
        default=0.2,
        help="Maximum distance threshold in meters for cKDTree nearest neighbor matching during retiling (default: 2.0m)",
    )

    parser.add_argument(
        "--instance-dimension",
        type=str,
        default="PredInstance",
        help="Name of the instance ID dimension in input files (default: PredInstance, fallback: treeID)",
    )

    parser.add_argument(
        "--tile-bounds-json",
        type=Path,
        required=True,
        help="Path to tile_bounds_tindex.json (required; used for neighbor graph)",
    )

    args = parser.parse_args()

    # Parse debug instance IDs
    debug_instance_ids = None
    if args.debug_instances:
        try:
            debug_instance_ids = set(int(x.strip()) for x in args.debug_instances.split(','))
        except ValueError:
            print(f"ERROR: Invalid --debug-instances format: {args.debug_instances}")
            print("Expected format: comma-separated integers (e.g., '485,73')")
            sys.exit(1)

    merge_tiles(
        input_dir=args.input_dir,
        original_tiles_dir=args.original_tiles_dir,
        output_merged=args.output_merged,
        output_tiles_dir=args.output_tiles_dir,
        tile_bounds_json=args.tile_bounds_json,
        original_input_dir=args.original_input_dir,
        buffer=args.buffer,
        overlap_threshold=args.overlap_threshold,
        correspondence_tolerance=args.correspondence_tolerance,
        max_volume_for_merge=args.max_volume_for_merge,
        border_zone_width=args.border_zone_width,
        num_threads=args.num_threads,
        enable_matching=not args.disable_matching,
        enable_volume_merge=not args.disable_volume_merge,
        skip_merged_file=args.skip_merged_file,
        verbose=args.verbose,
        retile_buffer=args.retile_buffer,
        retile_max_radius=args.retile_max_radius,
        debug_instance_ids=debug_instance_ids,
        match_all_instances=args.match_all_instances,
        instance_dimension=args.instance_dimension,
    )


if __name__ == "__main__":
    main()
