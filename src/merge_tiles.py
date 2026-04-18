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
import json
import math
import re
import shutil
import subprocess
import tempfile
import numpy as np
import laspy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from scipy.spatial import cKDTree
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# Force unbuffered output for real-time progress feedback
# (especially important when running in Docker/containers)
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# R/lidR → laspy dimension name mapping
# =============================================================================
# The standardization JSON (collection_summary.json from tool_standard) can mix
# R/lidR names, LAS 1.2 names, LAS 1.4 names, and already-normalized laspy
# names. Extra-byte dimensions (e.g. Amplitude, Reflectance, Deviation) are
# preserved as-is and fall through unchanged via the .get() default.
#
# In particular, LAS 1.2 `ScanAngleRank` / laspy `scan_angle_rank` and LAS 1.4
# `ScanAngle` / laspy `scan_angle` should all resolve to the same canonical
# dimension used by the remap path: `scan_angle`.

_DIMENSION_NAME_ALIASES = {
    "Intensity": "intensity",
    "intensity": "intensity",
    "ReturnNumber": "return_number",
    "return_number": "return_number",
    "NumberOfReturns": "number_of_returns",
    "number_of_returns": "number_of_returns",
    "ScanDirectionFlag": "scan_direction_flag",
    "scan_direction_flag": "scan_direction_flag",
    "EdgeOfFlightline": "edge_of_flight_line",
    "edge_of_flight_line": "edge_of_flight_line",
    "Classification": "classification",
    "classification": "classification",
    "ScannerChannel": "scanner_channel",
    "scanner_channel": "scanner_channel",
    "Synthetic_flag": "synthetic",
    "synthetic": "synthetic",
    "Keypoint_flag": "key_point",
    "key_point": "key_point",
    "Withheld_flag": "withheld",
    "withheld": "withheld",
    "Overlap_flag": "overlap",
    "overlap": "overlap",
    "ScanAngle": "scan_angle",
    "scan_angle": "scan_angle",
    "ScanAngleRank": "scan_angle",
    "scan_angle_rank": "scan_angle",
    "UserData": "user_data",
    "user_data": "user_data",
    "PointSourceID": "point_source_id",
    "point_source_id": "point_source_id",
    "gpstime": "gps_time",
    "gps_time": "gps_time",
    "R": "red",
    "red": "red",
    "G": "green",
    "green": "green",
    "B": "blue",
    "blue": "blue",
}

_RGB_STANDARD_DIMS = ("red", "green", "blue")
_COPC_READER_FALLBACK_WARNED: Set[str] = set()


def get_pdal_path() -> str:
    """Get the path to pdal executable."""
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def has_standard_rgb_dims(dim_names) -> bool:
    """Return True when all LAS RGB dimensions are present."""
    return all(name in dim_names for name in _RGB_STANDARD_DIMS)


def point_format_has_standard_rgb(point_format_id: int) -> bool:
    """Return True if the point format includes standard RGB fields."""
    try:
        return has_standard_rgb_dims(set(laspy.PointFormat(point_format_id).dimension_names))
    except Exception:
        return False


def point_format_with_standard_rgb(point_format_id: int) -> int:
    """Promote an RGB-less point format to the RGB-capable sibling when possible."""
    if point_format_has_standard_rgb(point_format_id):
        return point_format_id
    return {
        0: 2,
        1: 3,
        4: 5,
        6: 7,
        9: 10,
    }.get(point_format_id, point_format_id)


def list_pointcloud_files(input_dir: Path) -> List[Path]:
    """List LAS/LAZ/COPC files, preferring COPC over matching plain LAZ."""
    files = sorted(input_dir.glob("*.laz")) + sorted(input_dir.glob("*.las"))
    if not files:
        return []

    by_key: Dict[str, Path] = {}
    for path in sorted(files):
        key = path.name[:-9] if path.name.endswith(".copc.laz") else path.stem
        existing = by_key.get(key)
        if existing is None or path.name.endswith(".copc.laz"):
            by_key[key] = path
    return sorted(by_key.values())


def load_standardization_dims(json_path: Path) -> Set[str]:
    """Load reference_attribute_names from a collection_summary.json,
    convert R/lidR names to laspy names, and return as a set (minus X, Y, Z).

    Dimensions that are constant/all-zero across the entire collection
    (variance == 0 or absent from global_attribute_stats) are excluded
    automatically, since transferring them would be pointless.
    """
    with open(json_path) as f:
        data = json.load(f)
    collection = data["collection"]
    ref_names = collection["reference_attribute_names"]

    # Build set of attribute names that have actual variation
    global_stats = collection.get("global_attribute_stats", [])
    has_variation = set()
    for stat in global_stats:
        name = stat.get("name", "")
        variance = stat.get("variance", 0)
        if variance > 0:
            has_variation.add(name)

    result = set()
    skipped = []
    for n in ref_names:
        if n in ("X", "Y", "Z"):
            continue
        laspy_name = _DIMENSION_NAME_ALIASES.get(n, n)
        if has_variation and n not in has_variation:
            skipped.append(n)
            continue
        result.add(laspy_name)

    if skipped:
        print(f"  Standardization: skipping {len(skipped)} constant/zero dims: {skipped}", flush=True)

    return result


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


def normalize_tile_id(stem: str) -> str:
    """Return the base tile id by extracting the c##_r## pattern if present."""
    m = re.search(r"c\d+_r\d+", stem)
    if m:
        return m.group(0)
    return re.sub(
        r"(?:_segmented_remapped|_segmented|_remapped|_results|_subsampled_[\d.]+(?:cm|m))+$",
        "",
        stem,
    )


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
    bounds_field: str = "bounds",
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float]], List[Dict[str, Optional[int]]]]:
    """
    Build a neighbor graph from tile_bounds_tindex.json.

    Args:
        tile_bounds_json: Path to the tile bounds JSON.
        bounds_field: Preferred bounds field to use for matching/centers.
            Falls back to ``bounds`` when the requested field is absent.

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
    tile_buffer = float(data.get("tile_buffer", 0.0))

    json_bounds: List[Tuple[float, float, float, float]] = []
    centers: List[Tuple[float, float]] = []

    for tile in tiles:
        if bounds_field in tile:
            bx, by = tile[bounds_field]
        elif bounds_field == "planned_bounds" and "core" in tile:
            core = tile["core"]
            bx = [float(core[0][0]) - tile_buffer, float(core[0][1]) + tile_buffer]
            by = [float(core[1][0]) - tile_buffer, float(core[1][1]) + tile_buffer]
        elif "bounds" in tile:
            bx, by = tile["bounds"]
        else:
            raise ValueError(
                f"Tile entry missing bounds field '{bounds_field}' and fallback 'bounds': {tile}"
            )
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
    json_labels: Optional[List[str]] = None,
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

    # Fast path: canonical tile labels like c00_r00 can be matched directly to
    # JSON row/col labels when available.
    if json_labels is not None and len(json_labels) == len(json_bounds):
        label_to_json = {
            label: idx for idx, label in enumerate(json_labels) if label is not None
        }
        for name, _ in tile_items:
            json_idx = label_to_json.get(name)
            if json_idx is None or json_idx in used_json:
                continue
            tile_to_json[name] = json_idx
            json_to_tile[json_idx] = name
            used_json.add(json_idx)
        if len(tile_to_json) == len(tile_boundaries):
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

    tile_name = normalize_tile_id(filepath.stem)

    boundary = all_tiles.get(tile_name, compute_tile_bounds(points))

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
        # Use mutual overlap rather than one-sided containment.
        overlap_ratios[(inst_a, inst_b)] = min(ratio_a, ratio_b)

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


def _stream_merged_subset(
    merged_file: Path,
    bounds: Tuple[float, float, float, float],
    spatial_buffer: float,
    chunk_size: int = 1_000_000,
    dim_filter: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, laspy.ExtraBytesParams]]:
    """Stream a merged LAZ file and collect only points within *bounds* + buffer.

    Returns the same format as ``load_merged_file`` — (points, all_dims,
    extra_dim_params) — but only for the spatial subset.  Peak memory is
    bounded by the subset size, not the full file.

    Args:
        merged_file: Path to merged LAZ.
        bounds: (xmin, xmax, ymin, ymax) of the region of interest.
        spatial_buffer: Extra padding in metres around *bounds*.
        chunk_size: Points per streaming chunk.
        dim_filter: If provided, only collect these dimension names (plus
            XYZ which are always collected).  Reduces memory when callers
            only need a subset of dimensions (e.g. remap needs only
            3DTrees dims, not all standard dims).
    """
    xmin, xmax, ymin, ymax = bounds
    x_lo, x_hi = xmin - spatial_buffer, xmax + spatial_buffer
    y_lo, y_hi = ymin - spatial_buffer, ymax + spatial_buffer

    pts_x: List[np.ndarray] = []
    pts_y: List[np.ndarray] = []
    pts_z: List[np.ndarray] = []
    dim_lists: Dict[str, List[np.ndarray]] = defaultdict(list)
    extra_dim_params: Dict[str, laspy.ExtraBytesParams] = {}

    with laspy.open(str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        n_merged = reader.header.point_count
        std_dim_names = [
            n for n in reader.header.point_format.dimension_names
            if n not in ("X", "Y", "Z")
            and (dim_filter is None or n in dim_filter)
        ]
        for dim in reader.header.point_format.extra_dimensions:
            extra_dim_params[dim.name] = extra_bytes_params_from_dimension_info(dim)

        extra_dim_names_to_read = [
            n for n in extra_dim_params
            if dim_filter is None or n in dim_filter
        ]

        scanned = 0
        kept_total = 0
        n_chunks_est = (n_merged + chunk_size - 1) // chunk_size if chunk_size else 1
        for chunk in reader.chunk_iterator(chunk_size):
            cx = np.asarray(chunk.x)
            cy = np.asarray(chunk.y)
            mask = (cx >= x_lo) & (cx <= x_hi) & (cy >= y_lo) & (cy <= y_hi)
            scanned += len(cx)
            if not np.any(mask):
                # Print progress every ~20% of chunks
                pct = scanned / n_merged * 100 if n_merged else 100
                if int(pct) % 20 < int(len(cx) / n_merged * 100) + 1:
                    print(f" {pct:.0f}%", end="", flush=True)
                continue
            kept = int(mask.sum())
            kept_total += kept
            pts_x.append(cx[mask])
            pts_y.append(cy[mask])
            pts_z.append(np.asarray(chunk.z)[mask])
            for dim_name in std_dim_names:
                arr = getattr(chunk, dim_name, None)
                if arr is not None:
                    dim_lists[dim_name].append(np.asarray(arr)[mask])
            for dim_name in extra_dim_names_to_read:
                arr = getattr(chunk, dim_name, None)
                if arr is not None:
                    dim_lists[dim_name].append(np.asarray(arr)[mask])
            pct = scanned / n_merged * 100 if n_merged else 100
            if int(pct) % 20 < int(len(cx) / n_merged * 100) + 1:
                print(f" {pct:.0f}%", end="", flush=True)

    if not pts_x:
        return np.empty((0, 3), dtype=np.float64), {}, extra_dim_params

    points = np.column_stack([
        np.concatenate(pts_x), np.concatenate(pts_y), np.concatenate(pts_z),
    ])
    all_dims = {name: np.concatenate(arrs) for name, arrs in dim_lists.items() if arrs}
    return points, all_dims, extra_dim_params


def _write_points_with_dimensions_to_laz(
    output_path: Path,
    points: np.ndarray,
    dims: Dict[str, np.ndarray],
) -> Path:
    """Write XYZ plus named dimensions to a LAZ file."""
    if len(points) == 0:
        raise ValueError("Cannot write a LAZ file from an empty point set")

    promote_rgb_to_standard = has_standard_rgb_dims(dims.keys())
    point_format_id = 7 if promote_rgb_to_standard else 6

    header = laspy.LasHeader(point_format=point_format_id, version="1.4")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    output_las = laspy.LasData(header)
    output_las.x = points[:, 0]
    output_las.y = points[:, 1]
    output_las.z = points[:, 2]

    extra_dims_params = []
    for dim_name, dim_arr in dims.items():
        if promote_rgb_to_standard and dim_name in _RGB_STANDARD_DIMS:
            continue
        extra_dims_params.append(laspy.ExtraBytesParams(name=dim_name, type=dim_arr.dtype))
    if extra_dims_params:
        output_las.add_extra_dims(extra_dims_params)

    for dim_name, dim_arr in dims.items():
        setattr(output_las, dim_name, dim_arr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_las.write(
        str(output_path),
        do_compress=True,
        laz_backend=laspy.LazBackend.LazrsParallel,
    )
    del output_las
    return output_path


def _build_spatial_slices(
    bounds: Tuple[float, float, float, float],
    slice_count: int,
    slice_length: Optional[float] = None,
    target_points: Optional[int] = None,
    total_points: Optional[int] = None,
) -> List[Tuple[Tuple[float, float, float, float], str, bool]]:
    """Split XY bounds into thin spatial slices along the longer axis.

    Returns ``[(slice_bounds, axis, include_upper_edge), ...]`` where
    ``include_upper_edge`` is True only for the final slice along the chosen axis.
    """
    xmin, xmax, ymin, ymax = bounds
    x_span = max(0.0, float(xmax) - float(xmin))
    y_span = max(0.0, float(ymax) - float(ymin))
    if (slice_count <= 1 and not slice_length and not target_points) or (x_span == 0.0 and y_span == 0.0):
        return [(bounds, "x", True)]

    axis = "x" if x_span >= y_span else "y"
    span = x_span if axis == "x" else y_span
    if span == 0.0:
        return [(bounds, axis, True)]

    if target_points is not None and int(target_points) > 0 and total_points is not None and int(total_points) > 0:
        n_slices = max(1, int(np.ceil(float(total_points) / float(target_points))))
    elif slice_length is not None and float(slice_length) > 0:
        n_slices = max(1, int(np.ceil(span / float(slice_length))))
    else:
        n_slices = max(1, int(slice_count))
    step = span / n_slices
    slices: List[Tuple[Tuple[float, float, float, float], str, bool]] = []
    for idx in range(n_slices):
        is_last = idx == n_slices - 1
        start = (xmin if axis == "x" else ymin) + idx * step
        stop = (xmax if axis == "x" else ymax) if is_last else start + step
        if axis == "x":
            slice_bounds = (start, stop, ymin, ymax)
        else:
            slice_bounds = (xmin, xmax, start, stop)
        slices.append((slice_bounds, axis, is_last))
    return slices


def _snap_spatial_slices_to_header_grid(
    slice_specs: List[Tuple[Tuple[float, float, float, float], str, bool]],
    header,
) -> List[Tuple[Tuple[float, float, float, float], str, bool]]:
    """Snap slice boundaries to the point coordinate grid defined by a LAS header.

    This avoids tiny floating-point drift like ``585910.0800000001`` that can
    leave exact quantized coordinates outside both neighboring slices.
    """
    scales = tuple(float(v) for v in getattr(header, "scales", (0.0, 0.0, 0.0)))
    offsets = tuple(float(v) for v in getattr(header, "offsets", (0.0, 0.0, 0.0)))
    if len(scales) < 2 or len(offsets) < 2:
        return slice_specs

    def _snap(value: float, axis_idx: int) -> float:
        scale = scales[axis_idx]
        offset = offsets[axis_idx]
        if not np.isfinite(value) or scale <= 0:
            return float(value)
        raw = round((float(value) - offset) / scale)
        return float(offset + raw * scale)

    snapped = []
    for slice_bounds, axis, include_upper in slice_specs:
        xmin, xmax, ymin, ymax = slice_bounds
        if axis == "x":
            xmin = _snap(xmin, 0)
            xmax = _snap(xmax, 0)
        else:
            ymin = _snap(ymin, 1)
            ymax = _snap(ymax, 1)
        snapped.append(((xmin, xmax, ymin, ymax), axis, include_upper))
    return snapped


def _load_copc_subset_all_dims(
    path: Path,
    bounds: Tuple[float, float, float, float],
    halo: float = 1.0,
    work_dir: Optional[Path] = None,
) -> Tuple[Optional[laspy.LasData], int]:
    """Load a spatial COPC subset with all dimensions.

    Prefer ``laspy.CopcReader`` for in-memory subset reads. Fall back to the
    older PDAL -> temporary LAS path if direct COPC reads fail for a file.
    """
    xmin, xmax, ymin, ymax = bounds
    x_lo = xmin - halo
    x_hi = xmax + halo
    y_lo = ymin - halo
    y_hi = ymax + halo

    try:
        with laspy.CopcReader.open(str(path)) as reader:
            query_bounds = laspy.copc.Bounds(
                mins=np.array([x_lo, y_lo], dtype=np.float64),
                maxs=np.array([x_hi, y_hi], dtype=np.float64),
            )
            subset = reader.spatial_query(query_bounds)
            subset_count = len(subset)
            return (subset, subset_count) if subset_count > 0 else (None, 0)
    except Exception as exc:
        warn_key = str(path)
        if warn_key not in _COPC_READER_FALLBACK_WARNED:
            print(
                f"      CopcReader fallback for {path.name}: {exc}",
                flush=True,
            )
            _COPC_READER_FALLBACK_WARNED.add(warn_key)

    work_root = work_dir if work_dir is not None else Path(tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=work_root,
    ) as pf:
        pipeline_file = Path(pf.name)
        subset_file = pipeline_file.with_suffix(".las")
        pipeline = {
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": str(path),
                    "bounds": f"([{x_lo},{x_hi}],[{y_lo},{y_hi}])",
                },
                {
                    "type": "writers.las",
                    "filename": str(subset_file),
                    "forward": "all",
                    "extra_dims": "all",
                    "minor_version": 4,
                },
            ]
        }
        json.dump(pipeline, pf)

    try:
        result = subprocess.run(
            [get_pdal_path(), "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "PDAL pipeline failed")
        with laspy.open(str(subset_file), laz_backend=laspy.LazBackend.Lazrs) as reader:
            if reader.header.point_count == 0:
                return None, 0
        subset = laspy.read(str(subset_file), laz_backend=laspy.LazBackend.LazrsParallel)
        return subset, len(subset.points)
    finally:
        for tmp_path in (pipeline_file, subset_file):
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass


# =============================================================================
# Stage 6 (streaming): Retile merged → original tile grid
# =============================================================================


def retile_to_original_files_streaming(
    merged_file: Path,
    original_tiles_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    retile_buffer: float = 2.0,
    chunk_size: int = 1_000_000,
    instance_dimension: str = "PredInstance",
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
):
    """Fully streaming retiling — neither merged nor original files loaded fully.

    For each original tile:
      1. Stream merged LAZ → collect spatial subset (for KDTree)
      2. Build KDTree from merged subset
      3. Stream original tile in chunks, query KDTree, write output chunks

    Peak RAM ≈ merged spatial subset + KDTree + one original chunk.
    """
    import gc

    print(f"\n{'=' * 60}", flush=True)
    print("Retiling merged results to original tile files (streaming)", flush=True)
    print(f"{'=' * 60}", flush=True)

    original_files = sorted(original_tiles_dir.glob("*.laz"))
    if not original_files:
        original_files = sorted(original_tiles_dir.glob("*.las"))
    if not original_files:
        print(f"  No LAZ/LAS files found in {original_tiles_dir}", flush=True)
        return

    print(f"  Found {len(original_files)} original tile files", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
    threedtrees_set = set(threedtrees_dims) if threedtrees_dims else {instance_dimension}
    suffix = threedtrees_suffix or "SAT"

    # Get merged extra-dim metadata once
    with laspy.open(str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel) as mf:
        merged_extra_dim_params: Dict[str, laspy.ExtraBytesParams] = {
            dim.name: extra_bytes_params_from_dimension_info(dim)
            for dim in mf.header.point_format.extra_dimensions
        }

    tiles_to_process = []
    skipped = 0
    for orig_file in original_files:
        output_name = orig_file.name.replace(".copc.laz", ".laz")
        output_file = output_dir / output_name
        if output_file.exists():
            skipped += 1
        else:
            tiles_to_process.append((orig_file, output_file))

    if skipped > 0:
        print(f"  Skipping {skipped} already processed tiles", flush=True)
    if not tiles_to_process:
        print(f"  All tiles already processed!", flush=True)
        return

    import time as _time
    print(f"  Processing {len(tiles_to_process)} tiles...", flush=True)

    for ti, (orig_file, output_file) in enumerate(tiles_to_process):
        try:
            tile_start = _time.monotonic()
            print(f"  [{ti+1}/{len(tiles_to_process)}] {orig_file.name}...", flush=True)

            # 1. Get original tile header info (no points loaded)
            with laspy.open(str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel) as f:
                tile_bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
                n_orig = f.header.point_count
                orig_pf = f.header.point_format
                orig_version = f.header.version
                orig_offsets = f.header.offsets
                orig_scales = f.header.scales
                orig_extra_dims = list(orig_pf.extra_dimensions)

            # 2. Stream merged file — collect only this tile's subset
            print(f"    Collecting merged spatial subset...", end="", flush=True)
            local_merged_points, local_all_dims, _ = _stream_merged_subset(
                merged_file, tile_bounds, spatial_buffer, chunk_size,
            )
            print(f" {len(local_merged_points):,} pts", flush=True)
            if len(local_merged_points) == 0:
                print(f"    No merged points in tile region — skipping", flush=True)
                continue

            local_merged_instances = local_all_dims.pop(
                instance_dimension,
                np.zeros(len(local_merged_points), dtype=np.int32),
            )
            local_merged_extras = local_all_dims

            # 3. Build KDTree from merged subset
            print(f"    Building KDTree...", end="", flush=True)
            local_tree = cKDTree(local_merged_points)
            del local_merged_points
            print(" done", flush=True)

            # 4. Build output header
            out_header = laspy.LasHeader(
                point_format=orig_pf.id, version=orig_version,
            )
            out_header.offsets = orig_offsets
            out_header.scales = orig_scales

            # Branded names for merged dims
            inst_out_name = (
                f"3DT_{instance_dimension}_{suffix}"
                if instance_dimension in threedtrees_set
                else instance_dimension
            )
            merged_rename: Dict[str, str] = {instance_dimension: inst_out_name}
            for dim_name in local_merged_extras:
                if dim_name in threedtrees_set:
                    merged_rename[dim_name] = f"3DT_{dim_name}_{suffix}"
                else:
                    merged_rename[dim_name] = dim_name

            # Add extra dims to output header
            out_extra_names = set(out_header.point_format.dimension_names)
            extra_to_add = []
            for dim in orig_extra_dims:
                if dim.name not in out_extra_names:
                    extra_to_add.append(extra_bytes_params_from_dimension_info(dim, name=dim.name))
                    out_extra_names.add(dim.name)
            if inst_out_name not in out_extra_names:
                extra_to_add.append(laspy.ExtraBytesParams(name=inst_out_name, type=np.int32))
                out_extra_names.add(inst_out_name)
            for dim_name in local_merged_extras:
                out_name = merged_rename[dim_name]
                if out_name not in out_extra_names:
                    if dim_name in merged_extra_dim_params:
                        params = merged_extra_dim_params[dim_name]
                        extra_to_add.append(laspy.ExtraBytesParams(name=out_name, type=params.type))
                    else:
                        extra_to_add.append(laspy.ExtraBytesParams(
                            name=out_name, type=local_merged_extras[dim_name].dtype,
                        ))
                    out_extra_names.add(out_name)
            if extra_to_add:
                out_header.add_extra_dims(extra_to_add)

            # Pre-compute which original dims to copy
            orig_dim_names_to_copy = [
                d for d in orig_pf.dimension_names if d in out_extra_names
            ]

            # 5. Stream original tile + write output chunk by chunk
            unique_instances: set = set()
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with laspy.open(str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                with laspy.open(
                    str(output_file), mode="w", header=out_header,
                    do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel,
                ) as writer:
                    for chunk in reader.chunk_iterator(chunk_size):
                        n = len(chunk)
                        orig_pts = np.column_stack([chunk.x, chunk.y, chunk.z])
                        distances, indices = local_tree.query(orig_pts, workers=-1)
                        del orig_pts

                        point_record = laspy.ScaleAwarePointRecord.zeros(n, header=out_header)
                        # Copy all original dims from chunk
                        for dim_name in orig_dim_names_to_copy:
                            try:
                                setattr(point_record, dim_name, getattr(chunk, dim_name))
                            except Exception:
                                pass
                        # Set merged dims
                        setattr(point_record, inst_out_name, local_merged_instances[indices])
                        for dim_name, arr in local_merged_extras.items():
                            setattr(point_record, merged_rename[dim_name], arr[indices])

                        writer.write_points(point_record)

                        inst_chunk = local_merged_instances[indices]
                        inst_pos = inst_chunk[inst_chunk > 0]
                        if len(inst_pos) > 0:
                            unique_instances.update(np.unique(inst_pos).tolist())

            del local_tree, local_merged_instances, local_merged_extras
            gc.collect()

            tile_dt = _time.monotonic() - tile_start
            print(
                f"    Done: {n_orig:,} pts, "
                f"{len(unique_instances)} instances, {tile_dt:.1f}s",
                flush=True,
            )

        except Exception as e:
            print(f"  [{ti+1}/{len(tiles_to_process)}] FAILED: {e} → {orig_file.name}", flush=True)

    print(f"\n  ✓ Retiling complete: {len(tiles_to_process)} tiles processed (streaming)", flush=True)
    gc.collect()


# =============================================================================
# Stage 7 (streaming): Remap merged → original input files
# =============================================================================


def remap_to_original_input_files_streaming(
    merged_file: Path,
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    retile_buffer: float = 2.0,
    chunk_size: int = 1_000_000,
    spatial_slices: int = 50,
    spatial_chunk_length: Optional[float] = None,
    spatial_target_points: Optional[int] = None,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
    target_dims: Optional[Set[str]] = None,
):
    """Fully streaming remap — neither merged nor original files loaded fully.

    For each original input file:
      1. Stream merged LAZ → collect spatial subset (for KDTree)
      2. Build KDTree from merged subset
      3. Stream original file in chunks, query KDTree, write output chunks

    Peak RAM ≈ merged spatial subset + KDTree + one original chunk.
    """
    import gc

    print(f"\n{'=' * 60}", flush=True)
    print("Remapping to original input files (streaming)", flush=True)
    print(f"{'=' * 60}", flush=True)

    original_files = list_pointcloud_files(original_input_dir)
    if not original_files:
        print(f"  No original input files found in {original_input_dir}", flush=True)
        return

    print(f"  Found {len(original_files)} original input files", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if threedtrees_dims is None:
        threedtrees_dims_list = ["PredInstance", "PredSemantic"]
    else:
        threedtrees_dims_list = list(threedtrees_dims)
    threedtrees_dims_set = set(threedtrees_dims_list)

    # Get merged extra-dim metadata once
    with laspy.open(str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel) as mf:
        merged_extra_dim_params: Dict[str, laspy.ExtraBytesParams] = {
            dim.name: extra_bytes_params_from_dimension_info(dim)
            for dim in mf.header.point_format.extra_dimensions
        }

    files_to_process = []
    skipped = 0
    for input_file in original_files:
        output_name = input_file.name.replace(".copc.laz", ".laz")
        output_file = output_dir / output_name
        if output_file.exists():
            skipped += 1
        else:
            files_to_process.append((input_file, output_file))

    if skipped > 0:
        print(f"  Skipping {skipped} already processed files", flush=True)
    if not files_to_process:
        print(f"  All files already processed!", flush=True)
        return

    print(f"  Processing {len(files_to_process)} files...", flush=True)

    keep_original_dim = (
        lambda name: target_dims is None
        or name in target_dims
        or name in {"X", "Y", "Z", "x", "y", "z"}
    )

    total_matched = 0
    total_points = 0

    import time as _time
    stage_start = _time.monotonic()

    with tempfile.TemporaryDirectory(prefix="3dtrees_stage7_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        for fi, (input_file, output_file) in enumerate(files_to_process):
            try:
                file_start = _time.monotonic()
                print(f"  [{fi+1}/{len(files_to_process)}] {input_file.name}...", flush=True)

                # 1. Get original file header info (no points loaded)
                with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as f:
                    file_bounds = (f.header.x_min, f.header.x_max, f.header.y_min, f.header.y_max)
                    n_input = f.header.point_count
                    orig_pf = f.header.point_format
                    orig_version = f.header.version
                    orig_offsets = f.header.offsets
                    orig_scales = f.header.scales
                    orig_extra_dims = list(orig_pf.extra_dimensions)
                print(f"    Header: {n_input:,} pts, bounds x=[{file_bounds[0]:.1f}, {file_bounds[1]:.1f}] y=[{file_bounds[2]:.1f}, {file_bounds[3]:.1f}]", flush=True)

                branded_names = {}
                for dim_name in threedtrees_dims_set:
                    if dim_name in merged_extra_dim_params:
                        branded_names[dim_name] = (
                            f"3DT_{dim_name}_{threedtrees_suffix}"
                            if threedtrees_suffix
                            else f"3DT_{dim_name}"
                        )

                out_header = laspy.LasHeader(
                    point_format=orig_pf.id, version=orig_version,
                )
                out_header.offsets = orig_offsets
                out_header.scales = orig_scales

                out_extra_names = set(out_header.point_format.dimension_names)
                extra_to_add = []
                for dim in orig_extra_dims:
                    if keep_original_dim(dim.name) and dim.name not in out_extra_names:
                        extra_to_add.append(extra_bytes_params_from_dimension_info(dim))
                        out_extra_names.add(dim.name)
                for dim_name, out_name in branded_names.items():
                    if out_name not in out_extra_names:
                        if dim_name in merged_extra_dim_params:
                            params = merged_extra_dim_params[dim_name]
                            extra_to_add.append(laspy.ExtraBytesParams(name=out_name, type=params.type))
                        out_extra_names.add(out_name)
                if extra_to_add:
                    out_header.add_extra_dims(extra_to_add)

                orig_dims_to_copy = [
                    d for d in orig_pf.dimension_names
                    if keep_original_dim(d) and d in out_extra_names
                ]

                unique_instances: set = set()
                output_file.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                merged_dims_to_read = set(branded_names.keys())

                with laspy.open(
                    str(output_file), mode="w", header=out_header,
                    do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel,
                ) as writer:
                    if input_file.name.endswith(".copc.laz"):
                        slice_specs = _build_spatial_slices(
                            file_bounds,
                            spatial_slices,
                            spatial_chunk_length,
                            spatial_target_points,
                            n_input,
                        )
                        slice_specs = _snap_spatial_slices_to_header_grid(slice_specs, in_header)
                        print(
                            f"    Spatial slicing: {len(slice_specs)} slices along {slice_specs[0][1].upper()}",
                            flush=True,
                        )

                        for slice_idx, (slice_bounds, axis, include_upper) in enumerate(slice_specs, start=1):
                            subset, subset_count = _load_copc_subset_all_dims(
                                input_file,
                                slice_bounds,
                                halo=1.0,
                                work_dir=tmp_dir,
                            )
                            if subset is None or subset_count == 0:
                                print(f"      Slice {slice_idx}/{len(slice_specs)}: empty original subset", flush=True)
                                continue

                            sx = np.asarray(subset.x)
                            sy = np.asarray(subset.y)
                            sz = np.asarray(subset.z)
                            if axis == "x":
                                upper_mask = sx <= slice_bounds[1] if include_upper else sx < slice_bounds[1]
                                core_mask = (sx >= slice_bounds[0]) & upper_mask & (sy >= slice_bounds[2]) & (sy <= slice_bounds[3])
                            else:
                                upper_mask = sy <= slice_bounds[3] if include_upper else sy < slice_bounds[3]
                                core_mask = (sx >= slice_bounds[0]) & (sx <= slice_bounds[1]) & (sy >= slice_bounds[2]) & upper_mask
                            if not np.any(core_mask):
                                print(f"      Slice {slice_idx}/{len(slice_specs)}: no core points after halo crop", flush=True)
                                del subset
                                continue

                            orig_xyz = np.column_stack([sx[core_mask], sy[core_mask], sz[core_mask]])
                            local_pts, local_dims, _ = _stream_merged_subset(
                                merged_file,
                                slice_bounds,
                                0.0,
                                chunk_size,
                                dim_filter=merged_dims_to_read,
                            )

                            point_record = laspy.ScaleAwarePointRecord.zeros(len(orig_xyz), header=out_header)
                            for dim_name in orig_dims_to_copy:
                                arr = getattr(subset, dim_name, None)
                                if arr is not None:
                                    setattr(point_record, dim_name, np.asarray(arr)[core_mask])

                            if len(local_pts) > 0:
                                local_tree = cKDTree(local_pts)
                                _, indices = local_tree.query(orig_xyz, workers=-1)
                                del local_tree
                                for dim_name, out_name in branded_names.items():
                                    if dim_name in local_dims:
                                        vals = local_dims[dim_name][indices]
                                        setattr(point_record, out_name, vals)
                                        if np.issubdtype(vals.dtype, np.integer):
                                            pos = vals[vals > 0]
                                            if len(pos) > 0:
                                                unique_instances.update(np.unique(pos).tolist())
                                del local_pts, local_dims

                            writer.write_points(point_record)
                            written += len(orig_xyz)
                            pct = written / n_input * 100 if n_input else 100
                            elapsed = _time.monotonic() - file_start
                            rate = written / elapsed if elapsed > 0 else 0
                            print(
                                f"      Slice {slice_idx}/{len(slice_specs)}: {len(orig_xyz):,} core pts from {subset_count:,} in-bounds ({pct:.0f}%, {rate:,.0f} pts/s)",
                                flush=True,
                            )
                            del point_record, orig_xyz, subset, sx, sy, sz
                            gc.collect()
                    else:
                        n_chunks_expected = (n_input + chunk_size - 1) // chunk_size
                        spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
                        print(
                            f"    Non-COPC fallback: streaming {n_input:,} pts in ~{n_chunks_expected} chunks",
                            flush=True,
                        )

                        with laspy.open(str(input_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                            chunk_i = 0
                            for chunk in reader.chunk_iterator(chunk_size):
                                n = len(chunk)
                                orig_xyz = np.column_stack([chunk.x, chunk.y, chunk.z])
                                chunk_xmin = float(np.min(orig_xyz[:, 0]))
                                chunk_xmax = float(np.max(orig_xyz[:, 0]))
                                chunk_ymin = float(np.min(orig_xyz[:, 1]))
                                chunk_ymax = float(np.max(orig_xyz[:, 1]))
                                chunk_bounds = (chunk_xmin, chunk_xmax, chunk_ymin, chunk_ymax)

                                local_pts, local_dims, _ = _stream_merged_subset(
                                    merged_file,
                                    chunk_bounds,
                                    spatial_buffer,
                                    chunk_size,
                                    dim_filter=merged_dims_to_read,
                                )

                                point_record = laspy.ScaleAwarePointRecord.zeros(n, header=out_header)
                                for dim_name in orig_dims_to_copy:
                                    try:
                                        setattr(point_record, dim_name, getattr(chunk, dim_name))
                                    except Exception:
                                        pass

                                if len(local_pts) > 0:
                                    local_tree = cKDTree(local_pts)
                                    _, indices = local_tree.query(orig_xyz, workers=-1)
                                    del local_tree, local_pts
                                    for dim_name, out_name in branded_names.items():
                                        if dim_name in local_dims:
                                            vals = local_dims[dim_name][indices]
                                            setattr(point_record, out_name, vals)
                                            if np.issubdtype(vals.dtype, np.integer):
                                                pos = vals[vals > 0]
                                                if len(pos) > 0:
                                                    unique_instances.update(np.unique(pos).tolist())
                                    del local_dims

                                writer.write_points(point_record)
                                written += n
                                chunk_i += 1
                                pct = written / n_input * 100 if n_input else 100
                                elapsed = _time.monotonic() - file_start
                                rate = written / elapsed if elapsed > 0 else 0
                                print(
                                    f"      Chunk {chunk_i}/{n_chunks_expected}: {n:,} pts ({pct:.0f}%, {rate:,.0f} pts/s)",
                                    flush=True,
                                )
                                del orig_xyz

                gc.collect()

                total_matched += n_input
                total_points += n_input
                file_dt = _time.monotonic() - file_start
                print(
                    f"    Done: {n_input:,} pts, {len(unique_instances)} instances, {file_dt:.1f}s",
                    flush=True,
                )

            except Exception as e:
                print(f"  [{fi+1}/{len(files_to_process)}] FAILED: {e} → {input_file.name}", flush=True)

    overall_pct = (total_matched / total_points * 100) if total_points > 0 else 0
    print(
        f"\n  ✓ Remap complete: {len(files_to_process)} files, "
        f"{total_matched:,}/{total_points:,} matched ({overall_pct:.1f}%)",
        flush=True,
    )

    if files_to_process and total_matched > 0:
        first_input, first_output = files_to_process[0]
        if first_output.exists():
            _validate_common_dimensions_minmax(first_input, first_output)

    gc.collect()


# =============================================================================
# Legacy compatibility shims
# =============================================================================


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
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
):
    """Compatibility shim that materializes a temp LAZ and delegates to streaming."""
    del merged_extra_dim_params, num_threads, parallel_tiles
    import tempfile

    with tempfile.TemporaryDirectory(prefix="3dtrees_retile_") as tmp_dir:
        merged_laz = Path(tmp_dir) / "merged_for_retile.laz"
        _write_points_with_dimensions_to_laz(
            merged_laz,
            merged_points,
            {instance_dimension: merged_instances, **merged_extra_dims},
        )
        retile_to_original_files_streaming(
            merged_file=merged_laz,
            original_tiles_dir=original_tiles_dir,
            output_dir=output_dir,
            tolerance=tolerance,
            retile_buffer=retile_buffer,
            chunk_size=chunk_size,
            instance_dimension=instance_dimension,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
        )


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
    target_dims: Optional[Set[str]] = None,
):
    """Compatibility shim that materializes a temp LAZ and delegates to streaming."""
    del merged_extra_dim_params, num_threads
    import tempfile

    with tempfile.TemporaryDirectory(prefix="3dtrees_remap_") as tmp_dir:
        merged_laz = Path(tmp_dir) / "merged_for_originals.laz"
        _write_points_with_dimensions_to_laz(
            merged_laz,
            merged_points,
            merged_extra_dims,
        )
        remap_to_original_input_files_streaming(
            merged_file=merged_laz,
            original_input_dir=original_input_dir,
            output_dir=output_dir,
            tolerance=tolerance,
            retile_buffer=retile_buffer,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            target_dims=target_dims,
        )


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


class _OriginalFileCache:
    """LRU cache for original-file spatial subsets (KDTrees + dim arrays).

    Caches spatially cropped subsets keyed by file + XY window. When the
    merged file is streamed chunk-by-chunk, consecutive chunks are spatially
    close, so the same cropped subsets are often reused. The LRU policy evicts
    the least-recently-used entry when the cache is full, keeping peak RAM
    bounded.
    """

    def __init__(self, max_entries: int = 3, tmp_dir: Optional[Path] = None, chunk_size: int = 2_000_000) -> None:
        from collections import OrderedDict
        self._cache: 'OrderedDict[Tuple[Path, float, float, float, float], Tuple[Optional[cKDTree], Dict[str, np.ndarray]]]' = OrderedDict()
        self._max = max_entries
        self._tmp_dir = tmp_dir
        self._chunk_size = chunk_size

    def _load_copc_subset_with_reader(
        self,
        path: Path,
        dims_to_read: Dict[str, str],
        dim_dtypes: Dict[str, np.dtype],
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], float]:
        """Read a bounded COPC subset directly with laspy.CopcReader."""
        import time as _time

        read_start = _time.monotonic()
        with laspy.CopcReader.open(str(path)) as reader:
            query_bounds = laspy.copc.Bounds(
                mins=np.array([x_lo, y_lo], dtype=np.float64),
                maxs=np.array([x_hi, y_hi], dtype=np.float64),
            )
            points = reader.spatial_query(query_bounds)

        if points is None or len(points) == 0:
            return None, {}, _time.monotonic() - read_start

        xyz = np.column_stack([
            np.asarray(points.x),
            np.asarray(points.y),
            np.asarray(points.z),
        ])
        dim_arrays: Dict[str, np.ndarray] = {}
        for out_name, orig_name in dims_to_read.items():
            arr = getattr(points, orig_name, None)
            if arr is not None:
                dtype = dim_dtypes.get(out_name, np.float64)
                dim_arrays[out_name] = np.asarray(arr).astype(dtype, copy=False)

        return xyz, dim_arrays, _time.monotonic() - read_start

    def _load_copc_subset_with_pdal(
        self,
        path: Path,
        dims_to_read: Dict[str, str],
        dim_dtypes: Dict[str, np.dtype],
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], float]:
        """Read a bounded COPC subset via PDAL and return XYZ + requested dims."""
        import time as _time

        work_dir = self._tmp_dir if self._tmp_dir is not None else Path(tempfile.gettempdir())
        work_dir.mkdir(parents=True, exist_ok=True)
        read_start = _time.monotonic()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=work_dir,
        ) as pf:
            pipeline_file = Path(pf.name)
            subset_file = pipeline_file.with_suffix(".las")
            pipeline = {
                "pipeline": [
                    {
                        "type": "readers.copc",
                        "filename": str(path),
                        "bounds": f"([{x_lo},{x_hi}],[{y_lo},{y_hi}])",
                    },
                    {
                        "type": "writers.las",
                        "filename": str(subset_file),
                        "forward": "all",
                        "extra_dims": "all",
                        "minor_version": 4,
                    },
                ]
            }
            json.dump(pipeline, pf)

        try:
            result = subprocess.run(
                [get_pdal_path(), "pipeline", str(pipeline_file)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "PDAL pipeline failed")

            with laspy.open(str(subset_file), laz_backend=laspy.LazBackend.Lazrs) as f:
                n_pts = f.header.point_count
                if n_pts == 0:
                    return None, {}, _time.monotonic() - read_start
                points = f.read()

            xyz = np.column_stack([
                np.asarray(points.x),
                np.asarray(points.y),
                np.asarray(points.z),
            ])
            dim_arrays: Dict[str, np.ndarray] = {}
            for out_name, orig_name in dims_to_read.items():
                arr = getattr(points, orig_name, None)
                if arr is not None:
                    dtype = dim_dtypes.get(out_name, np.float64)
                    dim_arrays[out_name] = np.asarray(arr).astype(dtype, copy=False)

            return xyz, dim_arrays, _time.monotonic() - read_start
        finally:
            for tmp_path in (pipeline_file, subset_file):
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _subset_key(
        path: Path,
        bounds: Tuple[float, float, float, float],
        xy_buffer: float,
    ) -> Tuple[Path, float, float, float, float]:
        xmin, xmax, ymin, ymax = bounds
        return (
            path,
            round(xmin - xy_buffer, 3),
            round(xmax + xy_buffer, 3),
            round(ymin - xy_buffer, 3),
            round(ymax + xy_buffer, 3),
        )

    def get(
        self,
        path: Path,
        bounds: Tuple[float, float, float, float],
        xy_buffer: float,
    ) -> Optional[Tuple[Optional[cKDTree], Dict[str, np.ndarray]]]:
        """Return cached subset (tree, dims) or None. Marks entry as recently used."""
        if self._max <= 0:
            return None
        key = self._subset_key(path, bounds, xy_buffer)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def load(
        self,
        path: Path,
        dims_to_read: Dict[str, str],
        dim_dtypes: Dict[str, np.dtype],
        bounds: Tuple[float, float, float, float],
        xy_buffer: float,
    ) -> Tuple[Optional[cKDTree], Dict[str, np.ndarray]]:
        """Stream a spatial subset of an original file and build a KDTree.

        Returns (tree, dim_arrays).  If already cached, returns immediately.
        """
        import time as _time

        cached = self.get(path, bounds, xy_buffer)
        if cached is not None:
            return cached

        load_start = _time.monotonic()
        key = self._subset_key(path, bounds, xy_buffer)
        x_lo, x_hi, y_lo, y_hi = key[1:]

        # Evict if at capacity
        while self._max > 0 and len(self._cache) >= self._max:
            self._evict_lru()

        # Read point count from header (single-threaded to avoid OpenMP conflicts)
        with laspy.open(str(path), laz_backend=laspy.LazBackend.Lazrs) as f:
            n_pts = f.header.point_count

        prefix = "Cache loading subset" if self._max > 0 else "Loading subset"
        print(
            f"    {prefix}: {path.name} x=[{x_lo:.1f}, {x_hi:.1f}] "
            f"y=[{y_lo:.1f}, {y_hi:.1f}] from {n_pts:,} pts... ",
            end="",
            flush=True,
        )

        if path.name.endswith(".copc.laz"):
            try:
                xyz, dim_arrays, read_dt = self._load_copc_subset_with_reader(
                    path, dims_to_read, dim_dtypes, x_lo, x_hi, y_lo, y_hi,
                )
                if xyz is None:
                    total_dt = _time.monotonic() - load_start
                    result = (None, {})
                    if self._max > 0:
                        self._cache[key] = result
                    print(f"empty COPC subset (read {read_dt:.1f}s, total {total_dt:.1f}s)", flush=True)
                    return result
                kept_pts = len(xyz)
                print(f"kept {kept_pts:,} pts via COPC reader, tree... ", end="", flush=True)
            except Exception as e:
                print(f"COPC reader fallback to PDAL ({e})... ", end="", flush=True)
                try:
                    xyz, dim_arrays, read_dt = self._load_copc_subset_with_pdal(
                        path, dims_to_read, dim_dtypes, x_lo, x_hi, y_lo, y_hi,
                    )
                    if xyz is None:
                        total_dt = _time.monotonic() - load_start
                        result = (None, {})
                        if self._max > 0:
                            self._cache[key] = result
                        print(f"empty COPC subset (read {read_dt:.1f}s, total {total_dt:.1f}s)", flush=True)
                        return result
                    kept_pts = len(xyz)
                    print(f"kept {kept_pts:,} pts via PDAL fallback, tree... ", end="", flush=True)
                except Exception as p_e:
                    print(f"PDAL fallback failed ({p_e}); falling back to streamed scan... ", end="", flush=True)
                    xyz = None
                    dim_arrays = {}
                    read_dt = 0.0
        else:
            xyz = None
            dim_arrays = {}
            read_dt = 0.0

        if xyz is None:
            dim_lists: Dict[str, List[np.ndarray]] = defaultdict(list)
            xyz_parts: List[np.ndarray] = []

            # Single-pass streaming read: keep only points in the requested XY window.
            read_start = _time.monotonic()
            kept_pts = 0
            with laspy.open(str(path), laz_backend=laspy.LazBackend.Lazrs) as f:
                for chunk in f.chunk_iterator(self._chunk_size):
                    cx = np.asarray(chunk.x)
                    cy = np.asarray(chunk.y)
                    mask = (cx >= x_lo) & (cx <= x_hi) & (cy >= y_lo) & (cy <= y_hi)
                    if not np.any(mask):
                        continue
                    kept = int(mask.sum())
                    kept_pts += kept
                    xyz_parts.append(
                        np.column_stack([
                            cx[mask],
                            cy[mask],
                            np.asarray(chunk.z)[mask],
                        ])
                    )
                    for out_name, orig_name in dims_to_read.items():
                        arr = getattr(chunk, orig_name, None)
                        if arr is not None:
                            dim_lists[out_name].append(np.asarray(arr)[mask])

            read_dt = _time.monotonic() - read_start
            if kept_pts == 0:
                total_dt = _time.monotonic() - load_start
                result = (None, {})
                if self._max > 0:
                    self._cache[key] = result
                print(f"empty subset (read {read_dt:.1f}s, total {total_dt:.1f}s)", flush=True)
                return result

            xyz = np.concatenate(xyz_parts, axis=0)
            dim_arrays = {}
            for out_name, arrs in dim_lists.items():
                if arrs:
                    dtype = dim_dtypes.get(out_name, np.float64)
                    dim_arrays[out_name] = np.concatenate(arrs).astype(dtype, copy=False)
            print(f"kept {kept_pts:,} pts, tree... ", end="", flush=True)

        tree_start = _time.monotonic()
        tree = cKDTree(xyz, leafsize=32)
        tree_dt = _time.monotonic() - tree_start
        del xyz

        total_dt = _time.monotonic() - load_start
        result = (tree, dim_arrays)
        if self._max > 0:
            self._cache[key] = result
        print(
            f"{'cached' if self._max > 0 else 'done'} (read {read_dt:.1f}s, tree {tree_dt:.1f}s, total {total_dt:.1f}s)",
            flush=True,
        )
        return result

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if not self._cache:
            return
        key, (tree, dims) = self._cache.popitem(last=False)
        path = key[0]
        del tree
        for v in dims.values():
            del v
        dims.clear()
        gc.collect()
        print(f"    Cache evicted: {path.name} subset", flush=True)

    def clear(self) -> None:
        """Release all cached entries."""
        while self._cache:
            self._evict_lru()


def _scan_merged_dim_stats(
    merged_laz: Path,
    merged_dim_names: Set[str],
    skip_core: Set[str],
    chunk_size: int = 2_000_000,
) -> Dict[str, Tuple[float, float, int]]:
    """Lightweight streaming scan for per-dim min/max/nonzero_count.

    Only used in legacy mode (when ``target_dims is None``) to detect which
    merged dimensions are empty/constant and should be overwritten.
    """
    dim_stats: Dict[str, Tuple[float, float, int]] = {}
    with laspy.open(str(merged_laz), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            for dim_name in merged_dim_names - skip_core:
                arr = getattr(chunk, dim_name, None)
                if arr is None:
                    continue
                a = np.asarray(arr)
                c_min = float(np.min(a))
                c_max = float(np.max(a))
                c_nz = int(np.count_nonzero(a))
                if dim_name in dim_stats:
                    p_min, p_max, p_nz = dim_stats[dim_name]
                    dim_stats[dim_name] = (
                        min(p_min, c_min),
                        max(p_max, c_max),
                        p_nz + c_nz,
                    )
                else:
                    dim_stats[dim_name] = (c_min, c_max, c_nz)
    return dim_stats


def _add_original_dimensions_to_merged_impl(
    merged_laz: Path,
    original_input_dir: Path,
    output_path: Path,
    original_files: List[Path],
    tolerance: float,
    retile_buffer: float,
    distance_threshold: Optional[float],
    target_dims: Optional[Set[str]],
    source_extra_dims_to_keep: Optional[Set[str]],
    tmp_dir: Path,
    num_threads: int = 4,
    merge_chunk_size: int = 2_000_000,
    spatial_slices: int = 10,
    spatial_chunk_length: Optional[float] = None,
    spatial_target_points: Optional[int] = None,
) -> None:
    """Merged-centric streaming enrichment (v3).

    Streams the merged file chunk-by-chunk.  For each chunk, determines which
    original files overlap spatially, loads them into an LRU cache (KDTree +
    dimension arrays), queries each tree, keeps the closest match per merged
    point, and writes the enriched chunk immediately.

    Key properties:
    - No merged coordinates held in RAM (streamed one chunk at a time).
    - No memmaps or temp files.
    - Original files cached via LRU (typically 2-3 in cache at once).
    - Optional per-slice parallel subset prep across overlapping original files
      when ``num_threads > 1``.
    - KDTree queries still use ``workers=-1`` to consume all available CPUs.
    - Peak RAM is bounded by the number of overlapping original subsets loaded
      for the current slice.
    """

    skip_core = {"X", "Y", "Z"}
    CHUNK_SIZE = merge_chunk_size
    import time as _time

    phase_start = _time.monotonic()

    # ── Inspect merged input header ──────────────────────────────────────
    # NOTE: Use single-threaded Lazrs throughout enrichment to avoid
    # deadlocks with OpenMP threads persisting from earlier phases.
    print("  Inspecting merged input header...", flush=True)
    with laspy.open(str(merged_laz), laz_backend=laspy.LazBackend.Lazrs) as reader:
        n_merged = reader.header.point_count
        merged_header = reader.header
        merged_point_format = merged_header.point_format
        merged_dim_names = set(merged_point_format.dimension_names)
        merged_extra_dims = list(merged_point_format.extra_dimensions)
        if source_extra_dims_to_keep is not None:
            merged_extra_dims = [
                dim for dim in merged_extra_dims
                if dim.name in source_extra_dims_to_keep
            ]
    print(
        f"  Merged input: {n_merged:,} points, {len(merged_dim_names)} dimensions "
        f"({_time.monotonic() - phase_start:.1f}s)",
        flush=True,
    )

    if n_merged == 0:
        print("  Merged file is empty; nothing to enrich.", flush=True)
        return

    # ── Legacy dim-stats pre-scan (only when target_dims is None) ────────
    dim_stats: Dict[str, Tuple[float, float, int]] = {}
    if target_dims is None:
        phase_start = _time.monotonic()
        print("  Pre-scanning merged dims for legacy overwrite detection...", flush=True)
        dim_stats = _scan_merged_dim_stats(merged_laz, merged_dim_names, skip_core, chunk_size=CHUNK_SIZE)
        print(
            f"  Pre-scan done: {len(dim_stats)} dims analysed "
            f"({_time.monotonic() - phase_start:.1f}s)",
            flush=True,
        )

    # ── Phase 2: Detect original-file dimensions ────────────────────────
    phase_start = _time.monotonic()
    print(f"  Scanning original-file dimensions from {len(original_files)} files...", flush=True)
    orig_dims: Dict[str, np.dtype] = {}
    orig_extra_dim_info: Dict[str, object] = {}
    for i, orig_path in enumerate(original_files):
        file_start = _time.monotonic()
        print(f"    [{i+1}/{len(original_files)}] {orig_path.name}... ", end="", flush=True)
        with laspy.open(str(orig_path), laz_backend=laspy.LazBackend.Lazrs) as f:
            pf = f.header.point_format
            pt_dtype = None
            try:
                one = f.read_points(1)
                if one is not None and one.size > 0:
                    arr_raw = getattr(one, "array", one)
                    dt = getattr(arr_raw, "dtype", None)
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
        n_new = len([d for d in pf.dimension_names if d not in skip_core])
        print(f"{n_new} dims found ({_time.monotonic() - file_start:.1f}s)", flush=True)

    print(
        f"  Original dimension scan complete: {len(orig_dims)} unique dimensions "
        f"({_time.monotonic() - phase_start:.1f}s)",
        flush=True,
    )

    # ── Phase 3: Filter by target_dims if standardization mode ──────────
    requested_target_dims: Set[str] = set()
    if target_dims is not None:
        requested_target_dims = set(target_dims) - skip_core
        skipped_dims = set(orig_dims.keys()) - requested_target_dims
        if skipped_dims:
            print(f"  Standardization filter: skipping {len(skipped_dims)} dims not in target list: "
                  f"{sorted(skipped_dims)}", flush=True)
        orig_dims = {k: v for k, v in orig_dims.items() if k in requested_target_dims}
        orig_extra_dim_info = {k: v for k, v in orig_extra_dim_info.items() if k in requested_target_dims}
        missing_target_dims = sorted(requested_target_dims - set(orig_dims.keys()))
        if missing_target_dims:
            print(
                f"  Standardization warning: {len(missing_target_dims)} requested dims were not found "
                f"in the original files and cannot be transferred: {missing_target_dims}",
                flush=True,
            )

    # ── Phase 4: Determine which dims to add / overwrite ────────────────
    if target_dims is not None:
        # Standardization mode: add all requested original dims that exist, and
        # force-overwrite any of those names already present in merged.
        dims_to_add_new: Dict[str, np.dtype] = {}
        dims_to_overwrite: Dict[str, np.dtype] = {}
        for name in sorted(requested_target_dims):
            if name not in orig_dims:
                continue
            dtype = orig_dims[name]
            if name in merged_dim_names:
                dims_to_overwrite[name] = dtype
            else:
                dims_to_add_new[name] = dtype
        dims_to_add = {**dims_to_add_new, **dims_to_overwrite}
        orig_dim_to_read = {k: k for k in dims_to_add}
        orig_rename_for_merged: Dict[str, str] = {}
    else:
        # Legacy mode: use pre-computed dim_stats
        def _get_dim_stat_proxy(dim_name):
            if dim_name not in dim_stats:
                return None
            dmin, dmax, nz = dim_stats[dim_name]
            if nz == 0:
                return np.array([0, 0])
            return np.array([dmin, dmax])

        dims_to_add_new, dims_to_overwrite = _dims_to_fill_from_source(
            orig_dims, merged_dim_names, _get_dim_stat_proxy, skip=skip_core,
        )
        used_names = set(merged_dim_names) | set(dims_to_add_new.keys()) | set(dims_to_overwrite.keys())
        collision = (set(orig_dims.keys()) & merged_dim_names) - set(dims_to_overwrite.keys()) - skip_core
        orig_rename_for_merged = {}
        for name in sorted(collision):
            cand = f"{name}_original"
            out_name = cand if cand not in used_names else _next_available_suffix(name, used_names)
            orig_rename_for_merged[name] = out_name
            used_names.add(out_name)
        dims_to_add_renamed = {orig_rename_for_merged[n]: orig_dims[n] for n in collision}
        dims_to_add = {**dims_to_add_new, **dims_to_overwrite, **dims_to_add_renamed}
        orig_dim_to_read = {k: k for k in dims_to_add_new} | {k: k for k in dims_to_overwrite}
        orig_dim_to_read.update({out_name: name for name, out_name in orig_rename_for_merged.items()})

    if not dims_to_add:
        print("  No dimensions to add or replace; copying merged file.", flush=True)
        import shutil
        shutil.copy2(str(merged_laz), str(output_path))
        return

    for dim_name in sorted(dims_to_add_new.keys()):
        print(f"  Adding dimension: {dim_name}", flush=True)
    for dim_name in sorted(dims_to_overwrite.keys()):
        print(f"  Replacing dimension: {dim_name}", flush=True)
    for orig_name, out_name in sorted(orig_rename_for_merged.items()):
        print(f"  Adding dimension (collision): {orig_name} -> {out_name}", flush=True)

    spatial_buffer = max(tolerance * 2, 1.0) + retile_buffer
    max_dist = distance_threshold if distance_threshold is not None else spatial_buffer
    query_workers = max(1, int(num_threads))

    promote_rgb_to_standard = (
        has_standard_rgb_dims(orig_dims.keys())
        and not point_format_has_standard_rgb(merged_header.point_format.id)
    )

    # ── Construct enriched output schema/header ─────────────────────────
    phase_start = _time.monotonic()
    print("  Constructing enriched output schema (header + extra dims)...", flush=True)
    n_collision_dims = len(orig_rename_for_merged)
    print(
        f"    Plan: carry over merged dims, add {len(dims_to_add_new)} new, "
        f"replace {len(dims_to_overwrite)}, rename {n_collision_dims} collisions",
        flush=True,
    )
    preview_names = (
        sorted(dims_to_add_new.keys())
        + sorted(dims_to_overwrite.keys())
        + sorted(orig_rename_for_merged.values())
    )
    if preview_names:
        preview = ", ".join(preview_names[:8])
        suffix = " ..." if len(preview_names) > 8 else ""
        print(f"    Dimension preview: {preview}{suffix}", flush=True)

    carried_extra_params = []
    if promote_rgb_to_standard:
        print(
            f"    Promoting {', '.join(_RGB_STANDARD_DIMS)} to standard LAS RGB fields",
            flush=True,
        )
    for dim in merged_extra_dims:
        if promote_rgb_to_standard and dim.name in _RGB_STANDARD_DIMS:
            continue
        carried_extra_params.append(
            extra_bytes_params_from_dimension_info(dim, name=dim.name)
        )

    extra_params = []
    params_start = _time.monotonic()
    for name, dtype in dims_to_add_new.items():
        if promote_rgb_to_standard and name in _RGB_STANDARD_DIMS:
            continue
        if name in orig_extra_dim_info:
            extra_params.append(
                extra_bytes_params_from_dimension_info(orig_extra_dim_info[name], name=name)
            )
        else:
            extra_params.append(laspy.ExtraBytesParams(name=name, type=dtype))
    # Collision dims (legacy mode only)
    dims_to_add_renamed = {orig_rename_for_merged[n]: orig_dims.get(n, np.float64)
                           for n in orig_rename_for_merged} if orig_rename_for_merged else {}
    orig_name_by_out = {out_name: name for name, out_name in orig_rename_for_merged.items()}
    for out_name, dtype in dims_to_add_renamed.items():
        if promote_rgb_to_standard and out_name in _RGB_STANDARD_DIMS:
            continue
        orig_name = orig_name_by_out.get(out_name)
        if orig_name is not None and orig_name in orig_extra_dim_info:
            extra_params.append(
                extra_bytes_params_from_dimension_info(orig_extra_dim_info[orig_name], name=out_name)
            )
        else:
            extra_params.append(laspy.ExtraBytesParams(name=out_name, type=dtype))
    print(
        f"    Prepared {len(carried_extra_params) + len(extra_params)} extra-dimension descriptors "
        f"({_time.monotonic() - params_start:.1f}s)",
        flush=True,
    )

    header_copy_start = _time.monotonic()
    base_point_format_id = (
        point_format_with_standard_rgb(merged_header.point_format.id)
        if promote_rgb_to_standard
        else merged_header.point_format.id
    )
    out_header = laspy.LasHeader(
        point_format=base_point_format_id,
        version=merged_header.version,
    )
    out_header.offsets = merged_header.offsets
    out_header.scales = merged_header.scales
    print(
        f"    Base header: point_format={base_point_format_id}, "
        f"version={merged_header.version}, merged extra dims={len(merged_extra_dims)}",
        flush=True,
    )
    print(
        f"    Copied merged header metadata "
        f"({_time.monotonic() - header_copy_start:.1f}s)",
        flush=True,
    )

    # Add carried merged extra dims, plus any newly introduced ones.
    all_extra_params = carried_extra_params + extra_params
    if all_extra_params:
        add_dims_start = _time.monotonic()
        out_header.add_extra_dims(all_extra_params)
        print(
            f"    Added {len(all_extra_params)} extra dims to output schema "
            f"({_time.monotonic() - add_dims_start:.1f}s)",
            flush=True,
        )

    out_dim_names = set(out_header.point_format.dimension_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"  Output schema ready: {len(out_dim_names)} dimensions total "
        f"({_time.monotonic() - phase_start:.1f}s)",
        flush=True,
    )
    next_step = (
        "    Next: read original file bounding boxes, then enrich the merged COPC slice-by-slice."
        if merged_laz.name.endswith(".copc.laz")
        else "    Next: read original file bounding boxes, then stream the merged LAZ and enrich it chunk-by-chunk."
    )
    print(next_step, flush=True)

    # ── Pre-compute original file bounding boxes ────────────────────────
    phase_start = _time.monotonic()
    print("  Reading original file bounding boxes...", flush=True)
    orig_bboxes: Dict[Path, Tuple[float, float, float, float]] = {}
    eligible_files: List[Path] = []
    for orig_path in original_files:
        try:
            with laspy.open(str(orig_path), laz_backend=laspy.LazBackend.Lazrs) as f:
                hdr = f.header
                orig_bboxes[orig_path] = (hdr.x_min, hdr.x_max, hdr.y_min, hdr.y_max)
                eligible_files.append(orig_path)
                print(f"    {orig_path.name}: x=[{hdr.x_min:.1f}, {hdr.x_max:.1f}] "
                      f"y=[{hdr.y_min:.1f}, {hdr.y_max:.1f}] ({hdr.point_count:,} pts)", flush=True)
        except Exception as e:
            print(f"    {orig_path.name}: SKIPPED ({e})", flush=True)
            continue

    skipped_bounds = len(original_files) - len(eligible_files)
    if skipped_bounds > 0:
        print(f"  Skipped {skipped_bounds} original files (unreadable)", flush=True)
    print(
        f"  Bounding box scan complete: {len(eligible_files)}/{len(original_files)} files usable "
        f"({_time.monotonic() - phase_start:.1f}s)",
        flush=True,
    )

    print(
        f"\n  Starting nearest-neighbor enrichment pass: {n_merged:,} merged points, "
        f"{len(eligible_files)} original files",
        flush=True,
    )
    print(f"  Spatial buffer: {spatial_buffer:.1f}m, max match distance: {max_dist:.1f}m", flush=True)
    print(f"  Dimensions to transfer: {len(dims_to_add)} "
          f"({len(dims_to_add_new)} new, {len(dims_to_overwrite)} replace)", flush=True)
    print(f"  Worker usage: up to {query_workers} thread(s) for subset prep; KDTree queries use all available threads", flush=True)
    merged_is_copc = merged_laz.name.endswith(".copc.laz")
    if merged_is_copc:
        merged_bounds = (
            float(merged_header.x_min),
            float(merged_header.x_max),
            float(merged_header.y_min),
            float(merged_header.y_max),
        )
        slice_specs = _build_spatial_slices(
            merged_bounds,
            spatial_slices,
            spatial_chunk_length,
            spatial_target_points,
            n_merged,
        )
        slice_specs = _snap_spatial_slices_to_header_grid(slice_specs, merged_header)
        print(
            f"  Spatial slicing: {len(slice_specs)} slices along {slice_specs[0][1].upper()}",
            flush=True,
        )
        print("  This stage reads spatial COPC windows, matches points to originals, and rewrites the output.", flush=True)
    else:
        n_chunks = (n_merged + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"  Chunk size: {CHUNK_SIZE:,} points ({n_chunks} chunks expected)", flush=True)
        print("  This stage streams the merged LAZ, matches points to originals, and rewrites the output.", flush=True)

    # ── Main streaming loop ─────────────────────────────────────────────
    t_start = _time.monotonic()
    cache = _OriginalFileCache(max_entries=0, tmp_dir=tmp_dir, chunk_size=CHUNK_SIZE)
    total_written = 0
    total_matched = 0
    unit_num = 0

    with laspy.open(
        str(output_path), mode="w", header=out_header,
        do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel,
    ) as writer:
        def _process_block(
            label: str,
            cx: np.ndarray,
            cy: np.ndarray,
            cz: np.ndarray,
            merged_dim_getter,
            subset_count: Optional[int] = None,
        ) -> None:
            nonlocal total_written, total_matched
            t_chunk = _time.monotonic()
            overlap_dt = 0.0
            subset_prep_dt = 0.0
            query_dt = 0.0
            write_dt = 0.0
            n = len(cx)
            if n == 0:
                return

            chunk_xmin, chunk_xmax = float(cx.min()), float(cx.max())
            chunk_ymin, chunk_ymax = float(cy.min()), float(cy.max())

            overlap_start = _time.monotonic()
            overlapping: List[Path] = []
            for orig_path in eligible_files:
                ox_min, ox_max, oy_min, oy_max = orig_bboxes[orig_path]
                if (ox_max + spatial_buffer < chunk_xmin
                        or ox_min - spatial_buffer > chunk_xmax
                        or oy_max + spatial_buffer < chunk_ymin
                        or oy_min - spatial_buffer > chunk_ymax):
                    continue
                overlapping.append(orig_path)
            overlap_dt = _time.monotonic() - overlap_start

            best_dist = np.full(n, np.inf, dtype=np.float64)
            best_dims: Dict[str, np.ndarray] = {}
            for name, dtype in dims_to_add.items():
                best_dims[name] = np.zeros(n, dtype=dtype)

            chunk_pts = np.column_stack([cx, cy, cz])

            def _load_subset(orig_path: Path):
                ox_min, ox_max, oy_min, oy_max = orig_bboxes[orig_path]
                mask = (
                    (cx >= ox_min - spatial_buffer) & (cx <= ox_max + spatial_buffer) &
                    (cy >= oy_min - spatial_buffer) & (cy <= oy_max + spatial_buffer)
                )
                if not mask.any():
                    return orig_path, None

                query_idx = np.where(mask)[0]
                query_bounds = (
                    float(cx[query_idx].min()),
                    float(cx[query_idx].max()),
                    float(cy[query_idx].min()),
                    float(cy[query_idx].max()),
                )
                load_start = _time.monotonic()
                tree, dim_arrays = cache.load(
                    orig_path,
                    orig_dim_to_read,
                    dims_to_add,
                    query_bounds,
                    max_dist,
                )
                load_dt = _time.monotonic() - load_start
                return orig_path, (query_idx, tree, dim_arrays, load_dt)

            loaded_subsets: Dict[Path, Tuple[np.ndarray, Optional[cKDTree], Dict[str, np.ndarray], float]] = {}
            if overlapping:
                if query_workers > 1 and len(overlapping) > 1:
                    with ThreadPoolExecutor(max_workers=min(query_workers, len(overlapping))) as executor:
                        future_to_path = {
                            executor.submit(_load_subset, orig_path): orig_path
                            for orig_path in overlapping
                        }
                        for future in as_completed(future_to_path):
                            orig_path = future_to_path[future]
                            try:
                                loaded = future.result()
                            except Exception as e:
                                print(f"    Warning: could not load {orig_path.name}: {e}", flush=True)
                                continue
                            if loaded[1] is not None:
                                loaded_subsets[loaded[0]] = loaded[1]
                else:
                    for orig_path in overlapping:
                        try:
                            loaded = _load_subset(orig_path)
                        except Exception as e:
                            print(f"    Warning: could not load {orig_path.name}: {e}", flush=True)
                            continue
                        if loaded[1] is not None:
                            loaded_subsets[loaded[0]] = loaded[1]

            for orig_path in overlapping:
                loaded = loaded_subsets.get(orig_path)
                if loaded is None:
                    continue

                query_idx, tree, dim_arrays, load_dt = loaded
                subset_prep_dt += load_dt
                if tree is None:
                    continue

                query_pts = chunk_pts[query_idx]

                query_start = _time.monotonic()
                distances, orig_idx = tree.query(query_pts, k=1, workers=-1)
                if distances.ndim == 2:
                    distances = distances[:, 0]
                    orig_idx = orig_idx[:, 0]

                accept = (distances <= max_dist) & (distances < best_dist[query_idx])
                if not accept.any():
                    del query_idx, query_pts, distances, orig_idx, accept
                    continue

                acc_chunk_idx = query_idx[accept]
                acc_orig_idx = orig_idx[accept]
                best_dist[acc_chunk_idx] = distances[accept]

                for name, dim_arr in dim_arrays.items():
                    if name in best_dims:
                        best_dims[name][acc_chunk_idx] = dim_arr[acc_orig_idx]
                query_dt += _time.monotonic() - query_start

                del query_idx, query_pts, distances, orig_idx, accept
                del acc_chunk_idx, acc_orig_idx
                del tree
                for v in dim_arrays.values():
                    del v
                dim_arrays.clear()

            write_start = _time.monotonic()
            point_record = laspy.ScaleAwarePointRecord.zeros(n, header=out_header)
            for dim_name in merged_dim_names:
                try:
                    arr = merged_dim_getter(dim_name)
                    if arr is not None:
                        setattr(point_record, dim_name, arr)
                except Exception:
                    pass
            for name, arr in best_dims.items():
                if name in out_dim_names:
                    try:
                        target_dtype = getattr(point_record, name).dtype
                        if arr.dtype != target_dtype:
                            arr = arr.astype(target_dtype)
                    except (AttributeError, KeyError, TypeError):
                        pass
                    setattr(point_record, name, arr)

            writer.write_points(point_record)
            write_dt = _time.monotonic() - write_start

            matched_in_chunk = int((best_dist < np.inf).sum())
            total_matched += matched_in_chunk
            total_written += n
            chunk_dt = _time.monotonic() - t_chunk
            elapsed = _time.monotonic() - t_start
            pct = total_written / n_merged * 100
            rate = total_written / elapsed if elapsed > 0 else 0
            eta = (n_merged - total_written) / rate if rate > 0 else 0

            overlap_names = ", ".join(p.name for p in overlapping) if overlapping else "none"
            if subset_count is not None:
                print(
                    f"  {label}: {n:,} core pts from {subset_count:,} in-bounds, "
                    f"{matched_in_chunk:,} matched ({len(overlapping)} originals: {overlap_names}), "
                    f"{chunk_dt:.1f}s",
                    flush=True,
                )
            else:
                print(
                    f"  {label}: {n:,} pts, {matched_in_chunk:,} matched "
                    f"({len(overlapping)} originals: {overlap_names}), {chunk_dt:.1f}s",
                    flush=True,
                )
            print(
                f"    Timings: overlap {overlap_dt:.1f}s, subset prep {subset_prep_dt:.1f}s, "
                f"query/match {query_dt:.1f}s, write {write_dt:.1f}s",
                flush=True,
            )
            print(
                f"    Progress: {total_written:,}/{n_merged:,} ({pct:.1f}%), "
                f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s, rate {rate:,.0f} pts/s",
                flush=True,
            )

            del best_dist, best_dims, chunk_pts, point_record

        if merged_is_copc:
            for slice_idx, (slice_bounds, axis, include_upper) in enumerate(slice_specs, start=1):
                unit_num += 1
                subset, subset_count = _load_copc_subset_all_dims(
                    merged_laz,
                    slice_bounds,
                    halo=0.0,
                    work_dir=tmp_dir,
                )
                if subset is None or subset_count == 0:
                    print(f"  Slice {slice_idx}/{len(slice_specs)}: empty merged subset", flush=True)
                    continue

                sx = np.asarray(subset.x, dtype=np.float64)
                sy = np.asarray(subset.y, dtype=np.float64)
                sz = np.asarray(subset.z, dtype=np.float64)
                if axis == "x":
                    upper_mask = sx <= slice_bounds[1] if include_upper else sx < slice_bounds[1]
                    core_mask = (
                        (sx >= slice_bounds[0]) & upper_mask
                        & (sy >= slice_bounds[2]) & (sy <= slice_bounds[3])
                    )
                else:
                    upper_mask = sy <= slice_bounds[3] if include_upper else sy < slice_bounds[3]
                    core_mask = (
                        (sx >= slice_bounds[0]) & (sx <= slice_bounds[1])
                        & (sy >= slice_bounds[2]) & upper_mask
                    )
                if not np.any(core_mask):
                    print(f"  Slice {slice_idx}/{len(slice_specs)}: no core points after crop", flush=True)
                    del subset, sx, sy, sz
                    continue

                def _slice_dim_getter(dim_name, _subset=subset, _mask=core_mask):
                    arr = getattr(_subset, dim_name, None)
                    return None if arr is None else np.asarray(arr)[_mask]

                _process_block(
                    f"Slice {slice_idx}/{len(slice_specs)}",
                    sx[core_mask],
                    sy[core_mask],
                    sz[core_mask],
                    _slice_dim_getter,
                    subset_count=subset_count,
                )
                del subset, sx, sy, sz, core_mask
        else:
            with laspy.open(str(merged_laz), laz_backend=laspy.LazBackend.Lazrs) as reader:
                for chunk in reader.chunk_iterator(CHUNK_SIZE):
                    unit_num += 1
                    cx = np.asarray(chunk.x, dtype=np.float64)
                    cy = np.asarray(chunk.y, dtype=np.float64)
                    cz = np.asarray(chunk.z, dtype=np.float64)

                    def _chunk_dim_getter(dim_name, _chunk=chunk):
                        return getattr(_chunk, dim_name, None)

                    _process_block(
                        f"Chunk {unit_num}/{n_chunks}",
                        cx,
                        cy,
                        cz,
                        _chunk_dim_getter,
                    )
                    del chunk, cx, cy, cz

    cache.clear()
    gc.collect()

    total_elapsed = _time.monotonic() - t_start
    match_pct = total_matched / total_written * 100 if total_written > 0 else 0
    print(f"\n  Enrichment complete:", flush=True)
    print(f"    {total_written:,} points written, {total_matched:,} matched ({match_pct:.1f}%)", flush=True)
    print(f"    Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)

    # Per-dimension summary
    print(f"  Per-dimension summary:", flush=True)
    with laspy.open(str(output_path), laz_backend=laspy.LazBackend.Lazrs) as check:
        sample = check.read_points(min(100_000, n_merged))
        for dim_name in sorted(dims_to_add.keys()):
            arr = getattr(sample, dim_name, None)
            if arr is not None:
                a = np.asarray(arr)
                nz = int(np.count_nonzero(a))
                print(f"    {dim_name}: min={float(np.min(a)):.4f}, max={float(np.max(a)):.4f}, "
                      f"non-zero={nz}/{len(a)} ({nz/len(a)*100:.1f}%)", flush=True)

    print(f"  Saved enriched merged: {output_path}", flush=True)




def _bounds_overlap_2d(
    bounds_a: Tuple[float, float, float, float],
    bounds_b: Tuple[float, float, float, float],
    buffer: float = 0.0,
) -> bool:
    """Return True when two XY bboxes overlap, optionally with extra buffer."""
    return not (
        bounds_a[1] < bounds_b[0] - buffer
        or bounds_a[0] > bounds_b[1] + buffer
        or bounds_a[3] < bounds_b[2] - buffer
        or bounds_a[2] > bounds_b[3] + buffer
    )


def _prepare_collection_remap_metadata(
    collections: List[Path],
    target_dims: Optional[Set[str]] = None,
) -> List[dict]:
    """
    Scan collection schemas once and cache file bounds for remap.

    When ``target_dims`` is omitted, only extra dimensions are exported from the
    source collections. When it is provided, it is treated as a strict allowlist
    and may include both extra and standard LAS dimensions.
    """
    seen_dim_names: Dict[str, int] = {}
    collection_meta: List[dict] = []

    for coll_path in collections:
        coll_files = [coll_path] if coll_path.is_file() else list_pointcloud_files(coll_path)
        dim_entries = []
        scanned_names: set = set()
        file_entries = []

        for cf in coll_files:
            try:
                with laspy.open(str(cf), laz_backend=laspy.LazBackend.LazrsParallel) as f:
                    hdr = f.header
                    extra_names = {dim.name for dim in hdr.point_format.extra_dimensions}
                    file_entries.append({
                        "path": cf,
                        "bounds": (hdr.x_min, hdr.x_max, hdr.y_min, hdr.y_max),
                        "point_count": int(hdr.point_count),
                        "is_copc": cf.name.endswith(".copc.laz"),
                    })
                    for dim in hdr.point_format.dimensions:
                        if dim.name in scanned_names:
                            continue
                        is_standard_dim = dim.name not in extra_names
                        if target_dims is None and is_standard_dim:
                            continue
                        if target_dims is not None and dim.name not in target_dims:
                            continue
                        scanned_names.add(dim.name)
                        out_name = dim.name
                        count = seen_dim_names.get(dim.name, 0)
                        if count > 0:
                            out_name = f"{dim.name}_{count + 1}"
                        seen_dim_names[dim.name] = count + 1
                        dim_entries.append((
                            dim.name,
                            out_name,
                            np.dtype(dim.dtype),
                            None if is_standard_dim else extra_bytes_params_from_dimension_info(dim),
                        ))
            except Exception:
                continue

        collection_meta.append({
            "path": coll_path,
            "dim_entries": dim_entries,
            "files": file_entries,
            "all_copc": bool(file_entries) and all(entry["is_copc"] for entry in file_entries),
        })

    return collection_meta


def _load_collection_subset_for_bounds(
    coll_meta: dict,
    query_bounds: Tuple[float, float, float, float],
    spatial_buffer: float,
    chunk_size: int,
    work_dir: Optional[Path] = None,
) -> Tuple[Optional[cKDTree], Dict[str, np.ndarray], int, int]:
    """
    Load only the subset of one collection needed for the current spatial window.

    Returns ``(tree, dim_arrays, indexed_points, candidate_file_count)``.
    """
    orig_names = [entry[0] for entry in coll_meta["dim_entries"]]
    if not orig_names:
        return None, {}, 0, 0

    candidate_files = [
        entry for entry in coll_meta["files"]
        if _bounds_overlap_2d(entry["bounds"], query_bounds, buffer=spatial_buffer)
    ]
    if not candidate_files:
        return None, {}, 0, 0

    pts_x: List[np.ndarray] = []
    pts_y: List[np.ndarray] = []
    pts_z: List[np.ndarray] = []
    dim_bufs: Dict[str, List[np.ndarray]] = {name: [] for name in orig_names}

    for file_entry in candidate_files:
        cf = file_entry["path"]
        if file_entry["is_copc"]:
            subset, subset_count = _load_copc_subset_all_dims(
                cf,
                query_bounds,
                halo=spatial_buffer,
                work_dir=work_dir,
            )
            if subset is None or subset_count == 0:
                continue
            sx = np.asarray(subset.x)
            if sx.size == 0:
                del subset
                continue
            pts_x.append(sx)
            pts_y.append(np.asarray(subset.y))
            pts_z.append(np.asarray(subset.z))
            for orig_name in orig_names:
                arr = getattr(subset, orig_name, None)
                if arr is not None:
                    dim_bufs[orig_name].append(np.asarray(arr))
                else:
                    dim_bufs[orig_name].append(np.zeros(len(sx), dtype=np.float32))
            del subset
            continue

        with laspy.open(str(cf), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            for chunk in f.chunk_iterator(chunk_size):
                cx = np.asarray(chunk.x)
                cy = np.asarray(chunk.y)
                mask = (
                    (cx >= query_bounds[0] - spatial_buffer)
                    & (cx <= query_bounds[1] + spatial_buffer)
                    & (cy >= query_bounds[2] - spatial_buffer)
                    & (cy <= query_bounds[3] + spatial_buffer)
                )
                if not mask.any():
                    continue
                pts_x.append(cx[mask])
                pts_y.append(cy[mask])
                pts_z.append(np.asarray(chunk.z)[mask])
                for orig_name in orig_names:
                    arr = getattr(chunk, orig_name, None)
                    if arr is None:
                        dim_bufs[orig_name].append(np.zeros(int(mask.sum()), dtype=np.float32))
                    else:
                        dim_bufs[orig_name].append(np.asarray(arr)[mask])

    if not pts_x:
        return None, {}, 0, len(candidate_files)

    xyz = np.column_stack([
        np.concatenate(pts_x),
        np.concatenate(pts_y),
        np.concatenate(pts_z),
    ])
    dim_arrs = {name: np.concatenate(dim_bufs[name]) for name in orig_names}
    tree = cKDTree(xyz)
    n_points = len(xyz)
    del xyz
    return tree, dim_arrs, n_points, len(candidate_files)


def remap_collections_to_original_files(
    collections: List[Path],
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    retile_buffer: float = 2.0,
    chunk_size: int = 1_000_000,
    target_dims: Optional[Set[str]] = None,
    spatial_slices: int = 50,
    spatial_chunk_length: Optional[float] = None,
    spatial_target_points: Optional[int] = None,
) -> None:
    """Remap N segmented collections onto original input files in a single pass per file.

    For each original file, spatial subsets from all collections are loaded, one KDTree
    is built per collection, and all extra dims from all collections are written to the
    output file in one streaming pass. No 3DT branding — dimension names are copied as-is.
    If the same dimension name exists in more than one collection, later occurrences are
    suffixed with ``_2``, ``_3``, etc. to avoid collisions.

    Args:
        collections:        List of paths, each a folder of LAZ/LAS files (or a single
                            merged LAZ file) representing one segmentation collection.
        original_input_dir: Directory with original pre-tiling LAZ files.
        output_dir:         Directory to write enriched original files.
        tolerance:          KDTree match distance threshold in metres.
        retile_buffer:      Extra spatial buffer around each original file's bounds.
        chunk_size:         Points per streaming chunk.
    """
    import time as _time

    print(f"\n{'=' * 60}", flush=True)
    print(f"Remapping {len(collections)} collection(s) to original input files", flush=True)
    print(f"{'=' * 60}", flush=True)

    original_files = list_pointcloud_files(original_input_dir)
    if not original_files:
        print(f"  No original files found in {original_input_dir}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    spatial_buffer = tolerance * 2 + retile_buffer
    collection_meta = _prepare_collection_remap_metadata(collections, target_dims=target_dims)

    for ci, coll_meta in enumerate(collection_meta):
        dim_entries = coll_meta["dim_entries"]
        if dim_entries:
            mapped = ", ".join(
                out_name if orig_name == out_name else f"{orig_name}->{out_name}"
                for orig_name, out_name, _, _ in dim_entries
            )
        else:
            mapped = "(no selected dimensions discovered)"
        source_mode = "COPC subsets" if coll_meta["all_copc"] else "stream scan fallback"
        print(
            f"  Collection {ci + 1} dims from {coll_meta['path']}: {mapped} [{source_mode}]",
            flush=True,
        )

    skipped = 0
    with tempfile.TemporaryDirectory(prefix="3dtrees_multicol_remap_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        for fi, orig_file in enumerate(original_files):
            output_name = orig_file.name.replace(".copc.laz", ".laz")
            output_path = output_dir / output_name
            if output_path.exists():
                skipped += 1
                continue

            t0 = _time.monotonic()
            print(f"  [{fi + 1}/{len(original_files)}] {orig_file.name}...", flush=True)

            try:
                with laspy.open(str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel) as f:
                    orig_header = f.header
                    orig_pf = orig_header.point_format
                    orig_bounds = (
                        orig_header.x_min, orig_header.x_max,
                        orig_header.y_min, orig_header.y_max,
                    )
                    n_orig = orig_header.point_count

                all_output_dim_names = set(orig_pf.dimension_names)
                new_extra_params = []
                for dim in orig_pf.extra_dimensions:
                    new_extra_params.append(extra_bytes_params_from_dimension_info(dim))
                    all_output_dim_names.add(dim.name)

                all_new_out_names = []
                skipped_conflicting_dims = []
                for coll_meta in collection_meta:
                    for _, out_name, dtype, ebp in coll_meta["dim_entries"]:
                        if out_name in all_output_dim_names:
                            if ebp is None and out_name in orig_pf.dimension_names:
                                continue
                            skipped_conflicting_dims.append(out_name)
                            continue
                        new_extra_params.append(laspy.ExtraBytesParams(
                            name=out_name,
                            type=dtype,
                            description=(ebp.description or "") if ebp is not None else "",
                        ))
                        all_new_out_names.append(out_name)
                        all_output_dim_names.add(out_name)

                promote_rgb = (
                    has_standard_rgb_dims(all_output_dim_names)
                    and not point_format_has_standard_rgb(orig_pf.id)
                )
                out_pf_id = point_format_with_standard_rgb(orig_pf.id) if promote_rgb else orig_pf.id

                out_header = laspy.LasHeader(
                    point_format=out_pf_id,
                    version=orig_header.version,
                )
                out_header.offsets = orig_header.offsets
                out_header.scales = orig_header.scales

                filtered_extra_params = []
                for ep in new_extra_params:
                    if promote_rgb and ep.name in _RGB_STANDARD_DIMS:
                        continue
                    filtered_extra_params.append(ep)
                out_header.add_extra_dims(filtered_extra_params)

                if promote_rgb:
                    print(
                        f"    Promoting RGB to standard LAS fields: "
                        f"point_format {orig_pf.id} -> {out_pf_id}",
                        flush=True,
                    )

                added_dims_str = ", ".join(all_new_out_names) if all_new_out_names else "(none)"
                skipped_dims_str = (
                    ", ".join(sorted(set(skipped_conflicting_dims)))
                    if skipped_conflicting_dims else "(none)"
                )

                n_out = 0
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with laspy.open(
                    str(output_path), mode="w", header=out_header,
                    do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel,
                ) as writer:
                    if orig_file.name.endswith(".copc.laz"):
                        slice_specs = _build_spatial_slices(
                            orig_bounds,
                            spatial_slices,
                            spatial_chunk_length,
                            spatial_target_points,
                            n_orig,
                        )
                        slice_specs = _snap_spatial_slices_to_header_grid(slice_specs, orig_header)
                        print(
                            f"    Spatial slicing: {len(slice_specs)} slices along {slice_specs[0][1].upper()}",
                            flush=True,
                        )

                        for slice_idx, (slice_bounds, axis, include_upper) in enumerate(slice_specs, start=1):
                            subset, subset_count = _load_copc_subset_all_dims(
                                orig_file,
                                slice_bounds,
                                halo=1.0,
                                work_dir=tmp_dir,
                            )
                            if subset is None or subset_count == 0:
                                print(f"      Slice {slice_idx}/{len(slice_specs)}: empty original subset", flush=True)
                                continue

                            sx = np.asarray(subset.x)
                            sy = np.asarray(subset.y)
                            sz = np.asarray(subset.z)
                            if axis == "x":
                                upper_mask = sx <= slice_bounds[1] if include_upper else sx < slice_bounds[1]
                                core_mask = (
                                    (sx >= slice_bounds[0]) & upper_mask
                                    & (sy >= slice_bounds[2]) & (sy <= slice_bounds[3])
                                )
                            else:
                                upper_mask = sy <= slice_bounds[3] if include_upper else sy < slice_bounds[3]
                                core_mask = (
                                    (sx >= slice_bounds[0]) & (sx <= slice_bounds[1])
                                    & (sy >= slice_bounds[2]) & upper_mask
                                )
                            if not np.any(core_mask):
                                print(f"      Slice {slice_idx}/{len(slice_specs)}: no core points after halo crop", flush=True)
                                del subset
                                continue

                            orig_xyz = np.column_stack([sx[core_mask], sy[core_mask], sz[core_mask]])
                            out_chunk = laspy.ScaleAwarePointRecord.zeros(len(orig_xyz), header=out_header)
                            for dim_name in orig_pf.dimension_names:
                                arr = getattr(subset, dim_name, None)
                                if arr is not None:
                                    setattr(out_chunk, dim_name, np.asarray(arr)[core_mask])

                            coll_summaries = []
                            for ci, coll_meta in enumerate(collection_meta, start=1):
                                tree, dim_arrs, n_indexed, candidate_count = _load_collection_subset_for_bounds(
                                    coll_meta=coll_meta,
                                    query_bounds=slice_bounds,
                                    spatial_buffer=spatial_buffer,
                                    chunk_size=chunk_size,
                                    work_dir=tmp_dir,
                                )
                                coll_summaries.append(f"c{ci}:{n_indexed:,}/{candidate_count}")
                                if tree is None:
                                    continue
                                _, idxs = tree.query(orig_xyz, workers=-1)
                                for orig_name, out_name, dtype, _ in coll_meta["dim_entries"]:
                                    if orig_name in dim_arrs:
                                        setattr(out_chunk, out_name, dim_arrs[orig_name][idxs].astype(dtype))
                                del tree, dim_arrs

                            writer.write_points(out_chunk)
                            n_out += len(orig_xyz)
                            pct = n_out / n_orig * 100 if n_orig else 100
                            elapsed = _time.monotonic() - t0
                            rate = n_out / elapsed if elapsed > 0 else 0
                            print(
                                f"      Slice {slice_idx}/{len(slice_specs)}: {len(orig_xyz):,} core pts from {subset_count:,} in-bounds "
                                f"({pct:.0f}%, {rate:,.0f} pts/s) [{' '.join(coll_summaries)}]",
                                flush=True,
                            )
                            del out_chunk, orig_xyz, subset, sx, sy, sz
                            gc.collect()
                    else:
                        n_chunks_expected = (n_orig + chunk_size - 1) // chunk_size
                        print(
                            f"    Non-COPC original fallback: streaming {n_orig:,} pts in ~{n_chunks_expected} chunks",
                            flush=True,
                        )
                        with laspy.open(str(orig_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                            chunk_i = 0
                            for chunk in reader.chunk_iterator(chunk_size):
                                xyz_q = np.column_stack([
                                    np.asarray(chunk.x),
                                    np.asarray(chunk.y),
                                    np.asarray(chunk.z),
                                ])
                                chunk_bounds = (
                                    float(np.min(xyz_q[:, 0])),
                                    float(np.max(xyz_q[:, 0])),
                                    float(np.min(xyz_q[:, 1])),
                                    float(np.max(xyz_q[:, 1])),
                                )
                                out_chunk = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=out_header)
                                for dim_name in orig_pf.dimension_names:
                                    try:
                                        setattr(out_chunk, dim_name, np.asarray(getattr(chunk, dim_name)))
                                    except Exception:
                                        pass

                                coll_summaries = []
                                for ci, coll_meta in enumerate(collection_meta, start=1):
                                    tree, dim_arrs, n_indexed, candidate_count = _load_collection_subset_for_bounds(
                                        coll_meta=coll_meta,
                                        query_bounds=chunk_bounds,
                                        spatial_buffer=spatial_buffer,
                                        chunk_size=chunk_size,
                                        work_dir=tmp_dir,
                                    )
                                    coll_summaries.append(f"c{ci}:{n_indexed:,}/{candidate_count}")
                                    if tree is None:
                                        continue
                                    _, idxs = tree.query(xyz_q, workers=-1)
                                    for orig_name, out_name, dtype, _ in coll_meta["dim_entries"]:
                                        if orig_name in dim_arrs:
                                            setattr(out_chunk, out_name, dim_arrs[orig_name][idxs].astype(dtype))
                                    del tree, dim_arrs

                                writer.write_points(out_chunk)
                                n_out += len(chunk)
                                chunk_i += 1
                                pct = n_out / n_orig * 100 if n_orig else 100
                                elapsed = _time.monotonic() - t0
                                rate = n_out / elapsed if elapsed > 0 else 0
                                print(
                                    f"      Chunk {chunk_i}/{n_chunks_expected}: {len(chunk):,} pts "
                                    f"({pct:.0f}%, {rate:,.0f} pts/s) [{' '.join(coll_summaries)}]",
                                    flush=True,
                                )
                                del out_chunk, xyz_q

                print(
                    f"    {n_orig:,} pts → {n_out:,} pts written "
                    f"({_time.monotonic() - t0:.1f}s)",
                    flush=True,
                )
                print(f"    Added dims: {added_dims_str}", flush=True)
                if skipped_conflicting_dims:
                    print(
                        f"    Skipped conflicting dims already present in original schema: {skipped_dims_str}",
                        flush=True,
                    )

            except Exception as e:
                print(f"    Error processing {orig_file.name}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    if skipped:
        print(f"  Skipped {skipped} already-processed files.", flush=True)
    print(f"\n  Done: {len(original_files)} files processed to {output_dir}", flush=True)


def add_original_dimensions_to_merged(
    merged_laz: Path,
    original_input_dir: Path,
    output_path: Path,
    tolerance: float = 0.1,
    retile_buffer: float = 2.0,
    distance_threshold: Optional[float] = None,
    num_threads: int = 4,
    target_dims: Optional[Set[str]] = None,
    merge_chunk_size: int = 2_000_000,
    spatial_slices: int = 10,
    spatial_chunk_length: Optional[float] = None,
    spatial_target_points: Optional[int] = None,
) -> None:
    """
    Add dimensions from original input files to the merged point cloud by
    nearest-neighbor matching, and write an enriched merged LAZ file.

    Args:
        merged_laz: Path to the merged LAZ file.
        original_input_dir: Directory containing original LAZ/LAS files (pre-tiling).
        output_path: Path for the output merged LAZ with original dimensions added.
        tolerance: Distance tolerance for spatial buffer calculation.
        retile_buffer: Additional spatial buffer in meters.
        distance_threshold: Max distance for accepting a match (default: 2 * tolerance + retile_buffer).
        num_threads: Number of original files to process in parallel (default 4). Use 1 for sequential.
        target_dims: If provided (e.g. from standardization JSON), only these
            dimensions are transferred from originals, and they **overwrite**
            existing merged values.  Dimensions not in this set (e.g.
            PredInstance) are kept from merged.  When None, the legacy
            behaviour applies (add missing, overwrite empty/constant, rename
            collisions to _original).
    """
    if output_path.resolve() == Path(merged_laz).resolve():
        raise ValueError("output_path must differ from merged_laz to avoid overwriting input")

    original_files = list_pointcloud_files(original_input_dir)
    if not original_files:
        print("  No original input files found; skipping merged-with-originals output.", flush=True)
        return

    print(f"\n{'=' * 60}", flush=True)
    print("Adding original-file dimensions to merged point cloud", flush=True)
    print(f"{'=' * 60}", flush=True)

    # v3 streaming enrichment: no memmaps, no temp files needed.
    # tmp_dir kept for API compatibility but unused internally.
    tmp_dir = output_path.parent / f"_enrich_tmp_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _add_original_dimensions_to_merged_impl(
            merged_laz, original_input_dir, output_path, original_files,
            tolerance, retile_buffer, distance_threshold, target_dims, None, tmp_dir,
            num_threads=num_threads,
            merge_chunk_size=merge_chunk_size,
            spatial_slices=spatial_slices,
            spatial_chunk_length=spatial_chunk_length,
            spatial_target_points=spatial_target_points,
        )
    finally:
        import shutil as _shutil
        if tmp_dir.exists():
            _shutil.rmtree(tmp_dir, ignore_errors=True)


def enrich_collection_tiles_with_original_dimensions(
    tile_collection: Path,
    original_input_dir: Path,
    output_dir: Path,
    tolerance: float = 0.1,
    retile_buffer: float = 2.0,
    distance_threshold: Optional[float] = None,
    num_threads: int = 4,
    target_dims: Optional[Set[str]] = None,
    source_extra_dims_to_keep: Optional[Set[str]] = None,
    merge_chunk_size: int = 2_000_000,
    spatial_slices: int = 10,
    spatial_chunk_length: Optional[float] = None,
    spatial_target_points: Optional[int] = None,
) -> None:
    """
    Enrich a tile collection with original-file dimensions while keeping the
    tile geometry unchanged.

    Each input tile is processed independently with the same spatially chunked
    enrichment logic used for merged-file enrichment. This keeps the duplicate-
    free tile layout intact so the enriched outputs can be concatenated directly
    without global overlap deduplication.
    """
    import time as _time

    tile_collection = Path(tile_collection)
    original_input_dir = Path(original_input_dir)
    output_dir = Path(output_dir)

    tile_files = [tile_collection] if tile_collection.is_file() else list_pointcloud_files(tile_collection)
    if not tile_files:
        print(f"  No tile files found in {tile_collection}; skipping tile enrichment.", flush=True)
        return

    original_files = list_pointcloud_files(original_input_dir)
    if not original_files:
        print("  No original input files found; skipping tile enrichment.", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print("Enriching tiled remap source with original-file dimensions", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Tile source:  {tile_collection}", flush=True)
    print(f"  Originals:    {original_input_dir}", flush=True)
    print(f"  Output tiles: {output_dir}", flush=True)
    print(f"  Tile count:   {len(tile_files)}", flush=True)

    def _fmt_elapsed(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{seconds / 60:.1f} min"

    skipped = 0
    total_start = _time.monotonic()
    for idx, tile_path in enumerate(tile_files, start=1):
        output_name = tile_path.name.replace(".copc.laz", ".laz")
        output_path = output_dir / output_name
        if output_path.resolve() == tile_path.resolve():
            raise ValueError(
                f"Tile enrichment output would overwrite input tile: {tile_path}"
            )
        if output_path.exists():
            skipped += 1
            print(
                f"  [{idx}/{len(tile_files)}] {tile_path.name}: output already exists, skipping",
                flush=True,
            )
            continue

        tile_start = _time.monotonic()
        print(
            f"\n  [{idx}/{len(tile_files)}] Enriching tile {tile_path.name} "
            f"→ {output_name}",
            flush=True,
        )
        tmp_dir = output_dir / f"_enrich_tmp_{os.getpid()}_{idx:04d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            _add_original_dimensions_to_merged_impl(
                merged_laz=tile_path,
                original_input_dir=original_input_dir,
                output_path=output_path,
                original_files=original_files,
                tolerance=tolerance,
                retile_buffer=retile_buffer,
                distance_threshold=distance_threshold,
                target_dims=target_dims,
                source_extra_dims_to_keep=source_extra_dims_to_keep,
                tmp_dir=tmp_dir,
                num_threads=num_threads,
                merge_chunk_size=merge_chunk_size,
                spatial_slices=spatial_slices,
                spatial_chunk_length=spatial_chunk_length,
                spatial_target_points=spatial_target_points,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        print(
            f"  [{idx}/{len(tile_files)}] Tile enrichment finished in "
            f"{_fmt_elapsed(_time.monotonic() - tile_start)}",
            flush=True,
        )

    if skipped:
        print(f"  Skipped {skipped} already-enriched tile(s).", flush=True)
    total_elapsed = _time.monotonic() - total_start
    print(
        f"  Tile enrichment complete in {_fmt_elapsed(total_elapsed)}",
        flush=True,
    )


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
    standardization_json: Optional[Path] = None,
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
        merged_for_downstream = output_merged

        retile_to_original_files_streaming(
            merged_file=merged_for_downstream,
            original_tiles_dir=original_tiles_dir,
            output_dir=output_tiles_dir,
            tolerance=0.1,
            retile_buffer=retile_buffer,
            chunk_size=chunk_size,
            instance_dimension=instance_dimension,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
        )
        print(f"  ✓ Stage 6 completed: Retiled to original files")

        if original_input_dir is not None:
            print(f"\n{'=' * 60}")
            print("Stage 7: Remapping to Original Input Files")
            print(f"{'=' * 60}")
            
            original_output_dir = output_tiles_dir.parent / "original_with_predictions"
            remap_to_original_input_files_streaming(
                merged_file=merged_for_downstream,
                original_input_dir=original_input_dir,
                output_dir=original_output_dir,
                tolerance=retile_max_radius,
                retile_buffer=retile_buffer,
                chunk_size=chunk_size,
                threedtrees_dims=threedtrees_dims,
                threedtrees_suffix=threedtrees_suffix,
                target_dims=_target_dims,
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
        name = normalize_tile_id(f.stem)
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
    with tile_bounds_json.open() as f:
        json_data = json.load(f)
    json_labels = [
        f"c{int(tile['col']):02d}_r{int(tile['row']):02d}"
        if "col" in tile and "row" in tile else None
        for tile in json_data.get("tiles", [])
    ]
    json_bounds, centers, neighbors_idx = build_neighbor_graph_from_bounds_json(
        tile_bounds_json,
        bounds_field="planned_bounds",
    )
    print(f"  JSON tiles in bounds file: {len(json_bounds)}")

    tile_to_json, json_to_tile = _match_tiles_to_json_bounds(
        tile_boundaries,
        json_bounds,
        centers,
        json_labels=json_labels,
    )
    print("  Matched tiles to JSON bounds successfully")

    planned_tile_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    for tile_name, json_idx in tile_to_json.items():
        planned_tile_boundaries[tile_name] = json_bounds[json_idx]

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
    load_args = [(f, planned_tile_boundaries, buffer, neighbors_by_tile, instance_dimension) for f in laz_files]

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
    merged_file_for_downstream = output_merged

    if skip_merged_file:
        print(f"\n{'=' * 60}")
        print("Saving merged output (SKIPPED)")
        print(f"{'=' * 60}")
        print(f"  Skipped merged LAZ file creation (--skip_merged_file)")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(sorted_by_location)}")
        merged_file_for_downstream = output_merged.parent / (output_merged.stem + "_temp_for_downstream.laz")
        _write_points_with_dimensions_to_laz(
            merged_file_for_downstream,
            merged_points,
            {instance_dimension: merged_instances, **merged_extra_dims},
        )
    else:
        print(f"\n{'=' * 60}")
        print("Saving merged output")
        print(f"{'=' * 60}")

        output_merged.parent.mkdir(parents=True, exist_ok=True)

        # Write initial merged to a temp path so we can either enrich it or use as final
        merged_init = output_merged.parent / (output_merged.stem + "_init.laz")

        _write_points_with_dimensions_to_laz(
            merged_init,
            merged_points,
            {instance_dimension: merged_instances, **merged_extra_dims},
        )
        merged_file_for_downstream = merged_init
        gc.collect()

        print(f"  Saved merged (initial): {merged_init}")
        print(f"  Total points: {len(merged_points):,}")
        print(f"  Total instances: {len(sorted_by_location)}")

        import shutil
        # Add original-file dimensions and write directly to final output (so merged file is always enriched when requested)
        if original_input_dir is not None and transfer_original_dims_to_merged:
            enriched_output = output_merged
            try:
                add_original_dimensions_to_merged(
                    merged_file_for_downstream,
                    original_input_dir,
                    enriched_output,
                    tolerance=0.1,
                    retile_buffer=retile_buffer,
                    num_threads=num_threads,
                )
                print(f"  Enriched merged file with original-file dimensions: {output_merged}")
            except Exception as e:
                print(f"  Warning: Could not add original dimensions to merged file: {e}")
                try:
                    if enriched_output.exists():
                        enriched_output.unlink()
                except OSError:
                    pass
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
    
    retile_to_original_files_streaming(
        merged_file=merged_file_for_downstream,
        original_tiles_dir=original_tiles_dir,
        output_dir=output_tiles_dir,
        tolerance=0.1,
        retile_buffer=retile_buffer,
        chunk_size=chunk_size,
        instance_dimension=instance_dimension,
        threedtrees_dims=threedtrees_dims,
        threedtrees_suffix=threedtrees_suffix,
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
        remap_to_original_input_files_streaming(
            merged_file=merged_file_for_downstream,
            original_input_dir=original_input_dir,
            output_dir=original_output_dir,
            tolerance=0.1,
            retile_buffer=retile_buffer,
            chunk_size=chunk_size,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            target_dims=_target_dims,
        )
        print(f"  ✓ Stage 7 completed: Remapped to original input files")
    else:
        print(f"\n  Note: --original-input-dir not provided, skipping Stage 7 (remap to original input files)")

    if skip_merged_file and merged_file_for_downstream.exists():
        try:
            merged_file_for_downstream.unlink()
        except OSError:
            pass

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

    parser.add_argument(
        "--standardization-json",
        type=Path,
        default=None,
        help="Optional collection_summary.json; when provided, remap/original-dimension transfer only populates listed dimensions.",
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
        standardization_json=args.standardization_json,
    )


if __name__ == "__main__":
    main()
    _target_dims = None
    if standardization_json is not None:
        _target_dims = load_standardization_dims(standardization_json)
