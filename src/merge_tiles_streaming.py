#!/usr/bin/env python3
"""
Memory-Efficient Streaming Merge Pipeline for 3DTrees.

Replaces the in-memory merge stages (1-5) with a three-phase approach:
  Phase A: Extract lightweight metadata from each tile (one at a time)
  Phase B: Cross-tile matching + orphan recovery (metadata only)
  Phase C: Stream-write merged file tile-by-tile

Memory is bounded: only 1 tile loaded at a time, plus lightweight metadata.
"""

from __future__ import annotations

import gc
import json
import sys
import shutil
import numpy as np
import laspy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Reuse existing utilities from merge_tiles
from merge_tiles import (
    UnionFind,
    compute_tile_bounds,
    get_tile_bounds_from_header,
    build_neighbor_graph_from_bounds_json,
    has_standard_rgb_dims,
    normalize_tile_id,
    _match_tiles_to_json_bounds,
)

# Hash primes (same as used in compute_ff3d_overlap_ratios)
_HASH_PRIME_Y = np.int64(73856093)
_HASH_PRIME_Z = np.int64(19349669)
TILE_OFFSET = 100_000


# =============================================================================
# Data Structures
# =============================================================================


def _canonical_tile_name_for_json_index(
    json_idx: int,
    json_labels: List[Optional[str]],
) -> str:
    """Return a stable tile name sourced from tile_bounds_tindex.json."""
    if json_idx < len(json_labels):
        label = json_labels[json_idx]
        if label:
            return label
    return f"tile_{json_idx:04d}"


def _preferred_json_bounds_field(json_tiles: List[dict]) -> str:
    """Use actual_bounds for matching when available, otherwise fall back."""
    if any("actual_bounds" in tile for tile in json_tiles):
        return "actual_bounds"
    return "planned_bounds"


@dataclass
class InstanceMeta:
    """Lightweight per-instance metadata (no point arrays)."""

    global_id: int
    tile_name: str
    local_id: int
    centroid: np.ndarray  # (3,)
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    point_count: int
    owned_point_count: int = 0
    is_filtered: bool = False
    is_border: bool = False
    border_directions: List[str] = None  # e.g. ["east", "north"] for corner instances
    hash_set: Optional[Set[int]] = None  # spatial hashes for border instances only
    highpt_xy: Optional[np.ndarray] = None  # (2,) XY of the point with maximum Z
    lowpt_xy: Optional[np.ndarray] = None   # (2,) XY of the point with minimum Z


@dataclass
class TileMetadataResult:
    """Result from extracting metadata from a single tile."""

    tile_idx: int  # original index from laz_files enumeration
    tile_name: str
    filepath: Path
    boundary: Tuple[float, float, float, float]
    instances: Dict[int, InstanceMeta]  # global_id -> metadata
    filtered_local_ids: Set[int]
    kept_local_ids: Set[int]
    extra_dim_names: List[str]
    extra_dim_dtypes: Dict[str, np.dtype]


# =============================================================================
# Phase A: Metadata Extraction
# =============================================================================


def _compute_spatial_hashes(
    points: np.ndarray, scale: float = 10.0
) -> Set[int]:
    """Compute spatial hash set for a set of points (same grid as FF3D)."""
    grid = np.floor(points * scale).astype(np.int64)
    hashes = grid[:, 0] + grid[:, 1] * _HASH_PRIME_Y + grid[:, 2] * _HASH_PRIME_Z
    return set(hashes.tolist())


def _in_buffer_region(
    xy: Tuple[float, float],
    neighbors: Dict[str, Optional[str]],
    core_boundary: Tuple[float, float, float, float],
) -> bool:
    """Return True if xy is outside the core on a side that has a neighboring tile.

    Buffer only applies on sides where a neighbor exists — outer edges of the
    dataset are never considered buffer zone.
    """
    x, y = xy
    mn_x, mx_x, mn_y, mx_y = core_boundary
    return (
        (neighbors.get("west")  is not None and x < mn_x) or
        (neighbors.get("east")  is not None and x > mx_x) or
        (neighbors.get("south") is not None and y < mn_y) or
        (neighbors.get("north") is not None and y > mx_y)
    )


def _owned_region_mask(
    points_xy: np.ndarray,
    neighbors: Dict[str, Optional[str]],
    core_boundary: Optional[Tuple[float, float, float, float]],
) -> np.ndarray:
    """
    Return a mask for points owned by this tile.

    On sides with neighbors, ownership is clipped to the core. On outer edges,
    the full tile extent is owned. This matches the later overlap removal in
    Phase C and avoids discarding useful instances only because their centroid
    sits just outside the core.
    """
    owned = np.ones(len(points_xy), dtype=bool)
    if core_boundary is None:
        return owned

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    core_min_x, core_max_x, core_min_y, core_max_y = core_boundary

    if neighbors.get("west") is not None:
        owned &= x >= core_min_x
    if neighbors.get("east") is not None:
        owned &= x <= core_max_x
    if neighbors.get("south") is not None:
        owned &= y >= core_min_y
    if neighbors.get("north") is not None:
        owned &= y <= core_max_y

    return owned


def extract_tile_metadata(
    filepath: Path,
    tile_idx: int,
    tile_boundaries: Dict[str, Tuple[float, float, float, float]],
    tile_name: Optional[str],
    neighbors_for_tile: Optional[Dict[str, Optional[str]]],
    border_zone_width: float = 10.0,
    correspondence_tolerance: float = 0.05,
    instance_dimension: str = "PredInstance",
    chunk_size: int = 1_000_000,
    core_boundary: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[TileMetadataResult]:
    """
    Stream one tile in chunks and extract per-instance metadata without
    holding the full point array in memory.

    Per-instance centroid, bbox, point count, owned count and spatial
    hashes are accumulated incrementally.  Peak RAM ≈ one chunk +
    O(num_instances × avg_hash_set) instead of O(n_points).
    """
    if tile_name is None:
        tile_name = normalize_tile_id(filepath.stem)

    print(f"  [{tile_idx + 1}] Extracting metadata from {filepath.name}...")

    neighbors = neighbors_for_tile or {
        "east": None, "west": None, "north": None, "south": None,
    }

    scale = 1.0 / correspondence_tolerance

    # Per-instance accumulators (O(num_instances), not O(n_points))
    inst_sum: Dict[int, np.ndarray] = {}      # local_id -> [sum_x, sum_y, sum_z]
    inst_min: Dict[int, np.ndarray] = {}      # local_id -> [min_x, min_y, min_z]
    inst_max: Dict[int, np.ndarray] = {}      # local_id -> [max_x, max_y, max_z]
    inst_count: Dict[int, int] = {}           # local_id -> total point count
    inst_owned: Dict[int, int] = {}           # local_id -> owned point count
    inst_hashes: Dict[int, set] = {}          # local_id -> spatial hash set
    inst_highpt_xy: Dict[int, np.ndarray] = {}  # local_id -> [x, y] at max z
    inst_highpt_z: Dict[int, float] = {}        # local_id -> max z seen so far
    inst_lowpt_xy: Dict[int, np.ndarray] = {}   # local_id -> [x, y] at min z
    inst_lowpt_z: Dict[int, float] = {}         # local_id -> min z seen so far

    # For tile boundary fallback (running min/max of all points)
    tile_min = np.full(3, np.inf)
    tile_max = np.full(3, -np.inf)

    try:
        with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as f:
            n_points = f.header.point_count
            header_extra_dims = {
                dim.name: dim for dim in f.header.point_format.extra_dimensions
            }
            has_instance_dim = instance_dimension in header_extra_dims
            has_tree_id = "treeID" in header_extra_dims
            if not has_instance_dim:
                available = list(header_extra_dims.keys())
                fallback = "treeID" if has_tree_id else "(none — all IDs will be 0)"
                print(
                    f"    WARNING: dimension '{instance_dimension}' not found in {filepath.name}. "
                    f"Available extra dims: {available}. Falling back to: {fallback}",
                    flush=True,
                )

            extra_dim_names = []
            extra_dim_dtypes = {}
            for dim in f.header.point_format.extra_dimensions:
                if dim.name == instance_dimension or (
                    not has_instance_dim and dim.name == "treeID"
                ):
                    continue
                extra_dim_names.append(dim.name)
                extra_dim_dtypes[dim.name] = dim.dtype

            for chunk in f.chunk_iterator(chunk_size):
                cx = np.asarray(chunk.x)
                cy = np.asarray(chunk.y)
                cz = np.asarray(chunk.z)

                if has_instance_dim:
                    ci = np.asarray(getattr(chunk, instance_dimension), dtype=np.int32)
                elif has_tree_id:
                    ci = np.asarray(chunk.treeID, dtype=np.int32)
                else:
                    ci = np.zeros(len(chunk), dtype=np.int32)

                # Update tile-level bounds for fallback
                tile_min[0] = min(tile_min[0], float(cx.min()))
                tile_min[1] = min(tile_min[1], float(cy.min()))
                tile_min[2] = min(tile_min[2], float(cz.min()))
                tile_max[0] = max(tile_max[0], float(cx.max()))
                tile_max[1] = max(tile_max[1], float(cy.max()))
                tile_max[2] = max(tile_max[2], float(cz.max()))

                # Sort within chunk for efficient per-instance processing
                sort_idx = np.argsort(ci)
                ci_s = ci[sort_idx]
                cx_s = cx[sort_idx]
                cy_s = cy[sort_idx]
                cz_s = cz[sort_idx]
                del sort_idx

                unique_ids, first_idx, counts = np.unique(
                    ci_s, return_index=True, return_counts=True,
                )

                for uid_val, start, cnt in zip(unique_ids, first_idx, counts):
                    uid = int(uid_val)
                    if uid <= 0:
                        continue
                    end = start + int(cnt)
                    px = cx_s[start:end]
                    py = cy_s[start:end]
                    pz = cz_s[start:end]

                    pts = np.column_stack([px, py, pz])

                    if uid in inst_sum:
                        inst_sum[uid] += pts.sum(axis=0)
                        np.minimum(inst_min[uid], pts.min(axis=0), out=inst_min[uid])
                        np.maximum(inst_max[uid], pts.max(axis=0), out=inst_max[uid])
                        inst_count[uid] += len(pts)
                    else:
                        inst_sum[uid] = pts.sum(axis=0).copy()
                        inst_min[uid] = pts.min(axis=0).copy()
                        inst_max[uid] = pts.max(axis=0).copy()
                        inst_count[uid] = len(pts)
                        inst_owned[uid] = 0
                        inst_hashes[uid] = set()

                    # Track XY position of the highest point (max Z)
                    chunk_max_z_idx = int(pz.argmax())
                    chunk_max_z = float(pz[chunk_max_z_idx])
                    if uid not in inst_highpt_z or chunk_max_z > inst_highpt_z[uid]:
                        inst_highpt_z[uid] = chunk_max_z
                        inst_highpt_xy[uid] = np.array(
                            [float(px[chunk_max_z_idx]), float(py[chunk_max_z_idx])]
                        )

                    # Track XY position of the lowest point (min Z)
                    chunk_min_z_idx = int(pz.argmin())
                    chunk_min_z = float(pz[chunk_min_z_idx])
                    if uid not in inst_lowpt_z or chunk_min_z < inst_lowpt_z[uid]:
                        inst_lowpt_z[uid] = chunk_min_z
                        inst_lowpt_xy[uid] = np.array(
                            [float(px[chunk_min_z_idx]), float(py[chunk_min_z_idx])]
                        )

                    # Owned count
                    owned_mask = _owned_region_mask(pts[:, :2], neighbors, core_boundary)
                    inst_owned[uid] += int(owned_mask.sum())

                    # Spatial hashes (collect for all; prune non-border later)
                    inst_hashes[uid] |= _compute_spatial_hashes(pts, scale=scale)

                del ci_s, cx_s, cy_s, cz_s, cx, cy, cz, ci

    except Exception as e:
        print(f"    Error loading {filepath}: {e}")
        return None

    # Tile boundary fallback
    if tile_name in tile_boundaries:
        boundary = tile_boundaries[tile_name]
    else:
        boundary = (float(tile_min[0]), float(tile_max[0]),
                     float(tile_min[1]), float(tile_max[1]))

    # Build InstanceMeta from accumulated stats
    instance_metas: Dict[int, InstanceMeta] = {}
    filtered_ids: set = set()
    kept_ids: set = set()

    for local_id, count in inst_count.items():
        centroid = inst_sum[local_id] / count
        bbox_min = inst_min[local_id]
        bbox_max = inst_max[local_id]

        global_id = tile_idx * TILE_OFFSET + local_id
        owned_point_count = inst_owned[local_id]
        is_filtered = owned_point_count == 0
        if is_filtered:
            filtered_ids.add(local_id)
        else:
            kept_ids.add(local_id)

        # Border detection
        cx_val, cy_val = float(centroid[0]), float(centroid[1])
        border_dirs = []

        bx_min, by_min = float(bbox_min[0]), float(bbox_min[1])
        bx_max, by_max = float(bbox_max[0]), float(bbox_max[1])

        if core_boundary is not None:
            core_min_x, core_max_x, core_min_y, core_max_y = core_boundary
            if neighbors.get("west") is not None and (
                cx_val < core_min_x + border_zone_width or bx_min < core_min_x
            ):
                border_dirs.append("west")
            if neighbors.get("east") is not None and (
                cx_val > core_max_x - border_zone_width or bx_max > core_max_x
            ):
                border_dirs.append("east")
            if neighbors.get("south") is not None and (
                cy_val < core_min_y + border_zone_width or by_min < core_min_y
            ):
                border_dirs.append("south")
            if neighbors.get("north") is not None and (
                cy_val > core_max_y - border_zone_width or by_max > core_max_y
            ):
                border_dirs.append("north")

        is_border = len(border_dirs) > 0

        # Keep hash set only for border / helper-only instances
        hash_set = None
        needs_hash = is_border
        if not needs_hash and is_filtered and core_boundary is not None:
            if not (core_min_x <= cx_val <= core_max_x and
                    core_min_y <= cy_val <= core_max_y):
                needs_hash = True
                is_border = True
        if needs_hash:
            hash_set = inst_hashes.get(local_id)
        # else: discard collected hashes (non-border)

        instance_metas[global_id] = InstanceMeta(
            global_id=global_id,
            tile_name=tile_name,
            local_id=local_id,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            point_count=count,
            owned_point_count=owned_point_count,
            is_filtered=is_filtered,
            is_border=is_border,
            border_directions=border_dirs if border_dirs else None,
            hash_set=hash_set,
            highpt_xy=inst_highpt_xy.get(local_id),
            lowpt_xy=inst_lowpt_xy.get(local_id),
        )

    # Release accumulators
    del inst_sum, inst_min, inst_max, inst_count, inst_owned, inst_hashes, inst_highpt_xy, inst_highpt_z, inst_lowpt_xy, inst_lowpt_z
    gc.collect()

    print(
        f"    {n_points:,} points, {len(kept_ids)} kept, "
        f"{len(filtered_ids)} filtered, "
        f"{sum(1 for m in instance_metas.values() if m.is_border)} border"
    )

    return TileMetadataResult(
        tile_idx=tile_idx,
        tile_name=tile_name,
        filepath=filepath,
        boundary=boundary,
        instances=instance_metas,
        filtered_local_ids=filtered_ids,
        kept_local_ids=kept_ids,
        extra_dim_names=extra_dim_names,
        extra_dim_dtypes=extra_dim_dtypes,
    )


def _extract_tile_metadata_wrapper(args):
    """Top-level wrapper for ProcessPoolExecutor (must be pickleable)."""
    (filepath, tile_idx, tile_boundaries, tile_name, neighbors_for_tile,
     border_zone_width, correspondence_tolerance, instance_dimension,
     core_boundary, chunk_size) = args
    return extract_tile_metadata(
        filepath=filepath,
        tile_idx=tile_idx,
        tile_boundaries=tile_boundaries,
        tile_name=tile_name,
        neighbors_for_tile=neighbors_for_tile,
        border_zone_width=border_zone_width,
        correspondence_tolerance=correspondence_tolerance,
        instance_dimension=instance_dimension,
        core_boundary=core_boundary,
        chunk_size=chunk_size,
    )


# =============================================================================
# Phase B: Cross-Tile Matching + Orphan Recovery
# =============================================================================


def global_instance_matching(
    tile_results: List[TileMetadataResult],
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]],
    core_bounds_by_tile: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    overlap_threshold: float = 0.50,
    search_radius: float = 15.0,
    coverage_threshold: float = 0.50,
    proximity_merge_distance: float = 3.0,
    verbose: bool = False,
) -> Tuple[UnionFind, Dict[int, int]]:
    """
    Single-pass global instance matching replacing the old 4-function chain.

    Steps:
      1. Collect all instances with hash sets, build spatial index
      2. Hash-overlap matching (all cross-tile pairs within search_radius)
      3. Orphan recovery for unmatched filtered instances
      4. Proximity merge for complementary halves at core boundaries
      5. Build global_to_merged from union-find components

    Returns:
        uf: UnionFind with matched/recovered instances
        global_to_merged: mapping global_id -> merged_id
    """
    from scipy.spatial import cKDTree

    print(f"\n{'=' * 60}")
    print("Phase B: Global Instance Matching")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Step 1: Collect all instances and build spatial index
    # ------------------------------------------------------------------
    all_instances: Dict[int, InstanceMeta] = {}
    for result in tile_results:
        all_instances.update(result.instances)

    # Initialize Union-Find with all kept instances
    uf = UnionFind()
    for gid, meta in all_instances.items():
        if not meta.is_filtered:
            uf.make_set(gid, meta.point_count)

    # Separate instances with hash sets for spatial matching
    with_hash: List[Tuple[int, InstanceMeta]] = []
    for gid, meta in all_instances.items():
        if meta.hash_set is not None:
            with_hash.append((gid, meta))

    if len(with_hash) < 2:
        print("  < 2 instances with hash sets, skipping matching")
        components = uf.get_components()
        global_to_merged: Dict[int, int] = {}
        for merged_id, (root, members) in enumerate(components.items(), start=1):
            for gid in members:
                global_to_merged[gid] = merged_id
        return uf, global_to_merged

    # Build 2D KDTree from centroids
    gid_list = [gid for gid, _ in with_hash]
    centers = np.array([[m.centroid[0], m.centroid[1]] for _, m in with_hash])
    tree = cKDTree(centers)
    gid_to_idx = {gid: i for i, gid in enumerate(gid_list)}

    # ------------------------------------------------------------------
    # Step 2: Hash-overlap matching (all cross-tile pairs)
    # ------------------------------------------------------------------
    matched_gids: Set[int] = set()
    match_count = 0
    hash_checks = 0
    bbox_checks = 0

    for i, (gid_a, meta_a) in enumerate(with_hash):
        nearby_idxs = tree.query_ball_point(centers[i], r=search_radius)
        for j in nearby_idxs:
            gid_b = gid_list[j]
            if gid_b <= gid_a:
                continue  # avoid duplicate pairs

            meta_b = all_instances[gid_b]

            # Skip same tile — prevents over-merging within a tile
            if meta_a.tile_name == meta_b.tile_name:
                continue

            # Quick bbox overlap check
            bbox_checks += 1
            if not _bboxes_overlap_3d(meta_a, meta_b, tolerance=0.1):
                continue

            # Hash-set overlap
            hash_checks += 1
            intersection = len(meta_a.hash_set & meta_b.hash_set)
            if intersection == 0:
                continue

            ratio_a = intersection / len(meta_a.hash_set)
            ratio_b = intersection / len(meta_b.hash_set)
            # Require mutual support rather than one-sided containment.
            overlap = min(ratio_a, ratio_b)

            if overlap >= overlap_threshold:
                # Recover filtered helper-only instances into union-find if
                # needed. Even if they own no final output points, they can
                # still bridge equivalent crowns across tiles.
                for gid, meta in [(gid_a, meta_a), (gid_b, meta_b)]:
                    if meta.is_filtered:
                        uf.make_set(gid, meta.point_count)
                        meta.is_filtered = False
                        # Update tile result's kept/filtered sets
                        for r in tile_results:
                            if r.tile_name == meta.tile_name:
                                r.filtered_local_ids.discard(meta.local_id)
                                r.kept_local_ids.add(meta.local_id)
                                break

                uf.union(gid_a, gid_b)
                matched_gids.add(gid_a)
                matched_gids.add(gid_b)
                match_count += 1

                if verbose:
                    print(
                        f"    Match: {meta_a.tile_name}:{meta_a.local_id} <-> "
                        f"{meta_b.tile_name}:{meta_b.local_id} "
                        f"overlap={overlap:.1%}"
                    )

    print(f"  Step 2: {match_count} matches "
          f"({bbox_checks} bbox checks, {hash_checks} hash checks)")

    # ------------------------------------------------------------------
    # Step 3: Orphan recovery for unmatched filtered instances
    # ------------------------------------------------------------------
    # Build combined hash sets per kept component for coverage checks
    components = uf.get_components()
    component_hashes: Dict[int, Set[int]] = {}
    component_centers: Dict[int, np.ndarray] = {}

    for root, members in components.items():
        combined = set()
        sx, sy, n = 0.0, 0.0, 0
        for gid in members:
            meta = all_instances[gid]
            if meta.hash_set is not None:
                combined |= meta.hash_set
            sx += meta.centroid[0] * meta.point_count
            sy += meta.centroid[1] * meta.point_count
            n += meta.point_count
        if combined:
            component_hashes[root] = combined
            component_centers[root] = np.array([sx / n, sy / n])

    # Build KDTree of kept component centroids
    if component_centers:
        comp_roots = list(component_centers.keys())
        comp_pts = np.array([component_centers[r] for r in comp_roots])
        comp_tree = cKDTree(comp_pts)
    else:
        comp_roots = []
        comp_tree = None

    recovered = 0
    skipped = 0
    for gid, meta in with_hash:
        if not meta.is_filtered:
            continue
        if gid in matched_gids:
            continue

        # Check if covered by any kept component
        covered = False
        if comp_tree is not None:
            query_pt = np.array([meta.centroid[0], meta.centroid[1]])
            nearby = comp_tree.query_ball_point(query_pt, r=search_radius)
            for idx in nearby:
                root = comp_roots[idx]
                comp_h = component_hashes.get(root)
                if comp_h is None:
                    continue
                intersection = len(meta.hash_set & comp_h)
                coverage = intersection / len(meta.hash_set)
                if coverage >= coverage_threshold:
                    covered = True
                    break

        if covered:
            skipped += 1
        else:
            # Recover: not covered by any kept instance
            uf.make_set(gid, meta.point_count)
            meta.is_filtered = False
            for r in tile_results:
                if r.tile_name == meta.tile_name:
                    r.filtered_local_ids.discard(meta.local_id)
                    r.kept_local_ids.add(meta.local_id)
                    break
            recovered += 1

    print(f"  Step 3: recovered {recovered} orphans, skipped {skipped} (covered)")

    # ------------------------------------------------------------------
    # Step 4: Proximity merge for complementary halves at core boundaries
    # ------------------------------------------------------------------
    proximity_merged = 0
    if core_bounds_by_tile:
        # Collect unique core boundary lines
        x_boundaries: Set[float] = set()
        y_boundaries: Set[float] = set()
        for core in core_bounds_by_tile.values():
            x_boundaries.add(core[0])  # min_x
            x_boundaries.add(core[1])  # max_x
            y_boundaries.add(core[2])  # min_y
            y_boundaries.add(core[3])  # max_y

        boundary_tol = 1.0  # instance bbox must be within 1m of boundary

        # Find unmatched kept instances near core boundaries
        boundary_candidates: List[Tuple[int, InstanceMeta, str, float]] = []
        for gid, meta in all_instances.items():
            if meta.is_filtered or gid in matched_gids:
                continue
            if meta.hash_set is None:
                continue

            # Check if bbox touches any core boundary
            for bx in x_boundaries:
                if abs(meta.bbox_min[0] - bx) < boundary_tol or abs(meta.bbox_max[0] - bx) < boundary_tol:
                    boundary_candidates.append((gid, meta, "x", bx))
            for by in y_boundaries:
                if abs(meta.bbox_min[1] - by) < boundary_tol or abs(meta.bbox_max[1] - by) < boundary_tol:
                    boundary_candidates.append((gid, meta, "y", by))

        # Check pairs
        for i, (gid_a, meta_a, axis_a, bval_a) in enumerate(boundary_candidates):
            for j in range(i + 1, len(boundary_candidates)):
                gid_b, meta_b, axis_b, bval_b = boundary_candidates[j]

                if meta_a.tile_name == meta_b.tile_name:
                    continue
                if axis_a != axis_b or bval_a != bval_b:
                    continue
                # Already in same component?
                if uf.find(gid_a) == uf.find(gid_b):
                    continue

                # Centroid distance check
                dist = np.sqrt(
                    (meta_a.centroid[0] - meta_b.centroid[0]) ** 2
                    + (meta_a.centroid[1] - meta_b.centroid[1]) ** 2
                )
                if dist > proximity_merge_distance:
                    continue

                # Must be on opposite sides of the boundary
                if axis_a == "x":
                    if not (
                        (meta_a.centroid[0] < bval_a and meta_b.centroid[0] >= bval_a)
                        or (meta_b.centroid[0] < bval_a and meta_a.centroid[0] >= bval_a)
                    ):
                        continue
                    # Z range overlap
                    z_overlap = min(meta_a.bbox_max[2], meta_b.bbox_max[2]) - max(meta_a.bbox_min[2], meta_b.bbox_min[2])
                    z_union = max(meta_a.bbox_max[2], meta_b.bbox_max[2]) - min(meta_a.bbox_min[2], meta_b.bbox_min[2])
                    if z_union <= 0 or z_overlap / z_union < 0.3:
                        continue
                else:  # axis == "y"
                    if not (
                        (meta_a.centroid[1] < bval_a and meta_b.centroid[1] >= bval_a)
                        or (meta_b.centroid[1] < bval_a and meta_a.centroid[1] >= bval_a)
                    ):
                        continue
                    z_overlap = min(meta_a.bbox_max[2], meta_b.bbox_max[2]) - max(meta_a.bbox_min[2], meta_b.bbox_min[2])
                    z_union = max(meta_a.bbox_max[2], meta_b.bbox_max[2]) - min(meta_a.bbox_min[2], meta_b.bbox_min[2])
                    if z_union <= 0 or z_overlap / z_union < 0.3:
                        continue

                uf.union(gid_a, gid_b)
                proximity_merged += 1
                if verbose:
                    print(
                        f"    Proximity merge: {meta_a.tile_name}:{meta_a.local_id} <-> "
                        f"{meta_b.tile_name}:{meta_b.local_id} "
                        f"(boundary {axis_a}={bval_a:.1f}, dist={dist:.1f}m)"
                    )

    print(f"  Step 4: {proximity_merged} proximity merges at core boundaries")

    # ------------------------------------------------------------------
    # Step 5: Build global_to_merged from final components
    # ------------------------------------------------------------------
    components = uf.get_components()
    global_to_merged = {}
    for merged_id, (root, members) in enumerate(components.items(), start=1):
        for gid in members:
            global_to_merged[gid] = merged_id

    total_kept = sum(1 for m in all_instances.values() if not m.is_filtered)
    total_merged_ids = len(components)
    print(f"\n  Result: {total_kept} kept instances -> {total_merged_ids} merged IDs")

    return uf, global_to_merged


def _bboxes_overlap_3d(
    meta_a: InstanceMeta, meta_b: InstanceMeta, tolerance: float = 0.1
) -> bool:
    """Check if two instance bboxes overlap in XY (with tolerance)."""
    # XY overlap check with tolerance
    if (
        meta_a.bbox_max[0] + tolerance < meta_b.bbox_min[0]
        or meta_b.bbox_max[0] + tolerance < meta_a.bbox_min[0]
    ):
        return False
    if (
        meta_a.bbox_max[1] + tolerance < meta_b.bbox_min[1]
        or meta_b.bbox_max[1] + tolerance < meta_a.bbox_min[1]
    ):
        return False
    return True


def redistribute_small_instances(
    tile_results: List[TileMetadataResult],
    global_to_merged: Dict[int, int],
    min_points_reassign: int = 500,
    min_points_hull: int = 300,
    max_points_hull: int = 2000,
    max_volume_for_merge: float = 5.0,
    max_search_radius: float = 10.0,
    instance_dimension: str = "PredInstance",
    verbose: bool = False,
    chunk_size: int = 500_000,
) -> None:
    """
    Redistribute small merged instances before Phase C.

    Uses Phase A metadata (point counts, centroids, bboxes) to classify instances,
    then reads source tiles for hull computation on candidates.

    Updates global_to_merged in-place so Phase C writes with final IDs.

    Three tiers:
      - < min_points_reassign: unconditionally reassign to nearest instance
      - min_points_hull..max_points_hull: compute 3D convex hull from source tiles,
        reassign if volume < max_volume_for_merge
      - > max_points_hull: keep as-is
    """
    from scipy.spatial import cKDTree, ConvexHull

    print(f"\n{'=' * 60}")
    print("Phase B.5: Small Instance Redistribution")
    print(f"{'=' * 60}")

    # Aggregate per-merged-instance stats from Phase A metadata
    merged_point_counts: Dict[int, int] = defaultdict(int)
    merged_centroid_sum: Dict[int, np.ndarray] = {}
    merged_bbox_min: Dict[int, np.ndarray] = {}
    merged_bbox_max: Dict[int, np.ndarray] = {}
    # Track which global IDs belong to each merged ID (for hull point collection)
    merged_to_globals: Dict[int, List[int]] = defaultdict(list)

    for result in tile_results:
        for gid, meta in result.instances.items():
            if meta.is_filtered:
                continue
            mid = global_to_merged.get(gid)
            if mid is None:
                continue
            merged_point_counts[mid] += meta.point_count
            merged_to_globals[mid].append(gid)

            if mid not in merged_centroid_sum:
                merged_centroid_sum[mid] = np.zeros(2, dtype=np.float64)
                merged_bbox_min[mid] = meta.bbox_min.copy()
                merged_bbox_max[mid] = meta.bbox_max.copy()
            merged_centroid_sum[mid][0] += meta.centroid[0] * meta.point_count
            merged_centroid_sum[mid][1] += meta.centroid[1] * meta.point_count
            merged_bbox_min[mid] = np.minimum(merged_bbox_min[mid], meta.bbox_min)
            merged_bbox_max[mid] = np.maximum(merged_bbox_max[mid], meta.bbox_max)

    # Compute 2D XY centroids
    merged_centroids_2d: Dict[int, np.ndarray] = {}
    for mid, sums in merged_centroid_sum.items():
        count = merged_point_counts[mid]
        if count > 0:
            merged_centroids_2d[mid] = sums / count  # (x, y)

    # Classify instances
    tiny_instances = []      # < min_points_reassign: always reassign
    hull_candidates = []     # min_points_hull..max_points_hull: hull check
    large_instances = []     # > max_points_hull: keep

    for mid, c2d in merged_centroids_2d.items():
        count = merged_point_counts[mid]
        if count < min_points_reassign:
            tiny_instances.append((mid, c2d))
        elif min_points_hull <= count <= max_points_hull:
            hull_candidates.append((mid, c2d))
        else:
            large_instances.append((mid, c2d))

    print(f"  {len(large_instances)} large instances (> {max_points_hull} pts)")
    print(f"  {len(hull_candidates)} hull candidates ({min_points_hull}-{max_points_hull} pts)")
    print(f"  {len(tiny_instances)} tiny instances (< {min_points_reassign} pts)")

    # Remap: merged_id -> target_merged_id
    id_remap: Dict[int, int] = {}

    # Step 1: Reassign tiny instances to nearest non-tiny
    non_tiny = large_instances + hull_candidates
    if tiny_instances and non_tiny:
        ref_centroids = np.array([c for _, c in non_tiny])
        ref_ids = [mid for mid, _ in non_tiny]
        tree = cKDTree(ref_centroids)

        tiny_centroids = np.array([c for _, c in tiny_instances])
        distances, indices = tree.query(tiny_centroids)

        reassign_count = 0
        for i, (mid, _) in enumerate(tiny_instances):
            if distances[i] <= max_search_radius:
                target = ref_ids[indices[i]]
                id_remap[mid] = target
                reassign_count += 1
                if verbose:
                    print(
                        f"    Instance {mid}: {merged_point_counts[mid]} pts "
                        f"-> reassigned to {target} (dist={distances[i]:.1f}m)"
                    )
        print(f"  Reassigned {reassign_count}/{len(tiny_instances)} tiny instances")

    # Step 2: Hull-based check for medium instances
    if hull_candidates and large_instances:
        # Build index: global_id -> (tile_result, local_id) for hull candidate globals
        hull_candidate_ids = {mid for mid, _ in hull_candidates}
        hull_globals: Set[int] = set()
        for mid in hull_candidate_ids:
            hull_globals.update(merged_to_globals.get(mid, []))

        # Map global_id -> (filepath, local_id, tile_name) for reading points
        global_to_file: Dict[int, Tuple[Path, int]] = {}
        for result in tile_results:
            for gid, meta in result.instances.items():
                if gid in hull_globals:
                    global_to_file[gid] = (result.filepath, meta.local_id)

        # Collect 3D points per merged ID from source tiles
        # Group by filepath to minimize file opens
        file_to_globals: Dict[Path, List[Tuple[int, int]]] = defaultdict(list)
        for gid, (fpath, local_id) in global_to_file.items():
            file_to_globals[fpath].append((gid, local_id))

        # Collect hull vertices (not raw points) per merged ID to save memory.
        # After each source file, compute intermediate convex hulls and keep
        # only the vertices; the final hull can be computed from the union of
        # all intermediate hull vertices.
        hull_vertices: Dict[int, list] = defaultdict(list)

        print(f"  Reading hull candidate points from {len(file_to_globals)} source tiles...")
        for fpath, gid_lid_list in file_to_globals.items():
            local_ids_needed = {lid for _, lid in gid_lid_list}
            gid_by_lid = {lid: gid for gid, lid in gid_lid_list}
            # Accumulate raw points per mid for this file only
            _file_pts: Dict[int, list] = defaultdict(list)
            try:
                with laspy.open(
                    str(fpath), laz_backend=laspy.LazBackend.LazrsParallel
                ) as reader:
                    for chunk in reader.chunk_iterator(chunk_size):
                        inst_arr = np.array(getattr(chunk, instance_dimension))
                        chunk_lids = set(np.unique(inst_arr)) & local_ids_needed
                        if not chunk_lids:
                            continue
                        pts_xyz = np.column_stack([chunk.x, chunk.y, chunk.z])
                        for lid in chunk_lids:
                            gid = gid_by_lid.get(lid)
                            if gid is None:
                                continue
                            mid = global_to_merged.get(gid)
                            if mid is None or mid not in hull_candidate_ids:
                                continue
                            mask = inst_arr == lid
                            _file_pts[mid].append(pts_xyz[mask])
            except Exception as e:
                print(f"    Warning: Could not read {fpath.name}: {e}")
            # Reduce raw points to hull vertices before moving to next file
            for mid, pts_list in _file_pts.items():
                pts_3d = np.vstack(pts_list)
                if len(pts_3d) >= 4:
                    try:
                        h = ConvexHull(pts_3d)
                        hull_vertices[mid].append(pts_3d[h.vertices])
                    except Exception:
                        hull_vertices[mid].append(pts_3d)
                else:
                    hull_vertices[mid].append(pts_3d)
            del _file_pts

        # Build KDTree from large instances for hull reassignment
        large_centroids = np.array([c for _, c in large_instances])
        large_ids = [mid for mid, _ in large_instances]
        tree_large = cKDTree(large_centroids)

        hull_merge_count = 0
        hull_keep_count = 0
        for mid, c2d in hull_candidates:
            if mid in id_remap:
                continue
            pts_list = hull_vertices.get(mid)
            if not pts_list:
                continue
            pts_3d = np.vstack(pts_list)
            if len(pts_3d) < 4:
                # Too few points for hull, treat as tiny
                dist, idx = tree_large.query(c2d)
                if dist <= max_search_radius:
                    id_remap[mid] = large_ids[idx]
                    hull_merge_count += 1
                continue

            try:
                hull = ConvexHull(pts_3d)
                volume = hull.volume
            except Exception:
                bbox_size = pts_3d.max(axis=0) - pts_3d.min(axis=0)
                volume = float(np.prod(np.maximum(bbox_size, 0.01)))

            if volume < max_volume_for_merge:
                dist, idx = tree_large.query(c2d)
                if dist <= max_search_radius:
                    id_remap[mid] = large_ids[idx]
                    hull_merge_count += 1
                    if verbose:
                        print(
                            f"    Instance {mid}: {merged_point_counts[mid]} pts, "
                            f"{volume:.2f} m³ -> reassigned to {large_ids[idx]} (dist={dist:.1f}m)"
                        )
            else:
                hull_keep_count += 1
                if verbose:
                    print(
                        f"    Instance {mid}: {merged_point_counts[mid]} pts, "
                        f"{volume:.2f} m³ -> kept"
                    )

        print(
            f"  Hull check: {hull_merge_count} reassigned, "
            f"{hull_keep_count} kept (volume >= {max_volume_for_merge} m³)"
        )

    # Apply remap to global_to_merged
    if id_remap:
        remapped_count = 0
        for gid in list(global_to_merged.keys()):
            old_mid = global_to_merged[gid]
            new_mid = id_remap.get(old_mid)
            if new_mid is not None:
                global_to_merged[gid] = new_mid
                remapped_count += 1
        print(f"  Updated {remapped_count} global ID mappings")


# =============================================================================
# Phase C: Streaming Merged File
# =============================================================================


def _determine_merged_header(
    laz_files: List[Path],
    extra_dim_names: List[str],
    extra_dim_dtypes: Dict[str, np.dtype],
    instance_dimension: str = "PredInstance",
) -> Tuple[laspy.LasHeader, np.ndarray]:
    """
    Determine the merged file header by scanning all tile headers.
    Returns (header, global_min_coords).
    """
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for f in laz_files:
        try:
            with laspy.open(str(f), laz_backend=laspy.LazBackend.LazrsParallel) as las:
                global_min[0] = min(global_min[0], las.header.x_min)
                global_min[1] = min(global_min[1], las.header.y_min)
                global_min[2] = min(global_min[2], las.header.z_min)
                global_max[0] = max(global_max[0], las.header.x_max)
                global_max[1] = max(global_max[1], las.header.y_max)
                global_max[2] = max(global_max[2], las.header.z_max)
        except Exception:
            continue

    promote_rgb_to_standard = has_standard_rgb_dims(set(extra_dim_names))
    header = laspy.LasHeader(
        point_format=7 if promote_rgb_to_standard else 6,
        version="1.4",
    )
    header.offsets = global_min
    header.scales = np.array([0.001, 0.001, 0.001])

    # Add extra dimensions
    extra_params = [laspy.ExtraBytesParams(name=instance_dimension, type=np.int32)]
    for dim_name in extra_dim_names:
        if promote_rgb_to_standard and dim_name in {"red", "green", "blue"}:
            continue
        dtype = extra_dim_dtypes.get(dim_name, np.int32)
        extra_params.append(laspy.ExtraBytesParams(name=dim_name, type=dtype))
    header.add_extra_dims(extra_params)

    return header, global_min


def _update_instance_stats(
    points: np.ndarray,
    remapped: np.ndarray,
    instance_sum_xy: Dict[int, np.ndarray],
    instance_counts: Dict[int, int],
    instance_bbox_min: Dict[int, np.ndarray],
    instance_bbox_max: Dict[int, np.ndarray],
) -> None:
    """Accumulate per-instance metadata from a chunk (vectorized)."""
    unique_ids = np.unique(remapped)
    for mid_val in unique_ids:
        mid = int(mid_val)
        if mid <= 0:
            continue
        mask = remapped == mid
        pts = points[mask]
        n = int(mask.sum())

        if mid not in instance_sum_xy:
            instance_sum_xy[mid] = np.zeros(2, dtype=np.float64)
            instance_bbox_min[mid] = np.array([np.inf, np.inf, np.inf])
            instance_bbox_max[mid] = np.array([-np.inf, -np.inf, -np.inf])

        instance_sum_xy[mid][0] += pts[:, 0].sum()
        instance_sum_xy[mid][1] += pts[:, 1].sum()
        instance_counts[mid] += n
        instance_bbox_min[mid] = np.minimum(
            instance_bbox_min[mid], pts.min(axis=0)
        )
        instance_bbox_max[mid] = np.maximum(
            instance_bbox_max[mid], pts.max(axis=0)
        )


def _is_in_overlap_region(
    points: np.ndarray,
    neighbors: Dict[str, Optional[str]],
    core_boundary: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Return boolean mask for points in overlap regions (outside the core on sides
    that have neighbors).

    Uses exact core bounds from tile_bounds_tindex.json.
    """
    x, y = points[:, 0], points[:, 1]
    in_overlap = np.zeros(len(points), dtype=bool)

    core_min_x, core_max_x, core_min_y, core_max_y = core_boundary
    if neighbors.get("west") is not None:
        in_overlap |= x < core_min_x
    if neighbors.get("east") is not None:
        in_overlap |= x > core_max_x
    if neighbors.get("south") is not None:
        in_overlap |= y < core_min_y
    if neighbors.get("north") is not None:
        in_overlap |= y > core_max_y

    return in_overlap


def write_merged_streaming(
    tile_results: List[TileMetadataResult],
    global_to_merged: Dict[int, int],
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]],
    output_merged: Path,
    core_bounds_by_tile: Dict[str, Tuple[float, float, float, float]],
    correspondence_tolerance: float = 0.01,
    instance_dimension: str = "PredInstance",
    chunk_size: int = 1_000_000,
    skip_merged_file: bool = False,
    verbose: bool = False,
) -> Tuple[Optional[Path], Dict[int, Tuple[float, float]], Dict[int, int], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Stream-write the merged file tile by tile.

    Returns:
        (output_path, instance_centroids_xy, instance_counts,
         instance_bbox_min, instance_bbox_max)
    """
    print(f"\n{'=' * 60}")
    print("Phase C: Streaming Merged File Creation")
    print(f"{'=' * 60}")

    laz_files = [r.filepath for r in tile_results]

    # Collect all extra dim info across tiles
    all_extra_dim_names: List[str] = []
    all_extra_dim_dtypes: Dict[str, np.dtype] = {}
    seen_dims: Set[str] = set()
    for result in tile_results:
        for dim_name in result.extra_dim_names:
            if dim_name not in seen_dims:
                all_extra_dim_names.append(dim_name)
                seen_dims.add(dim_name)
        all_extra_dim_dtypes.update(result.extra_dim_dtypes)

    if all_extra_dim_names:
        print(f"  Extra dimensions: {', '.join(all_extra_dim_names)}")

    # Determine header from tile headers
    print("  Scanning tile headers for bounds...")
    header, global_min = _determine_merged_header(
        laz_files, all_extra_dim_names, all_extra_dim_dtypes, instance_dimension
    )

    # Track per-merged-instance metadata for renumbering and small volume merge
    # Running sums for centroid computation
    instance_sum_xy: Dict[int, np.ndarray] = {}  # merged_id -> [sum_x, sum_y]
    instance_counts: Dict[int, int] = defaultdict(int)  # merged_id -> point_count
    instance_bbox_min: Dict[int, np.ndarray] = {}  # merged_id -> [min_x, min_y, min_z]
    instance_bbox_max: Dict[int, np.ndarray] = {}  # merged_id -> [max_x, max_y, max_z]

    total_written = 0
    total_ground_dropped = 0
    total_deduped = 0
    total_filtered_removed = 0

    output_merged.parent.mkdir(parents=True, exist_ok=True)

    tile_name_to_idx = {r.tile_name: r.tile_idx for r in tile_results}

    with laspy.open(
        str(output_merged),
        mode="w",
        header=header,
        laz_backend=laspy.LazBackend.LazrsParallel,
    ) as writer:
        for tile_result in tile_results:
            tile_name = tile_result.tile_name
            tile_idx = tile_name_to_idx[tile_name]
            filepath = tile_result.filepath
            neighbors = neighbors_by_tile.get(tile_name, {})
            boundary = tile_result.boundary

            print(f"  Processing {filepath.name}...", end=" ", flush=True)

            # Build instance remap lookup from metadata
            # Size it from the tile's max known local ID
            max_local = max(
                (m.local_id for m in tile_result.instances.values()),
                default=0
            ) + 1
            inst_to_merged = np.full(max_local, -1, dtype=np.int32)
            inst_to_merged[0] = 0  # ground/unassigned stays 0

            for local_id in tile_result.kept_local_ids:
                if local_id <= 0:
                    continue
                gid = tile_idx * TILE_OFFSET + local_id
                merged_id = global_to_merged.get(gid, -1)
                if merged_id >= 0 and local_id < max_local:
                    inst_to_merged[local_id] = merged_id

            # Stream chunks: read -> remap/filter -> dedup -> write
            tile_written = 0
            tile_filtered = 0
            tile_ground_dropped = 0
            tile_deduped = 0

            try:
                with laspy.open(
                    str(filepath), laz_backend=laspy.LazBackend.LazrsParallel
                ) as f:
                    header_extra_dims = {
                        dim.name: dim
                        for dim in f.header.point_format.extra_dimensions
                    }
                    has_instance_dim = instance_dimension in header_extra_dims
                    has_tree_id = "treeID" in header_extra_dims
                    for chunk in f.chunk_iterator(chunk_size):
                        chunk_len = len(chunk)

                        # Extract XYZ
                        c_points = np.empty((chunk_len, 3), dtype=np.float64)
                        c_points[:, 0] = chunk.x
                        c_points[:, 1] = chunk.y
                        c_points[:, 2] = chunk.z

                        # Extract instances
                        if has_instance_dim:
                            c_instances = np.array(
                                getattr(chunk, instance_dimension), dtype=np.int32
                            )
                        elif has_tree_id:
                            c_instances = np.array(chunk.treeID, dtype=np.int32)
                        else:
                            c_instances = np.zeros(chunk_len, dtype=np.int32)

                        # Extract extra dims
                        c_extra: Dict[str, np.ndarray] = {}
                        for dim_name in all_extra_dim_names:
                            if dim_name in header_extra_dims:
                                c_extra[dim_name] = np.array(
                                    getattr(chunk, dim_name)
                                )
                            else:
                                dtype = all_extra_dim_dtypes.get(dim_name, np.int32)
                                c_extra[dim_name] = np.zeros(chunk_len, dtype=dtype)

                        # Remap instance IDs
                        safe = np.clip(c_instances, 0, max_local - 1)
                        remapped = inst_to_merged[safe]

                        # Determine which points are outside this tile's core
                        # (on sides that have neighbors).
                        core_bounds = core_bounds_by_tile.get(tile_name)
                        if core_bounds is None:
                            raise ValueError(
                                f"No core bounds found for tile {tile_name} in "
                                f"tile_bounds_tindex.json — cannot determine "
                                f"overlap region"
                            )
                        overlap = _is_in_overlap_region(
                            c_points, neighbors,
                            core_boundary=core_bounds,
                        )

                        # Remove:
                        #  - helper-only instances with no owned points
                        #    (remapped=-1)
                        #  - ALL points in the overlap region (both ground
                        #    and non-ground). Each tile only contributes its
                        #    core points.  Tiles independently subsample to
                        #    10cm, so overlap points from different tiles
                        #    have different coordinates and cannot be
                        #    hash-deduplicated.  Core regions partition the
                        #    area without gaps, giving exactly one copy of
                        #    every physical location.
                        filtered = remapped == -1
                        remove = filtered | overlap

                        n_filt = int(filtered.sum())
                        n_gd = int((overlap & (remapped == 0)).sum())
                        n_overlap_nonground = int(
                            (overlap & (remapped > 0)).sum()
                        )
                        tile_filtered += n_filt
                        tile_ground_dropped += n_gd
                        tile_deduped += n_overlap_nonground

                        if remove.any():
                            keep = ~remove
                            c_points = c_points[keep]
                            remapped = remapped[keep]
                            for dn in c_extra:
                                c_extra[dn] = c_extra[dn][keep]

                        if len(remapped) == 0:
                            continue

                        # Update per-instance metadata (vectorized)
                        _update_instance_stats(
                            c_points, remapped, instance_sum_xy,
                            instance_counts, instance_bbox_min,
                            instance_bbox_max,
                        )

                        # Write chunk to merged file
                        point_record = laspy.ScaleAwarePointRecord.zeros(
                            len(c_points), header=header
                        )
                        point_record.x = c_points[:, 0]
                        point_record.y = c_points[:, 1]
                        point_record.z = c_points[:, 2]
                        setattr(point_record, instance_dimension, remapped)
                        for dim_name in all_extra_dim_names:
                            if dim_name in c_extra:
                                setattr(point_record, dim_name, c_extra[dim_name])

                        writer.write_points(point_record)
                        tile_written += len(c_points)

            except Exception as e:
                print(f"ERROR: {e}")
                continue

            total_written += tile_written
            total_filtered_removed += tile_filtered
            total_ground_dropped += tile_ground_dropped
            total_deduped += tile_deduped

            print(
                f"{tile_written:,} written"
                + (f", {tile_ground_dropped:,} ground dropped" if tile_ground_dropped > 0 else "")
                + (f", {tile_deduped:,} overlap non-ground dropped" if tile_deduped > 0 else "")
                + (f", {tile_filtered:,} filtered" if tile_filtered > 0 else "")
            )

    gc.collect()

    print(f"\n  Total: {total_written:,} points written")
    print(f"  Overlap ground dropped: {total_ground_dropped:,} points")
    print(f"  Overlap non-ground dropped: {total_deduped:,} points")
    print(f"  Filtered (no owned points): {total_filtered_removed:,} points removed")

    # Compute final centroids for renumbering
    instance_centroids_xy: Dict[int, Tuple[float, float]] = {}
    for mid, sums in instance_sum_xy.items():
        count = instance_counts[mid]
        if count > 0:
            # (center_y, center_x) for north-to-south sorting
            instance_centroids_xy[mid] = (sums[1] / count, sums[0] / count)

    return output_merged, instance_centroids_xy, instance_counts, instance_bbox_min, instance_bbox_max


# =============================================================================
# Post-pass: Renumber Instances
# =============================================================================


def streaming_renumber_instances(
    merged_file: Path,
    instance_centroids_xy: Dict[int, Tuple[float, float]],
    instance_dimension: str = "PredInstance",
    chunk_size: int = 1_000_000,
    verbose: bool = False,
) -> Path:
    """
    Post-processing pass: renumber instances north-to-south.

    Reads merged file in chunks, remaps IDs, writes to new file.
    Returns path to final merged file.
    """
    print(f"\n{'=' * 60}")
    print("Post-pass: Instance Renumbering (north-to-south)")
    print(f"{'=' * 60}")

    # Get all active merged IDs
    active_ids = {mid for mid in instance_centroids_xy if mid > 0}

    # Sort by Y descending (north to south), then X ascending
    sorted_ids = sorted(
        active_ids,
        key=lambda mid: (
            -instance_centroids_xy.get(mid, (0, 0))[0],  # -Y (north first)
            instance_centroids_xy.get(mid, (0, 0))[1],  # +X (west first)
        ),
    )

    # Build renumbering: old_merged_id -> new_sequential_id
    renumber_remap: Dict[int, int] = {0: 0}
    for new_id, old_id in enumerate(sorted_ids, start=1):
        renumber_remap[old_id] = new_id

    # Check if any remapping is actually needed
    needs_remap = any(
        renumber_remap.get(mid, mid) != mid for mid in instance_centroids_xy
    )

    if not needs_remap:
        print(f"  No remapping needed, {len(sorted_ids)} final instances")
        return merged_file

    print(f"  Streaming rewrite with {len(sorted_ids)} final instances...")

    # Build numpy lookup table for fast remapping
    max_old_id = max(instance_centroids_xy.keys(), default=0) + 1
    lookup = np.arange(max_old_id, dtype=np.int32)
    for old_id, new_id in renumber_remap.items():
        if old_id < max_old_id:
            lookup[old_id] = new_id

    # Streaming rewrite
    output_final = merged_file.parent / (merged_file.stem + "_final.laz")

    with laspy.open(str(merged_file), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        out_header = reader.header

        with laspy.open(
            str(output_final),
            mode="w",
            header=out_header,
            laz_backend=laspy.LazBackend.LazrsParallel,
        ) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                inst_arr = getattr(chunk, instance_dimension)
                safe = np.clip(inst_arr, 0, max_old_id - 1)
                new_inst = lookup[safe]
                setattr(chunk, instance_dimension, new_inst)
                writer.write_points(chunk)

    # Replace original with final
    merged_file.unlink()
    shutil.move(str(output_final), str(merged_file))

    print(f"  Final instance count: {len(sorted_ids)}")
    print(f"  Saved: {merged_file}")

    return merged_file


# =============================================================================
# Optional: Save Filtered Tiles
# =============================================================================


def save_filtered_tiles(
    tile_results: List[TileMetadataResult],
    output_dir: Path,
    instance_dimension: str = "PredInstance",
    chunk_size: int = 1_000_000,
):
    """Save tiles with filtered instances removed (for debugging)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving filtered tiles to {output_dir}...")

    for result in tile_results:
        filepath = result.filepath
        kept_ids = result.kept_local_ids | {0}
        output_path = output_dir / f"{result.tile_name}.laz"

        try:
            with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as f:
                n_points = f.header.point_count
                header_extra_dims = {
                    dim.name: dim for dim in f.header.point_format.extra_dimensions
                }
                has_instance_dim = instance_dimension in header_extra_dims
                has_tree_id = "treeID" in header_extra_dims

                points = np.empty((n_points, 3), dtype=np.float64)
                instances = np.zeros(n_points, dtype=np.int32)
                extra_dims: Dict[str, np.ndarray] = {}

                for dim in f.header.point_format.extra_dimensions:
                    if dim.name != instance_dimension and not (
                        not has_instance_dim and dim.name == "treeID"
                    ):
                        extra_dims[dim.name] = np.zeros(n_points, dtype=dim.dtype)

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

            # Filter
            keep_mask = np.isin(instances, list(kept_ids))
            if keep_mask.sum() == 0:
                print(f"    Warning: {result.tile_name} empty after filtering, skipping")
                continue

            filtered_points = points[keep_mask]
            filtered_instances = instances[keep_mask]
            filtered_extras = {name: arr[keep_mask] for name, arr in extra_dims.items()}

            header = laspy.LasHeader(point_format=6, version="1.4")
            header.offsets = filtered_points[:, :3].min(axis=0)
            header.scales = [0.01, 0.01, 0.01]

            out_las = laspy.LasData(header)
            out_las.x = filtered_points[:, 0]
            out_las.y = filtered_points[:, 1]
            out_las.z = filtered_points[:, 2]

            extra_params = [
                laspy.ExtraBytesParams(name=instance_dimension, type=np.int32)
            ]
            for dim_name, arr in filtered_extras.items():
                extra_params.append(
                    laspy.ExtraBytesParams(name=dim_name, type=arr.dtype)
                )
            out_las.add_extra_dims(extra_params)

            setattr(out_las, instance_dimension, filtered_instances)
            for dim_name, arr in filtered_extras.items():
                setattr(out_las, dim_name, arr)

            out_las.write(str(output_path))

        except Exception as e:
            print(f"    Error processing {result.tile_name}: {e}")

    print(f"  Saved {len(tile_results)} filtered tiles")


# =============================================================================
# Filter Task: Buffer-Only Instance Removal (no merging)
# =============================================================================


def write_filtered_tiles_streaming(
    tile_results: List[TileMetadataResult],
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]],
    core_bounds_by_tile: Dict[str, Tuple[float, float, float, float]],
    output_dir: Path,
    instance_dimension: str = "PredInstance",
    chunk_size: int = 1_000_000,
    filter_anchor: str = "centroid",
    global_to_merged: Optional[Dict[int, int]] = None,
) -> None:
    """Write per-tile LAZ files with buffer-region instances removed.

    Filters instances whose anchor point falls in the buffer zone on sides that
    have a neighboring tile. Kept instances are written with all of their
    original points intact; they are not geometrically clipped to the core
    region. No cross-tile matching is performed. Streams output
    chunk-by-chunk — peak RAM per tile ≈ one chunk.

    Args:
        tile_results:        Phase-A metadata results (one per tile).
        neighbors_by_tile:   direction → neighbor name for each tile.
        core_bounds_by_tile: (min_x, max_x, min_y, max_y) core region per tile.
        output_dir:          Directory to write filtered LAZ files.
        instance_dimension:  Name of the instance-ID extra dim (default PredInstance).
        chunk_size:          Points per streaming chunk.
        filter_anchor:       Which point of each instance is checked against the buffer:
                             "centroid" (default), "highest_point", or "lowest_point".
        global_to_merged:    If provided (from Phase B.5), remap local instance IDs to
                             merged IDs before writing.  global_id = tile_idx*TILE_OFFSET+local_id.
    """
    from merge_tiles import extra_bytes_params_from_dimension_info

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print("Filter Task: Writing filtered tiles (buffer-only removal)")
    print(f"{'=' * 60}")

    n_tiles = len(tile_results)
    total_removed_pts = 0

    for ti, result in enumerate(tile_results):
        tile_name = result.tile_name
        filepath = result.filepath
        neighbors = neighbors_by_tile.get(
            tile_name, {"east": None, "west": None, "north": None, "south": None}
        )
        core_boundary = core_bounds_by_tile.get(tile_name)

        # Determine which LOCAL instance IDs to remove based on anchor point.
        ids_to_remove: Set[int] = set()
        if core_boundary is not None:
            for meta in result.instances.values():
                if filter_anchor == "highest_point":
                    anchor_xy = meta.highpt_xy
                elif filter_anchor == "lowest_point":
                    anchor_xy = meta.lowpt_xy
                else:  # "centroid" (default)
                    anchor_xy = meta.centroid[:2]

                if anchor_xy is not None and _in_buffer_region(
                    (float(anchor_xy[0]), float(anchor_xy[1])),
                    neighbors, core_boundary,
                ):
                    ids_to_remove.add(meta.local_id)

        suffix = "".join(filepath.suffixes) or ".laz"
        output_path = output_dir / f"{tile_name}{suffix}"
        print(
            f"  [{ti + 1}/{n_tiles}] {filepath.name}: "
            f"{len(result.instances)} instances, {len(ids_to_remove)} to remove",
            flush=True,
        )

        try:
            with laspy.open(str(filepath), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
                in_header = reader.header
                in_pf = in_header.point_format

                # Build output header mirroring input (preserve format, scales, extra dims)
                out_header = laspy.LasHeader(
                    point_format=in_pf.id,
                    version=in_header.version,
                )
                out_header.offsets = in_header.offsets
                out_header.scales = in_header.scales
                for dim in in_pf.extra_dimensions:
                    out_header.add_extra_dims([extra_bytes_params_from_dimension_info(dim)])

                has_instance_dim = instance_dimension in {
                    d.name for d in in_pf.extra_dimensions
                }
                n_in = in_header.point_count
                n_out = 0

                with laspy.open(
                    str(output_path), mode="w", header=out_header,
                    do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel,
                ) as writer:
                    for chunk in reader.chunk_iterator(chunk_size):
                        # 1. Remove instances whose centroid/highest-point is in buffer
                        if ids_to_remove and has_instance_dim:
                            instance_arr = np.asarray(
                                getattr(chunk, instance_dimension), dtype=np.int32
                            )
                            keep_mask = ~np.isin(instance_arr, list(ids_to_remove))
                        else:
                            keep_mask = np.ones(len(chunk), dtype=bool)

                        if keep_mask.any():
                            out_chunk = chunk[keep_mask]
                            # 2. Remap instance IDs via global_to_merged (Phase B.5 result)
                            if global_to_merged is not None and has_instance_dim:
                                local_ids = np.asarray(
                                    getattr(out_chunk, instance_dimension), dtype=np.int32
                                )
                                remapped = np.array([
                                    global_to_merged.get(
                                        result.tile_idx * TILE_OFFSET + int(lid), int(lid)
                                    )
                                    for lid in local_ids
                                ], dtype=np.int32)
                                setattr(out_chunk, instance_dimension, remapped)
                            writer.write_points(out_chunk)
                            n_out += int(keep_mask.sum())

            removed_pts = n_in - n_out
            total_removed_pts += removed_pts
            print(
                f"    {n_in:,} pts in → {n_out:,} pts out "
                f"({removed_pts:,} pts removed)",
                flush=True,
            )

        except Exception as e:
            print(f"    Error processing {tile_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(
        f"\n  Done: {n_tiles} tiles written to {output_dir} "
        f"({total_removed_pts:,} pts removed total)",
        flush=True,
    )


# =============================================================================
# Main Streaming Merge Orchestrator
# =============================================================================


def merge_tiles_streaming(
    input_dir: Path,
    original_tiles_dir: Path,
    output_merged: Path,
    output_tiles_dir: Path,
    tile_bounds_json: Path,
    original_input_dir: Optional[Path] = None,
    overlap_threshold: float = 0.5,
    correspondence_tolerance: float = 0.05,
    max_volume_for_merge: float = 5.0,
    border_zone_width: float = 10.0,
    min_cluster_size: int = 300,
    num_threads: int = 8,
    enable_matching: bool = True,
    enable_volume_merge: bool = True,
    skip_merged_file: bool = False,
    verbose: bool = False,
    retile_buffer: float = 2.0,
    retile_max_radius: float = 2.0,
    instance_dimension: str = "PredInstance",
    transfer_original_dims_to_merged: bool = True,
    threedtrees_dims: Optional[List[str]] = None,
    threedtrees_suffix: str = "SAT",
    save_filtered_tiles_flag: bool = False,
    enable_orphan_recovery: bool = True,
    standardization_json: Optional[Path] = None,
    merge_chunk_size: int = 2_000_000,
):
    """
    Memory-efficient streaming merge pipeline.

    Same interface as merge_tiles() but processes tiles one at a time.
    """
    from merge_tiles import (
        retile_to_original_files_streaming,
        remap_to_original_input_files_streaming,
        add_original_dimensions_to_merged,
        load_standardization_dims,
    )

    if not tile_bounds_json.exists():
        raise FileNotFoundError(f"tile_bounds_tindex.json not found: {tile_bounds_json}")

    print("=" * 60)
    print("3DTrees Streaming Merge Pipeline (Memory-Efficient)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output merged: {output_merged}" + (" (SKIPPED)" if skip_merged_file else ""))
    print(f"Output tiles: {output_tiles_dir}")
    print(f"Border zone width: {border_zone_width}m")
    print(f"Instance matching: {'ENABLED' if enable_matching else 'DISABLED'}")
    print(f"Volume merge: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    print(f"Workers: {num_threads}")
    print("=" * 60)

    _target_dims = None
    if standardization_json is not None:
        _target_dims = load_standardization_dims(standardization_json)

    # Check if merged output already exists -> skip to retiling (streaming)
    if output_merged.exists():
        print(f"\nMerged file already exists: {output_merged}")
        merged_for_downstream = output_merged

        retile_to_original_files_streaming(
            merged_file=merged_for_downstream,
            original_tiles_dir=original_tiles_dir,
            output_dir=output_tiles_dir,
            tolerance=retile_max_radius,
            retile_buffer=retile_buffer,
            instance_dimension=instance_dimension,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            chunk_size=merge_chunk_size,
        )

        if original_input_dir is not None:
            remap_to_original_input_files_streaming(
                merged_file=merged_for_downstream,
                original_input_dir=original_input_dir,
                output_dir=output_tiles_dir.parent / "original_with_predictions",
                tolerance=retile_max_radius,
                retile_buffer=retile_buffer,
                threedtrees_dims=threedtrees_dims,
                threedtrees_suffix=threedtrees_suffix,
                target_dims=_target_dims,
                chunk_size=merge_chunk_size,
            )
        return

    # Find input files
    laz_files = sorted(input_dir.glob("*.laz"))
    if not laz_files:
        laz_files = sorted(input_dir.glob("*.las"))
    if not laz_files:
        print(f"No LAZ/LAS files found in {input_dir}")
        return

    print(f"\nFound {len(laz_files)} tiles to merge")

    # Build neighbor graph from JSON
    print("  Loading neighbor graph from tile_bounds_tindex.json...")
    with tile_bounds_json.open() as f:
        json_data = json.load(f)
    json_labels = [
        f"c{int(tile['col']):02d}_r{int(tile['row']):02d}"
        if "col" in tile and "row" in tile else None
        for tile in json_data.get("tiles", [])
    ]
    bounds_field = _preferred_json_bounds_field(json_data.get("tiles", []))
    json_bounds, centers, neighbors_idx = build_neighbor_graph_from_bounds_json(
        tile_bounds_json,
        bounds_field=bounds_field,
    )

    # Extract core bounds from JSON (for accurate ground overlap removal)
    core_bounds_by_tile: Dict[str, Tuple[float, float, float, float]] = {}

    # Extract tile bounds from headers
    file_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    file_key_to_path: Dict[str, Path] = {}
    for f in laz_files:
        bounds = get_tile_bounds_from_header(f)
        if bounds:
            file_key = str(f)
            file_boundaries[file_key] = bounds
            file_key_to_path[file_key] = f

    tile_to_json, json_to_tile = _match_tiles_to_json_bounds(
        file_boundaries, json_bounds, centers, json_labels=json_labels
    )

    tile_name_by_key = {
        file_key: _canonical_tile_name_for_json_index(json_idx, json_labels)
        for file_key, json_idx in tile_to_json.items()
    }
    tile_name_by_path = {
        file_key_to_path[file_key]: tile_name
        for file_key, tile_name in tile_name_by_key.items()
        if file_key in file_key_to_path
    }
    json_idx_to_tile_name = {
        json_idx: tile_name_by_key[file_key]
        for file_key, json_idx in tile_to_json.items()
    }

    tile_boundaries: Dict[str, Tuple[float, float, float, float]] = {}
    for json_idx, tile_name in json_idx_to_tile_name.items():
        tile_boundaries[tile_name] = json_bounds[json_idx]

    # Parse core bounds from the JSON for accurate ground overlap removal
    _json_tiles = json_data.get("tiles", [])
    for json_idx, tile_name in json_idx_to_tile_name.items():
        if json_idx < len(_json_tiles) and "core" in _json_tiles[json_idx]:
            core = _json_tiles[json_idx]["core"]
            core_bounds_by_tile[tile_name] = (
                float(core[0][0]), float(core[0][1]),  # min_x, max_x
                float(core[1][0]), float(core[1][1]),  # min_y, max_y
            )

    # Build per-tile neighbor mapping
    neighbors_by_tile: Dict[str, Dict[str, Optional[str]]] = {}
    for json_idx, tile_name in json_idx_to_tile_name.items():
        nbrs = {"east": None, "west": None, "north": None, "south": None}
        for direction in ("east", "west", "north", "south"):
            n_idx = neighbors_idx[json_idx].get(direction)
            if n_idx is not None:
                nbrs[direction] = json_idx_to_tile_name.get(n_idx)
        neighbors_by_tile[tile_name] = nbrs

    # =========================================================================
    # Phase A: Metadata Extraction (parallel)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase A: Metadata Extraction ({num_threads} workers)")
    print(f"{'=' * 60}")

    tile_results: List[TileMetadataResult] = []

    # Build args for parallel extraction
    metadata_args = []
    for tile_idx, filepath in enumerate(laz_files):
        tile_name = tile_name_by_path.get(filepath)
        if tile_name is None:
            print(f"  Warning: no matched bounds entry for {filepath}, skipping")
            continue
        metadata_args.append((
            filepath,
            tile_idx,
            tile_boundaries,
            tile_name,
            neighbors_by_tile.get(tile_name),
            border_zone_width,
            correspondence_tolerance,
            instance_dimension,
            core_bounds_by_tile.get(tile_name),
            merge_chunk_size,
        ))

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(_extract_tile_metadata_wrapper, metadata_args))

    for result in results:
        if result is None:
            continue
        tile_results.append(result)

    total_instances = sum(
        len([m for m in r.instances.values() if not m.is_filtered])
        for r in tile_results
    )
    total_filtered = sum(
        len([m for m in r.instances.values() if m.is_filtered])
        for r in tile_results
    )
    print(f"\n  Phase A complete: {len(tile_results)} tiles, "
          f"{total_instances} kept instances, {total_filtered} filtered")

    # Optional: save filtered tiles
    if save_filtered_tiles_flag:
        filtered_dir = output_tiles_dir / "filtered_tiles"
        save_filtered_tiles(tile_results, filtered_dir, instance_dimension, chunk_size=merge_chunk_size)

    # =========================================================================
    # Phase B: Global Instance Matching + Orphan Recovery
    # =========================================================================
    if enable_matching:
        uf, global_to_merged = global_instance_matching(
            tile_results,
            neighbors_by_tile,
            core_bounds_by_tile,
            overlap_threshold=overlap_threshold,
            verbose=verbose,
        )
    else:
        print(f"\n{'=' * 60}")
        print("Phase B: Cross-Tile Matching (DISABLED)")
        print(f"{'=' * 60}")
        uf = UnionFind()
        global_to_merged = {}
        for result in tile_results:
            for gid, meta in result.instances.items():
                if not meta.is_filtered:
                    uf.make_set(gid, meta.point_count)
        components = uf.get_components()
        for merged_id, (root, members) in enumerate(components.items(), start=1):
            for gid in members:
                global_to_merged[gid] = merged_id

    # Small instance redistribution (before Phase C so remap is complete)
    if enable_volume_merge:
        redistribute_small_instances(
            tile_results=tile_results,
            global_to_merged=global_to_merged,
            min_points_reassign=500,
            min_points_hull=300,
            max_points_hull=2000,
            max_volume_for_merge=max_volume_for_merge,
            max_search_radius=10.0,
            instance_dimension=instance_dimension,
            verbose=verbose,
            chunk_size=max(100_000, merge_chunk_size // 4),
        )

    # =========================================================================
    # Phase C: Stream-write merged file
    # =========================================================================
    merged_file_for_downstream = output_merged

    if not skip_merged_file:
        (merged_path, inst_centroids, inst_counts,
         inst_bbox_min, inst_bbox_max) = write_merged_streaming(
            tile_results=tile_results,
            global_to_merged=global_to_merged,
            neighbors_by_tile=neighbors_by_tile,
            output_merged=output_merged,
            core_bounds_by_tile=core_bounds_by_tile,
            correspondence_tolerance=correspondence_tolerance,
            instance_dimension=instance_dimension,
            verbose=verbose,
            chunk_size=merge_chunk_size,
        )

        streaming_renumber_instances(
            merged_file=output_merged,
            instance_centroids_xy=inst_centroids,
            instance_dimension=instance_dimension,
            verbose=verbose,
            chunk_size=merge_chunk_size,
        )
        merged_file_for_downstream = output_merged

        # Enrich merged file with original dimensions if requested
        if original_input_dir is not None and transfer_original_dims_to_merged:
            try:
                # Compute target dims from standardization JSON if provided
                if _target_dims is not None:
                    print(f"  Standardization: filtering to {len(_target_dims)} dims from {standardization_json.name}")

                enriched = output_merged.parent / (output_merged.stem + "_enriched.laz")
                add_original_dimensions_to_merged(
                    merged_file_for_downstream,
                    original_input_dir,
                    enriched,
                    tolerance=0.1,
                    retile_buffer=retile_buffer,
                    num_threads=num_threads,
                    target_dims=_target_dims,
                    merge_chunk_size=merge_chunk_size,
                )
                shutil.move(str(enriched), str(output_merged))
                print(f"  Enriched merged file with original-file dimensions")
            except Exception as e:
                print(f"  Warning: Could not add original dimensions: {e}")
                try:
                    if enriched.exists():
                        enriched.unlink()
                except OSError:
                    pass

    else:
        print(f"\n  Merged file creation SKIPPED (--skip-merged-file)")
        # Still need to write merged for retiling - create a temporary one
        merged_path, *_ = write_merged_streaming(
            tile_results=tile_results,
            global_to_merged=global_to_merged,
            neighbors_by_tile=neighbors_by_tile,
            output_merged=output_merged.parent / "merged_temp.laz",
            core_bounds_by_tile=core_bounds_by_tile,
            correspondence_tolerance=correspondence_tolerance,
            instance_dimension=instance_dimension,
            verbose=verbose,
            chunk_size=merge_chunk_size,
        )
        merged_file_for_downstream = merged_path

    # =========================================================================
    # Stage 6: Retile to original files (streaming — no full merged load)
    # =========================================================================
    retile_to_original_files_streaming(
        merged_file=merged_file_for_downstream,
        original_tiles_dir=original_tiles_dir,
        output_dir=output_tiles_dir,
        tolerance=retile_max_radius,
        retile_buffer=retile_buffer,
        instance_dimension=instance_dimension,
        threedtrees_dims=threedtrees_dims,
        threedtrees_suffix=threedtrees_suffix,
        chunk_size=merge_chunk_size,
    )

    # =========================================================================
    # Stage 7: Remap to original input files (streaming, optional)
    # =========================================================================
    if original_input_dir is not None:
        remap_to_original_input_files_streaming(
            merged_file=merged_file_for_downstream,
            original_input_dir=original_input_dir,
            output_dir=output_tiles_dir.parent / "original_with_predictions",
            tolerance=retile_max_radius,
            retile_buffer=retile_buffer,
            threedtrees_dims=threedtrees_dims,
            threedtrees_suffix=threedtrees_suffix,
            target_dims=_target_dims,
            chunk_size=merge_chunk_size,
        )

    # Cleanup temp merged file if skip_merged_file
    if skip_merged_file:
        temp_merged = output_merged.parent / "merged_temp.laz"
        if temp_merged.exists():
            temp_merged.unlink()
    gc.collect()

    print(f"\n{'=' * 60}")
    print("Streaming merge complete!")
    print(f"{'=' * 60}")
