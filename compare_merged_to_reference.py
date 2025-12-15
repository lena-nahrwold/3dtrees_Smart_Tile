#!/usr/bin/env python3
"""
Compare merged predictions file against a reference file, only for the overlapping extent.

This script:
1. Loads both the merged file and reference file
2. Computes the overlapping extent (intersection of bounding boxes)
3. Filters both point clouds to only the overlapping region
4. Compares instance counts, point counts, and other metrics
"""

import argparse
import numpy as np
import laspy
from pathlib import Path
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: geopandas not available. GeoPackage files cannot be read.")


def get_bounds(points: np.ndarray) -> tuple:
    """Get bounding box of points: (min_x, max_x, min_y, max_y, min_z, max_z)."""
    return (
        np.min(points[:, 0]), np.max(points[:, 0]),
        np.min(points[:, 1]), np.max(points[:, 1]),
        np.min(points[:, 2]), np.max(points[:, 2])
    )


def get_overlap_bounds(bounds1: tuple, bounds2: tuple) -> tuple:
    """
    Compute overlapping bounding box.
    
    Returns:
        (min_x, max_x, min_y, max_y, min_z, max_z) of overlap, or None if no overlap
    """
    min_x1, max_x1, min_y1, max_y1, min_z1, max_z1 = bounds1
    min_x2, max_x2, min_y2, max_y2, min_z2, max_z2 = bounds2
    
    overlap_min_x = max(min_x1, min_x2)
    overlap_max_x = min(max_x1, max_x2)
    overlap_min_y = max(min_y1, min_y2)
    overlap_max_y = min(max_y1, max_y2)
    overlap_min_z = max(min_z1, min_z2)
    overlap_max_z = min(max_z1, max_z2)
    
    # Check if there's actual overlap
    if overlap_min_x >= overlap_max_x or overlap_min_y >= overlap_max_y or overlap_min_z >= overlap_max_z:
        return None
    
    return (overlap_min_x, overlap_max_x, overlap_min_y, overlap_max_y, overlap_min_z, overlap_max_z)


def filter_to_bounds(points: np.ndarray, instances: np.ndarray, species_ids: np.ndarray, bounds: tuple) -> tuple:
    """
    Filter points to only those within the specified bounds.
    
    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        species_ids: Array of species IDs
        bounds: (min_x, max_x, min_y, max_y, min_z, max_z)
    
    Returns:
        Filtered (points, instances, species_ids)
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    
    mask = (
        (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
        (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
        (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )
    
    return points[mask], instances[mask], species_ids[mask]


def load_reference_file(reference_file: str):
    """
    Load reference file - supports both LAZ and GeoPackage formats.
    
    Returns:
        (points, instances, species_ids, bounds)
        instances and species_ids may be None if not available
    """
    ref_path = Path(reference_file)
    
    if ref_path.suffix.lower() == '.gpkg':
        if not HAS_GEOPANDAS:
            raise ImportError("geopandas is required to read GeoPackage files. Install with: pip install geopandas")
        
        print(f"  Reading GeoPackage file...")
        gdf = gpd.read_file(reference_file)
        
        # Extract points from geometries
        # Handle both Point and Polygon geometries
        points_list = []
        instance_ids = []
        species_ids = []
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            
            # Get instance ID (try common column names)
            inst_id = None
            for col in ['tree_id', 'id', 'treeID', 'instance_id', 'PredInstance']:
                if col in gdf.columns:
                    inst_id = row[col]
                    break
            if inst_id is None:
                inst_id = idx + 1  # Use row index as ID
            
            # Get species ID if available
            species_id = None
            for col in ['species_id', 'species', 'speciesID', 'Species']:
                if col in gdf.columns:
                    species_id = row[col]
                    break
            
            # Extract points from geometry
            if geom.geom_type == 'Point':
                # Single point
                z_coord = geom.z if hasattr(geom, 'z') and geom.z is not None else 0.0
                points_list.append([geom.x, geom.y, z_coord])
                instance_ids.append(inst_id)
                species_ids.append(species_id if species_id is not None else 0)
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                # Extract boundary points from polygon
                if geom.geom_type == 'Polygon':
                    coords = list(geom.exterior.coords)
                else:  # MultiPolygon
                    coords = []
                    for poly in geom.geoms:
                        coords.extend(list(poly.exterior.coords))
                
                for coord in coords:
                    x, y = coord[0], coord[1]
                    z = coord[2] if len(coord) > 2 else 0.0
                    points_list.append([x, y, z])
                    instance_ids.append(inst_id)
                    species_ids.append(species_id if species_id is not None else 0)
        
        if len(points_list) == 0:
            raise ValueError("No valid geometries found in GeoPackage file")
        
        ref_points = np.array(points_list)
        ref_instances = np.array(instance_ids, dtype=np.int32)
        # Check if we have any non-zero species IDs
        has_species = any(s is not None and s != 0 for s in species_ids) if species_ids else False
        ref_species = np.array(species_ids, dtype=np.int32) if has_species else None
        
        print(f"  Extracted {len(ref_points)} points from {len(gdf)} geometries")
        print(f"  Columns available: {list(gdf.columns)}")
        if ref_instances is not None:
            print(f"  Unique instances: {len(np.unique(ref_instances))}")
        
        return ref_points, ref_instances, ref_species
    
    else:
        # Assume LAZ file
        print(f"  Reading LAZ file...")
        ref_las = laspy.read(reference_file, laz_backend=laspy.LazBackend.LazrsParallel)
        ref_points = np.vstack((ref_las.x, ref_las.y, ref_las.z)).T
        ref_instances = np.array(ref_las.PredInstance) if hasattr(ref_las, 'PredInstance') else None
        ref_species = np.array(ref_las.species_id) if hasattr(ref_las, 'species_id') else None
        
        return ref_points, ref_instances, ref_species


def compare_merged_to_reference(
    merged_file: str,
    reference_file: str,
    output_report: str = None,
):
    """
    Compare merged predictions against reference file for overlapping extent.
    
    Args:
        merged_file: Path to merged LAZ file with predictions
        reference_file: Path to reference file (LAZ or GeoPackage)
        output_report: Optional path to save comparison report
    """
    print("=" * 60)
    print("Compare Merged to Reference (Overlapping Extent Only)")
    print("=" * 60)
    
    # Load merged file
    print(f"\nLoading merged file: {merged_file}")
    merged_las = laspy.read(merged_file, laz_backend=laspy.LazBackend.LazrsParallel)
    merged_points = np.vstack((merged_las.x, merged_las.y, merged_las.z)).T
    merged_instances = np.array(merged_las.PredInstance)
    merged_species = np.array(merged_las.species_id) if hasattr(merged_las, 'species_id') else None
    
    print(f"  Merged file: {len(merged_points)} points")
    merged_bounds = get_bounds(merged_points)
    print(f"  Merged bounds: X=[{merged_bounds[0]:.2f}, {merged_bounds[1]:.2f}], "
          f"Y=[{merged_bounds[2]:.2f}, {merged_bounds[3]:.2f}], "
          f"Z=[{merged_bounds[4]:.2f}, {merged_bounds[5]:.2f}]")
    
    # Load reference file
    print(f"\nLoading reference file: {reference_file}")
    ref_points, ref_instances, ref_species = load_reference_file(reference_file)
    
    print(f"  Reference file: {len(ref_points)} points")
    ref_bounds = get_bounds(ref_points)
    print(f"  Reference bounds: X=[{ref_bounds[0]:.2f}, {ref_bounds[1]:.2f}], "
          f"Y=[{ref_bounds[2]:.2f}, {ref_bounds[3]:.2f}], "
          f"Z=[{ref_bounds[4]:.2f}, {ref_bounds[5]:.2f}]")
    
    # Compute overlapping extent
    print("\n--- Computing Overlapping Extent ---")
    overlap_bounds = get_overlap_bounds(merged_bounds, ref_bounds)
    
    if overlap_bounds is None:
        print("ERROR: No overlap between merged and reference files!")
        return
    
    min_x, max_x, min_y, max_y, min_z, max_z = overlap_bounds
    overlap_area = (max_x - min_x) * (max_y - min_y)
    overlap_volume = overlap_area * (max_z - min_z)
    
    print(f"  Overlap bounds: X=[{min_x:.2f}, {max_x:.2f}], "
          f"Y=[{min_y:.2f}, {max_y:.2f}], "
          f"Z=[{min_z:.2f}, {max_z:.2f}]")
    print(f"  Overlap area: {overlap_area:.2f} m²")
    print(f"  Overlap volume: {overlap_volume:.2f} m³")
    
    # Filter both point clouds to overlap region
    print("\n--- Filtering to Overlap Region ---")
    merged_points_overlap, merged_instances_overlap, merged_species_overlap = filter_to_bounds(
        merged_points, merged_instances, merged_species if merged_species is not None else np.zeros(len(merged_points), dtype=np.int32),
        overlap_bounds
    )
    
    if ref_instances is not None:
        ref_points_overlap, ref_instances_overlap, ref_species_overlap = filter_to_bounds(
            ref_points, ref_instances, ref_species if ref_species is not None else np.zeros(len(ref_points), dtype=np.int32),
            overlap_bounds
        )
    else:
        ref_points_overlap = filter_to_bounds(
            ref_points, np.zeros(len(ref_points), dtype=np.int32), np.zeros(len(ref_points), dtype=np.int32),
            overlap_bounds
        )[0]
        ref_instances_overlap = None
    
    print(f"  Merged points in overlap: {len(merged_points_overlap)} ({100*len(merged_points_overlap)/len(merged_points):.1f}% of total)")
    print(f"  Reference points in overlap: {len(ref_points_overlap)} ({100*len(ref_points_overlap)/len(ref_points):.1f}% of total)")
    
    # Compare metrics
    print("\n--- Comparison Metrics (Overlap Region Only) ---")
    
    # Point counts
    print(f"\nPoint Counts:")
    print(f"  Merged: {len(merged_points_overlap):,} points")
    print(f"  Reference: {len(ref_points_overlap):,} points")
    if len(ref_points_overlap) > 0:
        point_diff = len(merged_points_overlap) - len(ref_points_overlap)
        point_diff_pct = 100 * point_diff / len(ref_points_overlap)
        print(f"  Difference: {point_diff:+,} points ({point_diff_pct:+.1f}%)")
    
    # Instance counts
    if ref_instances_overlap is not None:
        merged_unique_instances = np.unique(merged_instances_overlap)
        merged_tree_instances = merged_unique_instances[merged_unique_instances > 0]
        
        ref_unique_instances = np.unique(ref_instances_overlap)
        ref_tree_instances = ref_unique_instances[ref_unique_instances > 0]
        
        print(f"\nInstance Counts:")
        print(f"  Merged trees: {len(merged_tree_instances)}")
        print(f"  Reference trees: {len(ref_tree_instances)}")
        instance_diff = len(merged_tree_instances) - len(ref_tree_instances)
        if len(ref_tree_instances) > 0:
            instance_diff_pct = 100 * instance_diff / len(ref_tree_instances)
            print(f"  Difference: {instance_diff:+,} trees ({instance_diff_pct:+.1f}%)")
        
        # Points per instance statistics
        print(f"\nPoints per Instance (Trees Only):")
        merged_points_per_instance = []
        for inst_id in merged_tree_instances:
            count = np.sum(merged_instances_overlap == inst_id)
            merged_points_per_instance.append(count)
        
        ref_points_per_instance = []
        for inst_id in ref_tree_instances:
            count = np.sum(ref_instances_overlap == inst_id)
            ref_points_per_instance.append(count)
        
        if merged_points_per_instance:
            print(f"  Merged - Mean: {np.mean(merged_points_per_instance):.0f}, "
                  f"Median: {np.median(merged_points_per_instance):.0f}, "
                  f"Min: {np.min(merged_points_per_instance)}, "
                  f"Max: {np.max(merged_points_per_instance)}")
        
        if ref_points_per_instance:
            print(f"  Reference - Mean: {np.mean(ref_points_per_instance):.0f}, "
                  f"Median: {np.median(ref_points_per_instance):.0f}, "
                  f"Min: {np.min(ref_points_per_instance)}, "
                  f"Max: {np.max(ref_points_per_instance)}")
    else:
        print("\nNote: Reference file does not contain PredInstance attribute")
    
    # Density comparison
    print(f"\nPoint Density:")
    merged_density = len(merged_points_overlap) / overlap_area if overlap_area > 0 else 0
    ref_density = len(ref_points_overlap) / overlap_area if overlap_area > 0 else 0
    print(f"  Merged: {merged_density:.2f} points/m²")
    print(f"  Reference: {ref_density:.2f} points/m²")
    if ref_density > 0:
        density_diff_pct = 100 * (merged_density - ref_density) / ref_density
        print(f"  Difference: {density_diff_pct:+.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"Overlap region: {overlap_area:.2f} m²")
    print(f"Merged points in overlap: {len(merged_points_overlap):,}")
    print(f"Reference points in overlap: {len(ref_points_overlap):,}")
    if ref_instances_overlap is not None:
        print(f"Merged trees in overlap: {len(merged_tree_instances)}")
        print(f"Reference trees in overlap: {len(ref_tree_instances)}")
    print("=" * 60)
    
    # Save report if requested
    if output_report:
        with open(output_report, 'w') as f:
            f.write("Comparison Report: Merged vs Reference (Overlapping Extent)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Merged file: {merged_file}\n")
            f.write(f"Reference file: {reference_file}\n\n")
            f.write(f"Overlap bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}], Z=[{min_z:.2f}, {max_z:.2f}]\n")
            f.write(f"Overlap area: {overlap_area:.2f} m²\n\n")
            f.write(f"Merged points: {len(merged_points_overlap):,}\n")
            f.write(f"Reference points: {len(ref_points_overlap):,}\n")
            if ref_instances_overlap is not None:
                f.write(f"Merged trees: {len(merged_tree_instances)}\n")
                f.write(f"Reference trees: {len(ref_tree_instances)}\n")
        print(f"\nReport saved to: {output_report}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare merged predictions file against reference file (overlapping extent only)"
    )
    parser.add_argument(
        "--merged_file",
        type=str,
        required=True,
        help="Path to merged LAZ file with predictions"
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        required=True,
        help="Path to reference LAZ file"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Optional path to save comparison report text file"
    )
    
    args = parser.parse_args()
    
    compare_merged_to_reference(
        merged_file=args.merged_file,
        reference_file=args.reference_file,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()

