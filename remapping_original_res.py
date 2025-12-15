import argparse
import numpy as np
from scipy.spatial import KDTree
import laspy
import os


def main(original_file, subsampled_file, output_file, num_threads=-1):
    """
    Remap predicted attributes from a subsampled point cloud back to the original resolution point cloud.

    This function takes an original point cloud file and a subsampled point cloud file (both LAS/LAZ),
    and for each point in the original, finds its nearest neighbor in the subsampled cloud. It then
    copies the predicted attributes (e.g., PredInstance, PredSemantic) from the subsampled cloud to the
    original cloud, creating new attributes if necessary. The updated original point cloud is saved to
    the specified output file.

    Args:
        original_file (str): Path to the original resolution LAS/LAZ file.
        subsampled_file (str): Path to the subsampled LAS/LAZ file with predicted attributes.
        output_file (str): Path to save the updated original LAS/LAZ file.
        num_threads (int): Number of threads for KDTree queries (-1 for all CPUs).
    """
    print(f"Using {num_threads} threads for processing" + (" (all CPUs)" if num_threads == -1 else ""))
    
    # Load the original point cloud with parallel LAZ reading
    original_las = laspy.read(original_file, laz_backend=laspy.LazBackend.LazrsParallel)
    original_points = np.vstack((original_las.x, original_las.y, original_las.z)).T

    # Load the subsampled point cloud with parallel LAZ reading
    subsampled_las = laspy.read(
        subsampled_file, laz_backend=laspy.LazBackend.LazrsParallel
    )
    subsampled_points = np.vstack(
        (subsampled_las.x, subsampled_las.y, subsampled_las.z)
    ).T

    print(f"Subsampled point cloud has {subsampled_las.header.point_count} points")
    print(f"Original point cloud has {original_las.header.point_count} points")

    # Create new attributes in the original point cloud if they don't exist
    extra_dims = original_las.point_format.extra_dimensions
    existing_dims = {dim.name for dim in extra_dims}

    if "PredInstance" not in existing_dims:
        original_las.add_extra_dim(
            laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
        )
    if "PredSemantic" not in existing_dims:
        original_las.add_extra_dim(
            laspy.ExtraBytesParams(name="PredSemantic", type=np.int32)
        )
    # if "species_id_x" not in existing_dims:
    #     original_las.add_extra_dim(laspy.ExtraBytesParams(name="species_id_x", type=np.int32))

    # Create a KD-tree for the subsampled point cloud
    print("Building KD-tree for nearest neighbor search...")
    tree = KDTree(subsampled_points)

    # Find the nearest neighbors in the subsampled point cloud for each point in the original point cloud
    print(f"Querying nearest neighbors with {num_threads} workers...")
    distances, indices = tree.query(original_points, workers=num_threads)

    # Map attributes from the subsampled point cloud to the original point cloud
    print("Mapping attributes to original point cloud...")
    original_las.PredInstance = subsampled_las.PredInstance[indices]
    # original_las.PredSemantic = subsampled_las.PredSemantic[indices]
    # original_las.species_id_x = subsampled_las.species_id_x[indices]

    # print a summary of the original point cloud wether it now has every attribute
    print("Original point cloud now has the following attributes:")
    for dim in original_las.point_format.extra_dimensions:
        print(f"{dim.name}: {dim.dtype}")

    # Save the updated original point cloud with LAZ compression (parallel)
    print("Writing compressed LAZ file with parallel compression...")
    with open(output_file, "wb") as f:
        original_las.write(
            f, do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel
        )
        f.flush()
        os.fsync(f.fileno())
    print(f"Updated point cloud saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map attributes from a subsampled point cloud back to the original point cloud."
    )
    parser.add_argument(
        "--original_file", type=str, help="Path to the original point cloud file"
    )
    parser.add_argument(
        "--subsampled_file",
        type=str,
        help="Path to the subsampled point cloud file with attributes",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the updated original point cloud file",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=-1,
        help="Number of threads for KDTree queries (-1 for all CPUs, default: -1)",
    )
    args = parser.parse_args()

    main(args.original_file, args.subsampled_file, args.output_file, args.num_threads)
