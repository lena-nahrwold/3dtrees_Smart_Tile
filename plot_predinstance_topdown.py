#!/usr/bin/env python
"""
Quick top-down visualization of PredInstance in a large LAZ file.

Usage
-----
From repo root:

    python tools/tool_smart_tile/plot_predinstance_topdown.py \
        /home/kg281/data/gfz/tiled_10m/merged/merged.laz \
        /home/kg281/data/gfz/tiled_10m/merged/merged_predinstance_topdown.png

The script:
  - streams the input with laspy.chunk_iterator
  - randomly subsamples up to a configurable maximum number of points
  - plots x/y from above, colored by PredInstance
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import laspy
import matplotlib.pyplot as plt
import numpy as np


def load_subsampled_points(
    path: Path,
    instance_dim: str = "PredInstance",
    max_points: int = 2_000_000,
    chunk_size: int = 5_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream the file and subsample points for plotting.

    Returns
    -------
    xs, ys, inst_ids : np.ndarray
        Arrays of equal length containing coordinates and instance IDs.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    # Open with chunked reading to avoid loading everything into RAM.
    with laspy.open(path) as reader:
        # LasReader exposes point_format via the header.
        dim_names = reader.header.point_format.dimension_names
        has_dim = instance_dim in dim_names
        if not has_dim:
            raise ValueError(
                f"Dimension '{instance_dim}' not found in file. "
                f"Available: {sorted(dim_names)}"
            )

        total_points = reader.header.point_count
        if total_points == 0:
            raise ValueError("Input file contains zero points.")

        # Target sampling rate relative to total points.
        keep_ratio = min(1.0, max_points / float(total_points))
        if keep_ratio <= 0.0:
            keep_ratio = max_points / float(total_points)

        xs_list: list[np.ndarray] = []
        ys_list: list[np.ndarray] = []
        inst_list: list[np.ndarray] = []

        # Derive a reasonable number of iterations based on chunk_size.
        # laspy's chunk_iterator takes "points_per_iteration".
        points_per_iter = max(1, min(chunk_size, total_points))

        for chunk in reader.chunk_iterator(points_per_iter):
            # Coordinates as float (already scaled by laspy).
            cx = np.asarray(chunk.x)
            cy = np.asarray(chunk.y)
            inst = np.asarray(getattr(chunk, instance_dim))

            n = len(cx)
            if n == 0:
                continue

            # Randomly subsample this chunk according to keep_ratio.
            if keep_ratio < 1.0:
                # Probability-based subsampling.
                mask = np.random.rand(n) < keep_ratio
                if not np.any(mask):
                    continue
                cx = cx[mask]
                cy = cy[mask]
                inst = inst[mask]

            xs_list.append(cx)
            ys_list.append(cy)
            inst_list.append(inst)

    if not xs_list:
        raise ValueError("No points selected for plotting after subsampling.")

    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    inst_ids = np.concatenate(inst_list)

    # If we still overshoot max_points (due to randomness), trim once more.
    if len(xs) > max_points:
        idx = np.random.choice(len(xs), size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]
        inst_ids = inst_ids[idx]

    return xs, ys, inst_ids


def plot_predinstance_topdown(
    laz_path: Path,
    output_path: Path,
    instance_dim: str = "PredInstance",
    max_points: int = 2_000_000,
    chunk_size: int = 5_000_000,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 8),
) -> None:
    """
    Create a top-down scatter plot of PredInstance.
    """
    print(f"Loading subsampled points from {laz_path} ...")
    xs, ys, inst_ids = load_subsampled_points(
        laz_path,
        instance_dim=instance_dim,
        max_points=max_points,
        chunk_size=chunk_size,
    )
    print(f"  Selected {len(xs):,} points for plotting.")

    # Normalize instances to 0..1 for colormap, but keep distinct categories.
    # Also compute centroids per PredInstance so we can annotate IDs.
    inst_unique, inv_idx = np.unique(inst_ids, return_inverse=True)

    # Index in [0, n_unique) for colormap
    idx_arr = inv_idx.astype(float)

    # Normalize indices to [0,1] for colormap.
    if len(inst_unique) > 1:
        idx_arr /= (len(inst_unique) - 1)

    # Compute centroids per instance for labeling.
    # Use bincount for efficiency.
    counts = np.bincount(inv_idx)
    sum_x = np.bincount(inv_idx, weights=xs)
    sum_y = np.bincount(inv_idx, weights=ys)
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = sum_x / np.maximum(counts, 1)
        cy = sum_y / np.maximum(counts, 1)

    # Create figure.
    print("Rendering figure ...")
    plt.figure(figsize=figsize, dpi=dpi)

    # Equal aspect to preserve geometry; alpha for dense plots.
    sc = plt.scatter(
        xs,
        ys,
        c=idx_arr,
        s=0.2,
        cmap="tab20",
        marker=".",
        linewidths=0,
        alpha=0.8,
    )

    # Add small text labels at instance centroids with the PredInstance ID.
    # This is primarily for debugging / QA; for many instances it will look busy.
    for i, pid in enumerate(inst_unique):
        # Skip non-positive IDs (e.g. background) if present
        if pid <= 0:
            continue
        plt.text(
            cx[i],
            cy[i],
            str(pid),
            fontsize=3,
            ha="center",
            va="center",
            color="black",
            alpha=0.8,
        )
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"PredInstance top-down: {laz_path.name}")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved figure to {output_path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-down PredInstance visualization for a merged LAZ file."
    )
    parser.add_argument(
        "input_laz",
        type=Path,
        help="Path to merged LAZ/LAZ file (must contain PredInstance dimension).",
    )
    parser.add_argument(
        "output_png",
        type=Path,
        nargs="?",
        help="Output PNG path (default: <input>_predinstance_topdown.png).",
    )
    parser.add_argument(
        "--instance-dim",
        default="PredInstance",
        help="Name of the instance dimension (default: PredInstance).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2_000_000,
        help="Maximum number of points to plot after subsampling.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Points per laspy chunk when streaming the file.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI (default: 300).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    input_laz: Path = args.input_laz
    output_png: Path

    if args.output_png is None:
        output_png = input_laz.with_name(input_laz.stem + "_predinstance_topdown.png")
    else:
        output_png = args.output_png

    plot_predinstance_topdown(
        input_laz,
        output_png,
        instance_dim=args.instance_dim,
        max_points=args.max_points,
        chunk_size=args.chunk_size,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

