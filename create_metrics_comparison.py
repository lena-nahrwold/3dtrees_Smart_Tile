#!/usr/bin/env python3
"""
Create metrics comparison visualization from comparison results.
Reads comparison_report.txt to extract metrics and creates visualizations.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


def parse_comparison_report(report_path: str) -> Dict[str, float]:
    """
    Parse metrics from the comparison_report.txt file.
    
    Returns:
        Dictionary with metrics
    """
    with open(report_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract point counts
    merged_points_match = re.search(r'Merged points in overlap:\s+([\d,]+)', content)
    ref_points_match = re.search(r'Reference points in overlap:\s+([\d,]+)', content)
    
    if merged_points_match:
        metrics['merged_points'] = int(merged_points_match.group(1).replace(',', ''))
    if ref_points_match:
        metrics['reference_points'] = int(ref_points_match.group(1).replace(',', ''))
    
    # Extract tree counts
    merged_trees_match = re.search(r'Merged trees in overlap:\s+(\d+)', content)
    ref_trees_match = re.search(r'Reference trees in overlap:\s+(\d+)', content)
    
    if merged_trees_match:
        metrics['merged_trees'] = int(merged_trees_match.group(1))
    if ref_trees_match:
        metrics['reference_trees'] = int(ref_trees_match.group(1))
    
    # Extract point difference
    point_diff_match = re.search(r'Difference:\s+([+-]?[\d,]+)\s+points\s+\(([+-]?[\d.]+)%\)', content)
    if point_diff_match:
        metrics['point_diff'] = int(point_diff_match.group(1).replace(',', ''))
        metrics['point_diff_pct'] = float(point_diff_match.group(2))
    
    # Extract tree difference
    tree_diff_match = re.search(r'Difference:\s+([+-]?[\d,]+)\s+trees\s+\(([+-]?[\d.]+)%\)', content)
    if tree_diff_match:
        metrics['tree_diff'] = int(tree_diff_match.group(1).replace(',', ''))
        metrics['tree_diff_pct'] = float(tree_diff_match.group(2))
    
    # Extract density
    merged_density_match = re.search(r'Merged:\s+([\d.]+)\s+points/m²', content)
    ref_density_match = re.search(r'Reference:\s+([\d.]+)\s+points/m²', content)
    density_diff_match = re.search(r'Difference:\s+([+-]?[\d.]+)%', content)
    
    if merged_density_match:
        metrics['merged_density'] = float(merged_density_match.group(1))
    if ref_density_match:
        metrics['reference_density'] = float(ref_density_match.group(1))
    if density_diff_match:
        metrics['density_diff_pct'] = float(density_diff_match.group(1))
    
    # Extract overlap area
    overlap_area_match = re.search(r'Overlap area:\s+([\d.]+)\s+m²', content)
    if overlap_area_match:
        metrics['overlap_area'] = float(overlap_area_match.group(1))
    
    # Extract points per instance stats
    merged_mean_match = re.search(r'Merged - Mean:\s+([\d.]+)', content)
    merged_median_match = re.search(r'Merged - Median:\s+([\d.]+)', content)
    merged_min_match = re.search(r'Merged - Min:\s+(\d+)', content)
    merged_max_match = re.search(r'Merged - Max:\s+(\d+)', content)
    
    ref_mean_match = re.search(r'Reference - Mean:\s+([\d.]+)', content)
    ref_median_match = re.search(r'Reference - Median:\s+([\d.]+)', content)
    ref_min_match = re.search(r'Reference - Min:\s+(\d+)', content)
    ref_max_match = re.search(r'Reference - Max:\s+(\d+)', content)
    
    if merged_mean_match:
        metrics['merged_mean_points_per_tree'] = float(merged_mean_match.group(1))
    if merged_median_match:
        metrics['merged_median_points_per_tree'] = float(merged_median_match.group(1))
    if merged_min_match:
        metrics['merged_min_points_per_tree'] = int(merged_min_match.group(1))
    if merged_max_match:
        metrics['merged_max_points_per_tree'] = int(merged_max_match.group(1))
    
    if ref_mean_match:
        metrics['reference_mean_points_per_tree'] = float(ref_mean_match.group(1))
    if ref_median_match:
        metrics['reference_median_points_per_tree'] = float(ref_median_match.group(1))
    if ref_min_match:
        metrics['reference_min_points_per_tree'] = int(ref_min_match.group(1))
    if ref_max_match:
        metrics['reference_max_points_per_tree'] = int(ref_max_match.group(1))
    
    return metrics


def plot_metrics_comparison(metrics: Dict[str, float], output_path: str):
    """
    Create visualization comparing merged vs reference metrics.
    """
    if len(metrics) == 0:
        print("ERROR: No metrics data found")
        return
    
    # Determine number of plots based on available metrics
    has_tree_metrics = 'merged_trees' in metrics and 'reference_trees' in metrics
    has_density = 'merged_density' in metrics and 'reference_density' in metrics
    has_points_per_tree = 'merged_mean_points_per_tree' in metrics
    
    num_plots = 2  # Always show point counts
    if has_tree_metrics:
        num_plots += 1
    if has_density:
        num_plots += 1
    if has_points_per_tree:
        num_plots += 1
    
    # Create subplots
    if num_plots <= 3:
        rows, cols = 1, num_plots
    elif num_plots <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot 1: Point counts
    ax = axes[plot_idx]
    categories = ['Merged', 'Reference']
    values = [
        metrics.get('merged_points', 0),
        metrics.get('reference_points', 0)
    ]
    colors = ['steelblue', 'green']
    bars = ax.bar(categories, values, color=colors, alpha=0.7)
    ax.set_ylabel('Point Count')
    ax.set_title('Point Counts (Overlap Region)')
    ax.tick_params(axis='x', rotation=0)
    # Add value labels
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
    # Add difference annotation
    if 'point_diff_pct' in metrics:
        diff_pct = metrics['point_diff_pct']
        ax.text(0.5, 0.95, f'Difference: {diff_pct:+.1f}%',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plot_idx += 1
    
    # Plot 2: Tree counts (if available)
    if has_tree_metrics:
        ax = axes[plot_idx]
        categories = ['Merged', 'Reference']
        values = [
            metrics.get('merged_trees', 0),
            metrics.get('reference_trees', 0)
        ]
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Tree Count')
        ax.set_title('Tree Counts (Overlap Region)')
        ax.tick_params(axis='x', rotation=0)
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        str(val), ha='center', va='bottom', fontsize=10)
        # Add difference annotation
        if 'tree_diff_pct' in metrics:
            diff_pct = metrics['tree_diff_pct']
            ax.text(0.5, 0.95, f'Difference: {diff_pct:+.1f}%',
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plot_idx += 1
    
    # Plot 3: Point density (if available)
    if has_density:
        ax = axes[plot_idx]
        categories = ['Merged', 'Reference']
        values = [
            metrics.get('merged_density', 0),
            metrics.get('reference_density', 0)
        ]
        bars = ax.bar(categories, values, color=['orange', 'red'], alpha=0.7)
        ax.set_ylabel('Point Density (points/m²)')
        ax.set_title('Point Density Comparison')
        ax.tick_params(axis='x', rotation=0)
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        # Add difference annotation
        if 'density_diff_pct' in metrics:
            diff_pct = metrics['density_diff_pct']
            ax.text(0.5, 0.95, f'Difference: {diff_pct:+.1f}%',
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plot_idx += 1
    
    # Plot 4: Points per tree statistics (if available)
    if has_points_per_tree:
        ax = axes[plot_idx]
        categories = ['Mean', 'Median', 'Min', 'Max']
        merged_values = [
            metrics.get('merged_mean_points_per_tree', 0),
            metrics.get('merged_median_points_per_tree', 0),
            metrics.get('merged_min_points_per_tree', 0),
            metrics.get('merged_max_points_per_tree', 0)
        ]
        ref_values = [
            metrics.get('reference_mean_points_per_tree', 0),
            metrics.get('reference_median_points_per_tree', 0),
            metrics.get('reference_min_points_per_tree', 0),
            metrics.get('reference_max_points_per_tree', 0)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        bars1 = ax.bar(x - width/2, merged_values, width, label='Merged', color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, ref_values, width, label='Reference', color='green', alpha=0.7)
        
        ax.set_ylabel('Points per Tree')
        ax.set_title('Points per Tree Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Merged vs Reference: Metrics Comparison (Overlap Region)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create metrics comparison visualization from comparison report"
    )
    parser.add_argument(
        "--comparison_report",
        type=str,
        required=True,
        help="Path to comparison_report.txt file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for metrics comparison PNG (default: same directory as report with name metrics_comparison.png)"
    )
    
    args = parser.parse_args()
    
    report_path = Path(args.comparison_report)
    
    if not report_path.exists():
        print(f"ERROR: Report file not found: {report_path}")
        exit(1)
    
    # Parse metrics
    print("Parsing metrics from report...")
    metrics = parse_comparison_report(str(report_path))
    
    if len(metrics) == 0:
        print("ERROR: No metrics found in report")
        exit(1)
    
    print(f"Found metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = report_path.parent / "metrics_comparison.png"
    
    # Create visualization
    print(f"\nCreating metrics comparison visualization...")
    plot_metrics_comparison(metrics, str(output_path))
    
    print(f"\nDone! Metrics comparison saved to: {output_path}")


if __name__ == "__main__":
    main()

