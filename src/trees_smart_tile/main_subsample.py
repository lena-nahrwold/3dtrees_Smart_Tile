#!/usr/bin/env python3
"""
Main subsampling script: Parallel subsampling to resolution 1 (2cm) and resolution 2 (10cm).

This script handles subsampling of tiled point clouds:
1. Subsample tiles to resolution 1 (default: 2cm)
2. Subsample resolution 1 files to resolution 2 (default: 10cm)

Files are split across available CPU cores for parallel processing.

COPC Optimizations:
- Uses COPC native bounds filtering in readers.copc (more efficient than filters.crop)
- Writes output as COPC format when input is COPC (better performance for subsequent steps)
- Leverages COPC's spatial indexing for efficient chunk-based processing
- Multi-threaded COPC writing for improved performance

Usage:
    python main_subsample.py --tiles_dir /path/to/tiles --res1 0.02 --res2 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# Import parameters
from parameters import TILE_PARAMS


def get_pdal_path() -> str:
    """Get the path to pdal executable."""
    import shutil
    # Use shutil.which to find pdal in PATH
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def get_cpu_count() -> int:
    """Get available CPU count."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def get_file_bounds(filepath: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get spatial bounds of a point cloud file using pdal info.
    
    Returns:
        Tuple of (minx, maxx, miny, maxy) or None on error
    """
    try:
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "info", "--metadata", str(filepath)],
            capture_output=True,
            text=True,
            check=True
        )
        
        import re
        minx = float(re.search(r'"minx":\s*([\d.-]+)', result.stdout).group(1))
        maxx = float(re.search(r'"maxx":\s*([\d.-]+)', result.stdout).group(1))
        miny = float(re.search(r'"miny":\s*([\d.-]+)', result.stdout).group(1))
        maxy = float(re.search(r'"maxy":\s*([\d.-]+)', result.stdout).group(1))
        
        return (minx, maxx, miny, maxy)
    except Exception:
        return None


def subsample_tile_chunk(args: Tuple[Path, str, float, Path, int, int]) -> Tuple[Path, int]:
    """
    Subsample a spatial chunk of a tile using PDAL with COPC-optimized bounds filter.
    
    COPC optimizations:
    - Uses bounds parameter directly in readers.copc (more efficient than filters.crop)
    - Output is always LAZ format (compressed LAS)
    
    Args:
        args: Tuple of (input_file, bounds_str, resolution, output_dir, chunk_idx, total_chunks)
    
    Returns:
        Tuple of (output_file, point_count)
    """
    input_file, bounds_str, resolution, output_dir, chunk_idx, total_chunks = args
    
    try:
        # Determine reader type - can read COPC or LAS
        is_copc = input_file.name.endswith('.copc.laz')
        reader_type = "readers.copc" if is_copc else "readers.las"
        
        # Always output as LAZ format (compressed LAS)
        chunk_file = output_dir / f"{input_file.stem}_chunk{chunk_idx}.laz"
        
        # Build pipeline - use COPC bounds filtering if available
        if is_copc:
            # COPC: Use bounds parameter directly in reader (most efficient)
            pipeline = {
                "pipeline": [
                    {
                        "type": reader_type,
                        "filename": str(input_file),
                        "bounds": bounds_str  # COPC native bounds filtering - very efficient
                    },
                    {
                        "type": "filters.voxelcentroidnearestneighbor",
                        "cell": resolution
                    },
                    {
                        "type": "writers.las",  # Always write as LAZ (compressed LAS)
                        "filename": str(chunk_file),
                        "compression": True,
                        "extra_dims": "all"  # Preserve extra dimensions like PredInstance
                    }
                ]
            }
        else:
            # LAS: Use filters.crop as fallback (LAS readers don't support bounds parameter)
            pipeline = {
                "pipeline": [
                    {
                        "type": reader_type,
                        "filename": str(input_file)
                    },
                    {
                        "type": "filters.crop",
                        "bounds": bounds_str
                    },
                    {
                        "type": "filters.voxelcentroidnearestneighbor",
                        "cell": resolution
                    },
                    {
                        "type": "writers.las",
                        "filename": str(chunk_file),
                        "compression": True,
                        "extra_dims": "all"
                    }
                ]
            }
        
        # Write and execute pipeline
        pipeline_file = output_dir / f"_pipeline_chunk{chunk_idx}.json"
        with open(pipeline_file, 'w') as f:
            json.dump(pipeline, f, indent=2)
        
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up pipeline
        if pipeline_file.exists():
            pipeline_file.unlink()
        
        if result.returncode != 0:
            print(f"      ⚠ Chunk {chunk_idx}/{total_chunks} error: {result.stderr[:100]}")
            return (None, 0)
        
        if not chunk_file.exists() or chunk_file.stat().st_size == 0:
            return (None, 0)
        
        # Get point count
        point_count = 0
        try:
            info_result = subprocess.run(
                [pdal_cmd, "info", "--metadata", str(chunk_file)],
                capture_output=True,
                text=True,
                check=True
            )
            import re
            match = re.search(r'"count":\s*(\d+)', info_result.stdout)
            if match:
                point_count = int(match.group(1))
        except Exception:
            pass
        
        print(f"      ✓ Chunk {chunk_idx}/{total_chunks}: {point_count:,} points")
        return (chunk_file, point_count)
        
    except Exception as e:
        print(f"      ✗ Chunk {chunk_idx}/{total_chunks} failed: {e}")
        return (None, 0)


def subsample_single_file(args: Tuple[Path, Path, float, Path, int]) -> Tuple[str, bool, str, int]:
    """
    Subsample a single file by splitting it into subtiles along X-axis and processing in parallel.
    
    Process:
    1. Split tile into num_threads subtiles along X-axis only
    2. Subsample each subtile in parallel using ProcessPoolExecutor (true CPU parallelism)
    3. Merge all subsampled subtiles back together
    
    Args:
        args: Tuple of (input_file, output_file, resolution, pipeline_dir, num_threads)
    
    Returns:
        Tuple of (filename, success, message, point_count)
    """
    input_file, output_file, resolution, pipeline_dir, num_threads = args
    try:
        print(f"    → Processing {input_file.name}...")
        
        # Get file bounds
        bounds = get_file_bounds(input_file)
        if not bounds:
            # Fall back to simple single-pass subsampling
            return subsample_simple(input_file, output_file, resolution, pipeline_dir)
        
        minx, maxx, miny, maxy = bounds
        
        # Split into num_threads subtiles along X-axis only
        grid_x = num_threads
        grid_y = 1
        
        # Calculate step size for X-axis
        x_step = (maxx - minx) / grid_x
        
        print(f"      Splitting into {num_threads} subtiles along X-axis only ({grid_x}x{grid_y} grid)")
        
        # Create chunk tasks - exactly num_threads chunks
        chunk_dir = pipeline_dir / f"{input_file.stem}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_tasks = []
        for chunk_idx in range(num_threads):
            chunk_minx = minx + chunk_idx * x_step
            chunk_maxx = minx + (chunk_idx + 1) * x_step
            # Keep full Y range for each chunk
            chunk_miny = miny
            chunk_maxy = maxy
            
            bounds_str = f"([{chunk_minx},{chunk_maxx}],[{chunk_miny},{chunk_maxy}])"
            chunk_tasks.append((input_file, bounds_str, resolution, chunk_dir, chunk_idx, num_threads))
        
        # Process chunks in parallel using ProcessPoolExecutor for true CPU parallelism
        chunk_files = []
        total_points = 0
        
        print(f"      → Subsampling {len(chunk_tasks)} subtiles in parallel...")
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(subsample_tile_chunk, task) for task in chunk_tasks]
            for future in as_completed(futures):
                chunk_file, point_count = future.result()
                if chunk_file and chunk_file.exists():
                    chunk_files.append(chunk_file)
                    total_points += point_count
        
        if not chunk_files:
            return (input_file.name, False, "No chunks produced", 0)
        
        print(f"      → Merging {len(chunk_files)} subsampled subtiles...")
        
        # Merge chunks using PDAL - always write as LAZ format
        # Chunk files are already LAZ from subsample_tile_chunk
        reader_type = "readers.las"  # Chunks are LAZ files
        
        # Ensure output is LAZ format
        if not output_file.name.endswith('.laz'):
            output_file = output_file.parent / (output_file.stem + '.laz')
        
        merge_pipeline = {
            "pipeline": [
                *[{"type": reader_type, "filename": str(f)} for f in chunk_files],
                {
                    "type": "filters.merge"
                },
                {
                    "type": "writers.las",  # Always write as LAZ (compressed LAS)
                    "filename": str(output_file),
                    "compression": True,
                    "extra_dims": "all"  # Preserve extra dimensions like PredInstance
                }
            ]
        }
        
        merge_pipeline_file = chunk_dir / "merge.json"
        with open(merge_pipeline_file, 'w') as f:
            json.dump(merge_pipeline, f, indent=2)
        
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(merge_pipeline_file)],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up chunks and temporary files
        for chunk_file in chunk_files:
            if chunk_file.exists():
                try:
                    chunk_file.unlink()
                except Exception:
                    pass
        
        # Clean up merge pipeline
        if merge_pipeline_file.exists():
            try:
                merge_pipeline_file.unlink()
            except Exception:
                pass
        
        # Remove chunk directory
        if chunk_dir.exists():
            try:
                import shutil
                shutil.rmtree(chunk_dir)
            except Exception:
                pass
        
        if result.returncode != 0:
            return (input_file.name, False, f"Merge failed: {result.stderr[:100]}", 0)
        
        if not output_file.exists() or output_file.stat().st_size == 0:
            return (input_file.name, False, "Output file empty", 0)
        
        print(f"    ✓ {input_file.name}: {total_points:,} points")
        return (input_file.name, True, "Success", total_points)
        
    except Exception as e:
        print(f"    ✗ {input_file.name}: {e}")
        return (input_file.name, False, str(e), 0)


def subsample_simple(input_file: Path, output_file: Path, resolution: float, pipeline_dir: Path) -> Tuple[str, bool, str, int]:
    """
    Simple single-pass subsampling (fallback method) with COPC reader optimization.
    
    Uses COPC reader for efficient reading, but always outputs LAZ format.
    
    Args:
        input_file: Input file path
        output_file: Output file path
        resolution: Voxel resolution
        pipeline_dir: Directory for pipeline files
    
    Returns:
        Tuple of (filename, success, message, point_count)
    """
    try:
        is_copc = input_file.name.endswith('.copc.laz')
        reader_type = "readers.copc" if is_copc else "readers.las"
        
        # Always output as LAZ format (compressed LAS)
        if not output_file.name.endswith('.laz'):
            output_file = output_file.parent / (output_file.stem + '.laz')
        
        pipeline = {
            "pipeline": [
                {"type": reader_type, "filename": str(input_file)},
                {"type": "filters.voxelcentroidnearestneighbor", "cell": resolution},
                {
                    "type": "writers.las",  # Always write as LAZ (compressed LAS)
                    "filename": str(output_file),
                    "compression": True,
                    "extra_dims": "all"  # Preserve extra dimensions like PredInstance
                }
            ]
        }
        
        pipeline_file = pipeline_dir / f"{input_file.stem}_simple.json"
        with open(pipeline_file, 'w') as f:
            json.dump(pipeline, f, indent=2)
        
        pdal_cmd = get_pdal_path()
        result = subprocess.run(
            [pdal_cmd, "pipeline", str(pipeline_file)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if pipeline_file.exists():
            pipeline_file.unlink()
        
        if result.returncode != 0:
            return (input_file.name, False, result.stderr[:200], 0)
        
        if not output_file.exists() or output_file.stat().st_size == 0:
            return (input_file.name, False, "Output file empty", 0)
        
        # Get point count
        point_count = 0
        try:
            info_result = subprocess.run(
                [pdal_cmd, "info", "--metadata", str(output_file)],
                capture_output=True,
                text=True,
                check=True
            )
            import re
            match = re.search(r'"count":\s*(\d+)', info_result.stdout)
            if match:
                point_count = int(match.group(1))
        except Exception:
            pass
        
        return (input_file.name, True, "Success", point_count)
        
    except Exception as e:
        return (input_file.name, False, str(e), 0)


def subsample_parallel(
    input_dir: Path,
    output_dir: Path,
    resolution: float,
    num_cores: int,
    num_threads: int,
    output_prefix: Optional[str] = None
) -> List[Path]:
    """
    Subsample all files in directory using parallel chunk processing.
    
    Files are processed sequentially (one at a time), but each file is split
    spatially into chunks along X-axis and processed in parallel.
    Uses PDAL voxelcentroidnearestneighbor filter.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory for output files
        resolution: Voxel resolution in meters
        num_cores: Not used (kept for compatibility)
        num_threads: Number of spatial chunks per file (from TILE_PARAMS['threads'])
        output_prefix: Optional prefix for output filenames
    
    Returns:
        List of created output file paths
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline directory
    pipeline_dir = output_dir / "pipelines"
    pipeline_dir.mkdir(exist_ok=True)
    
    # Find input files
    # Get all LAZ files (both .laz and .copc.laz)
    # Note: *.laz will match both .laz and .copc.laz, so we use set to deduplicate
    input_files = sorted(set(list(input_dir.glob("*.laz")) + list(input_dir.glob("*.copc.laz"))))
    
    if not input_files:
        print(f"    No input files found in {input_dir}")
        return []
    
    # Convert resolution to cm for filename
    res_cm = int(resolution * 100)
    
    # Prepare tasks
    tasks = []
    for input_file in sorted(input_files):
        # Generate output filename
        stem = input_file.stem
        # Remove .copc suffix if present
        if stem.endswith('.copc'):
            stem = stem[:-5]
        
        # Extract original base filename by removing prefixes and resolution suffixes
        import re
        base_name = stem
        
        # Remove resolution suffix patterns (e.g., "_1cm", "_10cm", "_subsampled0.01m", "_subsampled0.1m")
        base_name = re.sub(r'_subsampled[\d.]+m$', '', base_name)
        base_name = re.sub(r'_\d+cm$', '', base_name)
        
        # Remove output_prefix if present at the start (e.g., "output_dir_100m_")
        if output_prefix and base_name.startswith(output_prefix + '_'):
            base_name = base_name[len(output_prefix) + 1:]
        
        # Remove any remaining prefix patterns that look like "something_100m_" or "output_dir_100m_"
        base_name = re.sub(r'^[^_]+_\d+m_', '', base_name)
        
        # For tiled files, try to extract tile ID (c##_r##) pattern
        tile_match = re.search(r'(c\d+_r\d+)', base_name)
        if tile_match:
            # Keep tile ID for tiled files
            tile_id = tile_match.group(1)
            # Extract base name before tile ID if there's a prefix
            base_before_tile = base_name[:tile_match.start()]
            if base_before_tile and base_before_tile.endswith('_'):
                base_before_tile = base_before_tile[:-1]
            # Remove any remaining prefix from base_before_tile
            if base_before_tile:
                base_before_tile = re.sub(r'^[^_]+_\d+m_', '', base_before_tile)
                if base_before_tile:
                    output_name = f"{base_before_tile}_{tile_id}_subsampled_{res_cm}cm.laz"
                else:
                    output_name = f"{tile_id}_subsampled_{res_cm}cm.laz"
            else:
                output_name = f"{tile_id}_subsampled_{res_cm}cm.laz"
        else:
            # Single file or no tile ID - use clean base name
            # Remove any remaining prefix patterns
            base_name = re.sub(r'^[^_]+_\d+m_', '', base_name)
            output_name = f"{base_name}_subsampled_{res_cm}cm.laz"
        
        output_file = output_dir / output_name
        
        # Skip if already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"    ⊙ Skipping {input_file.name} (already exists)")
            continue
        
        tasks.append((input_file, output_file, resolution, pipeline_dir, num_threads))
    
    if not tasks:
        print(f"    ✓ All files already subsampled")
        return list(output_dir.glob("*.laz"))
    
    print(f"    Files to process: {len(tasks)}")
    print(f"    Processing mode: Sequential (one file at a time)")
    print(f"    Chunk parallelism: {num_threads} chunks per file (parallel)")
    print()
    
    # Process files sequentially, but chunks within each file in parallel
    successful = 0
    failed = 0
    total_points = 0
    
    for task in tasks:
        filename, success, message, point_count = subsample_single_file(task)
        if success:
            successful += 1
            total_points += point_count
        else:
            failed += 1
            print(f"    ✗ {filename}: {message}")
    
    # Clean up pipeline directory
    if pipeline_dir.exists() and not any(pipeline_dir.iterdir()):
        pipeline_dir.rmdir()
    
    print()
    print(f"    ═══ Summary ═══")
    print(f"    Complete: {successful} successful, {failed} failed")
    print(f"    Total points: {total_points:,}")
    
    return list(output_dir.glob("*.laz"))


def run_subsample_pipeline(
    tiles_dir: Path,
    res1: float = 0.01,
    res2: float = 0.1,
    num_cores: Optional[int] = None,
    num_threads: Optional[int] = None,
    output_prefix: Optional[str] = None,
    output_base_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Run the complete subsampling pipeline.
    
    Steps:
    1. Subsample tiles to resolution 1 (default: 2cm)
    2. Subsample resolution 1 files to resolution 2 (default: 10cm)
    
    Files are processed sequentially (one at a time), but each file is split
    spatially into num_threads chunks along X-axis and processed in parallel.
    
    Args:
        tiles_dir: Directory containing tile COPC files (input)
        res1: First resolution in meters (default: 0.01 = 1cm)
        res2: Second resolution in meters (default: 0.1 = 10cm)
        num_cores: Not used (kept for compatibility)
        num_threads: Number of parallel chunks per file (default: from TILE_PARAMS['threads'])
        output_prefix: Optional prefix for output filenames
        output_base_dir: Base directory for output (default: parent of tiles_dir)
    
    Returns:
        Tuple of (subsampled_res1_dir, subsampled_res2_dir)
    """
    # Auto-detect CPU count
    if num_cores is None:
        num_cores = get_cpu_count()
    
    # Get num_threads from TILE_PARAMS
    if num_threads is None:
        num_threads = TILE_PARAMS.get('threads', 5)
    
    # Convert to cm for display/filenames (but use simple directory names)
    res1_cm = int(res1 * 100)
    res2_cm = int(res2 * 100)
    
    # Define output directories - use output_base_dir if provided, otherwise use tiles_dir's parent
    if output_base_dir is None:
        output_base_dir = tiles_dir.parent
    
    # Create output directories directly under output_base_dir
    subsampled_res1_dir = output_base_dir / "subsampled_res1"
    subsampled_res2_dir = output_base_dir / "subsampled_res2"
    
    print("=" * 60)
    print("3DTrees Subsampling Pipeline")
    print("=" * 60)
    print(f"Input: {tiles_dir}")
    print(f"Resolution 1: {res1}m ({res1_cm}cm)")
    print(f"Resolution 2: {res2}m ({res2_cm}cm)")
    print(f"CPU cores: {num_cores}")
    print(f"Threads (chunks per file): {num_threads}")
    print()
    
    # Step 1: Subsample to resolution 1
    print("=" * 60)
    print(f"Step 1: Subsampling to {res1_cm}cm ({res1}m)")
    print("=" * 60)
    
    res1_files = subsample_parallel(
        input_dir=tiles_dir,
        output_dir=subsampled_res1_dir,
        resolution=res1,
        num_cores=num_cores,
        num_threads=num_threads,
        output_prefix=output_prefix
    )
    
    if not res1_files:
        raise ValueError(f"No files created in {subsampled_res1_dir}")
    
    print(f"\n  ✓ {res1_cm}cm subsampling complete: {len(res1_files)} files")
    print(f"  Output: {subsampled_res1_dir}")
    
    # Step 2: Subsample resolution 1 to resolution 2
    print()
    print("=" * 60)
    print(f"Step 2: Subsampling to {res2_cm}cm ({res2}m)")
    print("=" * 60)
    
    res2_files = subsample_parallel(
        input_dir=subsampled_res1_dir,
        output_dir=subsampled_res2_dir,
        resolution=res2,
        num_cores=num_cores,
        num_threads=num_threads,
        output_prefix=output_prefix
    )
    
    if not res2_files:
        raise ValueError(f"No files created in {subsampled_res2_dir}")
    
    print(f"\n  ✓ {res2_cm}cm subsampling complete: {len(res2_files)} files")
    print(f"  Output: {subsampled_res2_dir}")
    
    # Summary
    print()
    print("=" * 60)
    print("Subsampling Pipeline Complete")
    print("=" * 60)
    print(f"  Resolution 1 ({res1_cm}cm): {len(res1_files)} files in {subsampled_res1_dir}")
    print(f"  Resolution 2 ({res2_cm}cm): {len(res2_files)} files in {subsampled_res2_dir}")
    
    return subsampled_res1_dir, subsampled_res2_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Subsampling Pipeline - Parallel subsampling to multiple resolutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--tiles_dir", "-i",
        type=Path,
        required=True,
        help="Directory containing tile COPC files"
    )
    
    parser.add_argument(
        "--res1",
        type=float,
        default=TILE_PARAMS.get('resolution_1', 0.02),
        help=f"First resolution in meters (default: {TILE_PARAMS.get('resolution_1', 0.02)})"
    )
    
    parser.add_argument(
        "--res2",
        type=float,
        default=TILE_PARAMS.get('resolution_2', 0.1),
        help=f"Second resolution in meters (default: {TILE_PARAMS.get('resolution_2', 0.1)})"
    )
    
    parser.add_argument(
        "--num_cores",
        type=int,
        default=None,
        help="Number of CPU cores (default: auto-detect, not used for chunking)"
    )
    
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help=f"Number of spatial chunks per file for parallel processing (default: {TILE_PARAMS.get('threads', 5)})"
    )
    
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Optional prefix for output filenames"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.tiles_dir.exists():
        print(f"Error: Tiles directory does not exist: {args.tiles_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        res1_dir, res2_dir = run_subsample_pipeline(
            tiles_dir=args.tiles_dir,
            res1=args.res1,
            res2=args.res2,
            num_cores=args.num_cores,
            num_threads=args.num_threads,
            output_prefix=args.output_prefix
        )
        print(f"\nSubsampled files ready:")
        print(f"  Resolution 1: {res1_dir}")
        print(f"  Resolution 2: {res2_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

