# 3DTrees Smart Tiling Pipeline

**A high-performance point cloud processing pipeline for 3D tree segmentation: intelligent tiling, multi-resolution subsampling, prediction remapping, and cross-tile instance merging with species ID preservation.**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Detailed Usage](#detailed-usage)
7. [Pipeline Stages](#pipeline-stages)
8. [Parameters Reference](#parameters-reference)
9. [Input/Output Formats](#inputoutput-formats)
10. [Advanced Configuration](#advanced-configuration)
11. [Troubleshooting](#troubleshooting)
12. [Project Structure](#project-structure)
13. [Dependencies](#dependencies)
14. [License](#license)

---

## Overview

The **3DTrees Smart Tiling Pipeline** is a production-ready system designed to process large-scale LiDAR point clouds for individual tree segmentation. It addresses the fundamental challenge of processing massive datasets that exceed memory limits by intelligently dividing point clouds into manageable tiles, processing them independently, and then seamlessly merging the results.

### The Problem

Modern airborne and terrestrial LiDAR surveys can produce datasets with billions of points covering entire forests. Deep learning-based tree segmentation models typically operate on limited spatial extents due to memory constraints. Processing such data requires:

1. **Spatial partitioning** - Dividing large datasets into manageable tiles
2. **Buffer zones** - Handling tree instances that span tile boundaries
3. **Multi-resolution processing** - Subsampling for efficient neural network inference
4. **Prediction upscaling** - Remapping low-resolution predictions back to high-resolution data
5. **Instance merging** - Reconnecting tree instances split across tiles

### The Solution

This pipeline provides an end-to-end solution with two primary tasks:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TILE TASK                                          │
│                                                                                 │
│  Input LAZ/LAS Files                                                            │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│  │ COPC         │     │ Spatial      │     │ Smart        │                    │
│  │ Conversion   │────▶│ Index        │────▶│ Tiling       │                    │
│  │ (untwine)    │     │ (tindex)     │     │ (with buffer)│                    │
│  └──────────────┘     └──────────────┘     └──────────────┘                    │
│                                                   │                            │
│                                                   ▼                            │
│                                          ┌───────────────────┐                 │
│                                          │ Multi-Resolution  │                 │
│                                          │ Subsampling       │                 │
│                                          │ (2cm + 10cm)      │                 │
│                                          └───────────────────┘                 │
│                                                   │                            │
│                                                   ▼                            │
│                                           Outputs: tiles_100m/                 │
│                                                    ├─ c00_r00.copc.laz         │
│                                                    ├─ subsampled_2cm/          │
│                                                    └─ subsampled_10cm/         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              [External Segmentation]
                         (e.g., ForAINet, SegmentAnyTree)
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MERGE TASK                                         │
│                                                                                 │
│  Segmented 10cm Tiles (with PredInstance attribute)                            │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│  │ Prediction   │     │ Buffer       │     │ Cross-tile   │                    │
│  │ Remapping    │────▶│ Filtering    │────▶│ Instance     │                    │
│  │ (10cm→2cm)   │     │              │     │ Matching     │                    │
│  └──────────────┘     └──────────────┘     └──────────────┘                    │
│                                                   │                            │
│                                                   ▼                            │
│                              ┌───────────────────────────────────┐             │
│                              │ Deduplication + Small Volume Merge │            │
│                              └───────────────────────────────────┘             │
│                                                   │                            │
│                                                   ▼                            │
│                              ┌───────────────────────────────────┐             │
│                              │ Remap to Original Input Files     │             │
│                              └───────────────────────────────────┘             │
│                                                   │                            │
│                                                   ▼                            │
│                                      Unified Point Cloud                       │
│                               (Consistent Instance IDs Across Tiles)           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### High Performance
- **Parallel processing** using Python's `ProcessPoolExecutor` for multi-core utilization
- **COPC format optimization** for efficient spatial queries and streaming access
- **Memory-efficient chunking** - tiles are processed independently to minimize memory footprint
- **Parallel subsampling** - each tile is spatially divided into chunks processed concurrently

### Intelligent Tiling
- **Configurable tile size** - default 100m × 100m, adjustable for different use cases
- **Buffer zones** - overlapping regions (default 5m) ensure trees at boundaries are fully captured
- **Spatial indexing** - uses PDAL tindex for efficient data retrieval
- **Data-aligned grid** - tiles start from actual data extent, minimizing empty tiles

### Multi-Resolution Processing
- **Dual subsampling** - generates both 2cm and 10cm resolution outputs
- **Voxel-based downsampling** - uses PDAL's `voxelcentroidnearestneighbor` filter
- **Dimension preservation** - all extra dimensions (PredInstance, species_id, etc.) are maintained

### Smart Instance Merging
- **Centroid-based filtering** - removes duplicate instances in buffer zones
- **Overlap ratio matching** - identifies same trees across tile boundaries using point correspondence
- **Union-Find algorithm** - efficiently groups matched instances into unified trees
- **Species ID preservation** - always preserves species from the larger instance fragment
- **Small volume merging** - reassigns orphaned tree fragments to nearby larger instances
- **Original file remapping** - maps predictions back to original input files

---

## Pipeline Architecture

### Stage-by-Stage Breakdown

#### TILE TASK: Data Preparation

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **COPC Conversion** | Converts input LAZ/LAS files to Cloud-Optimized Point Cloud (COPC) format using `untwine`. Optional XYZ-only reduction for ~37% size savings. |
| 2 | **Spatial Index** | Creates a GeoPackage tindex using `pdal tindex` for efficient spatial queries across all input files. |
| 3 | **Tile Bounds** | Calculates optimal tile grid based on data extent, tile size, and buffer parameters. |
| 4 | **Tile Creation** | Extracts point data for each tile from COPC files using spatial bounds queries. |
| 5 | **Subsampling R1** | Downsamples tiles to resolution 1 (default: 2cm) using parallel chunk processing. |
| 6 | **Subsampling R2** | Further downsamples to resolution 2 (default: 10cm) for neural network inference. |

#### MERGE TASK: Result Integration

| Stage | Component | Description |
|-------|-----------|-------------|
| 0 | **Prediction Remapping** | Transfers PredInstance labels from 10cm predictions to 2cm resolution using KDTree nearest-neighbor lookup. |
| 1 | **Load and Filter** | Loads tiles, applies centroid-based buffer zone filtering to remove duplicate instances. |
| 2 | **Global ID Assignment** | Creates unique instance IDs across all tiles using tile-specific offsets. |
| 3 | **Cross-tile Matching** | Identifies matching instances in tile overlaps using overlap ratio and centroid distance (Union-Find grouping). |
| 4 | **Merge and Deduplicate** | Merges all tiles, removes duplicate points from buffer regions using 5cm tolerance. |
| 5 | **Small Volume Merge** | Reassigns tree fragments with volume < 4m³ to nearest large instance. |
| 6 | **Retiling** | Maps final instance IDs back to original tile boundaries for per-tile output. |
| 7 | **Original Remap** | Maps final instance IDs back to original input LAZ files (pre-tiling, optional). |

---

## Installation

### Conda Environment

```bash
# Create conda environment
mamba create -n 3dtrees -c conda-forge \
    python=3.10 \
    pdal=2.6 \
    python-pdal \
    gdal \
    laspy \
    lazrs-python \
    numpy \
    scipy \
    matplotlib-base \
    fiona \
    pyproj \
    geopandas \
    pydantic \
    pydantic-settings

# Activate environment
conda activate 3dtrees

# Build pdal_wrench from source (required for parallel operations)
git clone --depth 1 https://github.com/PDAL/wrench.git
cd wrench
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo cp pdal_wrench /usr/local/bin/
cd ../..

# Verify installation
python src/run.py --show-params
```

### System Requirements

- **Operating System**: Linux (tested on Ubuntu 20.04+), macOS, Windows with WSL2
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large datasets
- **CPU**: Multi-core processor recommended (parallel processing scales with cores)
- **Storage**: SSD recommended for I/O-intensive operations
- **PDAL**: Version 2.5 or higher
- **GDAL**: Version 3.0 or higher

---

## Quick Start

### Basic Tile Task

Process a directory of LAZ files into tiled, multi-resolution outputs:

```bash
python src/run.py --task tile \
    --input-dir /path/to/input \
    --output-dir /path/to/output
```

### Basic Merge Task

Merge segmented tiles back into a unified point cloud:

```bash
python src/run.py --task merge \
    --subsampled-10cm-folder /path/to/output/tiles_100m/subsampled_10cm \
    --original-input-dir /path/to/input
```

### View Current Parameters

```bash
python src/run.py --show-params
```

---

## Detailed Usage

### Tile Task Options

```bash
python src/run.py --task tile \
    --input-dir /path/to/input \           # Required: Directory with LAZ/LAS files
    --output-dir /path/to/output \         # Required: Output directory
    --tile-length 100 \                    # Tile size in meters (default: 100)
    --tile-buffer 5 \                      # Buffer overlap in meters (default: 5)
    --resolution-1 0.02 \                  # First resolution (default: 2cm)
    --resolution-2 0.1 \                   # Second resolution (default: 10cm)
    --workers 8 \                          # Parallel workers (default: 4)
    --threads 5 \                          # Threads per COPC writer (default: 5)
    --grid-offset 1.0 \                    # Grid offset from min coords (default: 1.0)
    --skip-dimension-reduction             # Keep all dimensions (default: XYZ-only)
```

### Merge Task Options

```bash
python src/run.py --task merge \
    --subsampled-10cm-folder /path/to/10cm \    # Path to segmented 10cm tiles
    --original-input-dir /path/to/input \       # Path to original input LAZ files
    --target-resolution 2 \                     # Target resolution in cm (default: 2)
    --buffer 10.0 \                             # Buffer zone distance (default: 10m)
    --overlap-threshold 0.3 \                   # Instance matching threshold (default: 30%)
    --max-centroid-distance 3.0 \               # Max centroid distance (default: 3m)
    --workers 8 \                               # Parallel workers (default: 4)
    --retile-buffer 1.0 \                       # Spatial buffer for retiling (default: 1m)
    --retile-max-radius 0.1 \                   # Max distance for retiling match (default: 0.1m)
    --disable-matching                          # Disable cross-tile matching
```

---

## Pipeline Stages

### Stage 1: COPC Conversion

**Purpose**: Convert input LAZ/LAS files to Cloud-Optimized Point Cloud (COPC) format.

**Technology**: Uses `untwine` for efficient COPC creation with spatial indexing.

**Options**:
- **Default (XYZ-only)**: Reduces dimensions to X, Y, Z only, achieving ~37% file size reduction
- **Full dimensions**: Use `--skip-dimension-reduction` to preserve all attributes

**Example**:
```bash
# XYZ-only (smaller files, faster processing)
python src/run.py --task tile --input-dir /data/in --output-dir /data/out

# Preserve all dimensions
python src/run.py --task tile --input-dir /data/in --output-dir /data/out --skip-dimension-reduction
```

### Stage 2: Spatial Indexing

**Purpose**: Create a spatial index (tindex) for efficient querying across all input files.

**Technology**: Uses `pdal tindex` to create a GeoPackage with file boundaries.

**Output**: `tindex_100m.gpkg` containing polygons representing each input file's extent.

### Stage 3: Tile Grid Calculation

**Purpose**: Compute optimal tile boundaries based on data extent and parameters.

**Algorithm**:
1. Load extent from tindex
2. Apply grid offset to starting coordinates
3. Create tiles of `tile_length` × `tile_length` meters
4. Add `tile_buffer` meters to each side
5. Generate job list with projected and geographic bounds

**Output Files**:
- `tile_bounds_tindex.json`: Complete tile metadata
- `tile_jobs_100m.txt`: Per-tile processing instructions
- `overview_copc_tiles.png`: Visualization of tiles and input files

### Stage 4: Tile Creation

**Purpose**: Extract points for each tile from COPC files using spatial queries.

**Process**:
1. For each tile, read bounds from job file
2. Query all COPC files that intersect tile bounds
3. Extract points using PDAL `readers.copc` with bounds parameter
4. Merge parts from multiple input files
5. Write as `c{col}_r{row}.copc.laz`

**Parallelization**: Multiple tiles processed concurrently (controlled by `--workers`)

### Stage 5-6: Multi-Resolution Subsampling

**Purpose**: Create downsampled versions for efficient neural network processing.

**Algorithm**: Voxel-based downsampling using `filters.voxelcentroidnearestneighbor`

**Process**:
1. Each tile is split spatially into chunks along X-axis
2. Chunks are processed in parallel
3. Results are merged back into single file

**Outputs**:
- `subsampled_2cm/`: Resolution_1 files
- `subsampled_10cm/`: Resolution_2 files

### Stage 0: Prediction Remapping

**Purpose**: Transfer predictions from low-resolution (10cm) back to high-resolution (2cm).

**Algorithm**: KDTree nearest-neighbor lookup

**Process**:
1. Load 10cm segmented file with `PredInstance` attribute
2. Load corresponding 2cm subsampled file
3. Build KDTree from 10cm points
4. Query nearest neighbors for all 2cm points
5. Transfer `PredInstance` (and `species_id` if present)

**Output**: `{tile_id}_segmented_remapped.laz`

### Stage 1: Load and Filter

**Purpose**: Load tiles and remove duplicate instances in overlapping buffer zones using centroid-based filtering.

**Algorithm**: Centroid-based filtering

**Process**:
1. Load all tile files with `PredInstance` attribute
2. For each tile, identify neighbors (east, west, north, south)
3. Compute instance centroids
4. Remove instances whose centroids fall within buffer zones facing neighbors
5. Retain instances for tiles that "own" them (based on centroid position)

### Stage 2: Global ID Assignment

**Purpose**: Create unique instance IDs across all tiles.

**Algorithm**: Tile-specific offsets

**Process**:
1. Calculate maximum instance ID per tile
2. Assign sequential offsets to each tile
3. Remap all instance IDs to global unique IDs

### Stage 3: Border Region Instance Matching

**Purpose**: Identify instances that span tile boundaries and group them.

**Algorithm**: Overlap ratio matching with Union-Find grouping

**Process**:
1. Define border zones on edges facing neighbors (buffer to buffer+`border_zone_width`)
2. For instances with centroids in border zones:
   - Match with neighboring tile instances using KDTree overlap ratios
   - Apply centroid proximity check (`max_centroid_distance`)
   - Group matched instances using Union-Find
3. Remap instance IDs so matched instances share the same ID

**Matching Criteria**:
1. **Centroid proximity**: Centroids within `max_centroid_distance` (default: 3m)
2. **Overlap ratio**: Fraction of points that correspond exceeds `overlap_threshold` (default: 30%)

### Stage 4: Merge and Deduplicate

**Purpose**: Merge all tiles and remove duplicate points from buffer regions.

**Algorithm**: KDTree-based point correspondence

**Process**:
1. Concatenate all tile points
2. Build KDTree of all points
3. For each point, check for duplicates within 5cm tolerance
4. Keep only the "canonical" copy (from lower tile index)
5. Write unified LAZ with consistent instance IDs

### Stage 5: Small Volume Instance Merging

**Purpose**: Reassign orphaned tree fragments to nearby larger instances.

**Criteria**:
- Instance convex hull volume < `max_volume_for_merge` (default: 4m³)
- Nearest large instance within search radius

**Species Preservation**: Species ID (if available) is always taken from the larger instance.

### Stage 6: Retiling to Original Files

**Purpose**: Map final instance IDs back to original tile boundaries for per-tile output.

**Algorithm**: cKDTree nearest-neighbor lookup

**Process**:
1. For each original tile file:
   - Filter merged points spatially to tile bounds + buffer
   - Build cKDTree from filtered merged points
   - Query nearest neighbor for all original tile points
   - Add PredInstance dimension with queried values
2. Save updated tile files

### Stage 7: Original File Remapping (Optional)

**Purpose**: Map final instance IDs back to the original input LAZ files (pre-tiling). Only runs if `--original-input-dir` is provided.

**Algorithm**: cKDTree nearest-neighbor lookup

**Process**:
1. For each original input LAZ file:
   - Load all points from the file
   - Filter merged points spatially to file bounds + `retile-buffer`
   - Build cKDTree from filtered merged points
   - Query nearest neighbor for all original points
   - Add PredInstance dimension with queried values
   - Handle unmatched points (distance > `retile-max-radius`) by setting PredInstance=0
2. Save updated files with new PredInstance dimension

**Parameters**:
- `--retile-buffer`: Controls how much to expand the bounding box when filtering merged points (default: 1.0m). Larger values include more merged points in the local KDTree, improving boundary coverage.
- `--retile-max-radius`: Maximum distance threshold for nearest neighbor matching (default: 0.1m). Should be set based on subsampling resolution (e.g., 0.1m for 2cm voxels).

---

## Parameters Reference

### Tile Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile-length` | 100 | Tile size in meters |
| `--tile-buffer` | 5 | Buffer overlap in meters |
| `--threads` | 5 | Spatial chunks per file during subsampling |
| `--workers` | 4 | Parallel file/tile processing |
| `--resolution-1` | 0.02 | First subsampling resolution (2cm) |
| `--resolution-2` | 0.1 | Second subsampling resolution (10cm) |
| `--grid-offset` | 1.0 | Grid offset from min coordinates |
| `--skip-dimension-reduction` | False | Keep all point dimensions |

### Merge Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-resolution` | 2 | Target resolution in cm |
| `--buffer` | 10.0 | Buffer distance for filtering (meters) |
| `--border-zone-width` | 10.0 | Width of border zone beyond buffer for instance matching (meters) |
| `--overlap-threshold` | 0.3 | Overlap ratio for instance matching (30%) |
| `--max-centroid-distance` | 3.0 | Max centroid distance to merge (meters) |
| `--correspondence-tolerance` | 0.05 | Point correspondence tolerance (meters) |
| `--max-volume-for-merge` | 4.0 | Max volume for small instance merge (m³) |
| `--original-input-dir` | None | Directory with original input LAZ files for final remap |
| `--skip-merged-file` | False | Skip creating merged LAZ file (only create retiled outputs) |
| `--workers` | 4 | Parallel processing (tile loading, KDTree queries) |
| `--retile-buffer` | 1.0 | Spatial buffer expansion for filtering merged points during retiling (meters) |
| `--retile-max-radius` | 0.1 | Maximum distance for cKDTree nearest neighbor matching during retiling (meters) |

### Understanding `--workers` vs `--threads`

These two parameters control different aspects of parallelism:

#### `--workers` (Global Parallelism)

Controls how many files/tasks run simultaneously using Python's `ProcessPoolExecutor`:

| Task | What `--workers` Controls |
|------|---------------------------|
| **Tile Task** | Parallel COPC conversions, parallel tile creation |
| **Merge Task** | Parallel tile loading, parallel convex hull computation, KDTree queries (`workers=-1` uses all CPUs) |

**Memory impact**: Higher values = more files in memory simultaneously

#### `--threads` (Per-File Chunking)

Controls spatial chunking during **subsampling only**:

- Each tile is split into `--threads` spatial chunks along the X-axis
- Chunks are processed in parallel using `ProcessPoolExecutor`
- Files are processed **sequentially** (one at a time), but each file's chunks run in parallel

```
Example with --threads=5:
  tile.laz → [chunk_0, chunk_1, chunk_2, chunk_3, chunk_4] → parallel subsample → merge
```

**Memory impact**: Higher values = smaller chunks = less memory per chunk



---

## Input/Output Formats

### Input Requirements

#### Tile Task
- **File formats**: LAZ (compressed) or LAS (uncompressed)
- **Coordinate system**: Should be in a projected CRS (e.g., UTM)
- **Directory structure**: Flat directory with LAZ/LAS files

#### Merge Task
- **Required attribute**: `PredInstance` (integer instance IDs)
- **Optional attributes**: `PredSemantic`, `species_id`
- **File naming**: `c{col}_r{row}*.laz` pattern

### Output Structure

```
output_dir/
├── copc_xyz/                    # COPC files (XYZ-only or full)
│   ├── input_file_1.copc.laz
│   └── input_file_2.copc.laz
│
├── tiles_100m/                  # Tiled point clouds
│   ├── c00_r00.copc.laz
│   ├── c00_r01.copc.laz
│   ├── c01_r00.copc.laz
│   │
│   ├── subsampled_2cm/          # Resolution 1 subsamples
│   │   ├── output_100m_c00_r00_2cm.laz
│   │   └── ...
│   │
│   └── subsampled_10cm/         # Resolution 2 subsamples
│       ├── output_100m_c00_r00_10cm.laz
│       └── ...
│
├── original_with_predictions/   # Original files with PredInstance
│   ├── input_file_1.laz
│   └── input_file_2.laz
│
├── tindex_100m.gpkg             # Spatial index
├── tile_bounds_tindex.json      # Tile metadata
├── tile_jobs_100m.txt           # Processing jobs
├── overview_copc_tiles.png      # Visualization
│
└── logs/                        # Processing logs
```

### Point Cloud Attributes

#### Tile Task Output
- `X`, `Y`, `Z`: 3D coordinates
- Additional dimensions if `--skip-dimension-reduction` is used

#### Merge Task Output
- `X`, `Y`, `Z`: 3D coordinates
- `PredInstance`: Global tree instance ID (consistent across tiles)
- `PredSemantic`: Semantic class (if present in input)
- `species_id`: Tree species ID (if present in input)

---

## Advanced Configuration

### Large Dataset Processing

For datasets exceeding 100GB:

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 200 \
    --tile-buffer 15 \
    --workers 32 \
    --threads 10
```

### High-Precision Processing

For research applications requiring maximum fidelity:

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 50 \
    --tile-buffer 10 \
    --resolution-1 0.01 \
    --resolution-2 0.05 \
    --skip-dimension-reduction
```

### Memory-Constrained Systems

For systems with limited RAM:

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 50 \
    --tile-buffer 5 \
    --workers 2 \
    --threads 2 \
    --resolution-1 0.03 \
    --resolution-2 0.15
```

---

## Troubleshooting

### Common Issues

#### "No LAZ/LAS files found"
- Ensure input files have `.laz` or `.las` extension (lowercase)
- Check that `input_dir` points to the correct directory
- Verify file permissions

#### "pdal: command not found"
- Install PDAL: `conda install -c conda-forge pdal`
- Verify installation: `pdal --version`
- Check PATH environment variable

#### "untwine: command not found"
- Build from source or install via conda: `conda install -c conda-forge untwine`
- Verify: `untwine --help`

#### "Memory allocation failed"
- Reduce `--tile-length` for smaller tiles
- Decrease `--workers` to limit concurrent memory usage
- Use `--resolution-1` and `--resolution-2` with larger values

#### "CRS mismatch" or "Coordinates appear projected"
- Ensure all input files are in the same coordinate reference system
- Use projected CRS (e.g., UTM) not geographic (WGS84)

#### "No PredInstance attribute found"
- Verify segmentation output includes `PredInstance` dimension
- Check attribute names (case-sensitive): `PredInstance`, not `predinstance`

### Debugging

Enable verbose output:

```bash
python src/run.py --task merge --verbose ...
```

Check processing logs in `output_dir/logs/`:

```bash
ls -la output_dir/logs/
cat output_dir/logs/c00_r00_convert.log
```

### Performance Tuning

#### Optimize for SSD
```bash
# Use more workers when I/O is fast
python src/run.py --task tile --workers 16 --threads 10 ...
```

#### Optimize for HDD
```bash
# Reduce parallel I/O
python src/run.py --task tile --workers 2 --threads 2 ...
```

#### Monitor resource usage
```bash
# Run with htop in another terminal
htop -p $(pgrep -f "python src/run.py")
```

---

## Project Structure

```
3dtrees_smart_tile/
├── src/                                 # Python source code
│   ├── run.py                          # Main CLI orchestrator
│   ├── parameters.py                   # Parameter configuration (Pydantic)
│   ├── main_tile.py                    # Tiling pipeline
│   ├── main_subsample.py               # Subsampling pipeline
│   ├── main_remap.py                   # Prediction remapping
│   ├── main_merge.py                   # Merge wrapper
│   ├── merge_tiles.py                  # Core merge implementation
│   ├── filter_buffer_instances.py      # Buffer zone filtering
│   ├── prepare_tile_jobs.py            # Tile job generation
│   ├── get_bounds_from_tindex.py       # Extent calculation
│   └── plot_tiles_and_copc.py          # Visualization
│
├── README.md                           # This documentation
└── README_PARAMETERS.md                # Parameter reference
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `run.py` | CLI entry point, task routing, parameter handling |
| `parameters.py` | Pydantic-based parameter definitions with CLI support |
| `main_tile.py` | COPC conversion, tindex creation, tile extraction |
| `main_subsample.py` | Parallel voxel-based subsampling |
| `main_remap.py` | KDTree-based prediction remapping |
| `main_merge.py` | Merge task orchestration |
| `merge_tiles.py` | Core merging algorithms (Union-Find, overlap ratio) |
| `filter_buffer_instances.py` | Centroid-based buffer zone filtering |
| `prepare_tile_jobs.py` | Tile grid calculation and job list generation |
| `get_bounds_from_tindex.py` | Extent extraction from spatial index |
| `plot_tiles_and_copc.py` | Matplotlib visualization of tiles |

---

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥3.10 | Runtime |
| PDAL | ≥2.5 | Point cloud processing |
| pdal_wrench | Latest | Parallel PDAL operations |
| laspy | Latest | LAZ/LAS file I/O |
| lazrs-python | Latest | LAZ compression |
| NumPy | Latest | Array operations |
| SciPy | Latest | KDTree spatial queries |
| pydantic | ≥2.0 | Parameter validation |
| pydantic-settings | Latest | CLI and env var support |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| matplotlib | Visualization |
| fiona | Vector file handling |
| pyproj | CRS transformations |
| geopandas | Geospatial operations |

### External Tools

| Tool | Purpose |
|------|---------|
| [untwine](https://github.com/hobuinc/untwine) | COPC conversion |
| [pdal_wrench](https://github.com/PDAL/wrench) | Parallel PDAL |

---

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{3dtrees_smart_tile,
  title = {3DTrees Smart Tiling Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-org/3dtrees_smart_tile}
}
```

---

**Questions or issues?** Open an issue on GitHub or contact the maintainers.
