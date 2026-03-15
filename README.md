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
11. [Docker & Automation](#docker--automation)
12. [Troubleshooting](#troubleshooting)
13. [Project Structure](#project-structure)
14. [Dependencies](#dependencies)
15. [License](#license)

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
│  │ Spatial      │     │ Tile Grid    │     │ Two-Phase    │                    │
│  │ Index        │────▶│ Calculation  │────▶│ Tiling       │                    │
│  │ (tindex)     │     │ (bounds)     │     │ (laspy+COPC) │                    │
│  └──────────────┘     └──────────────┘     └──────────────┘                    │
│                                                   │                            │
│                                                   ▼                            │
│                                          ┌───────────────────┐                 │
│                                          │ Multi-Resolution  │                 │
│                                          │ Subsampling       │                 │
│                                          │ (1cm + 10cm)      │                 │
│                                          └───────────────────┘                 │
│                                                   │                            │
│                                                   ▼                            │
│                                           Outputs: tiles_100m/                 │
│                                                    ├─ c00_r00.copc.laz         │
│                                                    ├─ subsampled_1cm/          │
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
│  │ (10cm→1cm)   │     │              │     │ Matching     │                    │
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
- **COPC format output** via untwine (with automatic PDAL fallback) for efficient spatial queries and streaming access
- **Memory-efficient chunking** - tiles are processed independently to minimize memory footprint
- **Parallel subsampling** - each tile is spatially divided into chunks processed concurrently

### Intelligent Tiling
- **Configurable tile size** - default 100m × 100m, adjustable for different use cases
- **Buffer zones** - overlapping regions (default 20m) ensure trees at boundaries are fully captured
- **Spatial indexing** - uses PDAL tindex for efficient data retrieval
- **Data-aligned grid** - tiles start from actual data extent, minimizing empty tiles
- **Smart tiling threshold** - optional single-file bypass for small datasets

### Multi-Resolution Processing
- **Dual subsampling** - generates both 1cm and 10cm resolution outputs (configurable)
- **Voxel-based downsampling** - uses PDAL's `voxelcentroidnearestneighbor` filter
- **Dimension preservation** - all extra dimensions (PredInstance, species_id, etc.) are maintained by default

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
| 1 | **Spatial Index** | Creates a GeoPackage tindex using `pdal tindex` for efficient spatial queries across all input files. |
| 2 | **Tile Bounds** | Calculates optimal tile grid based on data extent, tile size (default: 100m), and buffer (default: 20m) parameters. |
| 3a | **Phase 1: Distribute** | Reads each source LAZ/LAS file once (in memory-efficient chunks via laspy), distributes points to overlapping tiles as intermediate part files. Per-tile offsets prevent int32 overflow. |
| 3b | **Phase 2: COPC Conversion** | Merges part files and converts each tile to COPC format using untwine (fast, automatic fallback to PDAL). |
| 4 | **Subsampling R1** | Downsamples COPC tiles to resolution 1 (default: 1cm) using parallel spatial chunk processing. |
| 5 | **Subsampling R2** | Further downsamples to resolution 2 (default: 10cm) for neural network inference. |

#### MERGE TASK: Result Integration

| Stage | Component | Description |
|-------|-----------|-------------|
| 0 | **Prediction Remapping** | Transfers PredInstance labels from 10cm predictions to 1cm resolution using KDTree nearest-neighbor lookup. |
| 1 | **Load and Filter** | Loads tiles, applies centroid-based buffer zone filtering to remove duplicate instances. |
| 2 | **Global ID Assignment** | Creates unique instance IDs across all tiles using tile-specific offsets. |
| 3 | **Cross-tile Matching** | Identifies matching instances in tile overlaps using overlap ratio (Union-Find grouping). |
| 3b | **Orphan Recovery** | Recovers filtered instances that no neighbor "covers", so no trees are lost. |
| 4 | **Merge and Deduplicate** | Combines all tiles, removes duplicate points from overlapping buffer regions. |
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
    untwine \
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

Merge segmented tiles (requires **tile_bounds_tindex.json** from the Tile task):

```bash
python src/run.py --task merge \
    --subsampled-segmented-folder /path/to/subsampled_10cm \
    --subsampled-target-folder /path/to/subsampled_1cm \
    --tile_bounds_json /path/to/tile_bounds_tindex.json \
    --output-folder /path/to/out \
    --output-merged-laz /path/to/out/merged.laz
```

Optional: add `--original-input-dir /path/to/original` to also write per-original-file outputs with predictions.

### Basic Remap Task (merged file → original files)

Add all dimensions from a merged LAZ file to your original files:

```bash
python src/run.py --task remap_to_originals \
    --merged-laz /path/to/merged.laz \
    --original-input-dir /path/to/original/files \
    --output-dir /path/to/output
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
    --tile-buffer 20 \                     # Buffer overlap in meters (default: 20)
    --resolution-1 0.01 \                  # First resolution (default: 1cm)
    --resolution-2 0.1 \                   # Second resolution (default: 10cm)
    --workers 8 \                          # Parallel workers (default: 4)
    --threads 10                           # Threads per COPC writer (default: 10)
```

### Merge Task Options

**Required:** `--subsampled-segmented-folder`, `--subsampled-target-folder`, `--tile_bounds_json` (from Tile task).

```bash
python src/run.py --task merge \
    --subsampled-segmented-folder /path/to/10cm \   # Segmented 10cm tiles
    --subsampled-target-folder /path/to/1cm \      # Subsampled 1cm tiles (remap target)
    --tile_bounds_json /path/to/tile_bounds_tindex.json \
    --output-folder /path/to/out \                  # Optional; default: parent of segmented
    --output-merged-laz /path/to/out/merged.laz \  # Optional; merged LAZ path
    --original-input-dir /path/to/original \        # Optional; for per-original-file outputs
    --buffer 10.0 \                                # Buffer zone distance (default: 10m)
    --overlap-threshold 0.3 \                      # Instance matching (default: 0.3)
    --max-centroid-distance 3.0 \                   # Max centroid distance (default: 3m)
    --workers 8 \                                  # Parallel workers (default: 4)
    --disable-matching                             # Disable cross-tile matching
```

---

## Pipeline Stages

### Stage 1: Spatial Indexing

**Purpose**: Create a spatial index (tindex) for efficient querying across all input files.

**Technology**: Uses `pdal tindex` to create a GeoPackage with file boundaries.

**Output**: `tindex_100m.gpkg` containing polygons representing each input file's extent.

### Stage 2: Tile Grid Calculation

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

### Stage 3: Tile Creation (Two-Phase)

**Purpose**: Distribute points from source LAZ/LAS files into spatially partitioned tiles in COPC format.

#### Phase 1: Distribute

Each source file is read once using laspy (in memory-efficient chunks controlled by `--chunk-size`). Points are distributed to all overlapping tiles as intermediate `.las` part files. Per-tile offsets are computed from tile bounds to prevent int32 overflow in scaled coordinates.

**Parallelization**: One source file at a time, but tile writing is batched.

#### Phase 2: COPC Conversion

All part files for each tile are merged and converted to COPC format. The pipeline tries **untwine** first (fast, purpose-built for COPC generation) and automatically falls back to PDAL's `writers.copc` if untwine is not available.

**Parallelization**: Multiple tiles converted concurrently (controlled by `--workers`).

**Output**: `c{col}_r{row}.copc.laz` files in `tiles_{tile_length}m/`.

**Options**:
- **Default (preserve all dimensions)**: Keeps all point attributes (PredInstance, species_id, etc.)
- **XYZ-only**: Set `--skip-dimension-reduction false` to reduce to X, Y, Z only (useful for raw pre-segmentation data)

### Stage 4-5: Multi-Resolution Subsampling

**Purpose**: Create downsampled versions for efficient neural network processing.

**Algorithm**: Voxel-based downsampling using `filters.voxelcentroidnearestneighbor`

**Process**:
1. Each tile is split spatially into chunks along X-axis
2. Chunks are processed in parallel
3. Results are merged back into single file

**Outputs**:
- `subsampled_2cm/`: Resolution_1 files
- `subsampled_10cm/`: Resolution_2 files

### The Merge Process in Detail

The merge process takes segmented tiles (each with local `PredInstance` IDs) and produces a single unified point cloud where every tree has a globally unique ID, even trees that were split across tile boundaries. It also writes per-tile outputs and optionally maps predictions back to the original input files.

The diagram below shows the spatial layout. Tiles overlap by a configurable buffer (default 10 m). Trees near tile edges appear in both tiles with **different** local instance IDs. The merge must figure out which IDs in adjacent tiles refer to the same physical tree and unify them.

```
Tile A                           Tile B
┌──────────────────────┐  ┌──────────────────────┐
│                      │  │                      │
│   Tree 42            │  │            Tree 17   │
│      ╲               │  │               ╱      │
│       ╲   buffer     │  │    buffer    ╱       │
│        ╲  zone ──────┤  ├──── zone    ╱        │
│         ╲   ▓▓▓▓▓▓▓▓▓│  │▓▓▓▓▓▓▓▓▓  ╱         │
│          ╲  ▓overlap▓│  │▓overlap▓  ╱          │
│           ╲ ▓▓▓▓▓▓▓▓▓│  │▓▓▓▓▓▓▓▓▓ ╱           │
│            ╲──────────┤  ├──────────╱            │
│                      │  │                      │
│   Tree 42 and Tree 17 are the SAME physical    │
│   tree — the merge must unify them.            │
└──────────────────────┘  └──────────────────────┘
```

Below is each stage explained.

---

### Stage 0: Prediction Remapping (10 cm to 1 cm)

**What it does**: The segmentation model ran on 10 cm subsampled tiles. This stage transfers the `PredInstance` labels from the coarse 10 cm points to the finer 1 cm subsampled points using nearest-neighbor lookup, so subsequent stages work at 1 cm resolution.

**How it works**:
1. For each tile, load the 10 cm segmented file and the corresponding 1 cm subsampled file.
2. Build a cKDTree from the 10 cm points.
3. For every 1 cm point, find the closest 10 cm point and copy its `PredInstance` (and `species_id` if present).
4. Save the result as `{tile_id}_segmented_remapped.laz`.

**Why**: Segmentation at 10 cm is fast but loses detail. Remapping to 1 cm gives the merge much denser point clouds to work with, which improves overlap detection and deduplication accuracy.

---

### Stage 1: Load and Filter (Centroid-Based Buffer Filtering)

**What it does**: Loads all tiles and removes instances whose centroid is in the buffer zone on any side that has a neighbor tile. This eliminates most duplicates before the expensive matching stage.

**How it works**:

1. Load each LAZ tile; read XYZ coordinates, `PredInstance`, and any extra dimensions (e.g. `species_id`).
2. Use the `tile_bounds_tindex.json` file (from the Tile task) to determine which tiles are neighbors (east/west/north/south).
3. For each tile, compute the centroid of every instance.
4. An instance is **filtered** (removed) if its centroid falls in the buffer zone on a side that has a neighbor. The rule is: the tile with the lower column index (west) or lower row index (south) "owns" instances in the overlap.
5. Instances **not** in any buffer zone are **kept**.

**Result**: Each tile now has a set of "kept" instances and a set of "filtered" instances. Filtered instances are candidates for orphan recovery later.

```
Tile boundary:
┌──────────────────────────────────────┐
│ buffer │                     │ buffer│
│  zone  │     CORE AREA       │  zone │
│ (west) │  (instances here     │(east) │
│        │   are always kept)   │       │
│ ▓▓▓▓▓▓ │                     │▓▓▓▓▓▓│
│ filtered│                     │filtered│
│ if west │                     │if east│
│ neighbor│                     │neighbor│
└──────────────────────────────────────┘
```

---

### Stage 2: Global ID Assignment

**What it does**: Assigns globally unique IDs to every kept instance across all tiles.

**How it works**:
- Each tile gets an offset: `tile_idx * 100000`.
- A local instance ID `42` in tile 3 becomes global ID `300042`.
- A Union-Find data structure is initialized with one entry per kept instance, tracking its point count (used later to preserve the species ID from the larger fragment).

**Why**: Local IDs are only unique within one tile. Subsequent stages need IDs that are unique across the entire dataset.

---

### Stage 3: Border Region Instance Matching

**What it does**: Finds instances that represent the **same physical tree** across tile boundaries and groups them using Union-Find, so they end up with the same final ID.

**How it works**:

1. **Identify border instances**: For each tile, find kept instances whose centroid lies in the "border region" (the strip from `buffer` to `buffer + border_zone_width` meters from the tile edge, on sides with a neighbor). Only these instances can possibly match a counterpart in a neighbor tile.

2. **For each neighbor pair** (e.g. tile A east <-> tile B west): compare border instances from tile A facing east with border instances from tile B facing west:
   - **Bounding box check** (fast filter): Skip pairs whose 2D bounding boxes don't overlap or come within 10 cm. This eliminates most non-matching pairs cheaply.
   - **FF3D overlap ratio** (expensive check): For surviving pairs, compute point-to-point correspondence using hash-based grid matching (O(n) per pair). The overlap ratio is `max(intersection/size_a, intersection/size_b)` -- this handles asymmetric cases where one fragment is much larger than the other.
   - **Match decision**: If the overlap ratio exceeds `overlap_threshold` (default 0.3 = 30%), the pair is matched. Union-Find merges their global IDs into one group.

3. **Species ID preservation**: When instances are grouped, the species ID is always taken from the **larger** instance (by point count).

**Result**: A mapping from every global ID to a "merged ID". Matched instances share the same merged ID.

```
Tile A (east border)         Tile B (west border)
┌──────────┐                 ┌──────────┐
│          │   border zone   │          │
│ inst 42 ─┼─────────────────┼─ inst 17 │  overlap ratio = 0.65
│ (500 pts)│                 │ (480 pts)│  > threshold 0.3 → MATCH
│          │                 │          │
│ inst 99 ─┼─────────────────┼─ (none)  │  no counterpart → no match
│          │                 │          │
└──────────┘                 └──────────┘
```

---

### Stage 3b: Orphan Recovery

**What it does**: Recovers filtered instances that would otherwise be lost because no neighbor tile "owns" them (e.g. a tree whose centroid landed in the buffer zone of **both** tiles due to slightly different segmentation results).

**How it works**:

1. Compute bounding boxes **only** for filtered instances and kept instances in the border region (not all instances -- this is fast).
2. Build a cKDTree of the centers of all border-region kept instances.
3. For each filtered (orphan) instance:
   - Query the tree for kept instances within 30 m.
   - For each nearby kept instance, check if their points overlap (>50% of the orphan's points are within 1 m of the neighbor's points, using a per-neighbor cKDTree that is cached so each neighbor tree is built at most once).
   - If a covering instance is found, the orphan is skipped (it already exists in a neighbor tile).
   - If **no** covering instance is found, the orphan is **recovered**: it gets a new unique merged ID and is added back to the kept set.

**Why**: Without this step, trees at tile corners (where buffer zones of multiple tiles overlap) can be lost entirely.

---

### Stage 4: Merge and Deduplicate

**What it does**: Concatenates all kept points from all tiles into a single point cloud and removes duplicate points that exist in overlapping buffer regions.

**How it works**:

1. For each tile, remap local instance IDs to their final merged IDs (from Stage 3). Points belonging to filtered instances that were not recovered get ID `-1` and are discarded.
2. Concatenate all tile points, instance arrays, and extra dimension arrays.
3. **Deduplication**: Uses grid-based spatial hashing (not KDTree) for O(n) performance:
   - Divide the point cloud into 50 m grid cells.
   - Within each cell, hash each point's coordinates at 1 cm resolution.
   - Points with the same hash in the same cell are duplicates; keep the one with the higher instance ID.

**Result**: A single merged point cloud with no duplicate points and consistent instance IDs.

---

### Stage 5: Small Volume Instance Merging

**What it does**: Reassigns tiny tree fragments (orphaned clusters) to the nearest large instance.

**How it works**:

1. For each instance with a positive ID, compute its 3D convex hull volume (in parallel using ProcessPoolExecutor).
2. Classify instances as "small" if: volume < `max_volume_for_merge` (default 4 m3) **or** point count < `min_cluster_size` (default 300).
3. Build a cKDTree from the centroids of all "large" instances.
4. For each small instance, find the nearest large instance by centroid distance.
5. Reassign all points of the small instance to the nearest large instance (vectorized lookup table, no per-point loop).

**Species preservation**: The species ID of the receiving (larger) instance is kept unchanged. Small fragments do not overwrite species.

---

### Stage 6: Retiling to Original Tile Files

**What it does**: Maps the final merged instance IDs back onto each original tile's point cloud, so you get per-tile output files with globally consistent IDs.

**How it works**:

1. For each original tile file:
   - Read the tile's bounding box from its header.
   - Filter the merged point cloud to points within `tile bounds + spatial buffer`.
   - Build a cKDTree from those filtered merged points.
   - Read the original tile's points and query the tree for the nearest merged point for each original point.
   - Write the tile with the matched `PredInstance` (and any extra dims like `species_id`).

**Parallelism**: Can process multiple tiles concurrently using `--workers`.

---

### Stage 7: Original File Remapping (Optional)

**What it does**: If `--original-input-dir` is provided, maps the final merged instance IDs back to the original input LAZ files (before any tiling was done). This is useful when you want predictions on the exact original files.

**How it works**: Same algorithm as Stage 6 (spatial filter + cKDTree build + nearest-neighbor query), but applied to the original pre-tiling files instead of the tile files. Every point gets the instance ID of its nearest merged point (no distance cutoff).

---

## Parameters Reference

### Tile Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile-length` | 100 | Tile size in meters |
| `--tile-buffer` | 20 | Buffer overlap in meters |
| `--threads` | 10 | Threads per COPC writer |
| `--workers` | 4 | Parallel file/tile processing |
| `--resolution-1` | 0.01 | First subsampling resolution (1cm) |
| `--resolution-2` | 0.1 | Second subsampling resolution (10cm) |
| `--skip-dimension-reduction` | False | Skip XYZ-only reduction, keep all point dimensions |
| `--chunk-size` | 20000000 | Points per chunk when reading LAZ/LAS in Phase 1 (smaller = less peak RAM) |
| `--tiling-threshold` | None | File size threshold in MB for skipping tiling on single small files |

### Merge Task Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile_bounds_json` | **Required** | Path to tile_bounds_tindex.json from Tile task |
| `--buffer` | 10.0 | Buffer distance for filtering (meters) |
| `--border-zone-width` | 10.0 | Width of border zone beyond buffer for instance matching (meters) |
| `--overlap-threshold` | 0.3 | Overlap ratio for instance matching (30%) |
| `--max-centroid-distance` | 3.0 | Max centroid distance to merge (meters) |
| `--max-volume-for-merge` | 4.0 | Max volume for small instance merge (m³) |
| `--min-cluster-size` | 300 | Minimum cluster size in points for reassignment |
| `--original-input-dir` | None | Optional: directory with original input LAZ files for per-file outputs |
| `--skip-merged-file` | False | Skip creating merged LAZ file (only create retiled outputs) |
| `--disable-matching` | False | Disable cross-tile instance matching |
| `--disable-volume-merge` | False | Disable small volume instance merging |
| `--workers` | 4 | Parallel processing (tile loading, KDTree queries) |
| `--tolerance` | 5.0 | Max difference in meters for bounds matching (remap task) |

*Retile buffer is fixed internally at 2.0 m; correspondence tolerance is no longer a user parameter.*

### Understanding `--workers` vs `--threads`

These two parameters control different aspects of parallelism:

#### `--workers` (Global Parallelism)

Controls how many files/tasks run simultaneously using Python's `ProcessPoolExecutor`:

| Task | What `--workers` Controls |
|------|---------------------------|
| **Tile Task** | Parallel COPC conversions, parallel tile creation |
| **Merge Task** | Parallel tile loading, parallel convex hull computation, KDTree queries |

**Memory impact**: Higher values = more files in memory simultaneously

#### `--threads` (Per-File Chunking)

Controls spatial chunking during **subsampling only**:

- Each tile is split into `--threads` spatial chunks along the X-axis (default: 5)
- Chunks are processed in parallel using `ProcessPoolExecutor`
- Files are processed **sequentially** (one at a time), but each file's chunks run in parallel

```
Example with --threads=5:
  tile.laz → [chunk_0, chunk_1, chunk_2, chunk_3, chunk_4] → parallel subsample → merge
```

**Memory impact**: Higher values = smaller chunks = less memory per chunk

#### `--num-spatial-chunks`

Optional override for number of spatial chunks per tile during subsampling. If not specified, defaults to the value of `--workers`.



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
├── tiles_100m/                  # Tiled point clouds (100m default)
│   ├── c00_r00.copc.laz         # COPC tiles (Phase 2 output)
│   ├── c00_r01.copc.laz
│   ├── c01_r00.copc.laz
│   │
│   ├── subsampled_1cm/          # Resolution 1 subsamples (1cm default)
│   │   ├── output_100m_c00_r00_1cm.laz
│   │   └── ...
│   │
│   ├── subsampled_10cm/         # Resolution 2 subsamples
│   │   ├── output_100m_c00_r00_10cm.laz
│   │   └── ...
│   │
│   ├── segmented_remapped/      # Remapped predictions (merge task)
│   │   ├── c00_r00_segmented_remapped.laz
│   │   └── ...
│   │
│   └── output_tiles/            # Final per-tile outputs with merged IDs
│       ├── c00_r00.copc.laz
│       └── ...
│
├── original_with_predictions/   # Original files with PredInstance (merge task)
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
- All extra dimensions preserved by default (PredInstance, species_id, etc.)
- Set `--skip-dimension-reduction false` to reduce to XYZ only (for raw pre-segmentation data)

#### Merge Task Output
- `X`, `Y`, `Z`: 3D coordinates
- `PredInstance`: Global tree instance ID (consistent across tiles)
- `PredSemantic`: Semantic class (if present in input)
- `species_id`: Tree species ID (if present in input)

---

## Advanced Configuration

### Large Dataset Processing

For datasets exceeding 100GB with larger tiles:

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 500 \
    --tile-buffer 30 \
    --workers 32 \
    --threads 10
```

### High-Precision Processing

For research applications requiring maximum fidelity (already default):

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 300 \
    --tile-buffer 20 \
    --resolution-1 0.01 \
    --resolution-2 0.05
```

### Memory-Constrained Systems

For systems with limited RAM (smaller tiles, lower resolution):

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tile-length 100 \
    --tile-buffer 10 \
    --workers 2 \
    --threads 2 \
    --resolution-1 0.02 \
    --resolution-2 0.15
```

### Single Small File Processing

For processing single files without tiling:

```bash
python src/run.py --task tile \
    --input-dir /data/input \
    --output-dir /data/output \
    --tiling-threshold 1000  # Skip tiling if single file < 1000 MB
```

---

## Docker & Automation

The pipeline includes Docker support for containerized execution and automated workflows with resource monitoring.

### Docker Setup

Build the Docker image:

```bash
docker build -t 3dtrees_smart_tile .
```

Run the pipeline in Docker:

```bash
./run_docker.sh              # Tile task: input → tiled + subsampled (1cm, 10cm)
./run_docker_merge.sh        # Merge task: segmented 10cm + 1cm + tile_bounds JSON → merged.laz
./run_docker_remap.sh        # Remap task: merged.laz + original files → originals with all merged dimensions
```

Edit the path variables at the top of each script to match your data. Merge requires `tile_bounds_tindex.json` from the Tile task output.

### Automated Pipeline

For fully automated execution with resource monitoring, see [AUTOMATION_README.md](AUTOMATION_README.md).

Quick start:

```bash
./run_automated_pipeline.sh 1
```

This automated workflow:
- Downloads data from S3
- Runs tile task (COPC conversion, tiling, subsampling)
- Adds dummy PredInstance dimension (for testing)
- Runs remap_merge task with auto-detection of single vs multi-file workflows
- Tracks CPU and RAM usage throughout the process
- Generates resource usage logs

### Additional Documentation

- [AUTOMATION_README.md](AUTOMATION_README.md) - Detailed automation and Docker workflow guide
- [CLAUDE.md](CLAUDE.md) - Quick reference for AI assistants and developers
- [Dockerfile](Dockerfile) - Container configuration

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
- The pipeline automatically falls back to PDAL's `writers.copc` if untwine is not installed, so this is not an error — just slower COPC conversion.
- To install for better performance: `conda install -c conda-forge untwine`
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
├── AUTOMATION_README.md                # Automation and Docker guide
├── CLAUDE.md                           # Quick reference for developers
├── Dockerfile                          # Container configuration
├── run_automated_pipeline.sh           # Automated workflow orchestrator
├── run_docker.sh                       # Docker: Tile task
├── run_docker_merge.sh                 # Docker: Merge task (requires tile_bounds JSON)
├── run_docker_remap.sh                 # Docker: Remap task (merged file → original files)
└── .gitignore                          # Git ignore rules
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `run.py` | CLI entry point, task routing, parameter handling |
| `parameters.py` | Pydantic-based parameter definitions with CLI support |
| `main_tile.py` | Two-phase tiling (distribute + COPC conversion), tindex creation |
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
| PDAL | ≥2.5 | Point cloud processing, subsampling |
| untwine | Latest | Fast COPC conversion (auto-fallback to PDAL if unavailable) |
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
| [untwine](https://github.com/hobuinc/untwine) | COPC conversion (preferred, auto-fallback to PDAL) |

---

## License

MIT License

Copyright (c) 2026

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

Contributions are welcome! Please open an issue or pull request on the project repository.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{3dtrees_smart_tile,
  title = {3DTrees Smart Tiling Pipeline},
  author = {},
  year = {2026},
  url = {https://github.com/your-org/3dtrees_smart_tile}
}
```

---

**Questions or issues?** Open an issue on GitHub or contact the maintainers.
