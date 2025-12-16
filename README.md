# Tile Merger

Merge overlapping segmented point cloud tiles into a single homogeneous point cloud with consistent instance IDs.

TreeDivNet: For the data processing the data was tiled into 200m & 300m tiles with an overlap of 30m in each direction.  

## Overview

The tile merger takes **overlapping segmented LAZ tiles** (with potentially different instance segmentations in overlap regions) and produces a **single merged point cloud** where:

- Duplicate points are removed
- Same physical tree gets the same instance ID across all tiles
- Small fragments are merged to larger instances
- Species IDs are preserved from the largest instances

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Segmented LAZ tiles with overlap (c00_r00.laz, c00_r01.laz, ...)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Buffer Zone Filtering                                             │
│  - Remove instances whose XY centroid is in buffer zone on inner edges      │
│  - Only affects edges that have neighboring tiles                           │
│  - Keeps "whole" instances in each tile's core region                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Merge and Deduplicate                                             │
│  - Concatenate all tile points into single point cloud                      │
│  - Remove duplicate points (within 1mm tolerance)                           │
│  - When duplicates exist: keep point with higher instance ID                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Cross-Tile Instance Matching                                      │
│  - Find overlapping tile pairs                                              │
│  - Compute overlap ratio: max(intersection/size_A, intersection/size_B)     │
│  - Merge instances if: overlap >= threshold AND centroid_dist < max_dist    │
│  - Union-Find handles transitive merging (A=B, B=C → A=B=C)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Small Volume Instance Merging                                     │
│  - For instances with <10,000 points: calculate convex hull volume          │
│  - If volume < threshold: merge to nearest large instance by centroid       │
│  - Species ID always taken from larger instance                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: Retile to Original Files (Optional)                               │
│  - Map merged instance IDs back to original high-resolution tiles           │
│  - Uses KDTree for efficient point matching                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Single merged LAZ with PredInstance and species_id                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Merge

```bash
./main_merge.sh /path/to/segmented_tiles \
    --output-merged /path/to/merged.laz
```

### Full Pipeline with Retiling

```bash
./main_merge.sh /path/to/segmented_tiles \
    --original-tiles-dir /path/to/original_tiles \
    --output-merged /path/to/merged.laz \
    --output-tiles-dir /path/to/retiled_output
```

### Example with Custom Parameters

```bash
./main_merge.sh /data/segmented_remapped \
    --output-merged /data/merged.laz \
    --buffer 15.0 \
    --overlap-threshold 0.2 \
    --max-centroid-distance 5.0 \
    --max-volume-for-merge 6.0 \
    --verbose
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--buffer` | 10.0 | Buffer zone distance (meters) for filtering edge instances |
| `--overlap-threshold` | 0.3 | Minimum overlap ratio (0-1) to merge instances |
| `--max-centroid-distance` | 3.0 | Maximum centroid distance (meters) to merge instances |
| `--max-volume-for-merge` | 4.0 | Maximum convex hull volume (m³) for small instance merging |
| `--num-threads` | 8 | Number of threads for parallel processing |

### Optional Flags

| Flag | Description |
|------|-------------|
| `--disable-matching` | Skip cross-tile instance matching (Stage 3) |
| `--disable-overlap-check` | Merge based on centroid distance only (ignore overlap ratio) |
| `--disable-volume-merge` | Skip small volume instance merging (Stage 4) |
| `--verbose`, `-v` | Print detailed merge decisions |

## Input Requirements

### File Format
- LAZ or LAS files

### Naming Convention
Tiles must follow the pattern: `c{col}_r{row}*.laz`

Examples:
- `c00_r00_segmented_remapped.laz`
- `c01_r00_segmented.laz`
- `c00_r01.laz`

### Required Attributes
- **Instance IDs**: `PredInstance` or `treeID` (int32)

### Optional Attributes
- **Species IDs**: `species_id` (int32)

## Output

### Merged LAZ File
Single point cloud containing:
- All unique points (duplicates removed)
- `PredInstance`: Continuous instance IDs (1, 2, 3, ...)
- `species_id`: Species IDs preserved from largest instances

### Retiled Files (Optional)
Original tile files with updated:
- `PredInstance`: Merged instance IDs
- `species_id`: Merged species IDs

## Algorithm Details

### Buffer Zone Filtering

Instances are removed if their XY centroid falls within the buffer zone on an edge that has a neighboring tile:

```
┌──────────────────────────────────────┐
│            Buffer Zone               │
│  ┌────────────────────────────────┐  │
│  │                                │  │
│  │         Core Region            │  │
│  │    (instances kept here)       │  │
│  │                                │  │
│  └────────────────────────────────┘  │
│            Buffer Zone               │
└──────────────────────────────────────┘
         ↑ Only on edges with neighbors
```

### Overlap Ratio (FF3D-style)

The overlap ratio between two instances is calculated as:

```
overlap_ratio = max(intersection / size_A, intersection / size_B)
```

Where `intersection` is the number of corresponding points (within 5cm tolerance).

This is more lenient than IoU for asymmetric overlaps - if a small instance is fully contained in a larger one, it will still match.

### Species ID Preservation

Throughout all merging operations:
- When two instances merge, the **species_id from the larger instance** (by point count) is used
- This ensures taxonomic information is preserved from the most complete segmentation

## Preprocessing (Optional)

For more control, you can pre-filter buffer instances as a separate step:

```bash
./main_filter_buffer.sh /path/to/segmented_tiles \
    --output-dir /path/to/filtered \
    --buffer 10.0
```

This **removes points entirely** (not just sets instance to 0), useful when you want cleaner tile boundaries before merging.

## Troubleshooting

### Too many instances being merged together
- Increase `--overlap-threshold` (e.g., 0.5)
- Decrease `--max-centroid-distance` (e.g., 2.0)
- Use `--verbose` to see merge decisions

### Instances not merging that should
- Decrease `--overlap-threshold` (e.g., 0.1)
- Increase `--max-centroid-distance` (e.g., 5.0)
- Use `--disable-overlap-check` to merge based on distance only

### Small fragments remaining as separate instances
- Increase `--max-volume-for-merge` (e.g., 8.0)
- Check if the fragments are larger than 10,000 points (won't be processed)

### Duplicate points remaining
- Check if points are truly identical (the deduplication uses 1mm tolerance)
- Ensure input tiles have consistent coordinate systems

## Files

| File | Description |
|------|-------------|
| `main_merge.sh` | Shell wrapper script |
| `merge_tiles.py` | Main Python implementation |
| `filter_buffer_instances.py` | Standalone buffer filtering script |
| `main_filter_buffer.sh` | Shell wrapper for buffer filtering |
| `parameters.py` | Default parameter configuration |

## Dependencies

- Python 3.8+
- numpy
- laspy (with lazrs backend)
- scipy




