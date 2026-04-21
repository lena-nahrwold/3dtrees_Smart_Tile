# 3DTrees SmartTile

Point cloud tiling, border-instance filtering, and prediction remapping for the 3DTrees pipeline.

## Tasks

| Task | Purpose |
|------|---------|
| **tile** | COPC-normalize input files, create overlapping spatial tiles, write two subsampled output collections |
| **filter** | Remove duplicate instances in tile overlap zones; optionally remap filtered results onto target files |
| **remap** | Transfer prediction dimensions from segmented tiles back onto original (and/or subsampled) files |

## Quick start

```bash
# Show all parameters
python src/run.py --show-params

# Tile
python src/run.py --task tile \
    --input-dir /data/input --output-dir /data/output \
    --tile-length 300 --tile-buffer 20 \
    --resolution-1 0.01 --resolution-2 0.1 \
    --output-copc-res1 True \
    --output-copc-res2 False

# Filter (tile_bounds_json is optional — neighbors are auto-detected from cXX_rYY filenames)
# Segmented inputs may be LAZ/LAS/COPC.
python src/run.py --task filter \
    --segmented-folders /data/segmented \
    --tile-bounds-json /data/tile_bounds_tindex.json \
    --output-dir /data/output \
    --instance-dimension PredInstance_SAT

# Filter + remap in one step
python src/run.py --task filter \
    --segmented-folders /data/segmented \
    --output-dir /data/output \
    --instance-dimension PredInstance_SAT \
    --remap-merge True \
    --original-input-dir /data/originals \
    --subsampled-target-folder /data/subsampled \
    --transfer-original-dims-to-merged True

# Standalone remap (multiple segmentation sources)
# Source collections and remap targets may be LAZ/LAS/COPC.
python src/run.py --task remap \
    --segmented-folders /data/sat_tiles,/data/rct_tiles \
    --original-input-dir /data/originals \
    --output-dir /data/output \
    --produce-merged-file True \
    --transfer-original-dims-to-merged True
```

## Docker

```bash
docker build -t 3dtrees-smart-tile .

docker run -v /path/to/data:/data 3dtrees-smart-tile \
    --task filter --segmented-folders /data/segmented --output-dir /data/output
```

## Key behaviors

**Tile task**
- Converts all inputs to COPC via untwine (preserves all dimensions).
- Creates overlapping tiles on a `cXX_rYY` grid with configurable buffer.
- Writes two subsampled collections (default 1 cm and 10 cm); both res1 and res2 can be emitted as COPC.
- Outputs `tile_bounds_tindex.json` for downstream neighbor resolution.

**Filter task**
- `tile_bounds_json` is **optional**. When omitted, neighbors and border zone width are derived from `cXX_rYY` filename coordinates and spatial overlap of header bounds.
- Accepts segmented inputs as `.las`, `.laz`, or `.copc.laz`.
- Removes instances whose anchor (centroid by default) falls in the overlap zone of a neighbor tile.
- Omits filtered tile files that contain only buffer-region points and records that state in `filtered_tile_manifest.json` and, when available, an annotated output copy of `tile_bounds_tindex.json`.
- Small-cluster reassignment merges tiny fragments into nearby large instances.
- With `--remap-merge`, the filter tail runs a full remap onto originals and/or subsampled targets.

**Remap task**
- Supports multiple segmentation sources (comma-separated `--segmented-folders`).
- Accepts COPC collections and COPC remap targets in addition to plain LAZ/LAS.
- Transfers prediction dims to original files via KDTree nearest-neighbor matching.
- Enrichment step adds original-file attributes (intensity, RGB, etc.) to the merged output.
- RGB promotion: when `red`, `green`, `blue` are present, outputs use LAS point format 7 (standard RGB fields) instead of format 6.

## Output structure

```
output_dir/
  # tile task
  original_copc/             COPC-normalized sources
  subsampled_res1/           1 cm subsampled tiles
  subsampled_res2/           10 cm subsampled tiles
  tile_bounds_tindex.json    tile layout metadata
  overview_copc_tiles.png    tile grid preview

  # filter task
  filtered_tiles/            filtered LAZ tiles
  filtered_tiles_copc/       filtered COPC tiles (when prepared / reused)
  filtered_trees/            filtered tree TXT sidecars
  filtered_tile_manifest.json tile creation/omission state
  tile_bounds_tindex.json    annotated tile layout with per-tile filter_output status

  # remap / filter+remap
  original_with_predictions/ per-original-file outputs
  subsampled_with_predictions/ per-subsampled-file outputs
  merged_with_all_dims.laz   merged output (prediction dims only)
  merged_with_originals.laz  merged output enriched with original attributes
  tiles_with_original_dimensions/  enriched per-tile files
```

## Parameters

Run `python src/run.py --show-params` for the full list. Key parameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--task` | — | `tile`, `filter`, or `remap` |
| `--tile-length` | 300 | tile edge in meters |
| `--tile-buffer` | 20 | overlap buffer in meters |
| `--resolution-1` / `--resolution-2` | 0.01 / 0.1 | subsampling resolutions |
| `--output-copc-res1` / `--output-copc-res2` | True / False | emit res1/res2 outputs as `.copc.laz` instead of `.laz` |
| `--tile-bounds-json` | optional | tile layout JSON; filter auto-detects from filenames when absent |
| `--instance-dimension` | `PredInstance` | name of the instance-ID dimension |
| `--border-zone-width` | auto | derived from JSON, spatial overlap, or explicit override |
| `--segmented-folders` | — | comma-separated paths to segmented tile folders |
| `--original-input-dir` | — | original pre-tiling files for remap |
| `--subsampled-target-folder` | — | subsampled files to remap onto |
| `--produce-merged-file` | True | write a single merged LAZ |
| `--transfer-original-dims-to-merged` | True | enrich merged file with original attributes |
| `--standardization-json` | — | limits which original dims are transferred |
| `--chunk-size` | 20M | streaming chunk size (points) |
| `--workers` | 4 | parallel workers |

## Dependencies

Python 3.10+, laspy, lazrs, numpy, scipy, pydantic, pydantic-settings, matplotlib, fiona, pyproj, PDAL, untwine.

## License

MIT
