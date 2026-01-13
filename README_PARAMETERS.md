# Parameter Configuration Guide

The 3DTrees pipeline uses Pydantic BaseSettings for parameter configuration, supporting CLI arguments and environment variables.

## Quick Start

### View Current Parameters

```bash
python src/run.py --show-params
```

### Basic Tile Task

```bash
python src/run.py --task tile \
  --input-dir /path/to/input \
  --output-dir /path/to/output
```

### Basic Merge Task

```bash
python src/run.py --task merge \
  --subsampled-10cm-folder /path/to/10cm \
  --original-input-dir /path/to/input
```

## Priority Order

Parameters are applied in this order (highest priority last):

1. **Default values** (defined in `parameters.py`)
2. **Environment variables** (e.g., `TILE_LENGTH=150`)
3. **CLI arguments** (e.g., `--tile-length 150`)

## Parameter Reference

### Common Parameters

| CLI Argument | Default | Description |
|--------------|---------|-------------|
| `--task` | `tile` | Task to perform: `tile` or `merge` |
| `--input-dir` | None | Input directory with LAZ/LAS files (required for tile) |
| `--output-dir` | None | Output directory (required for tile) |
| `--workers` | 4 | Number of parallel workers |
| `--show-params` | False | Show current parameters and exit |

### Tile Task Parameters

| CLI Argument | Default | Description |
|--------------|---------|-------------|
| `--tile-length` | 100 | Tile size in meters |
| `--tile-buffer` | 5 | Buffer overlap in meters |
| `--threads` | 5 | Threads per COPC writer |
| `--resolution-1` | 0.02 | First subsampling resolution in meters (2cm) |
| `--resolution-2` | 0.1 | Second subsampling resolution in meters (10cm) |
| `--grid-offset` | 1.0 | Grid offset from min coordinates in meters |
| `--skip-dimension-reduction` | False | Keep all point dimensions (default: XYZ-only) |

### Merge Task Parameters

| CLI Argument | Default | Description |
|--------------|---------|-------------|
| `--subsampled-10cm-folder` | None | Path to segmented 10cm tiles |
| `--original-input-dir` | None | Path to original input LAZ files for final remap (Stage 7) |
| `--target-resolution` | 2 | Target resolution in cm for remapping |
| `--buffer` | 10.0 | Buffer distance for filtering in meters |
| `--overlap-threshold` | 0.3 | Overlap ratio for instance matching (0.3 = 30%) |
| `--max-centroid-distance` | 3.0 | Max centroid distance to merge instances in meters |
| `--correspondence-tolerance` | 0.05 | Point correspondence tolerance in meters |
| `--max-volume-for-merge` | 4.0 | Max convex hull volume for small instance merge in m³ |
| `--min-cluster-size` | 300 | Minimum cluster size in points |
| `--disable-matching` | False | Disable cross-tile instance matching |
| `--disable-volume-merge` | False | Disable small volume instance merging |
| `--verbose` | False | Print detailed merge decisions |

## Usage Examples

### Example 1: Large Tiles with More Workers

```bash
python src/run.py --task tile \
  --input-dir /data/input \
  --output-dir /data/output \
  --tile-length 200 \
  --tile-buffer 10 \
  --workers 16
```

### Example 2: Custom Resolutions

```bash
python src/run.py --task tile \
  --input-dir /data/input \
  --output-dir /data/output \
  --resolution-1 0.01 \
  --resolution-2 0.05
```

### Example 3: Preserve All Point Dimensions

```bash
python src/run.py --task tile \
  --input-dir /data/input \
  --output-dir /data/output \
  --skip-dimension-reduction
```

Note: By default, the pipeline reduces point clouds to XYZ-only for ~37% size reduction. Use `--skip-dimension-reduction` to keep all attributes (intensity, classification, RGB, etc.).

### Example 4: Using Environment Variables

```bash
export TILE_LENGTH=150
export TILE_BUFFER=8
export WORKERS=12

python src/run.py --task tile \
  --input-dir /data/input \
  --output-dir /data/output
```

### Example 5: Merge with Custom Parameters

```bash
python src/run.py --task merge \
  --subsampled-10cm-folder /data/10cm \
  --buffer 15.0 \
  --overlap-threshold 0.4 \
  --workers 16
```

### Example 6: Full Merge Pipeline with Original File Remap

```bash
python src/run.py --task merge \
  --subsampled-10cm-folder /data/output/tiles_100m/subsampled_10cm \
  --original-input-dir /data/input \
  --buffer 10.0 \
  --workers 8
```

This runs all merge stages including Stage 7 (remapping predictions back to original input files).

## Testing Parameters

Test your parameter configuration without running the pipeline:

```bash
# View defaults
python src/run.py --show-params

# View with overrides
python src/run.py --show-params --tile-length 150 --workers 8
```

## Parameter Format

CLI parameters use dashes (kebab-case) and can also use underscores:

```bash
# Both formats are equivalent:
--tile-length 150
--tile_length 150

# Multiple parameters
python src/run.py --task tile \
  --input-dir /data/input \
  --output-dir /data/output \
  --tile-length 150 \
  --tile-buffer 10 \
  --workers 8
```

## Environment Variables

Environment variables use UPPER_SNAKE_CASE:

| Environment Variable | CLI Equivalent |
|---------------------|----------------|
| `TASK` | `--task` |
| `INPUT_DIR` | `--input-dir` |
| `OUTPUT_DIR` | `--output-dir` |
| `WORKERS` | `--workers` |
| `TILE_LENGTH` | `--tile-length` |
| `TILE_BUFFER` | `--tile-buffer` |
| `RESOLUTION_1` | `--resolution-1` |
| `RESOLUTION_2` | `--resolution-2` |
| `BUFFER` | `--buffer` |
| `OVERLAP_THRESHOLD` | `--overlap-threshold` |
| `ORIGINAL_INPUT_DIR` | `--original-input-dir` |

## Tips

1. **Start with defaults**: Use `python src/run.py --show-params` to see current values
2. **Use CLI for most cases**: Direct CLI arguments are the simplest approach
3. **Use environment variables for deployment**: Set system-wide defaults via env vars
4. **Check parameter names**: Use dashes (`--tile-length`) or underscores (`--tile_length`)

## Complete Parameter List

**Common:** `--task`, `--input-dir`, `--output-dir`, `--workers`, `--show-params`

**Tile Task:** `--tile-length`, `--tile-buffer`, `--grid-offset`, `--threads`, `--resolution-1`, `--resolution-2`, `--skip-dimension-reduction`, `--num-spatial-chunks`

**Merge Task:** `--subsampled-10cm-folder`, `--original-input-dir`, `--target-resolution`, `--buffer`, `--overlap-threshold`, `--max-centroid-distance`, `--correspondence-tolerance`, `--max-volume-for-merge`, `--min-cluster-size`, `--disable-matching`, `--disable-volume-merge`, `--verbose`
