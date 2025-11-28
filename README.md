# 3DTrees Smart Tiling with Tindex

This directory contains scripts for efficient retiling of COPC files using tindex for spatial indexing.

## Overview

This approach uses:
- **Tindex (shapefile)** for efficient spatial queries to identify intersecting COPC files
- **Direct COPC readers with bounds** for efficient spatial filtering at the octree level
- **Parallel processing** of tiles

## Scripts

### Core Scripts

1. **`build_tindex.sh`** - Builds a tindex shapefile from COPC files
   ```bash
   ./build_tindex.sh <copc_dir> <output_tindex.shp> [srs]
   ```

2. **`get_bounds_from_tindex.py`** - Calculates tile bounds from tindex extent
   ```bash
   python get_bounds_from_tindex.py <tindex.shp> --tile-length=100 --tile-buffer=5
   ```

3. **`prepare_tile_jobs.py`** - Prepares tile job list from tindex
   ```bash
   python prepare_tile_jobs.py <tindex.shp> --tile-length=100 --tile-buffer=5
   ```

4. **`identify_copc_files_from_tindex.py`** - Identifies intersecting COPC files per tile using spatial queries
   ```bash
   python identify_copc_files_from_tindex.py <tindex.shp> <tile_bounds.json> --output=mapping.json
   ```

5. **`retiling.sh`** - Main retiling script (orchestrates the process)
   ```bash
   ./retiling.sh
   # Or with environment variables:
   MAX_TILE_PROCS=10 CONVERT_TO_COPC=true ./retiling.sh
   ```

### Conversion Scripts

6. **`convert_input_laz_to_copc.sh`** - Converts input LAZ files to COPC format
   ```bash
   ./convert_input_laz_to_copc.sh <input_laz_dir> <output_copc_dir> <threads>
   ```

7. **`convert_output_tiles_to_copc.sh`** - Converts output LAZ tiles to COPC format
   ```bash
   ./convert_output_tiles_to_copc.sh <input_laz_dir> <output_copc_dir> [threads] [tile_length]
   ```

### Configuration Files

8. **`laz_to_copc.json`** - PDAL pipeline template for LAZ to COPC conversion

## Workflow

The main `retiling.sh` script follows this workflow:

1. **Convert input to COPC** (if needed) - Converts raw LAZ files to COPC format
2. **Build tindex** - Creates spatial index from COPC files
3. **Calculate tile bounds** - Computes grid of tiles from tindex extent
4. **Identify intersecting files** - Uses spatial queries to find COPC files per tile
5. **Process tiles** - For each tile:
   - Extract from intersecting COPC files using direct reader with bounds
   - Merge parts into final tile
6. **Convert to COPC** (optional) - Convert output tiles to COPC format

## Usage

### Basic Usage

```bash
cd /home/kg281/projects/3dtrees_smart_tile
./retiling.sh
```

### With Custom Parameters

```bash
# Set environment variables before running
export tile_length=100
export tile_buffer=5
export MAX_TILE_PROCS=10
export CONVERT_TO_COPC=true  # Convert output tiles to COPC

./retiling.sh
```

### Step-by-Step Manual Execution

```bash
# 1. Convert input to COPC (if needed)
./convert_input_laz_to_copc.sh \
  /path/to/input_laz \
  /path/to/output_copc \
  5

# 2. Build tindex
./build_tindex.sh \
  /path/to/output_copc \
  /path/to/tindex.shp

# 3. Prepare tile jobs
python prepare_tile_jobs.py \
  /path/to/tindex.shp \
  --tile-length=100 \
  --tile-buffer=5

# 4. Identify intersecting files
python identify_copc_files_from_tindex.py \
  /path/to/tindex.shp \
  /path/to/tile_bounds.json \
  --output=mapping.json

# 5. Run retiling (or do steps 4-5 automatically with retiling.sh)
./retiling.sh

# 6. Convert output tiles to COPC (optional)
./convert_output_tiles_to_copc.sh \
  /path/to/tiles_100m \
  /path/to/tiles_100m_copc \
  5
```

## Advantages of This Approach

1. **Efficient Spatial Filtering** - Direct COPC readers with bounds only load overlapping octree nodes
2. **Fast File Identification** - Tindex spatial queries are O(log n) vs O(n) for VPC
3. **Scalable** - Works efficiently with thousands of COPC files
4. **Memory Efficient** - Only loads necessary spatial chunks

## Dependencies

- PDAL (with tindex support)
- Python packages: `fiona`, `shapely`, `pyproj`
- GDAL (for shapefile support)

Install Python dependencies:
```bash
pip install fiona shapely pyproj
```

## Configuration

Edit `retiling.sh` to change default paths and parameters:
- `tile_length` - Tile size in meters
- `tile_buffer` - Buffer around tiles in meters
- `MAX_TILE_PROCS` - Maximum parallel tile processes
- `threads` - Threads for COPC writing

