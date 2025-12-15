#!/bin/bash
# Main tiling script: retiling with overlap and subsampling to 2cm and 10cm
#
# Usage: main_tile.sh <input_dir> <output_dir> [tile_length] [tile_buffer] [threads] [num_threads]
#   input_dir: Input directory with LAZ/COPC files
#   output_dir: Output directory for all stages
#   tile_length: Tile size in meters (default: 100)
#   tile_buffer: Buffer size in meters (default: 5)
#   threads: Threads per COPC writer (default: 5)
#   num_threads: Number of parallel threads for subsampling (default: 4)
#
# Output structure:
#   {output_dir}/tiles_{length}m/              - Original tiles with overlap
#   {output_dir}/tiles_{length}m/subsampled_2cm/  - 2cm subsampled tiles
#   {output_dir}/tiles_{length}m/subsampled_10cm/ - 10cm subsampled tiles

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [tile_length] [tile_buffer] [threads] [num_threads]"
    echo "  input_dir: Input directory with LAZ/COPC files"
    echo "  output_dir: Output directory for all stages"
    echo "  tile_length: Tile size in meters (default: 100)"
    echo "  tile_buffer: Buffer size in meters (default: 5)"
    echo "  threads: Threads per COPC writer (default: 5)"
    echo "  num_threads: Number of parallel threads for subsampling (default: 4)"
    exit 1
fi

input_dir="$1"
output_dir="$2"
tile_length="${3:-100}"
tile_buffer="${4:-5}"
threads="${5:-5}"
num_threads="${6:-4}"

# Fixed resolutions: 2cm and 10cm
resolution_2cm=0.02
resolution_10cm=0.1

# Stage directories
tiles_dir="${output_dir}/tiles_${tile_length}m"
subsampled_2cm_dir="${tiles_dir}/subsampled_2cm"
subsampled_10cm_dir="${tiles_dir}/subsampled_10cm"

# Output prefix for subsampled files (e.g., "ULS_test_100m")
output_prefix="$(basename "$output_dir")_${tile_length}m"

echo "=========================================="
echo "Main Tiling Pipeline: Retiling + Subsample"
echo "=========================================="
echo "Input directory: $input_dir"
echo "Output directory: $output_dir"
echo "Tile length: ${tile_length}m"
echo "Tile buffer: ${tile_buffer}m"
echo "Threads: $threads (retiling), $num_threads (subsampling)"
echo "Resolutions: 2cm (${resolution_2cm}m) and 10cm (${resolution_10cm}m)"
echo ""

# Step 1: Retiling
echo "=========================================="
echo "Step 1: Retiling"
echo "=========================================="
echo "Running retiling.sh..."
# Use 1m offset: start grid at min_x + 1m and min_y + 1m of input files
grid_offset="1.0"
if ! bash "${SCRIPT_DIR}/retiling.sh" "$input_dir" "$output_dir" "$tile_length" "$tile_buffer" "$threads" "$grid_offset"; then
    echo "ERROR: Retiling failed!"
    exit 1
fi

echo ""
echo "✓ Retiling completed. Tiles saved to: ${tiles_dir}"
echo ""


# Step 2: Subsample to 2cm
echo "=========================================="
echo "Step 2: Subsample to 2cm (${resolution_2cm}m)"
echo "=========================================="
echo "Subsampling retiled COPC tiles to 2cm resolution..."

# Check if tiles directory exists and has files
if [ ! -d "$tiles_dir" ]; then
    echo "ERROR: Tiles directory not found: $tiles_dir"
    exit 1
fi

tile_count=$(find "$tiles_dir" -name "*.copc.laz" -o -name "*.laz" 2>/dev/null | wc -l)
if [ "$tile_count" -eq 0 ]; then
    echo "ERROR: No tile files found in: $tiles_dir"
    exit 1
fi

echo "Found $tile_count tile files to subsample"

# Create output directory for 2cm subsampling
mkdir -p "$subsampled_2cm_dir"

# Run subsampling on each tile file
if ! bash "${SCRIPT_DIR}/subsampling.sh" "$tiles_dir" "$resolution_2cm" "$num_threads" "$output_prefix"; then
    echo "ERROR: 2cm subsampling failed!"
    exit 1
fi

# Move subsampled files to our output directory
# subsampling.sh saves to <input_dir>/subsampled/
if [ -d "${tiles_dir}/subsampled" ]; then
    echo "Moving 2cm subsampled files..."
    mv "${tiles_dir}/subsampled"/* "$subsampled_2cm_dir/" 2>/dev/null || true
    rmdir "${tiles_dir}/subsampled" 2>/dev/null || true
fi

echo ""
echo "✓ 2cm subsampling completed. Files saved to: ${subsampled_2cm_dir}"
echo ""

# Step 3: Subsample to 10cm
echo "=========================================="
echo "Step 3: Subsample to 10cm (${resolution_10cm}m)"
echo "=========================================="
echo "Subsampling 2cm files to 10cm resolution..."

# Check if 2cm subsampled directory has files
subsampled_2cm_count=$(find "$subsampled_2cm_dir" -name "*.laz" 2>/dev/null | wc -l)
if [ "$subsampled_2cm_count" -eq 0 ]; then
    echo "ERROR: No 2cm subsampled files found in: $subsampled_2cm_dir"
    exit 1
fi

echo "Found $subsampled_2cm_count files to subsample to 10cm"

# Create output directory for 10cm subsampling
mkdir -p "$subsampled_10cm_dir"

# Run subsampling on 2cm files
if ! bash "${SCRIPT_DIR}/subsampling.sh" "$subsampled_2cm_dir" "$resolution_10cm" "$num_threads" "$output_prefix"; then
    echo "ERROR: 10cm subsampling failed!"
    exit 1
fi

# Move subsampled files to our output directory
if [ -d "${subsampled_2cm_dir}/subsampled" ]; then
    echo "Moving 10cm subsampled files..."
    mv "${subsampled_2cm_dir}/subsampled"/* "$subsampled_10cm_dir/" 2>/dev/null || true
    rmdir "${subsampled_2cm_dir}/subsampled" 2>/dev/null || true
fi

echo ""
echo "✓ 10cm subsampling completed. Files saved to: ${subsampled_10cm_dir}"
echo ""

# Summary
echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ Step 1: Retiling -> ${tiles_dir}"
echo "✓ Step 2: 2cm subsampling -> ${subsampled_2cm_dir}"
echo "✓ Step 3: 10cm subsampling -> ${subsampled_10cm_dir}"
echo ""
echo "All stages completed successfully!"
echo "=========================================="