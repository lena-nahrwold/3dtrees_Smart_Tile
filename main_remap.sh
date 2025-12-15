#!/bin/bash
# Remap predictions from 10cm subsampled files to target resolution (default 2cm)
#
# Usage: main_remap.sh <subsampled_10cm_folder> [--target_resolution <cm>] [--subsampled_target_folder <path>] [--output_folder <path>] [--num_threads <N>]
#   subsampled_10cm_folder: Path to folder containing *_results directories
#   --target_resolution: Target resolution in cm (default: 2)
#   --subsampled_target_folder: Path to target resolution subsampled folder (default: auto-derived)
#   --output_folder: Output folder for remapped files (default: auto-derived from input folder)
#   --num_threads: Number of CPU threads for Python processing (default: 8, used for KDTree queries)

set -euo pipefail

# Parse arguments
SUBSAMPLED_10CM_FOLDER=""
TARGET_RESOLUTION_CM=2
SUBSAMPLED_TARGET_FOLDER=""
OUTPUT_FOLDER=""
NUM_THREADS=8

while [[ $# -gt 0 ]]; do
    case $1 in
        --target_resolution)
            TARGET_RESOLUTION_CM="$2"
            shift 2
            ;;
        --subsampled_target_folder)
            SUBSAMPLED_TARGET_FOLDER="$2"
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --num_threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        *)
            if [ -z "$SUBSAMPLED_10CM_FOLDER" ]; then
                SUBSAMPLED_10CM_FOLDER="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input folder parameter is provided
if [ -z "$SUBSAMPLED_10CM_FOLDER" ]; then
    echo "Usage: $0 <subsampled_10cm_folder> [--target_resolution <cm>] [--subsampled_target_folder <path>] [--output_folder <path>] [--num_threads <N>]"
    echo "Example: $0 /path/to/subsampled_10cm --target_resolution 2 --subsampled_target_folder /path/to/subsampled_2cm --output_folder /path/to/output --num_threads 8"
    echo "  subsampled_10cm_folder: Path to folder containing *_results directories"
    echo "  --target_resolution: Target resolution in cm (default: 2)"
    echo "  --subsampled_target_folder: Path to target resolution subsampled folder (default: auto-derived)"
    echo "  --output_folder: Output folder for remapped files (default: auto-derived)"
    echo "  --num_threads: Number of CPU threads for Python processing (default: 8)"
    exit 1
fi

# Validate num_threads is a positive integer
if ! [[ "$NUM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num_threads must be a positive integer, got: $NUM_THREADS"
    exit 1
fi

# Validate input folder exists
if [ ! -d "$SUBSAMPLED_10CM_FOLDER" ]; then
    echo "Error: Input folder does not exist: $SUBSAMPLED_10CM_FOLDER"
    exit 1
fi

# Use provided target folder or auto-derive from input folder
if [ -z "$SUBSAMPLED_TARGET_FOLDER" ]; then
    SUBSAMPLED_TARGET_FOLDER="${SUBSAMPLED_10CM_FOLDER/subsampled_10cm/subsampled_${TARGET_RESOLUTION_CM}cm}"
fi

# Use provided output folder or auto-derive from input folder
if [ -z "$OUTPUT_FOLDER" ]; then
    OUTPUT_FOLDER="${SUBSAMPLED_10CM_FOLDER/subsampled_10cm/segmented_remapped}"
fi

echo "=========================================="
echo "Remap Predictions to Target Resolution"
echo "=========================================="
echo "Input folder (10cm): $SUBSAMPLED_10CM_FOLDER"
echo "Target resolution: ${TARGET_RESOLUTION_CM}cm"
echo "Target folder: $SUBSAMPLED_TARGET_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "CPU threads for processing: $NUM_THREADS"
echo ""

# Create output folder if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Get script directory to find remapping_original_res.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to process a single tile
process_tile() {
    local RESULTS_DIR="$1"
    local TARGET_RESOLUTION_CM="$2"
    local TARGET_RESOLUTION_M="$3"
    local SUBSAMPLED_TARGET_FOLDER="$4"
    local OUTPUT_FOLDER="$5"
    local SCRIPT_DIR="$6"
    local NUM_THREADS="$7"
    
    # Get the results folder name (e.g., c00_r00.copc_subsampled0.02m_subsampled0.1m_results)
    local RESULTS_DIRNAME=$(basename "$RESULTS_DIR")
    
    # Extract tile ID pattern c##_r## from anywhere in the filename
    # Pattern: c followed by 2+ digits, underscore, r, followed by 2+ digits
    local TILE_ID=$(echo "$RESULTS_DIRNAME" | grep -oE 'c[0-9]+_r[0-9]+' | head -1)
    
    # Validate tile ID was found and matches pattern c##_r## (where ## are digits)
    if [ -z "$TILE_ID" ] || ! [[ "$TILE_ID" =~ ^c[0-9]+_r[0-9]+$ ]]; then
        echo "Warning: Skipping directory - tile ID pattern c##_r## not found: $RESULTS_DIRNAME" >&2
        echo "  Expected pattern: c##_r## somewhere in filename (e.g., c00_r00, c01_r02)" >&2
        return 1
    fi
    
    # Path to segmented_pc.laz in results folder
    local SEGMENTED_FILE="$RESULTS_DIR/segmented_pc.laz"
    
    # Try multiple possible file patterns for the target resolution file
    # Pattern 1: c00_r00.copc_subsampled0.02m.laz
    local ORIGINAL_FILE="$SUBSAMPLED_TARGET_FOLDER/${TILE_ID}.copc_subsampled${TARGET_RESOLUTION_M}m.laz"
    
    # Pattern 2: c00_r00.copc_subsampled0.02m.laz (alternative naming)
    if [ ! -f "$ORIGINAL_FILE" ]; then
        ORIGINAL_FILE="$SUBSAMPLED_TARGET_FOLDER/${TILE_ID}.copc_subsampled${TARGET_RESOLUTION_CM}cm.laz"
    fi
    
    # Pattern 3: Try finding any file with tile ID at the start
    if [ ! -f "$ORIGINAL_FILE" ]; then
        ORIGINAL_FILE=$(find "$SUBSAMPLED_TARGET_FOLDER" -maxdepth 1 -name "${TILE_ID}*.laz" -type f | head -1)
    fi
    
    # Pattern 4: Try finding any file containing the tile ID pattern anywhere in filename
    if [ ! -f "$ORIGINAL_FILE" ]; then
        ORIGINAL_FILE=$(find "$SUBSAMPLED_TARGET_FOLDER" -maxdepth 1 -type f -name "*.laz" | grep -E ".*${TILE_ID}.*" | head -1)
    fi
    
    # Output file path
    local OUTPUT_FILE="$OUTPUT_FOLDER/${TILE_ID}_segmented_remapped.laz"
    
    # Validate that required files exist
    if [ ! -f "$SEGMENTED_FILE" ]; then
        echo "Warning: Segmented file not found: $SEGMENTED_FILE" >&2
        return 1
    fi
    
    if [ -z "$ORIGINAL_FILE" ] || [ ! -f "$ORIGINAL_FILE" ]; then
        echo "Warning: Target resolution (${TARGET_RESOLUTION_CM}cm) file not found for tile $TILE_ID" >&2
        echo "  Looked in: $SUBSAMPLED_TARGET_FOLDER" >&2
        echo "  Expected pattern: ${TILE_ID}.copc_subsampled${TARGET_RESOLUTION_M}m.laz" >&2
        # Show available files for debugging
        if [ -d "$SUBSAMPLED_TARGET_FOLDER" ]; then
            local available_files=$(find "$SUBSAMPLED_TARGET_FOLDER" -maxdepth 1 -type f -name "*.laz" | head -5)
            if [ -n "$available_files" ]; then
                echo "  Available files in target folder (showing first 5):" >&2
                echo "$available_files" | sed 's/^/    /' >&2
            else
                echo "  No .laz files found in target folder" >&2
            fi
        fi
        return 1
    fi
    
    echo "Processing tile: $TILE_ID"
    echo "  Segmented file (10cm): $SEGMENTED_FILE"
    echo "  Target file (${TARGET_RESOLUTION_CM}cm): $ORIGINAL_FILE"
    echo "  Output file: $OUTPUT_FILE"
    
    # Remap the file
    if ! python "$SCRIPT_DIR/remapping_original_res.py" \
        --subsampled_file "$SEGMENTED_FILE" \
        --original_file "$ORIGINAL_FILE" \
        --output_file "$OUTPUT_FILE" \
        --num_threads "$NUM_THREADS"; then
        echo "Error: Failed to process tile $TILE_ID" >&2
        return 1
    fi
    
    echo "Completed tile: $TILE_ID"
    return 0
}

# Format target resolution as meters (e.g., 2cm -> 0.02m)
TARGET_RESOLUTION_M=$(awk "BEGIN {printf \"%.2f\", $TARGET_RESOLUTION_CM / 100}")

# Find all results directories
RESULTS_DIRS=$(find "$SUBSAMPLED_10CM_FOLDER" -maxdepth 1 -type d -name "*_results" | sort)

# Count total tiles
TOTAL_TILES=$(echo "$RESULTS_DIRS" | grep -c . || echo 0)
if [ "$TOTAL_TILES" -eq 0 ]; then
    echo "No *_results directories found in $SUBSAMPLED_10CM_FOLDER"
    exit 1
fi

echo "Found $TOTAL_TILES tiles to process"
echo ""

# Process tiles sequentially (one at a time, but Python uses multiple CPUs)
CURRENT=0
FAILED=0
for RESULTS_DIR in $RESULTS_DIRS; do
    CURRENT=$((CURRENT + 1))
    echo "=========================================="
    echo "Processing tile $CURRENT of $TOTAL_TILES"
    echo "=========================================="
    
    if ! process_tile "$RESULTS_DIR" "$TARGET_RESOLUTION_CM" "$TARGET_RESOLUTION_M" "$SUBSAMPLED_TARGET_FOLDER" "$OUTPUT_FOLDER" "$SCRIPT_DIR" "$NUM_THREADS"; then
        FAILED=$((FAILED + 1))
        echo "Warning: Tile failed, continuing with next tile..."
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "Processing complete"
echo "=========================================="
echo "Total tiles: $TOTAL_TILES"
echo "Successful: $((TOTAL_TILES - FAILED))"
echo "Failed: $FAILED"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "Warning: $FAILED tile(s) failed to process. Check the output above for details."
    exit 1
fi

echo ""
echo "All tiles processed successfully."
