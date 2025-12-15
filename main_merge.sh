#!/bin/bash
# Tile Merger - Shell Wrapper
#
# Merges segmented LAZ tiles with instance and species ID preservation.
# Pipeline: buffer filtering → deduplication → instance matching → volume-based merging
# Optionally retiles the merged result back to original point cloud files.
#
# Usage: main_merge.sh <segmented_folder> [options]
#   segmented_folder: Path to folder containing *_segmented_remapped.laz files
#   --original-tiles-dir: Directory with original tile files for retiling (optional)
#   --output-merged: Output path for merged LAZ file (optional)
#   --output-tiles-dir: Output directory for retiled files (optional)
#   --buffer: Buffer distance in meters (default: 10.0)
#   --overlap-threshold: Overlap ratio threshold for instance matching (default: 0.3)
#   --max-centroid-distance: Max distance between centroids to merge (default: 3.0m)
#   --max-volume-for-merge: Max convex hull volume for small instance merging (default: 4.0 m³)
#   --disable-matching: Disable cross-tile instance matching
#   --disable-volume-merge: Disable small volume instance merging
#   --verbose: Print detailed merge decisions
#   --num-threads: Number of threads (default: 8)

set -euo pipefail

# Parse arguments
SEGMENTED_FOLDER=""
ORIGINAL_TILES_DIR=""
OUTPUT_MERGED=""
OUTPUT_TILES_DIR=""
BUFFER=10.0
OVERLAP_THRESHOLD=0.3
MAX_CENTROID_DISTANCE=3.0
MAX_VOLUME_FOR_MERGE=4.0
NUM_THREADS=8
DISABLE_MATCHING=false
DISABLE_OVERLAP_CHECK=false
DISABLE_VOLUME_MERGE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --original-tiles-dir)
            ORIGINAL_TILES_DIR="$2"
            shift 2
            ;;
        --output-merged)
            OUTPUT_MERGED="$2"
            shift 2
            ;;
        --output-tiles-dir)
            OUTPUT_TILES_DIR="$2"
            shift 2
            ;;
        --buffer)
            BUFFER="$2"
            shift 2
            ;;
        --overlap-threshold|--ff3d-threshold)
            OVERLAP_THRESHOLD="$2"
            shift 2
            ;;
        --max-centroid-distance)
            MAX_CENTROID_DISTANCE="$2"
            shift 2
            ;;
        --max-volume-for-merge)
            MAX_VOLUME_FOR_MERGE="$2"
            shift 2
            ;;
        --num-threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --disable-matching|--disable-ff3d)
            DISABLE_MATCHING=true
            shift
            ;;
        --disable-overlap-check)
            DISABLE_OVERLAP_CHECK=true
            shift
            ;;
        --disable-volume-merge)
            DISABLE_VOLUME_MERGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            if [ -z "$SEGMENTED_FOLDER" ]; then
                SEGMENTED_FOLDER="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input folder parameter is provided
if [ -z "$SEGMENTED_FOLDER" ]; then
    echo "Usage: $0 <segmented_folder> [options]"
    echo ""
    echo "Required:"
    echo "  segmented_folder: Path to folder containing segmented LAZ tiles"
    echo ""
    echo "Optional:"
    echo "  --original-tiles-dir <path>       Directory with original tile files for retiling"
    echo "  --output-merged <path>            Output path for merged LAZ file"
    echo "  --output-tiles-dir <path>         Output directory for retiled files"
    echo "  --buffer <meters>                 Buffer zone distance (default: 10.0)"
    echo "  --overlap-threshold <ratio>       Overlap ratio threshold for matching (default: 0.3)"
    echo "  --max-centroid-distance <meters>  Max centroid distance to merge (default: 3.0)"
    echo "  --max-volume-for-merge <m³>       Max convex hull volume for merging (default: 4.0)"
    echo "  --disable-matching                Disable cross-tile instance matching"
    echo "  --disable-overlap-check           Disable overlap ratio check (centroid distance only)"
    echo "  --disable-volume-merge            Disable small volume instance merging"
    echo "  --verbose, -v                     Print detailed merge decisions"
    echo "  --num-threads <num>               Number of threads (default: 8)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/segmented_remapped \\"
    echo "      --original-tiles-dir /path/to/original_tiles \\"
    echo "      --output-merged /path/to/merged.laz \\"
    echo "      --output-tiles-dir /path/to/output_tiles"
    exit 1
fi

# Validate input folder exists
if [ ! -d "$SEGMENTED_FOLDER" ]; then
    echo "Error: Input folder does not exist: $SEGMENTED_FOLDER"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set default output paths if not provided
if [ -z "$OUTPUT_MERGED" ]; then
    BASE_DIR=$(dirname "$SEGMENTED_FOLDER")
    OUTPUT_MERGED="${BASE_DIR}/merged.laz"
fi

if [ -z "$OUTPUT_TILES_DIR" ] && [ -n "$ORIGINAL_TILES_DIR" ]; then
    BASE_DIR=$(dirname "$SEGMENTED_FOLDER")
    OUTPUT_TILES_DIR="${BASE_DIR}/retiled"
fi

echo "=========================================="
echo "Tile Merger"
echo "=========================================="
echo "Input folder: $SEGMENTED_FOLDER"
echo "Original tiles: ${ORIGINAL_TILES_DIR:-'(not specified)'}"
echo "Output merged: $OUTPUT_MERGED"
echo "Output tiles: ${OUTPUT_TILES_DIR:-'(not specified)'}"
echo "Buffer: ${BUFFER}m"
if [ "$DISABLE_MATCHING" = true ]; then
    echo "Instance matching: DISABLED"
else
    if [ "$DISABLE_OVERLAP_CHECK" = true ]; then
        echo "Overlap check: DISABLED (centroid distance only)"
    else
        echo "Overlap threshold: ${OVERLAP_THRESHOLD}"
    fi
    echo "Max centroid distance: ${MAX_CENTROID_DISTANCE}m"
fi
if [ "$DISABLE_VOLUME_MERGE" = true ]; then
    echo "Volume merge: DISABLED"
else
    echo "Max volume for merge: ${MAX_VOLUME_FOR_MERGE} m³"
fi
echo "Verbose: $VERBOSE"
echo "Threads: $NUM_THREADS"
echo ""

# Build command
CMD="python \"$SCRIPT_DIR/merge_tiles.py\" \
    --input-dir \"$SEGMENTED_FOLDER\" \
    --output-merged \"$OUTPUT_MERGED\" \
    --buffer $BUFFER \
    --overlap-threshold $OVERLAP_THRESHOLD \
    --max-centroid-distance $MAX_CENTROID_DISTANCE \
    --max-volume-for-merge $MAX_VOLUME_FOR_MERGE \
    --num-threads $NUM_THREADS"

if [ -n "$ORIGINAL_TILES_DIR" ]; then
    CMD="$CMD --original-tiles-dir \"$ORIGINAL_TILES_DIR\""
fi

if [ -n "$OUTPUT_TILES_DIR" ]; then
    CMD="$CMD --output-tiles-dir \"$OUTPUT_TILES_DIR\""
fi

if [ "$DISABLE_MATCHING" = true ]; then
    CMD="$CMD --disable-matching"
fi

if [ "$DISABLE_OVERLAP_CHECK" = true ]; then
    CMD="$CMD --disable-overlap-check"
fi

if [ "$DISABLE_VOLUME_MERGE" = true ]; then
    CMD="$CMD --disable-volume-merge"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# Run merge script
eval $CMD

echo ""
echo "=========================================="
echo "Merge completed successfully!"
echo "=========================================="
