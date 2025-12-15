#!/bin/bash
# Buffer Filter Preprocessing - Shell Wrapper
#
# Removes instances (and all their points) whose centroids are in buffer zones
# facing neighboring tiles. This is a preprocessing step that should run before merging.
#
# Usage: main_filter_buffer.sh <input_dir> [options]
#   input_dir: Path to directory containing input LAZ tiles
#   --output-dir: Output directory for filtered files (required)
#   --buffer: Buffer distance in meters (default: 10.0)
#   --suffix: Suffix for output filenames (default: "_filtered")

set -euo pipefail

# Parse arguments
INPUT_DIR=""
OUTPUT_DIR=""
BUFFER=10.0
SUFFIX="_filtered"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --buffer)
            BUFFER="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if input directory is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Usage: $0 <input_dir> [options]"
    echo ""
    echo "Required:"
    echo "  input_dir: Path to directory containing input LAZ tiles"
    echo ""
    echo "Optional:"
    echo "  --output-dir, -o <path>    Output directory for filtered files (required)"
    echo "  --buffer <meters>          Buffer zone distance (default: 10.0)"
    echo "  --suffix <string>          Suffix for output filenames (default: '_filtered')"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/segmented_remapped \\"
    echo "      --output-dir /path/to/filtered \\"
    echo "      --buffer 10.0"
    exit 1
fi

# Check if output directory is provided
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
    exit 1
fi

# Validate input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Buffer Filter Preprocessing"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Buffer: ${BUFFER}m"
echo "Suffix: ${SUFFIX}"
echo ""

# Build command
CMD="python \"$SCRIPT_DIR/filter_buffer_instances.py\" \
    --input-dir \"$INPUT_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --buffer $BUFFER \
    --suffix \"$SUFFIX\""

# Run filter script
eval $CMD

echo ""
echo "=========================================="
echo "Filtering completed successfully!"
echo "=========================================="



