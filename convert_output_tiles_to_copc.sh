#!/bin/bash
# Convert output LAZ tiles to COPC format
# This script converts the final tiled output from LAZ to COPC format

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_laz_dir> <output_copc_dir> [threads] [tile_length]"
  echo "  input_laz_dir: Directory containing .laz tile files"
  echo "  output_copc_dir: Directory to write .copc.laz files"
  echo "  threads: Number of threads for COPC writing (default: 5)"
  echo "  tile_length: Optional tile length filter (e.g., '100m')"
  exit 1
fi

input_dir="$1"
output_dir="$2"
threads="${3:-5}"
tile_length="${4:-}"

if [ ! -d "$input_dir" ]; then
  echo "ERROR: Input directory does not exist: $input_dir"
  exit 1
fi

pipeline="$(dirname "$0")/laz_to_copc.json"

if [ ! -f "$pipeline" ]; then
  echo "ERROR: Pipeline file not found: $pipeline"
  exit 1
fi

mkdir -p "${output_dir}"

echo "Converting LAZ tiles to COPC format..."
echo "  Input: ${input_dir}"
echo "  Output: ${output_dir}"
echo "  Threads: ${threads}"

# Count files
total_files=$(find "${input_dir}" -maxdepth 1 -name "*.laz" | wc -l)
if [ "$total_files" -eq 0 ]; then
  echo "ERROR: No LAZ files found in ${input_dir}"
  exit 1
fi

echo "  Found ${total_files} LAZ files to convert"
echo ""

# Process files
processed=0
failed=0

while IFS= read -r laz; do
  filename=$(basename "$laz")
  
  # Skip if it's already a .copc.laz file
  if [[ "$filename" == *.copc.laz ]]; then
    echo "Skipping (already COPC): ${filename}"
    continue
  fi
  
  base="${filename%.laz}"
  out="${output_dir}/${base}.copc.laz"
  
  processed=$((processed + 1))
  echo "[${processed}/${total_files}] Converting ${filename}..."
  
  if pdal pipeline "${pipeline}" \
    --readers.las.filename="${laz}" \
    --writers.copc.filename="${out}" \
    --writers.copc.threads="${threads}" \
    > /dev/null 2>&1
  then
    if [ -f "$out" ] && [ -s "$out" ]; then
      echo "  ✓ Success: ${out}"
    else
      echo "  ✗ Failed: Output file not created or empty"
      failed=$((failed + 1))
      rm -f "$out" 2>/dev/null
    fi
  else
    echo "  ✗ Failed: Conversion error"
    failed=$((failed + 1))
    rm -f "$out" 2>/dev/null
  fi
done < <(find "${input_dir}" -maxdepth 1 -name "*.laz" | sort)

echo ""
echo "=== Conversion Summary ==="
echo "  Total files: ${total_files}"
echo "  Processed: ${processed}"
echo "  Successful: $((processed - failed))"
echo "  Failed: ${failed}"

if [ $failed -gt 0 ]; then
  echo ""
  echo "WARNING: ${failed} file(s) failed to convert"
  exit 1
fi

echo ""
echo "All files converted successfully!"
echo "Output COPC files in: ${output_dir}"

