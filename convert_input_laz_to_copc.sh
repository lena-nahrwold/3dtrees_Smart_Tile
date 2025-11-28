#!/bin/bash
# Convert input LAZ files to COPC format
# This is a standalone script for converting raw LAZ input files to COPC

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <input_laz_dir> <output_copc_dir> <threads>"
  echo "  input_laz_dir: Directory containing .laz files"
  echo "  output_copc_dir: Directory to write .copc.laz files"
  echo "  threads: Number of threads for COPC writing"
  exit 1
fi

input_dir="$1"
output_dir="$2"
threads="$3"
pipeline="$(dirname "$0")/laz_to_copc.json"

if [ ! -d "$input_dir" ]; then
  echo "ERROR: Input directory does not exist: $input_dir"
  exit 1
fi

if [ ! -f "$pipeline" ]; then
  echo "ERROR: Pipeline file not found: $pipeline"
  exit 1
fi

mkdir -p "${output_dir}"

echo "Converting LAZ files to COPC format..."
echo "  Input: ${input_dir}"
echo "  Output: ${output_dir}"
echo "  Threads: ${threads}"
echo ""

# Count files
total_files=$(find "${input_dir}" -name '*.laz' | wc -l)
if [ "$total_files" -eq 0 ]; then
  echo "ERROR: No LAZ files found in ${input_dir}"
  exit 1
fi

echo "Found ${total_files} LAZ files to convert"
echo ""

# Process files
processed=0
failed=0

while IFS= read -r laz; do
  rel="${laz#"${input_dir}/"}"
  base="${rel%.laz}"
  out="${output_dir}/${base}.copc.laz"
  mkdir -p "$(dirname "${out}")"
  
  processed=$((processed + 1))
  echo "[${processed}/${total_files}] Converting ${rel}..."
  
  if pdal pipeline "${pipeline}" \
    --readers.las.filename="${laz}" \
    --writers.copc.filename="${out}" \
    --writers.copc.threads="${threads}" \
    > /dev/null 2>&1
  then
    if [ -f "$out" ] && [ -s "$out" ]; then
      echo "  ✓ Success"
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
done < <(find "${input_dir}" -name '*.laz' | sort)

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

