#!/bin/bash
# Retiling script using tindex and direct COPC readers with bounds filtering
# This approach ensures efficient spatial filtering at the COPC level
#
# Usage: retiling.sh [input_dir] [output_dir] [tile_length] [tile_buffer] [threads] [parallel_conversion]
#   input_dir: Input directory with LAZ files (default: /home/kg281/data/output/pdal_experiments/uls_copc_input)
#   output_dir: Output directory (default: /home/kg281/data/output/pdal_experiments)
#   tile_length: Tile size in meters (default: 100)
#   tile_buffer: Buffer size in meters (default: 5)
#   threads: Threads per COPC writer (default: 5)
#   parallel_conversion: Parallel processes for COPC conversion if >10 files (default: 4)

set -euo pipefail

tile_length=${3:-"100"}
tile_buffer=${4:-"5"}
# MAX_TILE_PROCS: How many tiles to process in parallel (process-level parallelism)
MAX_TILE_PROCS=${MAX_TILE_PROCS:-3}
# threads: Number of threads for PDAL COPC writing/compression (internal thread-level parallelism)
threads=${5:-5}
# parallel_conversion: Number of parallel processes for COPC conversion (if >10 files)
parallel_conversion=${6:-4}


input_dir=${1:-"/home/kg281/data/output/pdal_experiments/uls_copc_input"}
output_dir=${2:-"/home/kg281/data/output/pdal_experiments"}
tiles_dir="${output_dir}/tiles_${tile_length}m"
copc_dir="${output_dir}/uls_copc"
log_folder="${output_dir}/logs"
tindex_file="${output_dir}/ULS_tiles_${tile_length}m_tindex.gpkg"

convert_script="/home/kg281/projects/3dtrees_smart_tile/convert_input_laz_to_copc.sh"
build_tindex_script="$(dirname "$0")/build_tindex.sh"

mkdir -p "${output_dir}"
mkdir -p "${tiles_dir}"
mkdir -p "${log_folder}"
mkdir -p "${copc_dir}"

# Step 1: Convert to COPC if needed
copc_list_file="${output_dir}/copc_files.txt"

if [ ! -s "${copc_list_file}" ]; then
  if [ ! -d "${input_dir}" ]; then
    echo "WARNING: Input directory does not exist: ${input_dir}"
    echo "Skipping COPC conversion. Assuming COPC files already exist in ${copc_dir}"
    else
      echo "=== Step 1: Preparing COPC files ==="
      
      # Count input files
      input_file_count=$(find "${input_dir}" -name "*.laz" 2>/dev/null | wc -l)
      
      if [ "$input_file_count" -gt 10 ]; then
        echo "  Found ${input_file_count} files. Using parallel conversion with ${parallel_conversion} processes"
        echo "  Each process uses ${threads} threads"
        
        # Create file list
        file_list=$(mktemp)
        find "${input_dir}" -name "*.laz" 2>/dev/null | sort > "${file_list}"
        total_files=$(wc -l < "${file_list}")
        files_per_process=$(( (total_files + parallel_conversion - 1) / parallel_conversion ))
        
        # Split files and run parallel conversions
        conv_pids=()
        
        for i in $(seq 0 $((parallel_conversion - 1))); do
          process_file_list="${file_list}_${i}"
          
          # Split files using split command or sed
          start_line=$((i * files_per_process + 1))
          end_line=$((start_line + files_per_process - 1))
          
          if [ $i -eq $((parallel_conversion - 1)) ]; then
            # Last process takes remaining files
            tail -n +${start_line} "${file_list}" > "${process_file_list}"
          else
            sed -n "${start_line},${end_line}p" "${file_list}" > "${process_file_list}"
          fi
          
          # Count files for this process
          process_file_count=$(wc -l < "${process_file_list}")
          if [ "$process_file_count" -eq 0 ]; then
            rm -f "${process_file_list}"
            continue
          fi
          
          # Run conversion for this subset in background
          (
            file_num=0
            echo "    [Process $((i+1))/${parallel_conversion}] Converting ${process_file_count} files..."
            while IFS= read -r laz_file; do
              file_num=$((file_num + 1))
              rel="${laz_file#"${input_dir}/"}"
              base="${rel%.laz}"
              out="${copc_dir}/${base}.copc.laz"
              mkdir -p "$(dirname "${out}")"
              
              if pdal pipeline "$(dirname "$0")/laz_to_copc.json" \
                --readers.las.filename="${laz_file}" \
                --writers.copc.filename="${out}" \
                --writers.copc.threads="${threads}" \
                > "${log_folder}/copc_conv_${i}_$(basename "${base}").log" 2>&1
              then
                if [ -f "$out" ] && [ -s "$out" ]; then
                  echo "    [Process $((i+1))/${parallel_conversion}] ✓ Finished ${file_num}/${process_file_count}: $(basename "${rel}")"
                else
                  echo "    [Process $((i+1))/${parallel_conversion}] ✗ Failed ${file_num}/${process_file_count}: $(basename "${rel}") - Output empty"
                  rm -f "$out" 2>/dev/null
                fi
              else
                echo "    [Process $((i+1))/${parallel_conversion}] ✗ Failed ${file_num}/${process_file_count}: $(basename "${rel}") - Conversion error"
                rm -f "$out" 2>/dev/null
              fi
            done < "${process_file_list}"
            echo "    [Process $((i+1))/${parallel_conversion}] Completed all ${process_file_count} files"
          ) &
          conv_pids+=($!)
        done
        
        # Wait for all conversion processes
        echo "  Waiting for ${#conv_pids[@]} parallel conversion processes..."
        for pid in "${conv_pids[@]}"; do
          wait $pid
        done
        
        # Cleanup
        rm -f "${file_list}" "${file_list}"_*
        
        echo "  Parallel conversion completed"
      else
        echo "  Found ${input_file_count} files. Using sequential conversion"
        bash "${convert_script}" "${input_dir}" "${copc_dir}" "${threads}"
      fi
  fi
  # Create list of existing COPC files
  find "${copc_dir}" -name "*.copc.laz" 2>/dev/null | sort > "${copc_list_file}" || touch "${copc_list_file}"
fi

if [ ! -s "${copc_list_file}" ]; then
  echo "ERROR: No COPC files found in ${copc_dir}."
  echo "Please either:"
  echo "  1. Set input_dir to point to your input LAZ files, or"
  echo "  2. Place COPC files directly in ${copc_dir}"
  exit 1
fi

# Step 2: Build tindex if it doesn't exist
if [ ! -f "${tindex_file}" ]; then
  echo "=== Step 2: Building tindex from COPC files ==="
  bash "${build_tindex_script}" "${copc_dir}" "${tindex_file}"
else
  echo "=== Step 2: Using existing tindex: ${tindex_file} ==="
fi

# Step 3: Calculate tile bounds and prepare jobs
echo "=== Step 3: Calculating tile bounds ==="
tile_jobs_file="${output_dir}/tile_jobs_${tile_length}m.txt"
tile_bounds_json="${output_dir}/tile_bounds_tindex.json"

eval "$(
  python "$(dirname "$0")/prepare_tile_jobs.py" "${tindex_file}" \
    --tile-length="${tile_length}" \
    --tile-buffer="${tile_buffer}" \
    --jobs-out="${tile_jobs_file}"
)"

# Step 4: Identify which COPC files intersect each tile
echo "=== Step 4: Identifying intersecting COPC files ==="
tile_copc_mapping="${output_dir}/tile_copc_mapping_${tile_length}m.json"

python "$(dirname "$0")/identify_copc_files_from_tindex.py" \
    "${tindex_file}" \
    "${tile_bounds_json}" \
    --output="${tile_copc_mapping}"

# Step 5: Process each tile
echo "=== Step 5: Processing tiles ==="

process_tile() {
    local label="$1"
    local proj_bounds="$2"
    local geo_bounds="$3"
    
    echo "Processing tile ${label}..." >&2
    
    # Get list of COPC files for this tile from mapping
    local copc_files=$(python -c "
import json
try:
    with open('${tile_copc_mapping}') as f:
        mapping = json.load(f)
    files = mapping['${label}']['copc_files']
    print(' '.join(f\"'{f}'\" for f in files))
except KeyError:
    print('')
" 2>/dev/null || echo "")
    
    if [ -z "$copc_files" ]; then
        echo "  No COPC files for tile ${label}" >&2
        return 0
    fi
    
    # Create temporary tile directory for parts
    local tile_dir="${tiles_dir}/${label}"
    mkdir -p "${tile_dir}"
    
    # Extract from each COPC file with bounds filtering using direct COPC reader
    local part_num=0
    local parts_created=0
    
    for copc_file in $copc_files; do
        # Remove quotes if present
        copc_file=$(echo "$copc_file" | sed "s/'//g")
        
        if [ ! -f "$copc_file" ]; then
            echo "  Warning: COPC file not found: $copc_file" >&2
            continue
        fi
        
        local part_file="${tile_dir}/part_${part_num}.laz"
        local pipeline_tmp="${log_folder}/${label}_part${part_num}_pipeline.json"
        
        # Create temporary pipeline JSON file
        # Note: writers.las doesn't support threads, but we can use it for readers.copc if needed
        # For now, threads parameter is mainly for COPC writer operations
        cat > "${pipeline_tmp}" <<EOF
[
    {
        "type": "readers.copc",
        "filename": "${copc_file}",
        "bounds": "${proj_bounds}"
    },
    {
        "type": "writers.las",
        "filename": "${part_file}",
        "forward": "all",
        "compression": true
    }
]
EOF
        
        # Use direct COPC reader with bounds for efficient spatial filtering
        if pdal pipeline "${pipeline_tmp}" > "${log_folder}/${label}_part${part_num}.log" 2>&1
        then
            rm -f "${pipeline_tmp}"
            if [ -f "$part_file" ] && [ -s "$part_file" ]; then
                parts_created=$((parts_created + 1))
            else
                rm -f "${part_file}" 2>/dev/null
            fi
        fi
        part_num=$((part_num + 1))
    done
    
    if [ $parts_created -eq 0 ]; then
        echo "  No parts created for tile ${label}" >&2
        rmdir "${tile_dir}" 2>/dev/null
        return 0
    fi
    
    # Merge all parts into final tile
    local final_tile="${tiles_dir}/${label}.laz"
    local parts=("${tile_dir}"/part_*.laz)
    
    if [ ${#parts[@]} -eq 0 ]; then
        echo "  No parts to merge for tile ${label}" >&2
        rmdir "${tile_dir}" 2>/dev/null
        return 1
    fi
    
    # Merge using pdal merge
    if pdal merge "${parts[@]}" "${final_tile}" \
        > "${log_folder}/${label}_merge.log" 2>&1
    then
        # Verify output file exists and has content
        if [ -f "${final_tile}" ] && [ -s "${final_tile}" ]; then
            # Clean up parts and temporary directory
            rm -f "${parts[@]}"
            rmdir "${tile_dir}" 2>/dev/null
            echo "  Completed tile ${label} (${parts_created} parts)" >&2
            return 0
        else
            echo "  Error: Merged file is empty for tile ${label}" >&2
            return 1
        fi
    else
        echo "  Error merging tile ${label}" >&2
        cat "${log_folder}/${label}_merge.log" >&2
        return 1
    fi
}

# Process tiles in parallel
max_procs=${MAX_TILE_PROCS}
pids=()
failed_tiles=()

while IFS='|' read -r label proj_bounds geo_bounds; do
    [[ -z "$label" ]] && continue
    
    process_tile "$label" "$proj_bounds" "$geo_bounds" &
    pids+=($!)
    
    if [ ${#pids[@]} -ge ${max_procs} ]; then
        wait ${pids[0]}
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failed_tiles+=("tile_${pids[0]}")
        fi
        pids=("${pids[@]:1}")
    fi
done < "${tile_jobs_file}"

# Wait for remaining jobs
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        failed_tiles+=("tile_${pid}")
    fi
done

if [ ${#failed_tiles[@]} -gt 0 ]; then
    echo "ERROR: Some tiles failed to process:"
    for tile in "${failed_tiles[@]}"; do
        echo "  - ${tile}"
    done
    exit 1
fi

echo "All tile pipelines completed successfully!"

# Step 6: Optionally convert output tiles to COPC format
if [ "${CONVERT_TO_COPC:-false}" = "true" ]; then
  echo ""
  echo "=== Step 6: Converting output tiles to COPC format ==="
  copc_tiles_dir="${output_dir}/tiles_${tile_length}m_copc"
  
  bash "$(dirname "$0")/convert_output_tiles_to_copc.sh" \
    "${tiles_dir}" \
    "${copc_tiles_dir}" \
    "${threads}" \
    "${tile_length}"
  
  echo "COPC tiles saved to: ${copc_tiles_dir}"
fi

echo "Generate the plot"
python "$(dirname "$0")/plot_tiles_and_copc.py" "${tindex_file}" "${tile_bounds_json}" "${output_dir}/tiles_${tile_length}m_plot.png"