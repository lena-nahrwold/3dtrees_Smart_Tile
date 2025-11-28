#!/bin/bash
# Check how many nodes are used from each COPC file during extraction

set -euo pipefail

mapping_json="${1:-/home/kg281/data/output/pdal_experiments/tile_copc_mapping_100m.json}"
tile_bounds_json="${2:-/home/kg281/data/output/pdal_experiments/tile_bounds_tindex.json}"
log_dir="${3:-/home/kg281/data/output/pdal_experiments/logs}"

if [ ! -f "$mapping_json" ]; then
  echo "ERROR: Mapping JSON not found: $mapping_json"
  exit 1
fi

echo "=== Checking COPC Node Usage ==="
echo "Mapping: $mapping_json"
echo ""

# Read mapping and process each tile
python3 <<PYTHON
import json
import subprocess
import re
from pathlib import Path

mapping_json = Path("${mapping_json}")
tile_bounds_json = Path("${tile_bounds_json}")
log_dir = Path("${log_dir}")

# Load data
with mapping_json.open() as f:
    mapping = json.load(f)

with tile_bounds_json.open() as f:
    tiles_data = json.load(f)

results = {}

for tile_label, tile_info in mapping.items():
    bounds = tile_info["bounds"]
    copc_files = tile_info["copc_files"]
    proj_bounds = f"([{bounds[0][0]},{bounds[0][1]}],[{bounds[1][0]},{bounds[1][1]}])"
    
    print(f"\n=== {tile_label} ===")
    print(f"Bounds: {proj_bounds}")
    print(f"COPC files: {len(copc_files)}")
    
    file_results = []
    
    for copc_file in copc_files:
        copc_path = Path(copc_file)
        if not copc_path.exists():
            print(f"  ⚠ {copc_path.name}: File not found")
            continue
        
        print(f"  Checking {copc_path.name}...")
        
        # Run extraction with verbose to get node count
        pipeline_file = log_dir / f"{tile_label}_{copc_path.stem}_check.json"
        log_file = log_dir / f"{tile_label}_{copc_path.stem}_nodes.log"
        
        # Create pipeline
        pipeline_content = f'''[
    {{
        "type": "readers.copc",
        "filename": "{copc_file}",
        "bounds": "{proj_bounds}"
    }},
    {{
        "type": "writers.null"
    }}
]'''
        
        pipeline_file.write_text(pipeline_content)
        
        # Run with verbose debug
        try:
            result = subprocess.run(
                ['pdal', 'pipeline', str(pipeline_file), '--verbose=8'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            log_file.write_text(result.stdout + result.stderr)
            
            # Extract node count from output
            output = result.stdout + result.stderr
            match = re.search(r'(\d+)\s+overlapping nodes', output)
            
            if match:
                used_nodes = int(match.group(1))
                
                # Try to get total nodes by loading full file
                full_result = subprocess.run(
                    ['pdal', 'pipeline', str(pipeline_file.parent / f'{copc_path.stem}_full.json'), '--verbose=8'],
                    capture_output=True,
                    text=True,
                    timeout=60
                ) if False else (None, None, "")
                
                # Create full load pipeline (no bounds)
                full_pipeline = f'''[
    {{
        "type": "readers.copc",
        "filename": "{copc_file}"
    }},
    {{
        "type": "writers.null"
    }}
]'''
                full_pipeline_file = log_dir / f"{copc_path.stem}_full_check.json"
                full_pipeline_file.write_text(full_pipeline)
                
                full_result = subprocess.run(
                    ['pdal', 'pipeline', str(full_pipeline_file), '--verbose=8'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                full_match = re.search(r'(\d+)\s+overlapping nodes', full_result.stdout + full_result.stderr)
                total_nodes = int(full_match.group(1)) if full_match else -1
                
                if total_nodes > 0:
                    percentage = (used_nodes / total_nodes) * 100
                    print(f"    Total nodes: {total_nodes:,}")
                    print(f"    Used nodes:  {used_nodes:,}")
                    print(f"    Efficiency:  {percentage:.1f}%")
                else:
                    print(f"    Used nodes:  {used_nodes:,}")
                    print(f"    Total nodes: unknown")
                
                file_results.append({
                    "file": str(copc_path.name),
                    "total_nodes": total_nodes if total_nodes > 0 else None,
                    "used_nodes": used_nodes,
                    "percentage": percentage if total_nodes > 0 else None
                })
                
                # Cleanup
                full_pipeline_file.unlink(missing_ok=True)
            else:
                print(f"    Could not extract node count from output")
        
        except Exception as e:
            print(f"    ERROR: {e}")
        finally:
            pipeline_file.unlink(missing_ok=True)
    
    results[tile_label] = file_results

# Print summary
print(f"\n=== Summary ===")
all_files = []
for tile_results in results.values():
    all_files.extend(tile_results)

if all_files:
    files_with_total = [f for f in all_files if f.get("total_nodes")]
    if files_with_total:
        total_nodes_all = sum(f["total_nodes"] for f in files_with_total)
        used_nodes_all = sum(f["used_nodes"] for f in files_with_total)
        avg_efficiency = sum(f["percentage"] for f in files_with_total) / len(files_with_total)
        
        print(f"  Files analyzed: {len(files_with_total)}")
        print(f"  Total nodes (all files): {total_nodes_all:,}")
        print(f"  Used nodes (all files): {used_nodes_all:,}")
        print(f"  Average efficiency: {avg_efficiency:.1f}%")
        print(f"  Overall efficiency: {(used_nodes_all/total_nodes_all*100):.1f}%")
PYTHON

