#!/usr/bin/env python3
"""
Check how many nodes are used from each COPC file during extraction.
This helps verify that COPC spatial filtering is working efficiently.
"""

import json
import subprocess
import sys
from pathlib import Path
import argparse
import re

try:
    import copclib as copc
except ImportError:
    print("ERROR: copc-lib not installed. Install with: pip install copc-lib")
    sys.exit(1)


def get_total_nodes(copc_file: Path) -> int:
    """Get total number of nodes in a COPC file."""
    try:
        reader = copc.FileReader(str(copc_file))
        hierarchy = reader.GetHierarchy()
        return len(hierarchy)
    except Exception as e:
        print(f"  ERROR reading {copc_file.name}: {e}", file=sys.stderr)
        return -1


def extract_with_bounds(copc_file: Path, bounds: str, output_log: Path) -> int:
    """Extract from COPC file with bounds and count overlapping nodes from log."""
    pipeline_json = output_log.with_suffix('.json')
    
    # Create pipeline
    pipeline_content = f'''[
    {{
        "type": "readers.copc",
        "filename": "{copc_file}",
        "bounds": "{bounds}"
    }},
    {{
        "type": "writers.null"
    }}
]'''
    
    pipeline_json.write_text(pipeline_content)
    
    # Run with verbose debug
    try:
        result = subprocess.run(
            ['pdal', 'pipeline', str(pipeline_json), '--verbose=8'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output_log.write_text(result.stdout + result.stderr)
        
        # Extract node count from output
        # Look for pattern like "(pdal pipeline readers.copc Debug) 358 overlapping nodes"
        match = re.search(r'(\d+)\s+overlapping nodes', result.stdout + result.stderr)
        if match:
            return int(match.group(1))
        
        return -1
    except subprocess.TimeoutExpired:
        return -1
    except Exception as e:
        print(f"  ERROR running pipeline: {e}", file=sys.stderr)
        return -1
    finally:
        if pipeline_json.exists():
            pipeline_json.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Check COPC node usage during extraction"
    )
    parser.add_argument(
        "mapping_json",
        type=Path,
        help="Path to tile_copc_mapping JSON file"
    )
    parser.add_argument(
        "tile_bounds_json",
        type=Path,
        help="Path to tile_bounds_tindex JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("copc_nodes_usage.json"),
        help="Output JSON with node usage statistics"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/home/kg281/data/output/pdal_experiments/logs"),
        help="Directory for log files"
    )
    args = parser.parse_args()
    
    # Load mappings
    with args.mapping_json.open() as f:
        mapping = json.load(f)
    
    with args.tile_bounds_json.open() as f:
        tiles_data = json.load(f)
    
    proj_crs = tiles_data.get("proj_srs", "EPSG:32632")
    
    results = {}
    
    # Process each tile
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
            
            # Get total nodes
            total_nodes = get_total_nodes(copc_path)
            if total_nodes < 0:
                print(f"    Failed to read total nodes")
                continue
            
            # Get used nodes
            log_file = args.log_dir / f"{tile_label}_{copc_path.stem}_nodes.log"
            used_nodes = extract_with_bounds(copc_path, proj_bounds, log_file)
            
            if used_nodes < 0:
                print(f"    Failed to extract used nodes")
                # Try reading from log file
                if log_file.exists():
                    content = log_file.read_text()
                    match = re.search(r'(\d+)\s+overlapping nodes', content)
                    if match:
                        used_nodes = int(match.group(1))
            
            if used_nodes >= 0 and total_nodes > 0:
                percentage = (used_nodes / total_nodes) * 100
                print(f"    Total nodes: {total_nodes:,}")
                print(f"    Used nodes:  {used_nodes:,}")
                print(f"    Efficiency:  {percentage:.1f}%")
                
                file_results.append({
                    "file": str(copc_path),
                    "total_nodes": total_nodes,
                    "used_nodes": used_nodes,
                    "percentage": percentage
                })
            else:
                print(f"    Total nodes: {total_nodes:,}")
                print(f"    Used nodes:  unknown")
        
        results[tile_label] = {
            "bounds": bounds,
            "files": file_results
        }
    
    # Save results
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Results saved to: {args.output}")
    
    # Print summary statistics
    all_files = []
    for tile_results in results.values():
        all_files.extend(tile_results["files"])
    
    if all_files:
        total_nodes_all = sum(f["total_nodes"] for f in all_files)
        used_nodes_all = sum(f["used_nodes"] for f in all_files)
        avg_efficiency = sum(f["percentage"] for f in all_files) / len(all_files)
        
        print(f"\nOverall Statistics:")
        print(f"  Files analyzed: {len(all_files)}")
        print(f"  Total nodes (all files): {total_nodes_all:,}")
        print(f"  Used nodes (all files): {used_nodes_all:,}")
        print(f"  Average efficiency: {avg_efficiency:.1f}%")
        print(f"  Overall efficiency: {(used_nodes_all/total_nodes_all*100):.1f}%")


if __name__ == "__main__":
    main()

