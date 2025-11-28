#!/usr/bin/env python3
"""
Display a clean summary of COPC node usage statistics.
"""

import json
import sys
from pathlib import Path

def main():
    mapping_json = Path("/home/kg281/data/output/pdal_experiments/tile_copc_mapping_100m.json")
    
    if not mapping_json.exists():
        print(f"ERROR: Mapping file not found: {mapping_json}")
        sys.exit(1)
    
    with mapping_json.open() as f:
        mapping = json.load(f)
    
    print("=" * 80)
    print("COPC Node Usage Summary")
    print("=" * 80)
    print()
    
    all_stats = []
    
    for tile_label, tile_info in sorted(mapping.items()):
        print(f"Tile: {tile_label}")
        print(f"  Bounds: [{tile_info['bounds'][0]}, {tile_info['bounds'][1]}]")
        print(f"  COPC files: {tile_info['file_count']}")
        print()
        
        for copc_file in tile_info['copc_files']:
            file_path = Path(copc_file)
            print(f"    • {file_path.name}")
        
        print()
    
    print("=" * 80)
    print("\nTo check actual node usage, run:")
    print("  bash check_copc_nodes_usage.sh")

if __name__ == "__main__":
    main()

