#!/bin/bash
# Build tindex from COPC files for efficient spatial queries

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <copc_dir> <output_tindex.shp|output_tindex.gpkg>"
  echo "  copc_dir: Directory containing .copc.laz files"
  echo "  output_tindex: Output tindex path (.shp or .gpkg recommended)"
  echo "Note: SRS is inferred from COPC files automatically"
  echo "      Use .gpkg for better SRS support"
  exit 1
fi

copc_dir="$1"
output_tindex="$2"

# Determine format and set driver
if [[ "${output_tindex}" == *.gpkg ]]; then
  ogr_driver="GPKG"
else
  ogr_driver="ESRI Shapefile"
fi

# Create directory for output
mkdir -p "$(dirname "${output_tindex}")"

# Create list of COPC files
copc_list=$(mktemp)
find "${copc_dir}" -name "*.copc.laz" | sort > "${copc_list}"

file_count=$(wc -l < "${copc_list}")
if [ "$file_count" -eq 0 ]; then
  echo "ERROR: No COPC files found in ${copc_dir}"
  rm "${copc_list}"
  exit 1
fi

echo "Found ${file_count} COPC files"
echo "Building tindex: ${output_tindex}"

# Build tindex using pdal tindex
# Note: SRS is typically inferred from COPC files, or can be set via --t_srs
# Use --stdin to read file list from file
# Use GeoPackage driver to avoid shapefile field length limitations
pdal tindex create \
  --tindex="${output_tindex}" \
  --stdin \
  --tindex_name="Location" \
  --ogrdriver="${ogr_driver}" \
  --fast_boundary \
  --write_absolute_path < "${copc_list}"

rm "${copc_list}"

echo "Tindex created successfully: ${output_tindex}"

