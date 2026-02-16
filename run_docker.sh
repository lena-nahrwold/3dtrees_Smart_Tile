#!/bin/bash
set -euo pipefail

INPUT_FOLDER=/home/kg281/data/gfz/segmented_gfz/
OUTPUT_FOLDER=/home/kg281/data/gfz/segmented_gfz/tiledsmall

mkdir -p "$OUTPUT_FOLDER"

docker run --rm -it \
  --cpuset-cpus="0-49" \
  --memory=200g \
  --runtime=nvidia \
  --gpus device=1 \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)":/src \
  -v "$INPUT_FOLDER":/in:ro \
  -v "$OUTPUT_FOLDER":/out \
  3dtrees_smart_tile \
  bash -lc 'python src/run.py --task tile --input_dir /in --output_dir /out --tile_length 10 --tile_buffer 5 --threads 10 --workers 2 --skip_dimension_reduction'
