# =============================================================================
# 3DTrees Smart Tile Pipeline - Dockerfile
# =============================================================================
# Single-stage build. Pipeline uses laspy + PDAL only (no untwine).
# =============================================================================

FROM condaforge/miniforge3:latest

# System deps: PDAL (subprocess), GDAL (required by fiona for reading tindex GeoPackage/shapefile)
RUN mamba install -n base -c conda-forge \
    python=3.10 \
    pdal \
    gdal \
    -y && \
    mamba clean --all -y

# Install uv for fast pip installs
RUN pip install --no-cache-dir uv

# Python deps (all used):
# laspy, lazrs – LAZ/LAS read/write (main_tile, main_subsample, merge_tiles, main_remap, filter_buffer_instances)
# numpy, scipy – arrays and cKDTree (merge_tiles, main_remap)
# matplotlib, fiona, pyproj – plot_tiles_and_copc.py, get_bounds_from_tindex.py, prepare_tile_jobs.py
# pydantic, pydantic-settings – parameters.py
RUN uv pip install --system \
    laspy \
    lazrs \
    numpy \
    scipy \
    matplotlib \
    fiona \
    pyproj \
    pydantic \
    pydantic-settings

# Verify PDAL
RUN pdal --version

# ===========================================
# Setup project
# ===========================================
WORKDIR /src

# Copy Python scripts from src/ folder
COPY src/ /src/

# Create a non-root user for running the application
# Create data directories with proper permissions (owned by appuser)
RUN mkdir -p /in /out /src/out
RUN chmod -R 755 /in /out /src/out

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/src
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib"
# Fix PROJ database path (conda location)
ENV PROJ_DATA="/opt/conda/share/proj"
ENV PROJ_LIB="/opt/conda/share/proj"
# Fix matplotlib config directory (writable location)
ENV MPLCONFIGDIR="/tmp/matplotlib"



# Set entrypoint to python run.py
# ===========================================
# Usage Examples:
# ===========================================
# Build:
#   docker build -t 3dtrees-smart-tile .
#
# Tile task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task tile --input_dir /data/input --output_dir /data/output
#
# Merge task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task merge --subsampled_10cm_folder /data/10cm
#
# Show parameters:
#   docker run 3dtrees-smart-tile --show-params
#
# Interactive shell:
#   docker run -it --entrypoint /bin/bash 3dtrees-smart-tile
# ===========================================
