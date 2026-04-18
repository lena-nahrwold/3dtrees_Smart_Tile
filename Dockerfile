# =============================================================================
# 3DTrees Smart Tile Pipeline - Dockerfile
# =============================================================================
# Single-stage build. Pipeline uses laspy + PDAL + untwine.
# Untwine is the default COPC conversion strategy (fast).
# =============================================================================

FROM condaforge/miniforge3:latest

# System deps: PDAL (subprocess), GDAL (required by fiona for reading tindex GeoPackage/shapefile)
RUN mamba install -n base -c conda-forge \
    python=3.10 \
    pdal \
    gdal \
    untwine \
    -y && \
    mamba clean --all -y

# Install uv for fast pip installs
RUN pip install --no-cache-dir uv

# Python deps (all used):
# laspy, lazrs – LAZ/LAS read/write (filter/remap pipeline, COPC helpers)
# numpy, scipy – arrays and cKDTree (merge_tiles-based remap/filter logic)
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

# Verify PDAL and untwine
RUN pdal --version
RUN untwine --help > /dev/null 2>&1 && echo "untwine OK" || echo "WARNING: untwine not available"

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
# Filter task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task filter --segmented-folders /data/segmented --tile-bounds-json /data/tile_bounds_tindex.json --output-dir /data/output
#
# Remap task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task remap --segmented-folders /data/segmented --original-input-dir /data/original --output-dir /data/output
#
# Show parameters:
#   docker run 3dtrees-smart-tile --show-params
#
# Interactive shell:
#   docker run -it --entrypoint /bin/bash 3dtrees-smart-tile
# ===========================================
