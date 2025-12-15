"""
Centralized parameter configuration for the 3DTrees smart tiling pipeline.

All default parameters are defined here and can be overridden via command-line arguments
or environment variables.
"""

# Tile task parameters
TILE_PARAMS = {
    'tile_length': 100,          # Tile size in meters
    'tile_buffer': 5,           # Buffer size in meters
    'threads': 5,                 # Threads per COPC writer
    'num_threads': 4,            # Number of parallel threads for subsampling
    'resolution_2cm': 0.02,       # First resolution in meters (2cm)
    'resolution_10cm': 0.1,      # Second resolution in meters (10cm)
    'grid_offset': 1.0            # Grid offset in meters
}

# Remap task parameters
REMAP_PARAMS = {
    'target_resolution_cm': 2,   # Target resolution in cm (default: 2cm, configurable)
    'num_threads': 8             # Number of threads for KDTree queries
}

# Merge task parameters
MERGE_PARAMS = {
    'buffer': 10.0,                  # Buffer distance for filtering (meters)
    'overlap_threshold': 0.3,        # Overlap ratio threshold for instance matching (0.3 = 30%)
    'max_centroid_distance': 3.0,    # Max centroid distance to merge instances (meters)
    'correspondence_tolerance': 0.05,# Max distance for point correspondence (meters) - MUST BE SMALL!
    'max_volume_for_merge': 4.0,     # Max convex hull volume for small instance merging (m³)
    'num_threads': 8                 # Number of threads for parallel processing
}

