"""
Centralized parameter configuration for the 3DTrees smart tiling pipeline.

All default parameters are defined here and can be overridden via:
1. Custom config file: python run.py --config my_config.py
2. CLI arguments: python run.py --param tile_length=150 --param resolution_1=0.03
3. Environment variables: TILE_LENGTH=150 python run.py ...

Parameters can also be loaded programmatically using load_params().
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


# Default tile task parameters
TILE_PARAMS = {
    'tile_length': 100,           # Tile size in meters
    'tile_buffer': 5,             # Buffer size in meters
    'threads': 5,                 # Threads per COPC writer
    'workers': 4,                 # Number of parallel workers for processing
    'resolution_1': 0.02,         # First resolution in meters (2cm)
    'resolution_2': 0.1,          # Second resolution in meters (10cm)
    'grid_offset': 1.0,           # Grid offset in meters
    'skip_dimension_reduction': False,  # Skip XYZ-only reduction, keep all dimensions
    # num_spatial_chunks: defaults to workers (auto-calculated)
    # Legacy names (for backwards compatibility)
    'resolution_2cm': 0.02,
    'resolution_10cm': 0.1,
}

# Default remap task parameters
REMAP_PARAMS = {
    'target_resolution_cm': 2,    # Target resolution in cm (default: 2cm, configurable)
    'workers': 4,                 # Number of parallel workers for KDTree queries
}

# Default merge task parameters
MERGE_PARAMS = {
    'buffer': 10.0,                   # Buffer distance for filtering (meters)
    'overlap_threshold': 0.3,         # Overlap ratio threshold for instance matching (0.3 = 30%)
    'max_centroid_distance': 3.0,     # Max centroid distance to merge instances (meters)
    'correspondence_tolerance': 0.05, # Max distance for point correspondence (meters) - MUST BE SMALL!
    'max_volume_for_merge': 4.0,      # Max convex hull volume for small instance merging (m³)
    'workers': 4,                     # Number of parallel workers for processing
    'verbose': True                 # Print detailed merge decisions
}


def load_params_from_file(config_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load parameters from a custom Python config file.
    
    The config file should define TILE_PARAMS, REMAP_PARAMS, and/or MERGE_PARAMS.
    
    Args:
        config_file: Path to Python config file
    
    Returns:
        Dictionary with 'TILE_PARAMS', 'REMAP_PARAMS', 'MERGE_PARAMS'
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Load the config file as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    params = {}
    if hasattr(config_module, 'TILE_PARAMS'):
        params['TILE_PARAMS'] = config_module.TILE_PARAMS
    if hasattr(config_module, 'REMAP_PARAMS'):
        params['REMAP_PARAMS'] = config_module.REMAP_PARAMS
    if hasattr(config_module, 'MERGE_PARAMS'):
        params['MERGE_PARAMS'] = config_module.MERGE_PARAMS
    
    return params


def load_params_from_env() -> Dict[str, Dict[str, Any]]:
    """
    Load parameter overrides from environment variables.
    
    Environment variables should be prefixed with TILE_, REMAP_, or MERGE_:
    - TILE_LENGTH=150
    - TILE_BUFFER=10
    - REMAP_NUM_THREADS=16
    - MERGE_BUFFER=15.0
    
    Returns:
        Dictionary with parameter overrides
    """
    params = {
        'TILE_PARAMS': {},
        'REMAP_PARAMS': {},
        'MERGE_PARAMS': {},
    }
    
    for key, value in os.environ.items():
        # Parse TILE_ prefix
        if key.startswith('TILE_'):
            param_name = key[5:].lower()
            params['TILE_PARAMS'][param_name] = _parse_value(value)
        # Parse REMAP_ prefix
        elif key.startswith('REMAP_'):
            param_name = key[6:].lower()
            params['REMAP_PARAMS'][param_name] = _parse_value(value)
        # Parse MERGE_ prefix
        elif key.startswith('MERGE_'):
            param_name = key[6:].lower()
            params['MERGE_PARAMS'][param_name] = _parse_value(value)
    
    return params


def parse_param_override(param_str: str) -> tuple[str, str, Any]:
    """
    Parse a parameter override string.
    
    Format: "category.param=value" or "param=value"
    Examples:
    - "tile_length=150"
    - "TILE.tile_length=150"
    - "resolution_1=0.03"
    
    Args:
        param_str: Parameter override string
    
    Returns:
        Tuple of (category, param_name, value)
    """
    if '=' not in param_str:
        raise ValueError(f"Invalid parameter format: {param_str}. Expected format: param=value")
    
    key, value = param_str.split('=', 1)
    
    # Check if category is specified
    if '.' in key:
        category, param_name = key.split('.', 1)
        category = category.upper()
        if not category.endswith('_PARAMS'):
            category = f"{category}_PARAMS"
    else:
        # Try to infer category from parameter name
        param_name = key
        category = _infer_category(param_name)
    
    parsed_value = _parse_value(value)
    
    return category, param_name, parsed_value


def _infer_category(param_name: str) -> str:
    """Infer parameter category from parameter name."""
    tile_params = ['tile_length', 'tile_buffer', 'threads', 'workers', 
                   'resolution_1', 'resolution_2', 'grid_offset', 'skip_dimension_reduction']
    remap_params = ['target_resolution_cm']
    
    if param_name in tile_params:
        return 'TILE_PARAMS'
    elif param_name in remap_params:
        return 'REMAP_PARAMS'
    else:
        return 'MERGE_PARAMS'


def _parse_value(value_str: str) -> Any:
    """Parse string value to appropriate Python type."""
    # Try boolean
    if value_str.lower() in ('true', 'yes', '1'):
        return True
    if value_str.lower() in ('false', 'no', '0'):
        return False
    
    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def load_params(
    config_file: Optional[Path] = None,
    param_overrides: Optional[list[str]] = None,
    use_env: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Load parameters with priority: CLI overrides > config file > env vars > defaults.
    
    Args:
        config_file: Optional path to custom config file
        param_overrides: List of parameter override strings (e.g., ["tile_length=150"])
        use_env: Whether to load from environment variables
    
    Returns:
        Dictionary with TILE_PARAMS, REMAP_PARAMS, MERGE_PARAMS
    """
    # Start with defaults
    params = {
        'TILE_PARAMS': TILE_PARAMS.copy(),
        'REMAP_PARAMS': REMAP_PARAMS.copy(),
        'MERGE_PARAMS': MERGE_PARAMS.copy(),
    }
    
    # Apply environment variables
    if use_env:
        env_params = load_params_from_env()
        for category in ['TILE_PARAMS', 'REMAP_PARAMS', 'MERGE_PARAMS']:
            params[category].update(env_params[category])
    
    # Apply config file
    if config_file:
        file_params = load_params_from_file(config_file)
        for category, values in file_params.items():
            params[category].update(values)
    
    # Apply CLI overrides (highest priority)
    if param_overrides:
        for override in param_overrides:
            category, param_name, value = parse_param_override(override)
            if category in params:
                params[category][param_name] = value
    
    return params


def print_params(params: Dict[str, Dict[str, Any]]):
    """Print current parameter configuration."""
    print("=" * 60)
    print("Current Parameters")
    print("=" * 60)
    
    for category in ['TILE_PARAMS', 'REMAP_PARAMS', 'MERGE_PARAMS']:
        if category in params:
            print(f"\n{category}:")
            for key, value in sorted(params[category].items()):
                print(f"  {key}: {value}")
    
    print("=" * 60)


if __name__ == "__main__":
    """CLI for viewing/testing parameter configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View/test parameter configuration")
    parser.add_argument("--config", type=Path, help="Custom config file")
    parser.add_argument("--param", action="append", help="Parameter override (param=value)")
    parser.add_argument("--no-env", action="store_true", help="Ignore environment variables")
    
    args = parser.parse_args()
    
    params = load_params(
        config_file=args.config,
        param_overrides=args.param,
        use_env=not args.no_env
    )
    
    print_params(params)
