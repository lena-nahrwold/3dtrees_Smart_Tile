"""
Centralized parameter configuration for the 3DTrees smart tiling pipeline.

Uses Pydantic BaseSettings for CLI argument parsing, environment variable support,
and parameter validation.

Usage:
    python run.py --task tile --input-dir /path/to/input --output-dir /path/to/output
    python run.py --task merge --subsampled-10cm-folder /path/to/10cm --original-input-dir /path/to/input
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, field_validator
from pathlib import Path
from typing import Optional


class Parameters(BaseSettings):
    """
    Pipeline parameters with CLI and environment variable support.
    
    All parameters can be passed via:
    - CLI arguments: --param-name value
    - Environment variables: PARAM_NAME=value
    """
    
    # ==========================================================================
    # Common parameters
    # ==========================================================================
    
    task: str = Field(
        "tile",
        description="Task to perform: 'tile' (tiling + subsampling) or 'merge' (remap + merge)",
    )
    
    input_dir: Optional[Path] = Field(
        None,
        description="Input directory with LAZ/LAS files (required for 'tile' task)",
        validation_alias=AliasChoices("input-dir", "input_dir"),
    )
    
    output_dir: Optional[Path] = Field(
        None,
        description="Output directory (required for 'tile' task)",
        validation_alias=AliasChoices("output-dir", "output_dir"),
    )
    
    workers: int = Field(
        4,
        description="Number of parallel workers for processing",
        validation_alias=AliasChoices("workers", "number-of-threads", "number_of_threads"),
    )
    
    show_params: bool = Field(
        False,
        description="Show current parameter configuration and exit",
        validation_alias=AliasChoices("show-params", "show_params"),
    )
    
    # ==========================================================================
    # Tile task parameters
    # ==========================================================================
    
    tile_length: Optional[int] = Field(
        100,
        description="Tile size in meters (only for 'tile' task)",
        validation_alias=AliasChoices("tile-length", "tile_length"),
    )
    
    tile_buffer: Optional[int] = Field(
        5,
        description="Buffer overlap in meters (only for 'tile' task)",
        validation_alias=AliasChoices("tile-buffer", "tile_buffer"),
    )
    
    threads: Optional[int] = Field(
        5,
        description="Threads per COPC writer (only for 'tile' task)",
    )
    
    resolution_1: Optional[float] = Field(
        0.02,
        description="First subsampling resolution in meters (2cm) (only for 'tile' task)",
        validation_alias=AliasChoices("resolution-1", "resolution_1"),
    )
    
    resolution_2: Optional[float] = Field(
        0.1,
        description="Second subsampling resolution in meters (10cm) (only for 'tile' task)",
        validation_alias=AliasChoices("resolution-2", "resolution_2"),
    )
    
    grid_offset: Optional[float] = Field(
        1.0,
        description="Grid offset from min coordinates in meters (only for 'tile' task)",
        validation_alias=AliasChoices("grid-offset", "grid_offset"),
    )
    
    skip_dimension_reduction: bool = Field(
        False,
        description="Skip XYZ-only reduction, keep all point dimensions (only for 'tile' task)",
        validation_alias=AliasChoices("skip-dimension-reduction", "skip_dimension_reduction"),
    )
    
    num_spatial_chunks: Optional[int] = Field(
        None,
        description="Number of spatial chunks per tile for subsampling (default: equals workers)",
        validation_alias=AliasChoices("num-spatial-chunks", "num_spatial_chunks"),
    )
    
    # ==========================================================================
    # Merge task parameters
    # ==========================================================================
    
    subsampled_10cm_folder: Optional[Path] = Field(
        None,
        description="Path to subsampled 10cm folder with segmented results (for 'merge' task)",
        validation_alias=AliasChoices("subsampled-10cm-folder", "subsampled_10cm_folder"),
    )
    
    subsampled_target_folder: Optional[Path] = Field(
        None,
        description="Path to target resolution subsampled folder (auto-derived if not specified)",
        validation_alias=AliasChoices("subsampled-target-folder", "subsampled_target_folder"),
    )
    
    segmented_remapped_folder: Optional[Path] = Field(
        None,
        description="Path to segmented remapped folder (for 'merge' task, skip remap step)",
        validation_alias=AliasChoices("segmented-remapped-folder", "segmented_remapped_folder"),
    )
    
    original_tiles_dir: Optional[Path] = Field(
        None,
        description="Directory with original tile files for retiling (for 'merge' task)",
        validation_alias=AliasChoices("original-tiles-dir", "original_tiles_dir"),
    )
    
    original_input_dir: Optional[Path] = Field(
        None,
        description="Directory with original input LAZ files for final remap (for 'merge' task)",
        validation_alias=AliasChoices("original-input-dir", "original_input_dir"),
    )
    
    output_merged_laz: Optional[Path] = Field(
        None,
        description="Output path for merged LAZ file (auto-derived if not specified)",
        validation_alias=AliasChoices("output-merged-laz", "output_merged_laz"),
    )
    
    output_tiles_folder: Optional[Path] = Field(
        None,
        description="Output folder for per-tile results (auto-derived if not specified)",
        validation_alias=AliasChoices("output-tiles-folder", "output_tiles_folder"),
    )
    
    output_folder: Optional[Path] = Field(
        None,
        description="Output folder for remapped files (auto-derived if not specified)",
        validation_alias=AliasChoices("output-folder", "output_folder"),
    )
    
    # Remap parameters
    target_resolution: Optional[int] = Field(
        2,
        description="Target resolution in cm for remapping (default: 2cm)",
        validation_alias=AliasChoices("target-resolution", "target_resolution"),
    )
    
    # Merge algorithm parameters
    buffer: Optional[float] = Field(
        10.0,
        description="Buffer distance for filtering in meters (for 'merge' task)",
    )
    
    overlap_threshold: Optional[float] = Field(
        0.3,
        description="Overlap ratio threshold for instance matching (0.3 = 30%)",
        validation_alias=AliasChoices("overlap-threshold", "overlap_threshold"),
    )
    
    max_centroid_distance: Optional[float] = Field(
        3.0,
        description="Max centroid distance to merge instances in meters",
        validation_alias=AliasChoices("max-centroid-distance", "max_centroid_distance"),
    )
    
    correspondence_tolerance: Optional[float] = Field(
        0.05,
        description="Max distance for point correspondence in meters (should be small ~5cm)",
        validation_alias=AliasChoices("correspondence-tolerance", "correspondence_tolerance"),
    )
    
    max_volume_for_merge: Optional[float] = Field(
        4.0,
        description="Max convex hull volume for small instance merging in m³",
        validation_alias=AliasChoices("max-volume-for-merge", "max_volume_for_merge"),
    )
    
    min_cluster_size: Optional[int] = Field(
        300,
        description="Minimum cluster size in points for reassignment",
        validation_alias=AliasChoices("min-cluster-size", "min_cluster_size"),
    )
    
    disable_matching: bool = Field(
        False,
        description="Disable cross-tile instance matching",
        validation_alias=AliasChoices("disable-matching", "disable_matching"),
    )
    
    disable_volume_merge: bool = Field(
        False,
        description="Disable small volume instance merging",
        validation_alias=AliasChoices("disable-volume-merge", "disable_volume_merge"),
    )
    
    verbose: bool = Field(
        False,
        description="Print detailed merge decisions",
    )
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    
    @field_validator(
        "input_dir",
        "output_dir",
    )
    @classmethod
    def validate_tile_required_params(cls, v, info):
        """Validate that tile task required parameters are provided."""
        # Note: Actual validation happens in run.py after instantiation
        # since we need to check the task value
        return v
    
    @field_validator(
        "tile_length",
        "tile_buffer",
        "resolution_1",
        "resolution_2",
        "threads",
    )
    @classmethod
    def validate_tile_params(cls, v, info):
        """Validate tile parameters are positive when provided."""
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v
    
    @field_validator(
        "buffer",
        "overlap_threshold",
        "max_centroid_distance",
        "correspondence_tolerance",
        "max_volume_for_merge",
    )
    @classmethod
    def validate_merge_params(cls, v, info):
        """Validate merge parameters are positive when provided."""
        if v is not None and v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v
    
    @field_validator("overlap_threshold")
    @classmethod
    def validate_overlap_threshold(cls, v):
        """Validate overlap threshold is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("overlap_threshold must be between 0 and 1")
        return v
    
    @field_validator("workers", "min_cluster_size")
    @classmethod
    def validate_positive_int(cls, v, info):
        """Validate integer parameters are positive."""
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v
    
    # ==========================================================================
    # Model configuration
    # ==========================================================================
    
    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        cli_ignore_unknown_args=True,
        env_prefix="",  # No prefix for env vars
        extra="ignore",  # Ignore unknown fields
    )


def print_params(params: Parameters):
    """Print current parameter configuration."""
    print("=" * 60)
    print("Current Parameters")
    print("=" * 60)
    
    print("\nCommon:")
    print(f"  task: {params.task}")
    print(f"  input_dir: {params.input_dir}")
    print(f"  output_dir: {params.output_dir}")
    print(f"  workers: {params.workers}")
    
    print("\nTile Task:")
    print(f"  tile_length: {params.tile_length}")
    print(f"  tile_buffer: {params.tile_buffer}")
    print(f"  threads: {params.threads}")
    print(f"  resolution_1: {params.resolution_1}")
    print(f"  resolution_2: {params.resolution_2}")
    print(f"  grid_offset: {params.grid_offset}")
    print(f"  skip_dimension_reduction: {params.skip_dimension_reduction}")
    
    print("\nMerge Task:")
    print(f"  subsampled_10cm_folder: {params.subsampled_10cm_folder}")
    print(f"  original_input_dir: {params.original_input_dir}")
    print(f"  target_resolution: {params.target_resolution}")
    print(f"  buffer: {params.buffer}")
    print(f"  overlap_threshold: {params.overlap_threshold}")
    print(f"  max_centroid_distance: {params.max_centroid_distance}")
    print(f"  correspondence_tolerance: {params.correspondence_tolerance}")
    print(f"  max_volume_for_merge: {params.max_volume_for_merge}")
    print(f"  min_cluster_size: {params.min_cluster_size}")
    print(f"  disable_matching: {params.disable_matching}")
    print(f"  verbose: {params.verbose}")
    
    print("=" * 60)


# Legacy compatibility: provide dict-like access for modules that need it
def get_tile_params(params: Parameters) -> dict:
    """Get tile parameters as a dictionary for legacy compatibility."""
    return {
        'tile_length': params.tile_length,
        'tile_buffer': params.tile_buffer,
        'threads': params.threads,
        'workers': params.workers,
        'resolution_1': params.resolution_1,
        'resolution_2': params.resolution_2,
        'grid_offset': params.grid_offset,
        'skip_dimension_reduction': params.skip_dimension_reduction,
    }


def get_merge_params(params: Parameters) -> dict:
    """Get merge parameters as a dictionary for legacy compatibility."""
    return {
        'buffer': params.buffer,
        'overlap_threshold': params.overlap_threshold,
        'max_centroid_distance': params.max_centroid_distance,
        'correspondence_tolerance': params.correspondence_tolerance,
        'max_volume_for_merge': params.max_volume_for_merge,
        'min_cluster_size': params.min_cluster_size,
        'workers': params.workers,
        'verbose': params.verbose,
    }


def get_remap_params(params: Parameters) -> dict:
    """Get remap parameters as a dictionary for legacy compatibility."""
    return {
        'target_resolution_cm': params.target_resolution,
        'workers': params.workers,
    }


# Legacy dict exports for backwards compatibility with modules that import them directly
TILE_PARAMS = {
    'tile_length': 100,
    'tile_buffer': 5,
    'threads': 5,
    'workers': 4,
    'resolution_1': 0.02,
    'resolution_2': 0.1,
    'grid_offset': 1.0,
    'skip_dimension_reduction': False,
}

REMAP_PARAMS = {
    'target_resolution_cm': 2,
    'workers': 4,
}

MERGE_PARAMS = {
    'buffer': 10.0,
    'overlap_threshold': 0.3,
    'max_centroid_distance': 3.0,
    'correspondence_tolerance': 0.05,
    'max_volume_for_merge': 4.0,
    'min_cluster_size': 300,
    'workers': 4,
    'verbose': True,
}


if __name__ == "__main__":
    """CLI for viewing/testing parameter configuration."""
    params = Parameters()
    print_params(params)
