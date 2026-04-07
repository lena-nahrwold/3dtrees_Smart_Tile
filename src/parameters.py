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
        description="Task to perform: 'tile' (tiling + subsampling), 'merge' (remap + merge), 'remap' (merged file -> original files), 'filter' (remove buffer-zone instances per tile, optional small-instance redistribution), or 'filter_remap' (filter then remap to original files, optional merged output)",
    )
    
    input_dir: Optional[Path] = Field(
        default=None,
        description="Input directory with LAZ/LAS files (required for 'tile' task)",
        validation_alias=AliasChoices("input-dir", "input_dir"),
    )
    
    output_dir: Optional[Path] = Field(
        default=None,
        description="Output directory (required for 'tile' task)",
        validation_alias=AliasChoices("output-dir", "output_dir"),
    )
    
    workers: int = Field(
        4,
        description="Number of parallel workers for processing",
        validation_alias=AliasChoices("workers", "number-of-threads", "number_of_threads"),
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
        20,
        description="Buffer overlap in meters (only for 'tile' task)",
        validation_alias=AliasChoices("tile-buffer", "tile_buffer"),
    )
    
    threads: Optional[int] = Field(
        10,
        description="Threads per COPC writer (only for 'tile' task)",
    )
    
    resolution_1: Optional[float] = Field(
        0.01,
        description="First subsampling resolution in meters (1cm) (only for 'tile' task)",
        validation_alias=AliasChoices("resolution-1", "resolution_1"),
    )
    
    resolution_2: Optional[float] = Field(
        0.1,
        description="Second subsampling resolution in meters (10cm) (only for 'tile' task)",
        validation_alias=AliasChoices("resolution-2", "resolution_2"),
    )
    
    
    skip_dimension_reduction: bool = Field(
        False,
        description="Skip XYZ-only reduction, keep all point dimensions. Set to False only for raw pre-segmentation data (only for 'tile' task)",
        validation_alias=AliasChoices("skip-dimension-reduction", "skip_dimension_reduction"),
    )
    
    instance_dimension: str = Field(
        "PredInstance",
        description="Name of the instance ID dimension in input files (default: PredInstance, fallback: treeID)",
        validation_alias=AliasChoices("instance-dimension", "instance_dimension"),
    )
    
    num_spatial_chunks: Optional[int] = Field(
        default=None,
        description="Number of spatial chunks per tile for subsampling (default: equals workers)",
        validation_alias=AliasChoices("num-spatial-chunks", "num_spatial_chunks"),
    )

    tiling_threshold: Optional[float] = Field(
        default=None,
        description="File size threshold in MB. If input folder has single file below this size, skip tiling (only for 'tile' task)",
        validation_alias=AliasChoices("tiling-threshold", "tiling_threshold"),
    )

    chunk_size: Optional[int] = Field(
        default=20_000_000,
        description="Points per chunk when reading LAZ/LAS in tiling Phase 1, and also for "
                    "chunkwise source COPC creation when that mode is enabled "
                    "(smaller = less peak RAM, more overhead; only for 'tile' task)",
        validation_alias=AliasChoices("chunk-size", "chunk_size"),
    )

    chunkwise_copc_source_creation: bool = Field(
        default=False,
        description="Stream source LAZ/LAS files into temporary LAS parts before final COPC creation. "
                    "Reduces peak RAM during source normalization, but uses more temporary disk space "
                    "and extra I/O (only for 'tile' task).",
        validation_alias=AliasChoices(
            "chunkwise-copc-source-creation",
            "chunkwise_copc_source_creation",
        ),
    )

    # ==========================================================================
    # Merge task parameters
    # ==========================================================================
    
    subsampled_10cm_folder: Optional[Path] = Field(
        default=None,
        description="Path to subsampled 10cm folder with segmented results (for 'merge' task)",
        validation_alias=AliasChoices("subsampled-10cm-folder", "subsampled_10cm_folder", "subsampled-segmented-folder"),
    )
    
    subsampled_target_folder: Optional[Path] = Field(
        default=None,
        description="Path to target resolution subsampled folder (auto-derived if not specified)",
        validation_alias=AliasChoices("subsampled-target-folder", "subsampled_target_folder"),
    )
    
    segmented_remapped_folder: Optional[Path] = Field(
        default=None,
        description="Path to segmented remapped folder (for 'merge' task, skip remap step)",
        validation_alias=AliasChoices("segmented-remapped-folder", "segmented_remapped_folder"),
    )
    
    original_tiles_dir: Optional[Path] = Field(
        default=None,
        description="Directory with original tile files for retiling (for 'merge' task)",
        validation_alias=AliasChoices("original-tiles-dir", "original_tiles_dir"),
    )

    tile_bounds_json: Optional[Path] = Field(
        default=None,
        description="Path to tile_bounds_tindex.json for neighbor graph and remap matching (merge task). If set, used instead of auto-derived paths.",
        validation_alias=AliasChoices("tile-bounds-json", "tile_bounds_json"),
    )
    
    original_input_dir: Optional[Path] = Field(
        default=None,
        description="Directory with original input LAZ files for final remap (for 'merge' task)",
        validation_alias=AliasChoices("original-input-dir", "original_input_dir"),
    )
    
    output_merged_laz: Optional[Path] = Field(
        default=None,
        description="Output path for merged LAZ file (auto-derived if not specified)",
        validation_alias=AliasChoices("output-merged-laz", "output_merged_laz"),
    )
    
    output_tiles_folder: Optional[Path] = Field(
        default=None,
        description="Output folder for per-tile results (auto-derived if not specified)",
        validation_alias=AliasChoices("output-tiles-folder", "output_tiles_folder"),
    )
    
    output_folder: Optional[Path] = Field(
        default=None,
        description="Output folder for remapped files (auto-derived if not specified)",
        validation_alias=AliasChoices("output-folder", "output_folder"),
    )

    # ==========================================================================
    # Remap-to-originals task parameters (one merged file -> original files folder)
    # ==========================================================================

    merged_laz: Optional[Path] = Field(
        default=None,
        description="Path to merged LAZ file (for 'remap' task). All dimensions from this file are added to original files.",
        validation_alias=AliasChoices("merged-laz", "merged_laz"),
    )

    output_merged_with_originals: Optional[Path] = Field(
        default=None,
        description="Path for merged LAZ with original-file dimensions added (for remap task). If unset, default to output_dir / merged_with_originals.laz.",
        validation_alias=AliasChoices("output-merged-with-originals", "output_merged_with_originals"),
    )

    transfer_original_dims_to_merged: bool = Field(
        True,
        description="Transfer original-file dimensions (e.g. Intensity, RGB) to the merged point cloud. Applies to merge task (single-file path) and remap task.",
        validation_alias=AliasChoices("transfer-original-dims-to-merged", "transfer_original_dims_to_merged"),
    )

    threedtrees_dims: str = Field(
        "PredInstance,PredSemantic",
        description="Comma-separated list of dimension names produced by 3DTrees to transfer to original files. These are renamed to 3DT_{name}_{suffix} in the output (e.g. 3DT_PredInstance_SAT).",
        validation_alias=AliasChoices("threedtrees-dims", "threedtrees_dims"),
    )

    threedtrees_suffix: str = Field(
        "SAT",
        description="Suffix for 3DTrees dimension branding (e.g. SAT → 3DT_PredInstance_SAT).",
        validation_alias=AliasChoices("threedtrees-suffix", "threedtrees_suffix"),
    )

    standardization_json: Optional[Path] = Field(
        default=None,
        description="Path to collection_summary.json from tool_standard. "
                    "When provided, only dimensions listed in "
                    "collection.reference_attribute_names (minus X,Y,Z) "
                    "are transferred from originals to the merged file, "
                    "overwriting any existing merged values.",
        validation_alias=AliasChoices("standardization-json", "standardization_json"),
    )

    merge_chunk_size: int = Field(
        default=2_000_000,
        description="Points per streaming chunk in merge pipeline. "
                    "Controls memory vs speed for all merge I/O (remap, enrichment, retiling). "
                    "Larger = faster but more RAM. Remap uses 3x this value.",
        validation_alias=AliasChoices("merge-chunk-size", "merge_chunk_size"),
    )

    # ==========================================================================
    # Remap task parameters
    # ==========================================================================

    source_folder: Optional[Path] = Field(
        default=None,
        description="Path to source LAZ files (e.g., segmented files) for 'remap' task",
        validation_alias=AliasChoices("source-folder", "source_folder"),
    )

    target_folder: Optional[Path] = Field(
        default=None,
        description="Path to target LAZ files (e.g., subsampled files) for 'remap' task",
        validation_alias=AliasChoices("target-folder", "target_folder"),
    )

    # Merge algorithm parameters
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
    
    max_volume_for_merge: Optional[float] = Field(
        5.0,
        description="Max 3D convex hull volume for small instance reassignment in m³",
        validation_alias=AliasChoices("max-volume-for-merge", "max_volume_for_merge"),
    )
    
    border_zone_width: Optional[float] = Field(
        10.0,
        description="Width of border zone inward from core boundary for instance matching (meters)",
        validation_alias=AliasChoices("border-zone-width", "border_zone_width"),
    )

    filter_anchor: str = Field(
        "centroid",
        description="For 'filter' task: which point of each instance is checked against the buffer zone. "
                    "Options: 'centroid' (default), 'highest_point', 'lowest_point'.",
        validation_alias=AliasChoices("filter-anchor", "filter_anchor"),
    )

    segmented_folders: str = Field(
        "",
        description="Comma-separated list of segmented tile folders (for filter_remap and remap tasks). "
                    "All extra dimensions from each folder are copied as-is to output files. "
                    "Dimensions with the same name across collections are suffixed _2, _3, etc.",
        validation_alias=AliasChoices("segmented-folders", "segmented_folders"),
    )

    produce_merged_file: bool = Field(
        False,
        description="For filter_remap task: also write a single merged.laz (and merged_with_originals.laz) "
                    "containing all points with original and segmentation dimensions combined.",
        validation_alias=AliasChoices("produce-merged-file", "produce_merged_file"),
    )

    remap_dims: Optional[str] = Field(
        None,
        description="Comma-separated extra dimension names to transfer during remap "
                    "(e.g. 'PredInstance,PredSemantic'). Default: all extra dims from the collection files.",
        validation_alias=AliasChoices("remap-dims", "remap_dims"),
    )

    remap_spatial_slices: int = Field(
        10,
        description="Number of spatial slices per original file for multi-collection remap "
                    "(higher = lower peak RAM, more subset queries).",
        validation_alias=AliasChoices("remap-spatial-slices", "remap_spatial_slices"),
    )

    remap_spatial_chunk_length: Optional[float] = Field(
        None,
        description="Optional spatial slice length in metres for remap/enrichment. "
                    "When set, this overrides the fixed slice count and derives the number "
                    "of slices from the file span.",
        validation_alias=AliasChoices("remap-spatial-chunk-length", "remap_spatial_chunk_length"),
    )

    remap_spatial_target_points: Optional[int] = Field(
        None,
        description="Optional target number of points per spatial slice for remap/enrichment. "
                    "When set, this overrides both fixed slice count and metre length and "
                    "derives the number of slices from the file header point count.",
        validation_alias=AliasChoices("remap-spatial-target-points", "remap_spatial_target_points"),
    )

    force_spatial_chunked_fusion: bool = Field(
        False,
        description="For multi-collection remap, skip the aligned-fusion check and go directly "
                    "to chunked spatial fusion before remapping.",
        validation_alias=AliasChoices("force-spatial-chunked-fusion", "force_spatial_chunked_fusion"),
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
    
    skip_merged_file: bool = Field(
        False,
        description="Skip creating merged LAZ file (only create retiled outputs)",
        validation_alias=AliasChoices("skip-merged-file", "skip_merged_file"),
    )

    save_filtered_tiles: bool = Field(
        False,
        description="Save filtered tiles (with buffer-zone instances removed) for debugging",
        validation_alias=AliasChoices("save-filtered-tiles", "save_filtered_tiles"),
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
        "chunk_size",
    )
    @classmethod
    def validate_tile_params(cls, v, info):
        """Validate tile parameters are positive when provided."""
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v
    
    @field_validator(
        "overlap_threshold",
        "max_centroid_distance",
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
    print(f"  instance_dimension: {params.instance_dimension}")
    
    print("\nTile Task:")
    print(f"  tile_length: {params.tile_length}")
    print(f"  tile_buffer: {params.tile_buffer}")
    print(f"  threads: {params.threads}")
    print(f"  chunk_size: {params.chunk_size}")
    print(f"  chunkwise_copc_source_creation: {params.chunkwise_copc_source_creation}")
    print(f"  resolution_1: {params.resolution_1}")
    print(f"  resolution_2: {params.resolution_2}")
    print(f"  skip_dimension_reduction: {params.skip_dimension_reduction}")
    
    print("\nMerge Task:")
    print(f"  subsampled_10cm_folder: {params.subsampled_10cm_folder}")
    print(f"  original_input_dir: {params.original_input_dir}")
    print(f"  overlap_threshold: {params.overlap_threshold}")
    print(f"  max_centroid_distance: {params.max_centroid_distance}")
    print(f"  max_volume_for_merge: {params.max_volume_for_merge}")
    print(f"  min_cluster_size: {params.min_cluster_size}")
    print(f"  disable_matching: {params.disable_matching}")
    print(f"  standardization_json: {params.standardization_json}")
    print(f"  merge_chunk_size: {params.merge_chunk_size:,}")
    print(f"  remap_spatial_slices: {params.remap_spatial_slices}")
    print(f"  remap_spatial_chunk_length: {params.remap_spatial_chunk_length}")
    print(f"  remap_spatial_target_points: {params.remap_spatial_target_points}")
    print(f"  force_spatial_chunked_fusion: {params.force_spatial_chunked_fusion}")
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
        'skip_dimension_reduction': params.skip_dimension_reduction,
        'chunk_size': params.chunk_size,
        'chunkwise_copc_source_creation': params.chunkwise_copc_source_creation,
    }


def get_merge_params(params: Parameters) -> dict:
    """Get merge parameters as a dictionary for legacy compatibility."""
    return {
        'overlap_threshold': params.overlap_threshold,
        'max_centroid_distance': params.max_centroid_distance,
        'max_volume_for_merge': params.max_volume_for_merge,
        'min_cluster_size': params.min_cluster_size,
        'workers': params.workers,
        'verbose': params.verbose,
        'instance_dimension': params.instance_dimension,
        'standardization_json': params.standardization_json,
        'merge_chunk_size': params.merge_chunk_size,
    }


def get_remap_params(params: Parameters) -> dict:
    """Get remap parameters as a dictionary for legacy compatibility."""
    return {
        'workers': params.workers,
        'instance_dimension': params.instance_dimension,
        'remap_spatial_slices': params.remap_spatial_slices,
        'remap_spatial_chunk_length': params.remap_spatial_chunk_length,
        'remap_spatial_target_points': params.remap_spatial_target_points,
        'force_spatial_chunked_fusion': params.force_spatial_chunked_fusion,
    }


# Legacy dict exports for backwards compatibility with modules that import them directly
TILE_PARAMS = {
    'tile_length': 100,
    'tile_buffer': 20,
    'threads': 10,
    'workers': 4,
    'resolution_1': 0.02,
    'resolution_2': 0.1,
    'skip_dimension_reduction': False,
    'chunk_size': 20_000_000,
    'chunkwise_copc_source_creation': False,
}

REMAP_PARAMS = {
    'target_resolution_cm': 2,
    'workers': 4,
    'remap_spatial_slices': 10,
    'remap_spatial_chunk_length': None,
    'remap_spatial_target_points': None,
    'force_spatial_chunked_fusion': False,
}

MERGE_PARAMS = {
    'overlap_threshold': 0.3,
    'max_centroid_distance': 3.0,
    'max_volume_for_merge': 5.0,
    'min_cluster_size': 300,
    'workers': 4,
    'verbose': True,
    'retile_buffer': 2.0,  # Fixed to 2.0m
    'merge_chunk_size': 2_000_000,
}


if __name__ == "__main__":
    """CLI for viewing/testing parameter configuration."""
    params = Parameters()
    print_params(params)
