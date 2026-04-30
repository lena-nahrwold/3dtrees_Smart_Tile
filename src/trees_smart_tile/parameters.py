"""
Centralized parameter configuration for the 3DTrees smart tile pipeline.

The public CLI exposes the active task entrypoints:
- ``tile``: COPC-first tiling plus two-stage subsampling
- ``filter``: buffer-zone filtering of segmented tiles, optionally followed by a remap tail
- ``remap``: remap segmented collection dimensions back to original files
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


TaskName = Literal["tile", "filter", "remap"]
FilterAnchor = Literal["centroid", "highest_point", "lowest_point"]


class Parameters(BaseSettings):
    """Pipeline parameters with CLI and environment variable support."""

    # Shared
    task: Optional[TaskName] = Field(
        default=None,
        description="Task to perform: 'tile', 'filter', or 'remap'.",
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="Output directory for the selected task.",
        validation_alias=AliasChoices("output-dir", "output_dir"),
    )
    workers: int = Field(
        default=4,
        description="Number of parallel workers.",
        validation_alias=AliasChoices("workers", "number-of-threads", "number_of_threads"),
    )
    chunk_size: int = Field(
        default=20_000_000,
        description="Points per streaming chunk for filtering, remap, and internal COPC preparation.",
        validation_alias=AliasChoices("chunk-size", "chunk_size"),
    )

    # Tile
    input_dir: Optional[Path] = Field(
        default=None,
        description="Input directory with LAZ/LAS files for the tile task.",
        validation_alias=AliasChoices("input-dir", "input_dir"),
    )
    tile_length: int = Field(
        default=100,
        description="Tile size in meters for the tile task.",
        validation_alias=AliasChoices("tile-length", "tile_length"),
    )
    tile_buffer: int = Field(
        default=20,
        description="Tile overlap buffer in meters for the tile task.",
        validation_alias=AliasChoices("tile-buffer", "tile_buffer"),
    )
    threads: int = Field(
        default=10,
        description="Threads per tile/subsampling worker for the tile task.",
        validation_alias=AliasChoices("threads"),
    )
    resolution_1: float = Field(
        default=0.01,
        description="First subsampling resolution in meters for the tile task.",
        validation_alias=AliasChoices("resolution-1", "resolution_1"),
    )
    resolution_2: float = Field(
        default=0.1,
        description="Second subsampling resolution in meters for the tile task.",
        validation_alias=AliasChoices("resolution-2", "resolution_2"),
    )
    output_copc_res1: bool = Field(
        default=True,
        description="Write res1 subsampling outputs as COPC ('.copc.laz') instead of standard LAZ ('.laz').",
        validation_alias=AliasChoices("output-copc-res1", "output_copc_res1"),
    )
    output_copc_res2: bool = Field(
        default=False,
        description="Write res2 subsampling outputs as COPC ('.copc.laz') instead of standard LAZ ('.laz').",
        validation_alias=AliasChoices("output-copc-res2", "output_copc_res2"),
    )
    dimension_reduction: bool = Field(
        default=True,
        description="Reduce subsampled outputs to minimal LAS dimensions for smaller files.",
        validation_alias=AliasChoices("dimension-reduction", "dimension_reduction"),
    )
    skip_dimension_reduction: Optional[bool] = Field(
        default=None,
        description="Deprecated inverse alias for dimension reduction. When set, overrides dimension_reduction as not skip_dimension_reduction.",
        validation_alias=AliasChoices("skip-dimension-reduction", "skip_dimension_reduction"),
    )
    num_spatial_chunks: Optional[int] = Field(
        default=None,
        description="Number of spatial chunks per tile during subsampling. Defaults to the worker count when omitted.",
        validation_alias=AliasChoices("num-spatial-chunks", "num_spatial_chunks"),
    )
    tiling_threshold: Optional[float] = Field(
        default=None,
        description="If the tile input folder contains a single file below this size in MB, skip tile generation after COPC normalization.",
        validation_alias=AliasChoices("tiling-threshold", "tiling_threshold"),
    )
    chunkwise_copc_source_creation: bool = Field(
        default=False,
        description="Build source COPC files through temporary chunked LAS parts to reduce peak RAM during tile normalization.",
        validation_alias=AliasChoices("chunkwise-copc-source-creation", "chunkwise_copc_source_creation"),
    )

    # Filter
    segmented_folders: str = Field(
        default="",
        description="Comma-separated list of segmented tile files and/or folders.",
        validation_alias=AliasChoices("segmented-folders", "segmented_folders"),
    )
    tile_bounds_json: Optional[Path] = Field(
        default=None,
        description="Path to tile_bounds_tindex.json used for filter neighbor/border logic.",
        validation_alias=AliasChoices("tile-bounds-json", "tile_bounds_json"),
    )
    instance_dimension: str = Field(
        default="PredInstance",
        description="Name of the instance ID dimension in segmented input files.",
        validation_alias=AliasChoices("instance-dimension", "instance_dimension"),
    )
    border_zone_width: Optional[float] = Field(
        default=None,
        description="Optional explicit border-zone width in meters; derived from tile_bounds_json when omitted.",
        validation_alias=AliasChoices("border-zone-width", "border_zone_width"),
    )
    filter_anchor: FilterAnchor = Field(
        default="centroid",
        description="Representative point used to classify an instance as border-adjacent.",
        validation_alias=AliasChoices("filter-anchor", "filter_anchor"),
    )
    min_cluster_size: int = Field(
        default=300,
        description="Minimum cluster size in points for reassignment.",
        validation_alias=AliasChoices("min-cluster-size", "min_cluster_size"),
    )
    max_volume_for_merge: float = Field(
        default=5.0,
        description="Maximum 3D convex hull volume for small-instance reassignment in m^3.",
        validation_alias=AliasChoices("max-volume-for-merge", "max_volume_for_merge"),
    )
    enable_volume_merge: bool = Field(
        default=True,
        description="Enable reassignment of very small kept instances.",
        validation_alias=AliasChoices("enable-volume-merge", "enable_volume_merge"),
    )
    remap_merge: bool = Field(
        default=False,
        description="After filtering, run the remap task on the filtered collection.",
        validation_alias=AliasChoices("remap-merge", "remap_merge"),
    )

    # Remap / filter-remap shared
    original_input_dir: Optional[Path] = Field(
        default=None,
        description="Directory with original input LAZ/LAS files for remap outputs.",
        validation_alias=AliasChoices("original-input-dir", "original_input_dir"),
    )
    subsampled_target_folder: Optional[Path] = Field(
        default=None,
        description="Optional target-resolution collection to receive remapped dimensions.",
        validation_alias=AliasChoices("subsampled-target-folder", "subsampled_target_folder"),
    )
    produce_merged_file: bool = Field(
        default=True,
        description="Also write a single merged output file.",
        validation_alias=AliasChoices("produce-merged-file", "produce_merged_file"),
    )
    transfer_original_dims_to_merged: bool = Field(
        default=True,
        description="When producing a merged file, enrich it with original-file attributes.",
        validation_alias=AliasChoices("transfer-original-dims-to-merged", "transfer_original_dims_to_merged"),
    )
    output_merged_with_originals: Optional[Path] = Field(
        default=None,
        description="Optional explicit path for merged output enriched with original attributes.",
        validation_alias=AliasChoices("output-merged-with-originals", "output_merged_with_originals"),
    )
    standardization_json: Optional[Path] = Field(
        default=None,
        description="Optional collection_summary.json to restrict which original attributes are transferred into merged outputs. Source prediction/remap dimensions in the merged file remain controlled independently by --remap-dims.",
        validation_alias=AliasChoices("standardization-json", "standardization_json"),
    )
    remap_dims: Optional[str] = Field(
        default=None,
        description="Optional strict allowlist of source dimensions to transfer during remap. When omitted, all extra dimensions are transferred. Listed names may be standard or extra dimensions. COPC-safe aliases may appear in outputs, for example 3DT_* source dimensions may be written as TDT_*.",
        validation_alias=AliasChoices("remap-dims", "remap_dims"),
    )

    @field_validator("workers", "threads", "tile_length", "tile_buffer", "chunk_size", "min_cluster_size")
    @classmethod
    def validate_positive_int(cls, value, info):
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value

    @field_validator("resolution_1", "resolution_2", "border_zone_width", "max_volume_for_merge", "tiling_threshold")
    @classmethod
    def validate_non_negative_float(cls, value, info):
        if value is not None and value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    model_config = SettingsConfigDict(
        case_sensitive=False,
        cli_parse_args=True,
        cli_ignore_unknown_args=True,
        env_prefix="",
        extra="ignore",
    )


def print_params(params: Parameters):
    """Print current parameter configuration."""
    print("=" * 60)
    print("Current Parameters")
    print("=" * 60)
    print(f"task: {params.task}")
    print(f"output_dir: {params.output_dir}")
    print(f"workers: {params.workers}")
    print(f"chunk_size: {params.chunk_size}")
    print(f"input_dir: {params.input_dir}")
    print(f"tile_length: {params.tile_length}")
    print(f"tile_buffer: {params.tile_buffer}")
    print(f"output_copc_res1: {params.output_copc_res1}")
    print(f"output_copc_res2: {params.output_copc_res2}")
    print(f"threads: {params.threads}")
    print(f"resolution_1: {params.resolution_1}")
    print(f"resolution_2: {params.resolution_2}")
    print(f"dimension_reduction: {params.dimension_reduction}")
    print(f"skip_dimension_reduction: {params.skip_dimension_reduction}")
    print(f"num_spatial_chunks: {params.num_spatial_chunks}")
    print(f"tiling_threshold: {params.tiling_threshold}")
    print(f"chunkwise_copc_source_creation: {params.chunkwise_copc_source_creation}")
    print(f"segmented_folders: {params.segmented_folders}")
    print(f"tile_bounds_json: {params.tile_bounds_json}")
    print(f"instance_dimension: {params.instance_dimension}")
    print(f"border_zone_width: {params.border_zone_width}")
    print(f"filter_anchor: {params.filter_anchor}")
    print(f"min_cluster_size: {params.min_cluster_size}")
    print(f"max_volume_for_merge: {params.max_volume_for_merge}")
    print(f"enable_volume_merge: {params.enable_volume_merge}")
    print(f"remap_merge: {params.remap_merge}")
    print(f"original_input_dir: {params.original_input_dir}")
    print(f"subsampled_target_folder: {params.subsampled_target_folder}")
    print(f"produce_merged_file: {params.produce_merged_file}")
    print(f"transfer_original_dims_to_merged: {params.transfer_original_dims_to_merged}")
    print(f"output_merged_with_originals: {params.output_merged_with_originals}")
    print(f"standardization_json: {params.standardization_json}")
    print(f"remap_dims: {params.remap_dims}")
    print("=" * 60)


# Internal compatibility constants retained for helper modules that still import them.
TILE_PARAMS = {
    "tile_length": 100,
    "tile_buffer": 20,
    "threads": 10,
    "workers": 4,
    "resolution_1": 0.02,
    "resolution_2": 0.1,
    "skip_dimension_reduction": False,
    "chunk_size": 20_000_000,
    "chunkwise_copc_source_creation": False,
}


if __name__ == "__main__":
    print_params(Parameters())
