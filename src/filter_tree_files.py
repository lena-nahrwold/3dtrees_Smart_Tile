from __future__ import annotations

from pathlib import Path

from filter_task_support import classify_tree_sidecar_file


def _tree_output_suffix_for_source(tree_file: Path) -> str:
    """Map a source tree sidecar name to the canonical filtered output suffix."""
    return classify_tree_sidecar_file(tree_file) or "_trees.txt"


def update_trees_files_with_global_ids(
    tile_results,
    global_to_merged,
    tree_texts_by_input_file,
    trees_output_dir,
    tile_offset: int,
):
    """
    Write filtered tree sidecars into a dedicated tree-files output directory.

    Each output file receives a new leading ``predinstance`` column containing
    the global cross-tile instance ID, while removed buffer-zone instances are
    dropped from the written tree rows.
    """
    trees_output_dir = Path(trees_output_dir)
    trees_output_dir.mkdir(parents=True, exist_ok=True)

    if not tree_texts_by_input_file:
        print("  Warning: no co-located tree .txt files found for the matched tiles")
        return

    n_written = 0
    for result in tile_results:
        tile_name = result.tile_name
        source_tree_files = tree_texts_by_input_file.get(Path(result.filepath), {})
        if not source_tree_files:
            print(f"  No tree text file for tile {tile_name}, skipping")
            continue

        local_to_new = {}
        for gid, meta in result.instances.items():
            if not meta.is_filtered and gid in global_to_merged:
                local_id = gid - result.tile_idx * tile_offset
                local_to_new[local_id] = global_to_merged[gid]

        for trees_file in source_tree_files.values():
            with open(trees_file, "r") as fh:
                lines = fh.readlines()

            out_lines = []
            data_idx = 0
            for line_no, line in enumerate(lines):
                if line_no == 0:
                    out_lines.append(line)
                elif line_no == 1:
                    out_lines.append("predinstance," + line.lstrip())
                else:
                    local_id = data_idx + 1
                    data_idx += 1
                    new_id = local_to_new.get(local_id)
                    if new_id is not None:
                        out_lines.append(f"{new_id}," + line.lstrip())

            out_name = f"{tile_name}{_tree_output_suffix_for_source(trees_file)}"
            out_path = trees_output_dir / out_name
            with open(out_path, "w") as fh:
                fh.writelines(out_lines)

            n_written += 1
            print(
                f"  Trees {trees_file.name}: "
                f"{len(local_to_new)}/{data_idx} trees kept -> {out_name}"
            )

    print(f"  Trees files written: {n_written}")
