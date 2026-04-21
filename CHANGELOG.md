# Changelog

## v2.0.1 - 2026-04-21

- Added filtered tile manifest output and annotated bounds JSON copies so omitted tiles are recorded explicitly.
- Skips writing buffer-only or empty filtered outputs, and skips remap follow-up work when no filtered tiles remain.
- Added `--output-copc-res2` and documented COPC support for segmented inputs, remap targets, and tiled outputs.
