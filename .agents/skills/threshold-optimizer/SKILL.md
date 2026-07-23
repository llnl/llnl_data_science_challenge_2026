---
name: threshold-optimizer
description: Compare threshold-based segmentations of a 3D CT `.npy` volume. Use when Codex needs to choose among the standard 0.3, 0.5, and 0.7 density thresholds and save comparable masks and slice visualizations.
---

# Threshold Optimizer

Use the configured `segment_ct_dataset` and `visualize_slice` MCP tools to compare the standard thresholds for a 3D NumPy CT volume.

1. Confirm that the input is an existing three-dimensional `.npy` volume. Do not modify it.
2. Create the requested output directory. If none is given, use a clearly named `threshold_comparison` subdirectory beside the input.
3. For each threshold `0.3`, `0.5`, and `0.7`, call `segment_ct_dataset` and save the mask as `mask_threshold_<threshold>.npy`.
4. For each mask, call `visualize_slice` at the requested slice index and axis. If none are supplied, use the midpoint of axis 0. Save each image as `slice_<index>_threshold_<threshold>.png`.
5. Load each saved mask only to count foreground voxels and calculate the foreground percentage. Do not overwrite tool outputs.
6. Write `threshold_comparison.md` in the output directory. Include the input path, slice/axis, threshold, foreground count, foreground percentage, mask path, image path, and a short qualitative comparison. Do not claim a single threshold is objectively best without user-provided criteria.

Return the report path and all generated artifact paths. Keep every generated file inside the comparison output directory.
