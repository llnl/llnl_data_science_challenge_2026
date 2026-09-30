---
name: threshold-comparison
description: Compare multiple segmentation thresholds on volumetric CT/TIFF data, writing one mask per threshold and a metrics report. Use when Task 5 threshold optimization or threshold comparison is requested.
---

# Threshold Comparison

Run the repository’s threshold comparison workflow for a volumetric dataset.

## Workflow

1. Use `data/9x9x9_octet_lattice/segmentation/threshold_optimizer.py` as the implementation reference.
2. Evaluate normalized thresholds `0.3`, `0.5`, and `0.7` unless the user specifies a different set.
3. Interpret a normalized threshold `t` as `voxel / dtype_max >= t`. For the current `uint16` TIFF, the equivalent cutoffs are `19661`, `32768`, and `45875`.
4. Process compressed TIFFs page-by-page; do not rely on memory mapping.
5. Save each binary mask separately as `segmented_mask_threshold_<threshold>.tif`.
6. Write `threshold_comparison_report.md` containing the input shape, dtype, equivalent cutoffs, foreground voxel counts, and foreground percentages.
7. Report the output paths and comparison metrics.

## Completed lattice run

The workflow was run on `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`, shape `(761, 815, 837)`, with `519,119,955` total voxels:

| Threshold | Foreground voxels | Foreground |
|---:|---:|---:|
| 0.3 | 519,051,227 | 99.9868% |
| 0.5 | 192,626,914 | 37.1064% |
| 0.7 | 35,093,774 | 6.7602% |

Outputs are in `data/9x9x9_octet_lattice/segmentation/threshold_comparison/`.
