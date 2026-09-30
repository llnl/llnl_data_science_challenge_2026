# Threshold comparison

- Input: `9x9x9_octet_lattice.tif`
- Shape: `(761, 815, 837)`
- Input dtype: `uint16`
- Thresholds are normalized intensity values, matching the Task 5 examples.

| Threshold | Equivalent uint16 cutoff | Foreground voxels | Foreground percent | Output |
|---:|---:|---:|---:|---|
| 0.3 | 19661 | 519,051,227 | 99.9868% | `segmented_mask_threshold_0.3.tif` |
| 0.5 | 32768 | 192,626,914 | 37.1064% | `segmented_mask_threshold_0.5.tif` |
| 0.7 | 45875 | 35,093,774 | 6.7602% | `segmented_mask_threshold_0.7.tif` |
