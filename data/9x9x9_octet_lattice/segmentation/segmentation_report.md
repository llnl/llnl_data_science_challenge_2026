# Task 6 Segmentation Report: 9x9x9 Octet Lattice

## Summary

- **Input:** `9x9x9_octet_lattice.tif`
- **Volume shape:** `(761, 815, 837)` (axis-0 slices, rows, columns)
- **Input dtype:** `uint16`
- **Selected method:** Global Otsu thresholding, with the threshold estimated from a reproducible page-wise sample using an 8-voxel stride along each sampled axis.
- **Selected threshold:** `40266`
- **Output mask:** `segmented_mask.tif` (`uint8`, 761 pages of shape `(815, 837)`)
- **Foreground voxels:** `57,733,074` (`11.1213%`)
- **Background voxels:** `461,386,881` (`88.8787%`)
- **Total voxels:** `519,119,955`

## Iteration history

1. Computed a global Otsu threshold of `40266` from the reproducible page-wise 8-voxel-stride sample.
2. The sampled foreground fraction at that threshold was `11.5739%`.
3. Wrote the full volume mask and diagnostic outputs. No additional threshold iteration was justified.

## Slice-380 validation

- Slice index `380` is valid for axis 0 (`0..760`).
- The extracted slice has shape `(815, 837)` and was successfully thresholded and written to `slice_380.png`.
- Slice-380 foreground: `32,837` pixels (`4.8137%`); background: `648,478` pixels (`95.1863%`).
- The slice mask contains the expected binary values (`0` background and `255` foreground in the PNG diagnostic).
- The row-wise foreground profile was generated for additional validation.

## Diagnostic outputs

- [Full segmented mask](segmented_mask.tif)
- [Slice-380 mask](slice_380.png)
- [Intensity histogram with selected threshold](diagnostic_histogram.png)
- [Slice-380 row-wise foreground profile](diagnostic_slice_profile.png)

## Limits and failures

- No processing failures were encountered.
- The threshold is global rather than spatially adaptive; intensity variation across the volume may therefore affect local segmentation quality.
- Otsu estimation used a reproducible strided sample to bound memory use; the final mask was produced by processing TIFF pages sequentially.
- The source TIFF compression tag was `1` (uncompressed). The page-wise implementation also avoids requiring the complete source volume in memory.
