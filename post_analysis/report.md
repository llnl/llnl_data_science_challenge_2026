# Full-TIFF CT Material Occupancy Report

## Result

The source scan contains **519,119,955 voxels** with shape `761 × 815 × 837`.

| Definition | Voxels | Percent |
|---|---:|---:|
| Literal nonzero (`value > 0`) | 519,119,954 | **99.9999998%** |
| CT material using Stage 3 cutoff (`value >= 40000`) | 58,892,807 | **11.3447%** |
| CT material using the Stage 2 Otsu cutoff (`value >= 40081`) | 58,534,972 | **11.2758%** |

## Interpretation

The meaningful CT-material estimate is **11.3447%** using the Stage 3 threshold
of 40,000. Literal nonzero occupancy is effectively 100% because the scan's
background is encoded with nonzero intensity values; therefore, `value > 0`
does not distinguish material from background for this TIFF.

## Method

- Source: `part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
- Data type: big-endian unsigned 16-bit (`uint16`)
- The TIFF was read as a read-only memory map and processed in Z chunks.
- Material threshold: `>= 40000`, matching the Stage 3 visualizer.
- Voxel intensity range: `0–65535`
- Mean intensity: `34296.1466`

See [summary.json](summary.json) for machine-readable values and
[intensity_histogram.csv](intensity_histogram.csv) for the full 16-bin
intensity distribution.
