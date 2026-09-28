# 2x Slice-wise Registration Diagnostic

## Result

**A piecewise Z-dependent XY displacement is suggested: the upper volume prefers a different shift than the lower volume.**

The source TIFF was processed by read-only memmap and downsampled 2x in Z, Y,
and X. The registered graph was rasterized on the same grid. For each 16-source-voxel Z slab, the diagnostic searched XY translations from
-12 to 12 source voxels and selected the shift that
maximized registered-CAD/CT-material overlap.

## Measured trends

| Quantity | Value |
|---|---:|
| X shift trend | -0.0029 source voxels per Z voxel (R²=0.052) |
| Y shift trend | -0.0025 source voxels per Z voxel (R²=0.039) |
| Trend magnitude | 0.0039 source voxels per Z voxel |
| Core lower-Z median shift | (0.0, 0.0) source voxels |
| Core upper-Z median shift | (-2.0, -2.0) source voxels |
| Lower-to-upper shift change | 2.83 source voxels |
| Mean overlap gain after local shift | 0.0192 |
| Maximum overlap gain | 0.0561 |

The core-slab analysis excludes the outer 40 source-voxel margins and slabs with
less than 0.15 zero-shift overlap. In the retained region, the preferred shift
changes from approximately `(0, 0)` in the lower-Z
part to `(-2, -2)` in the upper-Z part. This is
consistent with a possible layer-wise or piecewise registration displacement,
but it is not proof by itself because the search is quantized at 2 source voxels.
Missing struts, thresholding,
partial-volume effects, and nearby lattice material can produce local overlap
improvements even without registration error. A smooth linear shift with Z is
the signature expected from tilt/shear; abrupt step changes are more consistent
with layer-wise displacement.

## Files

- [Slice-wise shift results](slab_shift_results.csv)
- [Shift and overlap plots](slice_wise_shift_diagnostic.png)
- [2x registration overlay](registration_overlay_2x.png)
- [Machine-readable summary](summary.json)
