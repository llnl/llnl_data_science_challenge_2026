# Per-Strut Average Thickness Analysis

## Result

Across all **18,468 struts**, the mean thickness proxy is **342.3 µm** and the
median is **337.4 µm**. The 5th–95th percentile range is **160.5–535.5 µm**.

See [strut_thickness_histogram.png](strut_thickness_histogram.png) for the
histogram and [per_strut_thickness.csv](per_strut_thickness.csv) for the
individual strut values.

## Measurement and CAD reference

The inventory provides CT material occupancy rather than a direct segmented
cross-sectional diameter. The reported thickness is therefore an
occupancy-derived effective diameter:

```text
thickness_proxy = 2 × nominal_radius × sqrt(CT_material_occupancy)
```

The CAD registration JSON lists the nominal strut `thickness` parameter as
`0.1` for every strut. That parameter is retained in the per-strut CSV, but it
does not itself specify microns. For a physical reference, the Stage 2
calibration used a nominal radius of 6 source voxels. With the scan scale of
58.09 µm/source voxel:

- Nominal CAD/reference radius: **348.54 µm**
- Nominal CAD/reference diameter: **697.08 µm**

The histogram marks 697.08 µm with a dashed red line. It also marks the
all-strut mean and median.

## Classification summary

| Classification | Count | Mean proxy (µm) | Median proxy (µm) |
|---|---:|---:|---:|
| Nominal | 17,746 | 354.4 | 341.1 |
| Missing_Intentional | 87 | 46.8 | 38.8 |
| Missing_Unintentional | 630 | 43.1 | 0.0 |
| Expected_Missing_But_Material_Present | 5 | 273.0 | 183.5 |

This is a scale-aware screening proxy, not direct metrology: occupancy can be
reduced by missing material, partial-volume effects, threshold choice, and
junction overlap. A true average geometric thickness would require extracting
cross-sections perpendicular to every registered strut centerline from the
segmented CT volume.
