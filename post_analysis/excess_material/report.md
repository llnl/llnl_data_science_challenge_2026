# Excess-material analysis

## Interpretation

The excess mask is CT material found outside the nominal CAD occupancy, within a narrow CAD-local region. It is evidence of geometric disagreement, not a defect-type label by itself.

## Global mask measurements

- Mask shape: `(381, 408, 419)` in `(Z, Y, X)` order.
- Nonzero excess voxels: `301,338` (0.4627% of the full mask grid).
- CAD occupancy voxels: `13,364,749`; excess/CAD voxel ratio: `2.255%`.
- Missing-material voxels: `3,897,608`; excess/missing overlap: `0`.
- Connected excess components: `1,901`; largest component: `170,734` voxels (56.66% of excess).
- Component volume uses the pipeline mask spacing of `116.18 µm` per voxel.
- Z-boundary concentration: `270,075` voxels (89.63%) lie in the first or last 30 mask slices.

## Relationship to reported defect types

The inventory has no explicit `Bent` or `Inflated` labels. Excess voxels were therefore assigned to the nearest registered strut centerline as a spatial association, not as a ground-truth classification.

- A missing-class strut with unusually high external excess can be **displacement/bending-like**: the nominal centerline is empty while material appears nearby.
- A nominal or expected-material-present strut with unusually high external excess can be **thickening/inflation-like**: material remains in the CAD strut and also extends outside it.
- The top-1% strut threshold was `403` assigned excess voxels; these candidates are listed in `top_excess_struts.csv`.

In this dataset, the excess is dominated by a systematic Z-end pattern and nominal struts, so it does **not** provide convincing evidence for a population of bent or inflated defects. The five `Expected_Missing_But_Material_Present` struts collectively account for only 344 excess voxels; two struts account for 233 and 99 of those voxels. The strongest unintentional-missing association is 50 voxels. These are follow-up candidates, not confirmed defect types.

A definitive bent-versus-inflated decision requires a local centerline/surface fit or cross-sectional radius measurement; the current mask alone cannot make that distinction.

## Outputs

- `excess_material_overview.png`: projections, Z profile, and component sizes.
- `excess_by_strut_classification.png`: excess association by inventory class.
- `excess_strut_locations_xy.png`: XY locations of struts with excess material.
- `excess_components.csv`: connected-component measurements.
- `excess_by_strut.csv`: per-strut excess associations.
- `excess_class_summary.csv`: class-level aggregation.
- `top_excess_struts.csv`: highest-excess candidates and heuristic pattern.
- `top_excess_by_class.csv`: top ten excess-associated struts within each inventory class.
