# Overly Thick and Overly Thin Strut Screening

## Result

This analysis screens all 18,468 registered struts using the existing
occupancy-derived effective diameter proxy. The nominal inventory class is
used as the reference population. Struts below its 5th percentile are labeled
**overly thin** and struts above its 95th percentile are labeled **overly
thick**.

It finds **1,601 overly thin struts (8.67%)** and **888 overly thick struts
(4.81%)**. The remaining 15,979 struts fall within the nominal reference
interval.

The exact counts and thresholds are in [summary.json](summary.json). The
candidate-level measurements are in [candidate_struts.csv](candidate_struts.csv),
and the full table is in [per_strut_thickness_thinness.csv](per_strut_thickness_thinness.csv).

## Interpretation

The thickness proxy is:

```text
effective diameter = 2 × 348.54 µm × sqrt(CT material occupancy)
```

The nominal-class reference cutoffs are approximately **210 µm** for overly
thin and **545 µm** for overly thick. These are distribution-based screening
limits, not physical design tolerances. The CT inventory reports sampled tube
occupancy, which can be affected by missing material, partial-volume effects,
threshold selection, junction overlap, and registration.

The overly thin group should be interpreted together with the existing
`Missing_Intentional` and `Missing_Unintentional` classifications. The overly
thick group is a high-occupancy candidate set; high occupancy alone does not
prove inflation. Where available, `excess_voxels` from the excess-material
analysis is included in the candidate table as supporting spatial evidence.

## Visualizations

- [thickness_thinness_distribution.png](thickness_thinness_distribution.png): distribution and cutoffs.
- [candidates_by_classification.png](candidates_by_classification.png): candidate counts by existing classification.
- [candidate_locations_xy.png](candidate_locations_xy.png): XY spatial distribution of candidates.

For direct geometric thickness, cross-sections perpendicular to each registered
strut centerline would be required; this screening analysis does not claim that
level of metrology.
