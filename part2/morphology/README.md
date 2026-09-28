# Strut morphology screening

`strut_morphology.py` analyzes registered CAD edges one at a time. It opens the
CT TIFF as a read-only `tifffile.memmap`, extracts a bounded centerline-aligned
ROI, excludes `node_exclusion_distance_um` at both ends, and thresholds only
that ROI. A 3-D Euclidean distance transform is then computed with the
configured anisotropic voxel spacing. Diameter is twice the distance at the
scan-derived transverse material centroid. Occupancy and gap length are
sampled along the expected CAD line; bending uses the deviation and path length
of those scan-derived centroids relative to the straight registered CAD line.

The Stage 2a `missingness_classification` is copied unchanged. Rows classified
as `Missing_Intentional` or `Missing_Unintentional` are excluded from morphology
analysis rather than relabeled. Morphology calls are deliberately provisional:
`suspected_thin`, `suspected_thick`, `suspected_broken`, and `suspected_bent`.

## Reproducible commands

From the repository root, the smoke test processes the first 16 non-missing
struts in ascending registered strut ID order:

```powershell
python part2/morphology/strut_morphology.py --config part2/morphology/morphology_config.json
```

It writes `strut_morphology_metrics.csv` and `.json` in this directory. The
script refuses to replace either file unless `--overwrite` is explicit. After
reviewing the smoke-test rows, the full-lattice command is:

```powershell
python part2/morphology/strut_morphology.py --config part2/morphology/morphology_config.json --full --overwrite
```

The full run analyzes every inventory row except the two existing missingness
classes. The labels and thresholds must be validated against reviewed CT/CAD
examples before being treated as engineering decisions.
