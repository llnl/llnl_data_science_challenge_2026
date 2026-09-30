# Stage 3: Cross-Sectional Defect Analysis

## Purpose

This stage extends the authoritative Stage 2a missing-material inventory into
an explainable screening analysis of non-missing registered struts. It preserves
Stage 2a's `Missing_Intentional` and `Missing_Unintentional` labels, then
evaluates material-bearing struts for thin, inflated, bent, and broken evidence.

The implementation is [`analyze_strut_defects.py`](analyze_strut_defects.py).
It consumes the affine-registered graph, the Stage 2a inventory, and the raw
CT TIFF. The inventory alone is insufficient for this analysis because it
contains aggregate tube statistics rather than cross-sectional shape profiles.

## Method

Each graph edge is trimmed by 20% at each endpoint to avoid junction material.
At 21 evenly spaced stations, the analyzer samples a disk perpendicular to the
registered nominal centerline. The disk is sampled directly from the TIFF
memmap; it is not saved as a crop or used to construct a dense CT mask.

For each station the analyzer measures material occupancy, intensity,
equivalent material radius, material-centroid offset from the nominal
centerline, and the material fraction outside the nominal-radius disk. It
reduces those station measurements to per-strut features:

- Thickness: median and 10th-percentile equivalent radius, plus radius
  variation.
- Loss: station occupancy and longest consecutive low-material run.
- Inflation/excess: material fraction in the sampled outer annulus.
- Bending: maximum/RMS centroid displacement and the second difference of
  centroid displacement along the strut.

The equivalent radius is a sampling-derived proxy:

```text
equivalent radius = voxel_size × sqrt(material samples in disk / pi)
```

It is not a direct metrology reconstruction of the strut surface.

## Labels and scores

High-quality `Nominal` Stage 2a struts define robust feature reference values.
For a feature `x`, the positive abnormality score is based on its deviation
from the nominal median in robust MAD units. Thinness uses negative radius
deviation; inflation uses positive radius or outer-annulus deviation; bending
uses centroid displacement or curvature; and breaking uses a long
low-material run.

Scores at or above the default threshold of 3 are screening candidates. The
output preserves multiple signals: a strut can have `primary_defect=Bent` and
`secondary_defects=Thin`, for example. These are not final metrology labels
until validated on reviewed examples or labeled defect templates.

`Missing_Intentional` and `Missing_Unintentional` retain their Stage 2a labels.
`Expected_Missing_But_Material_Present` is analyzed but automatically marked
`needs_review` because CAD intent and CT evidence disagree.

## Memory design

The source TIFF is opened with `tifffile.memmap`. Processing uses bounded
strut batches (96 by default), with arrays of approximately:

```text
batch size × station count × cross-section sample count
```

The sampled CT values are immediately reduced to numerical features and then
released before the next batch. No dense full-volume material mask, complete
cross-section image stack, or all-strut voxel cache is held in memory.

Station measurements are optional and stream directly to CSV. This makes the
peak memory independent of the number of registered struts, apart from small
per-strut feature arrays used for robust reference calibration.

## Outputs

The default output directory is `stage_3_defect_analysis/output/`.

- `defect_analysis_by_strut.csv`: one row for every processed strut. This is
  the compatibility and review table.
- `defect_analysis_by_station.csv`: optional long-format station evidence,
  written only with `--write-stations`.
- `defect_class_summary.csv`: primary-label counts.
- `defect_analysis_summary.json`: inputs, parameters, robust reference values,
  counts, and limitations.
- `defect_analysis_by_strut.parquet`: optional compressed columnar export,
  written with `--write-parquet` when `pyarrow` is installed.

Parquet is the preferred canonical format for repeated downstream queries
because column filters can avoid parsing irrelevant fields. CSV remains useful
for spreadsheet review and tool compatibility. Do not split the canonical
per-strut table by defect type: labels can be revised and one strut can have
multiple defect signals. If a future dataset requires partitioning, partition
Parquet by stable dataset/run or unit-cell block rather than by label.

## Running the analysis

From `part2/`:

```bash
python3 stage_3_defect_analysis/analyze_strut_defects.py \
  --write-stations \
  --write-parquet
```

For a fast smoke test that does not create the full station table:

```bash
python3 stage_3_defect_analysis/analyze_strut_defects.py --max-struts 10
```

The runtime environment must provide `numpy` and `tifffile`; `pyarrow` is only
needed for the optional Parquet export.

## Limitations and required validation

- The present thin, inflated, bent, and broken calls are robust outlier
  screening—not validated ground truth classifications.
- A global registration error can resemble bending. The reference distribution
  and localized per-edge shape profile reduce but do not eliminate this risk.
- The outer-annulus metric measures excess material, but does not yet prove
  that it is attached to the strut. Local connected-component analysis is the
  required next step for a production inflation or dross classifier.
- Threshold, registration, radius, and station-count sensitivity must be
  evaluated against manually reviewed struts and the labeled unit-cell defect
  templates before using the scores for final quality decisions.
