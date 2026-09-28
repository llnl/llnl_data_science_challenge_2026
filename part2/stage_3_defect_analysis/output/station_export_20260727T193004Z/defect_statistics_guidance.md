# Statistical Analysis Guidance for Stage 3 Defect Data

## Purpose and scope

This document describes how to use the station-level export
`defect_analysis_by_station.csv` together with
`defect_analysis_by_strut.csv` to quantify and statistically analyze the
Stage 3 defect signals. The data are screening measurements, not validated
ground-truth defect annotations. The most important next step before making
quality claims is manual review of representative struts and, if possible,
creation of an independently labeled validation set.

The export contains 18,468 struts and 21 longitudinal cross-sectional stations
per strut, for 387,828 station records. The 21 records belonging to one strut
are repeated measurements and must not be treated as 21 independent samples.
For most population-level comparisons, the strut is the primary experimental
unit.

## Files and identifiers

Use the files as follows:

| File | Unit of observation | Main use |
|---|---|---|
| `defect_analysis_by_station.csv` | One strut × station | Local profiles along a strut; station-level plots and repeated-measures models |
| `defect_analysis_by_strut.csv` | One strut | Primary population statistics, defect rates, ranking, and classification analysis |
| `defect_analysis_summary.json` | One analysis run | Parameters, nominal reference baselines, counts, and limitations |
| `defect_class_summary.csv` | One primary class | Quick class counts and fractions |

Join the two CSV files on `strut_id`. The station table deliberately does not
repeat the per-strut labels and scores, so that the long-format file stays
compact.

## Meaning of the station variables

| Variable | Interpretation | Recommended use |
|---|---|---|
| `position_fraction` | Longitudinal position after endpoint trimming; 0.20 through 0.80 at 21 stations | Align profiles across struts |
| `position_x_um`, `position_y_um`, `position_z_um` | Registered station center position in micrometres | Spatial plots and spatial dependence checks |
| `material_occupancy` | Fraction of valid samples above the material threshold in the nominal-radius disk | Primary material-loss/thickness signal |
| `mean_intensity` | Mean sampled intensity in the nominal-radius disk | Secondary density/contrast signal; interpret with occupancy |
| `equivalent_radius_um` | Radius proxy derived from material occupancy | Thin/inflated comparisons; not a direct surface reconstruction |
| `centroid_offset_um` | Distance between sampled material centroid and nominal centerline | Local displacement and bending evidence |
| `outside_nominal_fraction` | Material fraction in the sampled outer annulus | Inflation/excess-material screening signal |
| `valid_sample_fraction` | Fraction of expected cross-section samples inside the image bounds | Quality control; low values should not be interpreted as defects |

The current run used a material threshold of 40,081 intensity units, a voxel
size of 58.09 micrometres, a nominal radius of 6 voxels, and 21 stations. These
parameters are recorded in `defect_analysis_summary.json` and should be
reported with every derived result.

## Recommended analysis workflow

### 1. Perform quality control first

Check that each strut has exactly 21 station records, station IDs run from 0
through 20, and `valid_sample_fraction` is acceptable. Flag or exclude a
record only according to a predeclared rule; do not silently drop low-quality
stations. A useful initial report includes:

- counts of missing or non-finite values by variable;
- the distribution of `valid_sample_fraction`;
- the number of struts with fewer than 21 usable stations;
- the number of struts marked `needs_review` in the per-strut table;
- distributions stratified by acquisition region or unit-cell block, if those
  identifiers are available.

The current output marks 1,952 struts as needing review. That flag should be
used for prioritization and sensitivity analysis, not automatically treated as
a defect label.

### 2. Make strut-level summaries for primary inference

Reduce the station table to one row per strut before comparing defect groups.
Useful summaries include:

- median and 10th percentile `equivalent_radius_um`;
- median and minimum `material_occupancy`;
- median and maximum `outside_nominal_fraction`;
- median, maximum, and RMS `centroid_offset_um`;
- fraction of stations below a declared occupancy threshold;
- longest consecutive run of low-occupancy stations;
- longitudinal slope or curvature of the centroid-offset profile;
- the number of valid stations.

The existing per-strut table already contains many of these summaries and the
Stage 3 scores. Prefer those fields for reproducing the current classifications;
derive new summaries from the station table when testing an alternative
definition.

### 3. Define a control population before estimating effect sizes

For an initial reference population, use struts that are both Stage 2 nominal
and Stage 3 nominal, or use a manually reviewed nominal subset. The current
run's robust reference population and median/MAD baselines are in the JSON
summary. Do not estimate a nominal threshold from the same observations after
labeling them with that threshold and then describe the result as independent
validation.

For a variable `x`, a robust standardized deviation can be reported as:

```text
robust_z = (x - nominal_median) / (1.4826 × nominal_MAD)
```

Use a reversed sign for thinness when larger radius means healthier material.
Report the raw physical units as well as any robust score. A score is useful
for ranking, but it is not automatically a probability of defect.

### 4. Analyze each defect with measurements suited to its mechanism

| Defect | Primary measurements | Useful summaries/tests | Important caution |
|---|---|---|---|
| Thin | `equivalent_radius_um`, `material_occupancy`, radius percentile | Compare median/p10 radius to nominal; estimate the fraction below a predeclared physical limit; use robust effect sizes | A low radius can reflect sampling, registration, or partial image coverage |
| Inflated | `outside_nominal_fraction`, equivalent radius | Compare outer-annulus fraction and maximum profile values; model the probability of exceeding an engineering limit | The annulus is an excess-material proxy and does not establish attached dross |
| Bent | `centroid_offset_um` plus station positions | Plot aligned profiles; compare maximum/RMS offset and derived curvature; use a profile model if shape matters | Global registration error can resemble bending |
| Broken | low-occupancy station runs, occupancy minimum and gap fraction | Estimate gap prevalence and gap length; use a run-length or binary profile analysis | Do not infer a broken strut from one invalid or low-quality station |
| Missing intentional/unintentional | Per-strut `stage2_classification` | Report as separate Stage 2 classes and compare against CT-derived signals descriptively | These labels are preserved from Stage 2a and are not reclassified by Stage 3 |

For defect prevalence, report a numerator, denominator, percentage, and a
confidence interval. Wilson intervals are suitable for a simple proportion;
cluster bootstrap intervals are preferable when the denominator is assembled
from grouped unit cells or other dependent regions.

## Statistical methods

### Descriptive statistics

For each primary class, report sample count, median, interquartile range, and
10th/90th percentiles for the main strut-level measurements. Plot raw points or
empirical distributions in addition to boxplots; the classes are imbalanced,
with Nominal and Inflated much larger than Broken, Bent, or Thin.

Use medians and quantiles as the default because the scores and spatial
measurements are skewed and may contain extreme values. Means and standard
deviations can be added when they have a physical interpretation.

### Two-group comparisons

For an exploratory comparison between a defect group and a nominal control:

- report the median difference and Hodges–Lehmann estimate or a bootstrap
  confidence interval;
- report a rank-biserial or Cliff's delta effect size;
- use Mann–Whitney only as a secondary test, not as the main scientific result;
- use a permutation test stratified by unit-cell block if spatial grouping is
  material.

Do not report only a p-value. With thousands of struts, very small and
practically irrelevant differences can be statistically significant.

### Multiple classes and multiple signals

For simultaneous comparisons across Thin, Inflated, Bent, Broken, and several
measurements, control the false discovery rate using Benjamini–Hochberg, or
preselect one primary endpoint per defect. State whether the analysis is
exploratory or confirmatory before looking at the results.

### Station-level repeated-measures analysis

Use a station-level model only when the longitudinal profile itself is the
question. Include `strut_id` as a random intercept or use cluster-robust
standard errors. A typical continuous model is:

```text
measurement ~ defect_class + position_fraction + defect_class:position_fraction
              + (1 | strut_id)
```

The interaction tests whether the profile changes differently along a strut
for different classes. For binary low-material status, use a mixed logistic
model or GEE with `strut_id` as the subject. If mixed-model convergence is
poor, aggregate to the strut level or use cluster bootstrap rather than
pretending the 21 stations are independent.

For bounded outcomes such as occupancy or fractions, consider a beta model
after handling exact zeros and ones explicitly, or use a robust transformed
model as a sensitivity analysis. Clearly state the handling of boundary values.

### Predictive validation

If the goal is to improve defect classification, create a manually reviewed
label set and split it by strut—not by station. Prefer splits by unit-cell
block or spatial region when nearby struts may be correlated. Report:

- sensitivity/recall, specificity, precision, and negative predictive value;
- precision-recall curves for rare defects such as Broken;
- ROC-AUC only as a secondary metric;
- calibration or observed defect prevalence by score bin;
- bootstrap confidence intervals with resampling at the strut or block level.

Do not tune a threshold and evaluate it on the same reviewed examples.

## Minimal Python starting point

The following pattern creates strut-level summaries and compares a chosen
measurement with a nominal control. It uses the CSV files in this folder.

```python
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

station = pd.read_csv("defect_analysis_by_station.csv")
strut = pd.read_csv("defect_analysis_by_strut.csv")

station_summary = (
    station.groupby("strut_id")
    .agg(
        station_count=("station_id", "size"),
        valid_station_count=("valid_sample_fraction", lambda x: np.isfinite(x).sum()),
        radius_median=("equivalent_radius_um", "median"),
        radius_p10=("equivalent_radius_um", lambda x: x.quantile(0.10)),
        occupancy_median=("material_occupancy", "median"),
        occupancy_min=("material_occupancy", "min"),
        excess_median=("outside_nominal_fraction", "median"),
        excess_max=("outside_nominal_fraction", "max"),
        offset_median=("centroid_offset_um", "median"),
        offset_max=("centroid_offset_um", "max"),
    )
    .reset_index()
    .merge(strut[["strut_id", "stage2_classification", "primary_defect"]], on="strut_id")
)

control = station_summary.query(
    "stage2_classification == 'Nominal' and primary_defect == 'Nominal'"
)
thin = station_summary.query("primary_defect == 'Thin'")

print(station_summary.groupby("primary_defect").size())
print(station_summary.groupby("primary_defect")["radius_median"].describe())
print("Nominal-vs-Thin radius test:", mannwhitneyu(
    control["radius_median"].dropna(),
    thin["radius_median"].dropna(),
    alternative="two-sided",
))
```

For a first profile visualization:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plot_data = station.merge(
    strut[["strut_id", "primary_defect"]], on="strut_id", how="left"
)
sns.lineplot(
    data=plot_data,
    x="position_fraction",
    y="material_occupancy",
    hue="primary_defect",
    estimator="median",
    errorbar=("pi", 50),
)
plt.ylabel("Median material occupancy")
plt.xlabel("Trimmed position along strut")
plt.tight_layout()
```

The profile plot is descriptive. For inferential error bars or p-values, use a
strut-clustered bootstrap or a repeated-measures model.

## Recommended deliverables for a formal analysis

1. A data-quality table showing record counts, invalid samples, and exclusions.
2. A class-prevalence table with confidence intervals.
3. A strut-level distribution table with effect sizes against nominal controls.
4. Profile plots for occupancy, radius, excess fraction, and centroid offset.
5. A threshold-sensitivity table showing how prevalence changes across
   reasonable material thresholds and score cutoffs.
6. A validation table from independently reviewed struts, split by strut or
   spatial block.
7. A provenance section recording the input files, analysis parameters, code
   version, exclusions, and random seeds for resampling.

## Limitations and interpretation rules

- Thin, Inflated, Bent, and Broken are robust outlier screening calls, not
  confirmed physical defect labels.
- The existing Inflated class is large (4,187 struts, 22.7%); inspect its
  spatial and visual distribution for threshold or registration artifacts.
- The Stage 2 missing labels should remain separate from material-bearing
  defect classes.
- Coordinates and station measurements may be spatially correlated through
  shared junctions, unit cells, or registration errors.
- A statistically significant association is not evidence of causation or
  manufacturing impact without process variables and an appropriate design.
- Any engineering acceptance limit should be defined from metrology,
  performance requirements, or reviewed examples—not selected only because it
  maximizes separation in this dataset.

