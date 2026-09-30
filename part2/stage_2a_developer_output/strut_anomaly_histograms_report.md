# Stage 2a Strut-Anomaly Histogram Report

This report explains the four panels in
[`strut_anomaly_histograms.png`](strut_anomaly_histograms.png). The figure
summarizes 18,468 registered struts:

- 17,746 classified as `Nominal`
- 87 classified as `Missing_Intentional`
- 630 classified as `Missing_Unintentional`
- 5 classified as `Expected_Missing_But_Material_Present`

The gray histograms show the complete inventory. Red outlines show all
non-nominal rows. Green outlines identify the struts expected to be missing
according to the 0.5% CAD model.

## 1. Material occupancy

**What it measures:**

Material occupancy is the fraction of sampled tube voxels along a strut whose
CT intensity is above the Otsu material threshold. A value near 1 means most
sampled voxels contain material; a value near 0 means the sampled strut path is
mostly empty.

**How to read it:**

The large gray distribution represents the ordinary variation across the
lattice. The red and green distributions are concentrated near zero, showing
that the detected anomaly classes have much less CT material than typical
struts. The dashed line at 0.08 is the occupancy reference used by the
diagnostic plot and is close to the low-occupancy defect group.

**Result in this run:**

The median occupancy was approximately 0.239 for nominal struts, 0.003 for
intentional omissions, and 0 for unintentional missing calls. The five
expected-but-present struts had higher occupancy, with a median of about 0.069,
which explains why they were not classified as clean intentional omissions.

**Interpretation:**

This is the strongest direct signal in the figure for distinguishing a missing
strut from a nominal strut. It is still a sampled occupancy measurement, not a
full 3D reconstruction of the strut.

## 2. Longitudinal gap proxy

**What it measures:**

The pipeline samples each strut at 21 positions along its trimmed length. This
panel reports the longest consecutive run of stations with very low material,
converted to micrometres using the 58.09 µm source-voxel scale.

**How to read it:**

Most struts have a gap value near zero, producing the tall gray peak at the
left. Non-nominal struts form a separate group at large gap values, close to
1,750–2,044 µm. A large gap indicates that the low-material region extends
along much of the sampled strut rather than being a small isolated fluctuation.

**Result in this run:**

Intentional missing struts had a median longest gap of approximately 2,044 µm.
Unintentional missing calls had the same median, while nominal struts had a
median of zero. This separation supports the interpretation that many flagged
rows correspond to near-complete strut-length voids.

**Interpretation:**

This panel helps distinguish a structural interruption from ordinary
partial-volume variation. It does not by itself determine whether the cause is
an intentional CAD omission, a manufacturing omission, or another defect type.

## 3. Occupancy-derived effective radius

**What it measures:**

This panel converts occupancy into an effective-radius proxy using

`effective radius = nominal radius × sqrt(occupancy)`.

The dashed line marks the nominal reference radius of approximately 349 µm.

**How to read it:**

Most nominal struts cluster around roughly 150–220 µm in this proxy, with a
separate high-occupancy group near the nominal reference. Non-nominal rows are
concentrated near zero or at substantially smaller values.

**Result in this run:**

The proxy is consistent with the occupancy result: missing-strut candidates
appear as strongly reduced effective radii, while nominal struts occupy the
larger-radius portion of the distribution.

**Important limitation:**

This is not a direct diameter or wall-thickness measurement. It assumes that
filled cross-sectional area scales with the measured occupancy and that the
nominal sampled strut behaves like a filled circular tube. Use it for relative
comparison and screening, not as a metrology result.

## 4. Missing-volume diagnostic

**What it measures:**

This panel shows the estimated missing volume for rows with positive missing
volume. The values are calculated from the nominal sampled tube volume and the
fraction of the tube classified as empty. The x-axis is logarithmic so that
small and large values can be viewed together. The dashed line is the physical
3×3×3 source-voxel noise floor, approximately 0.005293 mm³.

**How to read it:**

The large gray concentration near approximately 0.5–0.75 mm³ represents the
positive-volume inventory. The red outline is concentrated in the same high
volume range, indicating that the flagged anomalies generally involve a large
fraction of a nominal strut volume rather than only sub-voxel noise.

**Result in this run:**

The median estimated missing volume was approximately 0.741 mm³ for
intentional missing struts and 0.743 mm³ for unintentional missing calls. No
reported defect fell below the recorded physical noise floor.

**Interpretation:**

This panel provides a scale-aware measure of defect magnitude. It supports the
claim that the reported missing-strut candidates are substantial relative to
the sampling tube. It should not be interpreted as a precise volume of a
physical fracture, because the estimate depends on tube radius, trimming,
occupancy thresholding, and the assumption that the sampled tube represents
the actual strut geometry.

## Overall interpretation

All four panels tell a consistent story: the non-nominal population is
characterized by low CT occupancy, long low-material runs, a small
occupancy-derived radius, and large estimated missing volume. The occupancy
and longitudinal-gap panels are the most direct evidence for missing material.
The radius and volume panels are derived diagnostic quantities that make the
signal easier to compare in physical units.

These histograms support anomaly screening, but they do not independently
validate the defect labels. In particular, the plots cannot by themselves
separate intentional missing struts from unintentional manufacturing defects;
that distinction comes from comparison with the baseline and expected CAD
models in the main Stage 2a pipeline.
