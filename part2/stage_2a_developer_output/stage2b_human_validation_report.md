# Stage 2b Human Validation Report

## Final verdict

**PASSED** at validator iteration 2. Stage 2a passed the independent inventory, physical-noise, morphology, and coarse-alignment gates at the required 58.09 µm/source-voxel scale.

## Prior Stage 2b iteration

Iteration 1 was `NEEDS_REVISION`. Its preserved proof is `stage2b_revision_justification_iter_1.png`. It found 92 known CAD omissions with recall 18.5%, 5,235 reported defects, and CAD missing fraction 67.5%.



## Initial state and checks

- Loaded 18,468 inventory rows and 722 reported defect rows from the declared payload.
- Confirmed every reported `voxel_size_um` is 58.09 and the median reported strut length is 3244.07 µm.
- Opened the 1 GB source TIFF read-only and sampled it at 2x. Masks were read as memmaps. No full-resolution dense volume or 3D mesh was created.
- Used headless Matplotlib for validation evidence; no new failure frame was required for the passing revision.

## Statistical and geometric findings

- 87 of 92 known 0.5% CAD omissions are classified `Missing_Intentional` (94.6% recall); 5 are said to contain material.
- 630 non-design struts are labeled missing or broken (3.4% of inventory), below the 10% rejection gate.
- The missing mask occupies 29.2% of the CAD raster and 34.5% of its voxels are in the outer one-voxel CAD layer; the shell fraction is below the 75% drift gate.
- The expected-mask missing fraction is 79.2%, versus 29.0% for non-expected CAD. Median CT occupancy is 0.004 for expected omissions versus 0.235 elsewhere, providing strong separation.
- A coarse effective-8x translation audit changes CAD occupancy from 0.251 at zero shift to 0.251 at ZYX shift (0, 0, 0). The optimum remains at zero, so this audit finds no residual coarse translational drift.
- 0/722 reported defects fall below the 3x3x3 source-voxel physical floor (0.005293 mm³). 0 sphericity values lie outside (0,1], and 1 defect has aspect ratio below 1; the morphology checks therefore pass their rejection gates.
- Stage 2a emits neither `Bent` nor `Inflated` classifications in this revision. The excess mask contains 301,338 voxels and has 0 voxels overlapping the missing mask, but it is not associated with individual strut classifications. Thus bent/inflated typology remains untested rather than falsely asserted.

## Required revision

None.

## Artifacts

- No failure frame generated for passing iteration 2; iteration 1 proof remains `stage2b_revision_justification_iter_1.png`
- Machine-readable audit metrics: `stage2b_validation_metrics.json`
