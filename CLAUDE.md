# CLAUDE.md

Guidance for Claude Code when working in this repository.

Repository conventions (structure, setup, coding style, testing, commits/PRs)
live in `AGENTS.md` — read that file too; treat it as authoritative for those
topics rather than duplicating them here.

## Project

LLNL Data Science Challenge 2026 — building an agentic-AI workflow for analyzing
X-ray CT scans of additively manufactured (LPBF) octet-truss lattice structures:
segmentation, slicing, skeletonization, and defect detection (missing, thin,
broken, bent struts; dross). See `README.md` and `DATA_SCIENCE_CHALLENGE_2026.pdf`
for the full challenge description.

## Current Status

- [x] Phase 1 — MCP tools (`segment_ct_dataset`, `visualize_slice`, `skeletonize`),
      skills, and a segmentation subagent built against the `9x9x9_octet_lattice`
      and `unitcell` sample data.
- [ ] Phase 2 (in progress) — working with the `data/missing_struts` dataset
      (real CT `.tif` stacks of 9x9x9 octet lattices with 0%, 0.5%, and 1%
      nominal missing struts, 3 replicates each). Goal: detect missing
      junctions/struts directly from the `.tif` files.

  Current approach (see `src/strut_cylinder_segmentation.py`): segment the
  volume once at the full-volume Otsu threshold, then sweep a cylinder along
  each nominal strut and count segmented voxels inside it. The per-strut
  statistic is `area = n_segmented / sampled_length`, a cross-section in
  voxels², compared against the nominal design cross-section (12.3 vox²).

  Note the raw/nominal design JSONs and STLs are NOT aligned with the TIFF
  coordinate system; only the `registered_jsons/` variants line up with their
  corresponding scan (see `data/missing_struts/file_names.txt` and
  `data/9x9x9_octet_lattice/note.txt` for which TIFF/JSON pairs correspond).

### Three findings that dominate any coordinate-sampling approach

1. **The specimen's bottom face was machined off after printing.** Any mid-stack
   slice shows the lattice ending in a sawtooth around y≈720, with no material
   where the design's last junction row sits (y≈760). Those junctions exist in
   the JSON but not in the part. Before excluding them they were 324 of 325
   flagged struts and *all* empty cylinders, while every other layer flagged
   0.00% — the entire apparent signal, none of it a defect. The face is the
   max-Y layer of the **nominal** JSON (`data/missing_struts/octet_truss_9x9x9.json`),
   whose Y maps to registered y with r=1.0000; registered coordinates are
   rotated, so the layer cannot be found by thresholding registered y.

2. **The `registered_jsons/` alignment is not exact — it carries a ~0.8% x-scale
   error**, ~5.6 voxels of drift across the specimen, against struts only ~4
   voxels thick. Symptom: flagged fraction climbing with x (5% near x≈200 to 80%
   near x≈720). Two fixes, both verified: sample with a cylinder radius ≥7 so a
   drifted strut still falls inside, or refit an affine (mean node offset
   2.13 → 0.87 vox).

3. **Cylinders must exclude the strut ends.** Twelve struts meet at a junction
   and the blob stays bright when any one is absent, so a full-span cylinder can
   never register an interior strut as empty — after excluding the bottom face
   it found 1 strut in 17460. Trimming 25% off each end restores sensitivity.

  With all three handled (`--radii 8 --end-trim 0.25`, bottom face excluded):
  **95 struts flagged (0.54%), 87 fully empty (0.50%)** against a 0.5% nominal
  (~92 struts), flat across x, y and z at 0.29–0.99% — no positional gradient.

  Statistics that do *not* work, both verified on real data: the cylinder **mean
  intensity** (a solid/background mixture, so a healthy strut averages near the
  volume Otsu itself — flags 35% at r=3, 50% at r=5), and **Otsu on the derived
  area distribution** (modes are ~3%/97%, so it lands mid-population and flags
  48%). Segment first at the volume Otsu, then count; cut on design geometry.
