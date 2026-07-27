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
- [x] Phase 2a — junction detection packaged for agents: `src/junction_scan.py`
      (science), the `scan_lattice_junctions` and `visualize_junction_overlay`
      MCP tools, and the `scan-junctions` skill. The skill runs alignment and
      systematic checks *before* reporting stochastic candidates, and treats the
      sampling radius as its one free parameter.
- [x] Phase 2b — the tools carry **no specimen-specific defect knowledge** and
      take only a volume and a registered JSON. An `exclude_bottom_face` flag
      hardcoded this one specimen's machining artifact; the other eight scans
      will not share it. The agent now finds a systematic region from the
      component sizes, the band fractions and the overlays, then drops it with
      `exclude_junction_ids` read off the per-junction CSV.

**Current focus is missing junctions, not missing struts.** The strut detector
works and its findings are recorded below, but validating it is parked.

Note the raw/nominal design JSONs and STLs are NOT aligned with the TIFF
coordinate system; only the `registered_jsons/` variants line up with their
corresponding scan (see `data/missing_struts/file_names.txt` and
`data/9x9x9_octet_lattice/note.txt` for which TIFF/JSON pairs correspond).
Only 1 of the 9 scans named in `file_names.txt` is present locally
(`0point5dash1`, byte-identical to `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`),
and `registered_jsons/` holds only its companion — so **no 0% control specimen
is available** for validating a detector against a known-clean part.

## Missing junctions — solved and validated

The detector now lives in `src/junction_scan.py`; `mark_junction_candidates_tiff.py`
is a thin script over it, and the `scan-junctions` skill is how an agent drives
it. All three produce the same numbers below.

`src/mark_junction_candidates_tiff.py` finds **2 missing junctions** in the
`0point5dash1` scan: merged ids 513 at (141, 685, 499) and 2682 at
(615, 682, 420), both degree 12. The user visually inspected the stack and
confirmed there are exactly 2 — so this is 2/2 with no false positives or
negatives. Three independent statistics agree on the same pair (node-intensity
sphere, strut-center probe, cylinder segmented-voxel counts), and all 24 struts
incident to them are independently flagged *and* fully empty by the strut
detector.

Consequence worth carrying: **strut removal in this dataset is clustered, not
i.i.d.** Two junctions with all 12 incident struts absent is impossible under
independent removal at 0.5% (0.005¹²). An earlier analysis dismissed both as
machining artifacts on exactly that reasoning and was wrong. Do not estimate
expected defect counts from independent-removal probabilities.

### Three findings that make junction detection work

1. **The lattice JSONs list a physical junction many times.** 10206 entries at
   3430 distinct positions, in groups of 1, 2, 4 or 8, with the junction's
   incident struts split across the duplicates. Unmerged, degrees run 1–8
   instead of the lattice's 3–12, every junction-level count is inflated
   threefold, and any test of the form "every incident strut is missing" is
   answered for a fragment. `merge_colocated_junctions()` in
   `overlay_registered_nodes_histogram.py` merges by position; `load_lattice()`
   routes through it and returns the entry→junction map, which is what carries a
   per-entry property (the nominal design's y) onto the merged lattice. Struts
   are not duplicated, so merging leaves strut-level results untouched.

2. **Cut node intensities against the volume's Otsu, never their own.** Otsu
   maximizes `w0·w1·(m0−m1)²`, which assumes comparably sized classes. A few
   hundred dark nodes against ~9400 bright ones flattens the objective — 5%
   variation from 34000 to 48000 — and puts its argmax at 43583, inside the
   material mode, past the one real gap (empty from 35500 to 39500). It also
   makes the answer depend on which nodes are scored, so it *degrades* as
   exclusions improve: on a cleaned set it returned 52952 and flagged 21.6%.
   The volume's own Otsu (40049) is measured where the balanced-bimodal
   assumption holds (11.3% material), lands mid-trough, and does not move when
   junctions are excluded.

3. **The node-sampling radius must absorb the registration drift** (see strut
   finding 2 below for the drift itself). At `MAX_SAMPLE_RADIUS_VOXELS = 2` the
   probe flagged 377 junctions beyond x≈700 — 53% of that band against 0.2%
   elsewhere — purely because the JSON sat off the struts. At radius 8 that band
   flags nothing. Unlike struts, a junction can tolerate a wide probe: the
   nearest distinct junction is 55.8 voxels away.

4. **`dark_components.largest / n_candidates` separates a systematic absence
   from misalignment.** Absent material is contiguous; junctions that merely
   missed their struts are scattered. Measured on the reference volume against
   three lattices — dark count, largest component, ratio:

   | Lattice | r=4 | r=8 | r=12 |
   |---|---|---|---|
   | Clean (machined face) | 304, 296 → 0.97 | 173, 171 → 0.99 | 6, 1 → 0.17 |
   | Rotated 2° about z | 716, 392 → 0.55 | 288, 102 → 0.35 | 152, 73 → 0.48 |
   | Phantom cell layer | 665, 662 → 1.00 | 534, 532 → 1.00 | 367, 363 → 0.99 |

   The rotated lattice never gets above 0.55 at any radius. Caveat in the top
   row: a one-layer-thick real absence still fades at r=12, because the probe
   reaches the struts behind it — so judge at r=4 and r=8, and do not use r=12
   alone to dismiss a region. The collapse measures how *thin* the absence is,
   not whether it is real; the phantom slab is two cell layers deep and never
   recovers.

Plus the machined-off bottom face, which is shared with the strut analysis and
described as strut finding 1. It is no longer special-cased in the junction
tools: they rediscover it every run as one component of 171, and a caller drops
it with `exclude_junction_ids`. `mark_junction_candidates_tiff.py` still uses
`bottom_layer_junctions()` directly, which is the right place for it — that
script is hardcoded to this specimen, and the detector is not.

## Missing struts — working, unvalidated, parked

`src/strut_cylinder_segmentation.py`: segment the volume once at the full-volume
Otsu threshold, then sweep a cylinder along each nominal strut and count
segmented voxels inside it. The per-strut statistic is
`area = n_segmented / sampled_length`, a cross-section in voxels², compared
against the nominal design cross-section (12.3 vox²).

With `--radii 8 --end-trim 0.25` and the bottom face excluded: **95 struts
flagged (0.54%), 87 fully empty (0.50%)** against a 0.5% nominal (~92 struts),
flat across x, y and z at 0.29–0.99% — no positional gradient. Those 95
decompose into 24 incident to the two missing junctions (all confirmed) and 71
isolated. The 8 flagged-but-not-empty struts are all isolated and may be *thin*
rather than missing — a separate defect class. The 71 have not been validated
against anything.

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

  Statistics that do *not* work, both verified on real data: the cylinder **mean
  intensity** (a solid/background mixture, so a healthy strut averages near the
  volume Otsu itself — flags 35% at r=3, 50% at r=5), and **Otsu on the derived
  area distribution** (modes are ~3%/97%, so it lands mid-population and flags
  48%). Segment first at the volume Otsu, then count; cut on design geometry.
