---
name: scan-junctions
description: Scan the junctions of a lattice CT volume against its registered design JSON, separate systematic problems (misalignment, a missing region or face) from stochastically absent junctions, and write a findings report. Use when asked to QC, sanity-check, or scan a lattice CT scan for missing junctions, or as the precursor check before any strut-level analysis.
---

# Scan Junctions

Score every junction of a registered lattice against a CT volume, decide whether
the scan and the lattice can be trusted together at all, and only then report
individual missing junctions.

The order matters. A dark junction is not by itself evidence of a defect: a
misregistered lattice, an unprinted or machined-away region, and a genuinely
absent junction all produce the same flag. Claiming stochastic defects on a
misaligned scan is the failure mode this skill exists to prevent, so the
alignment verdict gates everything after it.

## Inputs

- **Volume** — a `.tif`/`.tiff` CT stack or a converted `.npy`.
- **Registered JSON** — the lattice already registered to *this scan's* voxel
  coordinates, normally from a `registered_jsons/` directory. Nominal and raw
  design JSONs are in a different coordinate system and will flag nearly
  everything; check the scan/JSON pairing before starting.
- **Nominal JSON** (required) — the same lattice as designed, on a clean integer
  grid. Registered coordinates are rotated, so the design grid is the only place
  a face or slab of the part can be *named*. Every scan reports a dark fraction
  per design layer, and that is the statistic that finds a missing region on a
  specimen you have never seen before. The two files must list the same entries
  in the same order; a mismatched pair is rejected rather than guessed at.

Never modify an input. Put every artifact in one output directory named for
the scan.

## Workflow

1. **Convert once.** If the volume is a TIFF, call `convert_tiff_volume` to make
   a `.npy` and use that for every later call. The scan and overlay tools accept
   TIFFs directly, but each call would re-decode the stack.

2. **Check coarse alignment before measuring anything.** Call
   `visualize_junction_overlay` with `mode="mip"` and look at it. Then take one
   `mode="slice"` view per axis (`axis` 0, 1, 2) near the middle of the
   specimen. You are asking one question: do the junction markers sit on the
   lattice in the image?

   - Markers on the nodes → alignment **PASS**, continue.
   - Markers on a lattice-shaped cloud but consistently offset, rotated, or
     scaled → alignment **FAIL**. Report it and stop. Do not report missing
     junctions from a misaligned lattice.
   - Markers unrelated to the image content → wrong JSON for this scan. Stop and
     say so.

3. **Baseline scan at radius 8.** Call `scan_lattice_junctions` with
   `radius=8`, no exclusions. Read the returned counts and open the summary JSON.

4. **Sweep the radius.** Repeat at `radius=4` and `radius=12`. Outputs are
   tagged by radius, so nothing is overwritten. How the dark count *moves* is
   the main evidence separating misalignment from real absence.

5. **Classify what you found**, using the table below. Report every systematic
   finding you identify — a machined face is a real, reportable property of the
   specimen, not something to silently drop.

6. **Stochastic pass.** Only once systematic effects are identified: re-run at
   `radius=8` naming each absent region in `exclude_design_layers` (`"y=max"`,
   `"z=min"`, `"x=19"`, comma-separated) and any leftover region in
   `exclude_junction_ids`. What survives is the missing-junction list.

   Exclude only what you have shown is missing from the *part*. Excluding a
   layer because it is inconvenient turns a systematic finding into a silent
   one, and the excluded junctions are still scored and counted either way.

7. **Verify each surviving candidate by eye.** For each, call
   `visualize_junction_overlay` with `mode="slice"` and `slice_index` set to
   that candidate's `z`. Confirm the red circle covers empty space while its
   neighbors show material. Drop any candidate that plainly sits on material and
   say why.

8. **Write `junction_scan_report.md`** in the output directory (see Report).

## Classification

Read these from the summary JSON: **`dark_fraction_by_design_layer`** first,
then `dark_fraction_by_band`, `dark_fraction_by_octant`, `dark_components`, and
each candidate's `dark_neighbor_count` and `component_size`.

`dark_fraction_by_design_layer` is the primary systematic test. It groups
junctions by their coordinate on the *design* grid, one record per layer per
axis, so a region absent from the part appears as a layer at or near a dark
fraction of 1 — whichever face of whichever specimen it lands on.

| Signal | Reading |
|---|---|
| One or more design layers at ~100%, the **same layers** at every radius | A region absent from the part: machined face, unprinted end, lattice past the scanned field. Systematic, and reportable as a finding. |
| Layers hot but never reaching 100%, and the **hottest layer moves** between radii — often to a different axis | Misalignment or drift. Not defects. |
| Dark count falls steeply with radius, spread over many components | Supports misalignment; read with the row above. |
| `component_size` 1 and `dark_neighbor_count` 0, stable across radii | A stochastically missing junction. This is the reportable defect. |
| No layer above a few percent, no component above size 1 | Clean at junction level. |

Measured on the reference scan against deliberately broken lattices — the
darkest design layer at each radius:

| Lattice | r=4 | r=8 | r=12 |
|---|---|---|---|
| Clean (machined face) | y=18, 100% | y=18, 94.5% | y=18, 2.2% |
| Rotated 2° | x=18, 95.6% | x=18, 61.9% | **y=18**, 46.4% |
| Phantom cell layer | x=19, 100.0% | x=19, 100.0% | x=19, 100.0% |

Three things to take from that table:

- **The rotated lattice never reaches 100% and its worst layer changes axis**
  between r=8 and r=12. A merely displaced junction has material nearby; a wide
  enough probe finds it, and which layer looks worst is then an accident.
- **The phantom slab does not move at all** — 180 of 180, at every radius. Its
  junctions sit in air, so no radius can find material for them.
- **A one-layer-thick absence still fades at a wide radius.** The real machined
  face collapses from 100% to 2.2% at r=12 because the probe reaches the struts
  behind it. So judge a layer at r=4 and r=8; do not use r=12 alone to dismiss
  one. Depth, not authenticity, is what that collapse measures.

## Choosing the radius

Start at **8**. It is validated on the 9x9x9 octet specimens and is the default.

The radius exists to absorb residual registration error, and it is the only free
parameter here. Too small and correctly-placed material is missed because the
lattice sits slightly off it; too large and the probe reaches neighbouring
struts and hides a real absence.

- **Sweep 4, 8, 12** to see which effects are radius-driven.
- **Below 4**: only with visually confirmed tight alignment. At radius 2 on the
  reference scan, drift alone flagged 53% of the junctions beyond x≈700 against
  0.2% elsewhere — all false.
- **Above 12**: diagnosis only. Junctions on that specimen are 55.8 voxels apart
  so neighbouring *junctions* do not bleed in until ~20, but struts radiate from
  every junction and a wide probe finds them, masking genuine absence. Do not
  report candidates from a radius above 12.
- If the flag count is still falling between 8 and 12, suspect alignment rather
  than raising the radius further.

## Report

Write `junction_scan_report.md` containing:

- Inputs (volume, registered JSON, nominal JSON) and every parameter used.
- **Alignment verdict: PASS / FAIL / UNCERTAIN**, with the overlay images cited
  as evidence.
- Systematic findings, each with the statistic that shows it (which design
  layer, at what dark fraction, how it moved with radius), and what you excluded
  as a result. Name every excluded layer explicitly.
- The radius sweep as a small table: radius against dark count, candidate count,
  largest component, and the darkest design layer.
- Surviving candidates: junction id, x, y, z, intensity, degree, and whether you
  visually confirmed it.
- Anything you could not determine. If a check could not be run, say so plainly
  instead of implying it passed.

State counts as measured. Do not estimate how many defects to *expect* from a
nominal defect rate — removals in this dataset are clustered, not independent,
and that reasoning has already produced a wrong conclusion once.

## Validation

- The scan reports `degree` between 3 and 12. Degrees of 1–8 mean co-located
  JSON entries were not merged and every junction-level count is wrong.
- `n_junctions` is smaller than `n_entries`, typically by about a factor of 3.
- The Otsu threshold is the full-volume one and does not change between runs on
  the same volume, whatever is excluded.
- Every candidate you report has been looked at in a slice overlay.

## Notes

- **A junction goes dark only when every incident strut is absent**, and
  interior junctions have twelve. This scan therefore has little sensitivity to
  individual missing struts — they leave both endpoints bright. A clean junction
  report does not mean the specimen is defect-free; it means strut-level
  analysis can proceed on a trustworthy alignment.
- **Merged junction ids are not JSON entry ids.** Each junction's `entry_ids`
  column in the CSV maps back to the source file.
- **Thresholding is always against the full volume's Otsu**, never against the
  sampled node intensities. A node-fitted threshold lands inside the material
  mode and gets worse as exclusions improve.
- **Coordinates**: JSON positions are `[x, y, z]`; the volume is indexed
  `(z, y, x)`. Slice `axis` 0/1/2 means z/y/x. Design-layer names like `y=18`
  are on the *design* grid and are not voxel coordinates.
- **No defect is assumed.** Nothing in the tools knows that the 9x9x9 specimens
  were machined on one face; that face is found, every run, by its design layer
  reading 100% dark. A different specimen with a different systematic — or none
  at all — is measured the same way. Do not carry a previous scan's exclusions
  into a new one.
- The per-junction CSV carries `design_x,design_y,design_z`, so you can group
  by layer or check a candidate's neighbourhood without re-running the scan.
- Registered JSON filenames in this dataset contain spaces; quote them.

### Reference result

On `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif` with its registered JSON
and `data/missing_struts/octet_truss_9x9x9.json` as nominal, at radius 8: 3430
junctions merged from 10206 entries, degree 3–12, full-volume Otsu 40049, 173
dark. Design layer `y=18` reads 171/181 dark and holds one component of 171 —
the machined-off face, found from the statistic rather than assumed. Excluding
it (`exclude_design_layers="y=max"`, 181 excluded) leaves exactly **2**
candidates — junctions 513 at (141, 685, 499) and 2682 at (615, 682, 420), both
degree 12, both size-1 components with no dark neighbours. Human inspection of
that stack confirms exactly 2 missing junctions. A run on this scan that reports
a different number has a bug or a changed parameter, not a discovery.
