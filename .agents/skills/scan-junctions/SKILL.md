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
- **Nominal JSON** (optional) — the design lattice on a clean integer grid.
  Only needed to exclude a machined-off bottom face.

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

6. **Stochastic pass.** Only once systematic effects are identified and
   excluded: re-run at `radius=8` with `exclude_bottom_face` (if a machined face
   was found and the nominal JSON is available) and/or `exclude_junction_ids`
   for any other systematic region. What survives is the missing-junction list.

7. **Verify each surviving candidate by eye.** For each, call
   `visualize_junction_overlay` with `mode="slice"` and `slice_index` set to
   that candidate's `z`. Confirm the red circle covers empty space while its
   neighbors show material. Drop any candidate that plainly sits on material and
   say why.

8. **Write `junction_scan_report.md`** in the output directory (see Report).

## Classification

Read these from the summary JSON: `dark_fraction_by_band`,
`dark_fraction_by_octant`, `dark_components`, and each candidate's
`dark_neighbor_count` and `component_size`.

| Signal | Reading |
|---|---|
| High dark fraction nearly everywhere, falling sharply as radius grows | Misalignment or heavy drift. Not defects. |
| Dark fraction climbing monotonically along one axis, radius-sensitive | Residual registration drift on that axis (a scale error). Raise the radius or refit the registration. |
| One band or octant far hotter than the rest, **stable** across radii, with a large `dark_components.largest` | A missing region: a machined face, an unprinted corner, or lattice extending past the scan. Systematic. |
| `component_size` 1 and `dark_neighbor_count` 0, stable across radii | A stochastically missing junction. This is the reportable defect. |
| Nothing hot, no components above size 1 | Clean at junction level. |

The discriminator between the middle two rows is radius sensitivity: drift
shrinks as the probe widens, absent material does not.

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
- Systematic findings, each with the statistic that shows it (which band, which
  component size, how it moved with radius), and what you excluded as a result.
- The radius sweep as a small table: radius against dark count, candidate count,
  and largest component.
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
  `(z, y, x)`. Slice `axis` 0/1/2 means z/y/x.
- **The machined-off bottom face** on the 9x9x9 specimens is the maximum-Y layer
  of the *nominal* design. Registered coordinates are rotated, so it cannot be
  found by thresholding registered y — this is why `exclude_bottom_face` needs
  the nominal JSON.
- Registered JSON filenames in this dataset contain spaces; quote them.

### Reference result

On `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif` with its registered JSON
at radius 8: 3430 junctions merged from 10206 entries, degree 3–12, full-volume
Otsu 40049, 173 dark. Of those, 171 form one component on the machined-off
bottom face. Excluding it leaves exactly **2** candidates — junctions 513 at
(141, 685, 499) and 2682 at (615, 682, 420), both degree 12, both size-1
components with no dark neighbours. Human inspection of that stack confirms
exactly 2 missing junctions. A run on this scan that reports a different number
has a bug or a changed parameter, not a discovery.
