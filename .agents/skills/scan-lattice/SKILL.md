---
name: scan-lattice
description: Scan a lattice CT volume against its registered design JSON for missing junctions and missing struts, separating systematic problems (misalignment, a missing region or face) from stochastic defects, and write a findings report. Use when asked to QC, sanity-check, or scan a lattice CT scan for defects.
---

# Scan Lattice

Score every junction and every strut of a registered lattice against a CT
volume, decide whether the scan and the lattice can be trusted together at all,
and only then report individual defects.

The order matters, and it is junctions first, struts second. A dark junction is
not by itself evidence of a defect: a misregistered lattice, an unprinted or
machined-away region, and a genuinely absent junction all produce the same flag.
Claiming stochastic defects on a misaligned scan is the failure mode this skill
exists to prevent, so the misalignment grade gates everything after it — the
strut phase included. The junction phase is also what fixes the sampling radius
and identifies any systematic region, both of which the strut phase reuses
rather than re-deriving.

## Inputs

- **Volume** — a `.tif`/`.tiff` CT stack or a converted `.npy`.
- **Registered JSON** — the lattice already registered to *this scan's* voxel
  coordinates, normally from a `registered_jsons/` directory. Nominal and raw
  design JSONs are in a different coordinate system and will flag nearly
  everything; check the scan/JSON pairing before starting.

That is the whole input. The tools know nothing about any particular specimen,
and there is no defect that they assume or special-case — every systematic
finding on this page is something you read out of the scan itself.

Never modify an input. Put every artifact in one output directory named for
the scan.

## MCP server

Every tool named on this page — `convert_tiff_volume`, `scan_lattice_junctions`,
`scan_lattice_struts`, `visualize_junction_overlay` — is served by the
**ct-segmentation** MCP server
(`src/mcp_server.py`, registered in `.mcp.json` for Claude Code and
`~/.codex/config.toml` for Codex). The server also carries `visualize_slice`,
which renders a slice with no overlay, for when you want the image without the
markers.

Confirm the tools are in your tool list before starting. Both ways the setup
breaks are silent: a missing dependency stops the *whole* server rather than
one tool, and a config pointing at a different checkout starts a server that
simply lacks the newer tools — in either case there is no error to see. If a
tool this page names is missing, **stop and report that** rather than
re-deriving the scan with ad-hoc scripts: only the versioned detector produces
numbers comparable across runs, and a hand-rolled scan silently measures
something else. A human setting up a session can verify the server with
`python evals/check_mcp_server.py`.

## Workflow

Steps 1–7 are the junction phase, steps 8–10 the strut phase. Do not start the
strut phase on a MAJOR misalignment.

1. **Convert once.** If the volume is a TIFF, call `convert_tiff_volume` to make
   a `.npy` and use that for every later call. The scan and overlay tools accept
   TIFFs directly, but each call would re-decode the stack.

2. **Check coarse alignment before measuring anything.** Call
   `visualize_junction_overlay` with `mode="mip"` and look at it. Then take one
   `mode="slice"` view per axis (`axis` 0, 1, 2) near the middle of the
   specimen. Pass `radius=4` — the baseline radius — so the slice views draw
   the same spheres the baseline scan will use. You are asking one question:
   how far do the junction markers sit from the lattice in the image?

   The registered JSON is never perfectly aligned with the volume — absorbing
   the residual is what the radius sweep exists for — so grade the misalignment
   rather than pass/fail it:

   - **NONE** — markers sit centred on the nodes. Continue.
   - **MINOR** — markers sit on the lattice but visibly off-centre, or drift
     across the specimen by less than roughly a junction's own radius (the
     spheres in the slice view have a known radius, which gives you the scale
     to estimate it). Continue: the sweep must absorb it, and the report must
     say so.
   - **MAJOR** — markers form a lattice-shaped cloud that is offset, rotated, or
     scaled so far that no sampling sphere could reach the true node positions
     without also reaching the neighbouring junctions. Report it and stop. Do
     not report missing junctions from a grossly misaligned lattice.
   - Markers unrelated to the image content → wrong JSON for this scan. Stop and
     say so.

3. **Baseline scan.** Call `scan_lattice_junctions` with `radius=4`. Read the
   returned counts and open the summary JSON.

4. **Sweep the radius.** Repeat at larger sample radii, e.g. `radius=8` and
   `radius=12`. The radius sets the size of the voxel neighbourhood sampled
   around each junction to decide whether it is dark. It should always be much
   smaller than the spacing between junctions — a good scale is the approximate
   radius of a junction itself. Outputs are tagged by radius, so nothing is
   overwritten. A larger radius can absorb a slight misalignment between a
   junction's expected and actual positions, so how the dark count *moves*
   across the sweep is the main evidence separating misalignment from real
   absence.

   Verify the radii visually as you go: in a `mode="slice"` overlay the circles
   are the sampling spheres drawn to scale, so at each radius you are weighing
   you can see directly whether the sphere covers its junction despite the
   residual misalignment and whether it stays clear of the neighbouring
   junctions and their struts. Use the same views to chase any anomaly — a band
   with an elevated dark fraction, a component that exists at only one radius —
   before classifying it.

5. **Classify what you found**, using the table below. Report every systematic
   finding you identify — a machined face is a real, reportable property of the
   specimen, not something to silently drop.

6. **Stochastic pass.** Only once systematic effects are identified: take the
   CSV from your chosen reporting radius (see Choosing the radius) and separate
   its dark set into the systematic region you found and everything
   else. That remainder is the missing-junction list — usually the rows with
   `component_size` 1, but use the ids of the component or band you actually
   identified. Nothing is re-run; a junction's darkness does not depend on what
   any other junction is doing.

   Set aside only what you have shown is missing from the *part*, and say in the
   report what you set aside and why. Dropping junctions because they are
   inconvenient turns a systematic finding into a silent one.

7. **Verify each surviving candidate by eye.** For each, call
   `visualize_junction_overlay` with `mode="slice"` and `slice_index` set to
   that candidate's `z`. Confirm the red circle covers empty space while its
   neighbors show material. Drop any candidate that plainly sits on material and
   say why.

8. **Scan the struts, once.** Call `scan_lattice_struts` at **the same radius
   you settled on for the junction report**. There is no strut radius sweep and
   no second free parameter to tune: the radius is chosen from the junction
   slice overlays, and the strut scan inherits it. Each strut is sampled at its
   midpoint alone, which the tool fixes.

9. **Account for the systematic region at strut level.** Each row of the strut
   CSV carries the merged ids of its two endpoint junctions. A flagged strut
   with an endpoint inside a systematic region you identified in step 5 belongs
   to that region, not to the defect list — a strut into a machined-off face is
   absent for the same reason the face is. Report the count separately; do not
   drop it silently.

   Then split what remains in two: struts incident to a junction you reported as
   missing, and isolated ones. The first group is corroboration — a missing
   junction means all twelve of its struts are absent, so finding them is a
   consistency check on the junction result, not twelve extra defects. The
   isolated ones are the strut-level finding.

10. **Write `lattice_scan_report.md`** in the output directory (see Report).

## Classification

Read these from the summary JSON: `dark_components`, `dark_fraction_by_band`,
`dark_fraction_by_octant`, and each candidate's `dark_neighbor_count` and
`component_size`. Then look at the images.

The one number that does most of the work is **`dark_components.largest` divided
by `n_dark`**: the share of the dark set sitting in a single connected
lump. Absent *material* is contiguous. Junctions that merely missed their struts
are scattered.

| Signal | Reading |
|---|---|
| `largest / n_dark` near 1 at r=4 and r=8, concentrated in one band or octant | A region absent from the part: machined face, unprinted end, lattice past the scanned field. Systematic, and reportable as a finding. |
| `largest / n_dark` well below 1 (roughly 0.6 or less), dark count falling steeply with radius, spread over many components | Misalignment or drift. Not defects. |
| Markers visibly off the lattice in the overlays | Misalignment, whatever the numbers say. The images outrank the statistics here. |
| `component_size` 1 and `dark_neighbor_count` 0, stable across radii | A stochastically missing junction. This is the reportable defect. |
| Nothing hot, no component above size 1 | Clean at junction level. |

Measured on the reference scan against deliberately broken lattices — dark
count, largest component, and their ratio:

| Lattice | r=4 | r=8 | r=12 |
|---|---|---|---|
| Clean (machined face) | 304, 296 → 0.97 | 173, 171 → 0.99 | 6, 1 → 0.17 |
| Rotated 2° | 716, 392 → 0.55 | 288, 102 → 0.35 | 152, 73 → 0.48 |
| Phantom cell layer | 665, 662 → 1.00 | 534, 532 → 1.00 | 367, 363 → 0.99 |

Two things to take from that table:

- **The rotated lattice never gets above 0.55.** A displaced junction has
  material nearby and its neighbours land differently, so the dark set stays
  broken up no matter how many of them there are.
- **A one-layer-thick absence fades at a wide radius.** The real machined face
  drops to 6 dark junctions at r=12 because the probe reaches the struts behind
  it. That is why you judge at r=4 and r=8 and do not use r=12 alone to dismiss
  anything: the collapse measures how *thin* the absence is, not whether it is
  real. The phantom slab is two layers deep and never recovers.

Once you believe a region is absent from the part, get its junction ids from the
CSV — the rows in that component, or the rows in the band you identified — and
account for them separately from the rest of the dark set in the report.

## Struts

`scan_lattice_struts` samples each strut at the **midpoint** of its
junction-to-junction span, with the same sphere probe and the same full-volume
Otsu threshold the junction scan uses. A strut is **missing** when that point is
dark. There is one sample point and one class; the tool fixes this and you
cannot change it.

The midpoint is used because it is the only placement that works. A probe near a
strut's end measures its junction, whose material stays bright when any one of
its twelve struts is absent. On the reference scan junction-local material
reaches about 12 voxels along an absent strut's own axis, while the registration
drift needs a sampling radius of 7 or more. Against a 55.8-voxel strut the
midpoint sits 27.9 voxels from either junction, so a radius-8 sphere clears both
by about 8 voxels. Quarter-span points sit 14 voxels out and would need a radius
below 2 — no radius satisfies both demands there.

| Signal | Reading |
|---|---|
| `missing`, endpoint junction inside a systematic region | Belongs to that region. Report the count; it is not a separate defect. |
| `missing`, endpoint junction reported missing | Corroboration of the junction finding, not an independent defect. |
| `missing`, isolated | A missing strut. This is the reportable strut-level defect. |
| Missing fraction climbing steadily along one axis | The radius is too small for the drift, exactly as at junction level. Go back to the junction overlays. |

Two blind spots, which the report must state rather than let a clean result
imply were checked:

- **A thin strut reads present.** The statistic is a maximum over a sphere, so a
  strut that is printed under-thickness still has a bright voxel at its
  midpoint.
- **A partial strut reads present.** A break that does not cover the midpoint
  leaves the sample point bright. The scan therefore bounds *absence*, not
  integrity. The summary JSON reports `n_partial`, but at one sample point that
  count is structurally zero — check `n_points_per_strut` before reading
  anything into it, and never report it as evidence that no breaks exist.

There is no strut-level image tool. Verify a strut candidate by taking a
`visualize_junction_overlay` slice at its `z` — the strut runs between two of
the circles you can see — and by checking that the junction scan's own numbers
for its two endpoints are consistent with what the strut result claims.

## Choosing the radius

The radius exists to absorb residual registration error, and it is the only free
parameter here — one number, chosen once at junction level and reused for
struts. Too small and correctly-placed material is missed because the lattice
sits slightly off it; too large and the probe reaches neighbouring
struts and hides a real absence.

Choose the reporting radius **from the slice overlays, not from the counts**.
At a good radius the spheres visibly cover their junctions despite the residual
misalignment while staying clear of the neighbouring junctions and their
struts; report from the smallest radius that does both. Do not pick the radius
by waiting for the dark count to stabilise — a thin systematic absence keeps
the count falling long after the drift flags are gone, and on some scans it
never settles. The count's movement across the sweep is evidence for
*classifying* what you found, not a criterion for the radius.

If no radius covers the junctions without also reaching their neighbours, that
is a MAJOR misalignment — report it and stop rather than widening further.

## Visual tooling

`visualize_junction_overlay` is the image tool, and it always re-scores the
junctions internally: pass the `radius` of the scan you are checking, since the
markers are coloured from that result — green for bright, red for dark — and
the picture changes with the radius.

- **`mode="mip"`** — a maximum-intensity projection along `axis` with every
  junction marked. Use it for the misalignment grade and for seeing where the
  dark set concentrates. It cannot show which voxels a flag came from.
- **`mode="slice"`** — a single plane in which each nearby junction's sampling
  sphere is drawn to scale: a junction `d` voxels off the plane appears as a
  circle of radius `sqrt(radius² − d²)`, so the picture shows exactly the
  voxels its flag was computed from, and junctions more than one radius off the
  plane are not drawn. This is the view for verifying a candidate (a red circle
  over empty space, its neighbours green on material), for judging whether a
  sphere reaches the next junction, and for reading residual misalignment
  (bright material sitting consistently off-centre in its circle).
- `slice_index=-1` picks the median junction coordinate along `axis`, a
  mid-specimen default; set it to a candidate's own coordinate to inspect that
  candidate. `axis` 0/1/2 slices along z/y/x.
- To scrub the whole stack instead of single planes, re-run
  `scan_lattice_junctions` with `write_marked_tiff=True` for an RGB TIFF with
  the candidates painted. It costs roughly three bytes per input voxel in
  memory and on disk, so request it once, at the radius you settled on.

## Report

Write `lattice_scan_report.md` containing:

- **Inputs and parameters** — the volume, the registered JSON, every radius run,
  and the reporting radius, which is shared by both phases.
- **Misalignment: NONE / MINOR / MAJOR**, with the overlay images cited. For
  MINOR, state what residual the sweep had to absorb; MAJOR ends the report
  here.

### Junctions

- **Systematic verdict: YES / NO.** Was there a missing region or other
  systematic issue? If so, describe each issue and cite the evidence. For each
  systematic issue, write the ids and positions of the affected junctions to
  `junction_scan_systematic_<issue>.json`, replacing `<issue>` with a short name
  for the issue, and cite these files in the report.
- **Stochastic verdict: YES / NO.** Were any junctions dark but not part of a
  systematic issue? If so, describe each finding and cite the evidence. For
  each stochastic finding, write the ids and positions of the affected
  junctions to `junction_scan_stochastic_<issue>.json`, named the same way, and
  cite these files in the report.
- **Radius sweep table** — radius against dark count and largest component.
  This table is the evidence for the two judgments above: it shows which part
  of the dark set was drift (the flags that melted away as the radius grew) and
  which persisted. Include it so the classification can be audited without
  re-running the scan. State the reporting radius alongside it and cite the
  slice overlay that justified the choice.
- **Surviving candidates** — junction id, x, y, z, intensity, degree, and
  whether the slice overlay confirmed it.
- **Statistics**: number dark / total, number dark (systematic) / total,
  number dark (stochastic) / total, and number dark but dismissed on visual
  inspection / total. The last three together account for every dark junction
  at the reporting radius.

### Struts

- **Systematic verdict: YES / NO**, and the count of missing struts attributed
  to each systematic region, with the endpoint junctions that put them there.
  Write their ids and positions to `strut_scan_systematic_<issue>.json`.
- **Corroborating struts** — those incident to a junction you reported missing.
  Give the count and say explicitly that they are a consistency check on the
  junction finding rather than independent defects.
- **Stochastic verdict: YES / NO** for the isolated missing struts that remain.
  Write their ids and positions to `strut_scan_stochastic_<issue>.json`.
- **Statistics**: number missing / total, split into systematic, corroborating
  and isolated. Those must account for every missing strut at the reporting
  radius.
- **Sensitivity not claimed** — state plainly that this scan detects *absence*
  only: a thin strut and a strut broken away from its midpoint both read as
  present, so a clean strut result does not mean the struts are sound.

### Additional notes

Any other observations made during the scan, including any issues with the
volume or the registered JSON. If a check could not be run, say so plainly
instead of implying it passed.

State counts as measured. Do not estimate how many defects to *expect* from a
nominal defect rate — removals in this dataset are clustered, not independent,
and that reasoning has already produced a wrong conclusion once.

## Validation

- The junction scan reports `degree` between 3 and 12. Degrees of 1–8 mean
  co-located JSON entries were not merged and every junction-level count is
  wrong.
- `n_junctions` is smaller than `n_entries`, typically by about a factor of 3,
  and the strut scan reports the same `n_junctions` as the junction scan.
- The Otsu threshold is the full-volume one, identical in both scans, and does
  not change between runs on the same volume.
- Every junction candidate you report has been looked at in a slice overlay.
- Every strut you report as missing has both endpoint junctions accounted for:
  either both are bright (an isolated missing strut) or one is a junction you
  already reported.

## Notes

- **A junction goes dark only when every incident strut is absent**, and
  interior junctions have twelve. The junction scan therefore has little
  sensitivity to individual missing struts — they leave both endpoints bright.
  That is the whole reason the strut phase exists, and why a clean junction
  result is a precondition for it rather than a conclusion.
- **Struts are not merged and their ids are the JSON's own**, unlike junction
  ids. Only the junction entries are duplicated.
- **Merged junction ids are not JSON entry ids.** Each junction's `entry_ids`
  column in the CSV maps back to the source file.
- **Thresholding is always against the full volume's Otsu**, never against the
  sampled node intensities. A node-fitted threshold lands inside the material
  mode, and it moves with whichever nodes you feed it.
- **Coordinates**: JSON positions are `[x, y, z]`; the volume is indexed
  `(z, y, x)`. Slice `axis` 0/1/2 means z/y/x.
- **No defect is assumed.** Nothing in the tools knows that the 9x9x9 specimens
  were machined on one face; you find it, every run, from the component sizes
  and the images. A different specimen with a different systematic — or none at
  all — is read the same way. Never carry a previous scan's conclusions into a
  new one.
- Both CSVs have a row for **every** element, flagged or not — the junction one
  with its component size and dark-neighbour count, the strut one with its
  endpoint junction ids and how many of its sample points were dark. That is
  where you get the ids of a region you want to account for separately, and it
  lets you re-band or re-group the specimen without re-running either scan.
- Registered JSON filenames in this dataset contain spaces; quote them.
