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
`scan_lattice_struts`, `visualize_junction_overlay`, `visualize_lattice_element`
— is served by the **ct-segmentation** MCP server
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

7. **Verify a sample of the surviving candidates by eye.** For each one in the
   sample, call `visualize_lattice_element` with `element="junction"` and the
   candidate's merged id. You get three projections of the sphere's own
   neighbourhood, which is what settles a candidate — a single slice through a
   junction can miss material sitting a few voxels off the plane. Confirm the
   circle covers empty space while the surrounding struts show material. Drop
   any candidate that plainly sits on material and say why.

   How many to look at, and how to pick them: see *How much to look at*.

8. **Sweep the strut parameters.** `scan_lattice_struts` has two: `radius` and
   `cap_voxels`. Start at the radius you settled on for junctions, then vary
   **one at a time**. Outputs are tagged with both numbers, so nothing
   overwrites.

   Set the cap by *measuring* the junction bleed, not by guessing it: the struts
   incident to the junctions you just reported missing are known to be absent,
   so the smallest cap at which they all read `missing` is the cap that clears
   the bleed. This is why the strut phase runs after the junction phase and not
   beside it. See *Choosing the strut parameters* below for the full method,
   the fallback when no junction was found missing, and what to record.

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

10. **Triage a sample of the `partial`s**, and of the isolated missing struts.
    Take an element view of each one in the sample (*How much to look at*). A
    partial is evidence, not a verdict, and two different things produce it —
    see *Reading a partial* below.

11. **Write `lattice_scan_report.md`** in the output directory (see Report).

## How much to look at

Looking is how a candidate is settled, but looking at *every* candidate is not
what makes a result sound, and on a scan with hundreds of flags it is not
possible. Verify a sample and state what the sample was. The same rule applies
to junction candidates, to isolated missing struts, and to `partial`s:

- **Ten or fewer of a kind — look at all of them.** Small sets are cheap, and
  one wrong flag is a large fraction of the finding.
- **More than ten — look at at least ten**, chosen rather than taken off the top
  of the CSV. Spread them across the specimen (both ends of the drift axis, more
  than one octant) and include the borderline cases, where a misclassification
  is most likely: for junctions the dark ones with the highest intensity, for
  struts the `partial`s with only one dark segment. A sample drawn from one
  corner tests one corner.
- **A failure in the sample is a result about the whole set, not about one row.**
  If a sampled candidate plainly sits on material, the parameters or the region
  attribution are wrong for everything flagged the same way. Say so, fix it, and
  re-scan — do not drop the row and keep the total.

Report the sample: how many of each kind you viewed, out of how many, how you
chose them, and what they showed. A count backed by a stated sample is a
measurement; a count backed by "spot-checked" is not.

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

`scan_lattice_struts` wraps a **cylinder** of `radius` around each strut's axis,
cuts `cap_voxels` off each end, divides what remains into **three** abutting
segments, and takes the brightest voxel in each. A segment is dark when that
maximum falls below the same full-volume Otsu threshold the junction scan uses:

```
-[===|===|===]-      - = cap, not scored      [...] = the scored span
```

| Dark segments | Class |
|---|---|
| all three | `missing` |
| some but not all | `partial` |
| none | `present` |

The segment count is fixed at three by the tool. The two parameters you do
control do genuinely separate jobs, and that separation is the whole reason for
the shape:

- **`radius` is perpendicular to the axis**, so it only ever answers the
  registration drift. Too small and correctly-printed struts read absent because
  the lattice sits slightly off them; the symptom is a flagged fraction climbing
  along one axis.
- **`cap_voxels` is along the axis**, so it only ever answers the junction
  reach. Twelve struts meet at a junction and the blob stays bright when any one
  is absent, so a segment reaching into it reports on the junction instead of
  the strut. On the reference scan that material extends about 12 voxels along
  an absent strut's own axis.

Because they act on different axes, you can clear the junction bleed without
narrowing the probe. The sphere probe this replaced could not: its single radius
had to satisfy both demands at once, no value did, and it was pinned to one
sample point at the midpoint with the `partial` class unreachable.

| Signal | Reading |
|---|---|
| `missing`, endpoint junction inside a systematic region | Belongs to that region. Report the count; it is not a separate defect. |
| `missing`, endpoint junction reported missing | Corroboration of the junction finding, not an independent defect. |
| `missing`, isolated | A missing strut. This is the reportable strut-level defect. |
| `partial` | Evidence, not a verdict. Triage it — see below. |
| Missing fraction climbing steadily along one axis | The radius is too small for the drift, exactly as at junction level. |
| Many absent-looking struts reading `partial` rather than `missing` | The cap is too small and the junctions are bleeding into the end segments. |

One blind spot the report must state rather than let a clean result imply was
checked: **a thin strut reads present.** The statistic is a maximum, so a strut
printed under-thickness still has a bright voxel in every segment. This scan
bounds *absence* and under-thickness is a separate defect class it does not see.

### Reading a partial

Two different things put a strut in this class, and an element view tells them
apart:

- **Junction bleed** — an *end* segment is bright and the rest dark, because the
  cap did not clear the endpoint junction's material. This is a probe artifact:
  the strut is really absent. Fix it by raising `cap_voxels`, and check that the
  strut moves to `missing`.
- **A real break** — material over part of the span and none over the rest, with
  the bright part not hugging a junction. This is the defect class the capped
  cylinder exists to recover.

If raising the cap converts a partial to missing, it was bleed. If it survives a
cap that visibly clears both junctions, report it as a break candidate.

## Choosing the strut parameters

A count is not evidence for a parameter — it is what the parameter produces. So
neither parameter is chosen by picking the number that gives an agreeable answer.
The cap has a **measurement** that fixes it, described first; the radius is
chosen from images.

### The cap: measure the bleed with struts you already know are absent

The cap exists to clear the endpoint junctions' material, so setting it needs to
know how far that material reaches. You can measure that on this specimen rather
than assume it, using struts whose answer you already know.

**A junction goes dark only when every incident strut is absent.** So for any
junction the junction phase reported missing, all twelve of its struts are
*known* absent, and every one of them must read `missing`. Any that read
`partial` are being lit by their own junction — which makes them a direct
readout of how far the bleed reaches, not a proxy for it.

The method:

1. **Take the struts incident to the missing junctions you found in step 6.**
   Get their ids from the strut CSV: any row whose `junction0` or `junction1` is
   one of those junctions. A degree-12 junction contributes 12 struts.
2. **Run `scan_lattice_struts` at increasing `cap_voxels`**, holding the radius
   at the junction phase's value. A ladder like 6, 8, 10, 12, 14, 16, 20 is
   enough; outputs are tagged by both parameters so nothing overwrites.
3. **Count how many of those known-absent struts read `missing`** at each cap.
   The count climbs with the cap and then saturates at all of them.
4. **The cap is the smallest value where they all read `missing`.** Below it the
   junctions are still bleeding into the end segments; above it you are only
   giving up span.

Worked on the reference scan (`0point5dash1`, 24 struts incident to two
confirmed missing junctions, radius 8) — **as an illustration of the method, not
values to reuse**:

| `cap_voxels` | 6 | 8 | 10 | 12 | **14** | 16 | 20 |
|---|---|---|---|---|---|---|---|
| known-absent struts reading `missing` | 3/24 | 7/24 | 13/24 | 21/24 | **24/24** | 24/24 | 24/24 |
| total `partial` | 128 | 112 | 74 | 38 | **23** | 18 | 10 |

14 is the answer there. Note what the `partial` row would have told you on its
own: it falls monotonically, so "fewest partials" would have chosen cap 20 and
thrown away a third of the scored span for nothing. **The saturation point is
the criterion, not the partial count.**

**Do not go past it.** Every voxel of cap is span that stops being scored, and a
break sitting inside a cap is invisible. A larger cap looks better on every
count in the table while quietly buying that blindness, which is exactly why the
criterion is "smallest that saturates".

**If the junction phase found no missing junction**, this gauge is unavailable
and you must say so in the report. Two fallbacks, in order of preference:

- **Use a systematic region instead.** Struts running into a machined-off face
  or an unprinted corner are known absent for the same reason and work
  identically as a gauge.
- **Fall back to the images.** Raise the cap until, in
  `visualize_lattice_element` views of several flagged struts, both scored ends
  visibly stop short of the junction blobs. This is weaker — it is your judgment
  of where a blob ends rather than a measurement — and the report must say the
  cap was set by eye.

### The radius: check coverage in the images

Start at the radius the junction phase settled on. It is bounded on both sides
and you confirm it by looking:

- **Too small** and the drift leaves strut material outside the cylinder. Take
  views of two or three struts you expect to be sound, mid-specimen and at high
  drift: the material must sit inside the dashed band along the whole scored
  span. The other symptom is a missing fraction climbing steadily along one
  axis.
- **Too large** and the cylinder reaches sideways into neighbouring material.
  The gauge above catches this too — a radius that is too wide starts *losing*
  known-absent struts back to `partial` or `present`.

Take the smallest radius that covers the struts and keeps the gauge saturated.

### Verify before reporting

At your chosen pair, take `visualize_lattice_element` views of at least:

- **two or three struts you expect to be sound** — the cylinder covers their
  material along the whole scored span;
- **two or three flagged struts** — both caps visibly stop short of the junction
  blobs, and the scored span crosses only empty volume;
- **a sample of the `partial`s** (*How much to look at*) — to classify each as
  bleed or break.

**Record the sweep as you go** — every pair tried, what its images showed, and
why you rejected it or kept it. The report has to reproduce this reasoning, and
reconstructing it afterwards from counts alone is not possible.

## Choosing the junction radius

The junction phase has one free parameter, and this is it. The radius exists to
absorb residual registration error: too small and correctly-placed material is
missed because the lattice sits slightly off it; too large and the probe reaches
neighbouring struts and hides a real absence. The strut phase starts from
whatever you settle on here and adds a second parameter of its own — see
*Choosing the strut parameters*.

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

There are two image tools. `visualize_junction_overlay` answers questions about
the *specimen* — is the lattice aligned, where does the dark set sit.
`visualize_lattice_element` answers questions about *one element*, and is what
you verify candidates and choose strut parameters with.

### `visualize_lattice_element`

Give it `element="junction"` or `element="strut"` and the id the scan CSV
reported. It crops that element's own probe region, pads it, and returns three
max-intensity projections — along z, y and x — with the probe drawn to scale,
plus the element's measured intensities and status in the reply.

Three projections rather than one slice, because a single plane through a
55-voxel strut shows a few voxels of it and material a little off the plane is
invisible. An element dark in all three views is dark throughout its box.

- For a **junction** the circle is the sampling sphere, coloured red when dark
  and green when bright.
- For a **strut** each scored segment is drawn along the axis, red when dark and
  green when bright, with ticks at the segment boundaries. The dotted line
  running past both ends is the excluded cap, and both junction blobs stay in
  frame — which is how you see whether the cap clears them. The dashed lines at
  ±`radius` show the probe's width; they are exact only when the strut lies in
  the projection plane and an over-estimate otherwise, which the other two
  projections cover.

It scores only the element you name, so it is cheap and does not re-run a scan.
Pass the same `radius` and `cap_voxels` as the scan you are checking, or you are
looking at a different measurement than the one you are judging.

### `visualize_junction_overlay`

This one always re-scores the junctions internally: pass the `radius` of the
scan you are checking, since the markers are coloured from that result — green
for bright, red for dark — and the picture changes with the radius.

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
  mid-specimen default. `axis` 0/1/2 slices along z/y/x. To inspect a single
  candidate, prefer `visualize_lattice_element`.
- To scrub the whole stack instead of single planes, re-run
  `scan_lattice_junctions` with `write_marked_tiff=True` for an RGB TIFF with
  the candidates painted. It costs roughly three bytes per input voxel in
  memory and on disk, so request it once, at the radius you settled on.

## Report

Write `lattice_scan_report.md` containing:

- **Final parameters**, stated plainly and near the top, on their own labelled
  lines — not buried in prose:

  ```
  Junction sampling radius: <r> voxels
  Strut cylinder radius:    <r> voxels
  Strut end cap:            <c> voxels
  Segments per strut:       3 (fixed by the tool)
  ```

  Repeat these values in your closing message to the user. Every count in the
  report is conditional on them, so a reader who takes nothing else away must
  still take these.
- **Inputs** — the volume, the registered JSON, and the output directory.
- **Misalignment: NONE / MINOR / MAJOR**, with the overlay images cited. For
  MINOR, state what residual the sweep had to absorb; MAJOR ends the report
  here.

### Parameter selection

The parameters are your judgment, so this section has to make that judgment
auditable. A count reported without saying which images justified the parameters
behind it is an incomplete report. Include:

- **The junction radius sweep table** — radius against dark count and largest
  component — and which slice overlay justified the reporting radius.
- **The strut sweep table** — every `(radius, cap_voxels)` pair tried, against
  `n_missing`, `n_partial`, the worst band fraction, and **the bleed gauge**:
  how many of the known-absent struts read `missing` at that setting, as a
  fraction of the total. Name which struts the gauge used and why they are known
  absent (which missing junction they are incident to, or which systematic
  region they run into). If no gauge was available, say so here and state that
  the cap was set by eye.
- **Your reasoning, in your own words**, one short paragraph per parameter:
  what you varied, which element views you looked at (cite them by filename),
  what those images showed, and why the value you chose beat its neighbours.
  Answer specifically:
  - **where the gauge saturated**, and that the cap you chose is the smallest
    value that saturates it — not a larger one that also passes;
  - why the radius covers the struts despite the drift, **and no wider**.
- **Rejected settings** — at least one pair you tried and rejected, with the
  reason. For example: *"cap 8: 92 struts read `partial` rather than `missing`,
  and `strut_1234_r8_c8_proj_z.png` shows the junction blob reaching into the
  first segment — junction bleed, not breaks."* A sweep reported with no
  rejections is not a sweep, and reads as defaults accepted without testing.

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
- **Surviving candidates** — junction id, x, y, z, intensity, degree, and, for
  those in the verified sample, whether the element view confirmed it. Say which
  ones were viewed, how many out of how many, and how they were chosen. (The
  radius sweep table itself lives in *Parameter selection*; refer to it here
  rather than repeating it.)
- **Statistics**: number dark / total, number dark (systematic) / total,
  number dark (stochastic) / total, and number dark but dismissed on visual
  inspection / total. The last three together account for every dark junction
  at the reporting radius. The dismissed count is over the sample viewed, so
  give the sample size beside it.

### Struts

- **Systematic verdict: YES / NO**, and the count of missing struts attributed
  to each systematic region, with the endpoint junctions that put them there.
  Write their ids and positions to `strut_scan_systematic_<issue>.json`.
- **Corroborating struts** — those incident to a junction you reported missing.
  Give the count and say explicitly that they are a consistency check on the
  junction finding rather than independent defects.
- **Stochastic verdict: YES / NO** for the isolated missing struts that remain.
  Write their ids and positions to `strut_scan_stochastic_<issue>.json`.
- **Partial struts** — the count, and for each one in the viewed sample whether
  the element view showed junction bleed (a bright *end* segment, meaning the
  cap is short) or a genuine mid-span break. Give the image filename for each.
  State the sample: how many were viewed out of how many, and how they were
  chosen. If any were reclassified by raising the cap, say so. Report break
  candidates as candidates, not as confirmed breaks.
- **Statistics**: number missing / total, split into systematic, corroborating
  and isolated; plus number partial / total split into bleed and break
  candidates. Those must account for every flagged strut at the reporting
  parameters. The bleed/break split is over the sample viewed — give the sample
  size beside it, and do not extrapolate the split to the unviewed rest.
- **Sensitivity not claimed** — state plainly that this scan detects *absence*
  only: a thin strut reads as present, and so does a break that falls entirely
  inside one of the end caps. A clean strut result does not mean the struts are
  sound.

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
- A stated sample of the junction candidates you report has been looked at in an
  element view — all of them when there are ten or fewer, at least ten and
  spread across the specimen otherwise — and nothing in the sample turned out to
  sit on material.
- Every strut you report as missing has both endpoint junctions accounted for:
  either both are bright (an isolated missing strut) or one is a junction you
  already reported. This one is a bookkeeping check on the CSV, not a visual
  one, so it covers every row.
- A sample of the `partial`s has been looked at and classified as bleed or
  break, on the same rule, and the report gives the sample size next to the
  split.
- **The bleed gauge is saturated: every strut incident to a junction you
  reported missing reads `missing` itself.** If any reads `partial`, the cap is
  too short — the junction's own material is bleeding into the end segments —
  and the parameters are not settled. This is the check the cap was chosen to
  pass, so a report that fails it has its counts measured at the wrong setting.
- **The cap is the smallest value that saturates the gauge.** If the next value
  down also saturates it, you scored less span than you needed to.
- The report states the final parameters on their own lines, gives the gauge
  column in the sweep table, and names at least one rejected setting.

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
  endpoint junction ids and how many of its segments were dark. That is
  where you get the ids of a region you want to account for separately, and it
  lets you re-band or re-group the specimen without re-running either scan.
- Registered JSON filenames in this dataset contain spaces; quote them.
