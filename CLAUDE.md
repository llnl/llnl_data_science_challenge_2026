# CLAUDE.md

Guidance for Claude Code when working in this repository.

Repository conventions (structure, setup, coding style, testing, commits/PRs)
live in `AGENTS.md` — read that file too; treat it as authoritative for those
topics rather than duplicating them here.

## Philosophy
Prefer *simple* over complex. Always think twice before adding a feature. When weighing solutions, simple solutions should be weighed much higher than complex ones — this is a scientific project, code must be verifiable, maintainable, and interpretable. Complex solutions cost in these areas. Resist the tendency to over-engineer. Think like a scientist, not a software engineer. 

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
      MCP tools, and the `scan-lattice` skill. The skill runs alignment and
      systematic checks *before* reporting stochastic candidates, and treats the
      sampling radius as its one free parameter.
- [x] Phase 2b — the tools carry **no specimen-specific defect knowledge** and
      take only a volume and a registered JSON. An `exclude_bottom_face` flag
      hardcoded this one specimen's machining artifact; the other eight scans
      will not share it. The agent now finds a systematic region from the
      component sizes, the band fractions and the overlays, and accounts for it
      separately when reading the per-junction CSV. The tools report every dark
      junction and suppress nothing — an `exclude_junction_ids` parameter came
      and went, since a junction's darkness never depended on any other
      junction's status and excluding only changed the bookkeeping.
- [x] Phase 2c — strut detection rebuilt on the junction detector's sphere
      probe: `src/strut_scan.py`, the `scan_lattice_struts` MCP tool, and the
      `scan-junctions` skill widened to `scan-lattice` covering both phases. The
      cylinder/cross-section detector is deleted. Two free parameters (sphere
      radius, number of bisections) replace the cylinder's three, the agent gets
      only one of them, and it is the *same* radius the junction phase chose.
      Bisections are pinned to **1** — the midpoint alone — because that is the
      only placement whose probe clears the junctions at a radius wide enough to
      absorb the registration drift (finding 4 below).
- [x] Phase 2d — the strut probe is now a **capped cylinder**: radius acts
      perpendicular to the strut and answers the drift, `cap_voxels` acts along
      it and answers the junction reach. Because the two demands act on
      different axes they no longer collide, so the span *can* be subdivided and
      the `partial` class is reachable for the first time. Three segments are
      fixed by the MCP tool; the agent sees `radius` and `cap_voxels` and
      chooses them from images, using the new `visualize_lattice_element` tool
      (`src/element_views.py`) which renders one junction or one strut as three
      orthogonal max projections with the probe drawn to scale. The
      `scan-lattice` skill now requires the agent to report *how* it chose its
      parameters — a sweep table, the images it judged from, and at least one
      rejected setting — and to print the final parameters plainly.

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
is a thin script over it, and the `scan-lattice` skill is how an agent drives
it. All three produce the same numbers below.

`src/mark_junction_candidates_tiff.py` finds **2 missing junctions** in the
`0point5dash1` scan: merged ids 513 at (141, 685, 499) and 2682 at
(615, 682, 420), both degree 12. The user visually inspected the stack and
confirmed there are exactly 2 — so this is 2/2 with no false positives or
negatives. Independent statistics agree on the same pair (node-intensity sphere,
strut-center probe, and the retired cylinder's segmented-voxel counts), and all
24 struts incident to them are flagged by the strut detector at every radius and
bisection count tried.

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

4. **`dark_components.largest / n_dark` separates a systematic absence
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
described as strut finding 1. It is no longer special-cased in either detector:
they rediscover it every run — one junction component of 171, and 328 missing
struts — report it alongside everything else, and leave the caller to account
for it separately.
`mark_junction_candidates_tiff.py` still uses `bottom_layer_junctions()`
directly, which is the right place for it — that script is hardcoded to this
specimen, and the detector is not.

## Missing struts — capped cylinder, cross-validated against the retired sphere

`src/strut_scan.py` wraps a cylinder of `radius` around each strut's axis, cuts
`cap_voxels` off each end, tiles what remains into `2**n - 1` abutting segments,
and takes the brightest voxel in each, cut against the full-volume Otsu. A strut
is `missing` when every segment is dark, `partial` when some but not all are.
**The MCP tool fixes `n = 2`** — three segments — leaving the agent `radius` and
`cap_voxels`. Full derivation in `docs/strut_flagging_criteria.md`.

**The two parameters act on different axes, and that is the whole point.**
`radius` is perpendicular to the strut and answers only the registration drift;
`cap_voxels` is along it and answers only the endpoint junctions' reach. The
sphere probe this replaced had one radius for both jobs, no value satisfied
them together (finding 4 below), and the `partial` class was therefore
unreachable. It is reachable now.

The cylinder/cross-section detector (`strut_cylinder_segmentation.py`) is still
**deleted** and is not what this is: its mean-intensity and area-cut statistics
stay dead, and its `bottom_layer_junctions()` remains in
`mark_junction_candidates_tiff.py`, the one script hardcoded to this specimen.

### The cap is measured, not assumed

The 24 struts incident to the two confirmed missing junctions are all genuinely
absent, so any reading `partial` are being lit by their own junction. That makes
them a direct readout of the bleed. At `radius = 8` on `0point5dash1`:

| `cap_voxels` | 6 | 8 | 10 | 12 | **14** | 16 | 20 |
|---|---|---|---|---|---|---|---|
| confirmed struts reading `missing` | 3/24 | 7/24 | 13/24 | 21/24 | **24/24** | 24/24 | 24/24 |
| total `missing` | 305 | 323 | 362 | 397 | **417** | 422 | 427 |
| total `partial` | 128 | 112 | 74 | 38 | **23** | 18 | 10 |

14 is the smallest cap that clears it, matching the independently measured
junction reach (0% of probes lit at 14 voxels). Bigger is not safer — cap is
span that stops being scored, and a break inside a cap is invisible.

At `cap = 14`, radius 6 is too narrow (58 partials, and 3 struts called missing
that a sphere at the same radius found material in) and radius 10 too wide
(loses 3 of the 24 confirmed struts to neighbouring material). Radius 8, the
junction phase's, sits in the flat middle.

### Cross-validation on `0point5dash1` (full lattice, 18468 struts, r=8)

| Method | flagged | missing | partial |
|---|---|---|---|
| sphere at midpoint (retired) | 429 | 429 | — unreachable |
| **cylinder, cap 14, 3 segments (shipped)** | **440** | **417** | **23** |

- The cylinder flags **all 429** struts the sphere flagged, at **every**
  `(radius, cap)` pair tried. Nothing the old detector found is lost anywhere in
  the sweep, which is what makes the cap safe to tune.
- Its `missing` set is a strict **subset** of the sphere's — no strut is
  cylinder-missing and sphere-present at r=8. The 12 struts the sphere called
  missing and the cylinder calls partial have material off the midpoint, which
  a single sample point could not see. A further 11 partials have a bright
  midpoint and were called `present` outright by the sphere.
- All **24 struts** incident to junctions 513 and 2682 read `missing`.
- The machined face is rediscovered unprompted: **328 of 417** missing struts
  have an endpoint in the 171-junction dark component.
- x-band missing fractions at r=8, cap=14 are 2.24/2.35/1.82/2.64% — flat, so
  no drift gradient survives.

A `partial` is evidence, not a verdict: junction bleed and a real break both
produce it, and they are separated by looking, with `visualize_lattice_element`.

### Four findings that dominate any coordinate-sampling approach

1. **The specimen's bottom face was machined off after printing.** Any mid-stack
   slice shows the lattice ending in a sawtooth around y≈720, with no material
   where the design's last junction row sits (y≈760). Those junctions exist in
   the JSON but not in the part. The sphere detector rediscovers it unprompted:
   328 of its 349 missing struts lie there. The face is the max-Y layer of the
   **nominal** JSON (`data/missing_struts/octet_truss_9x9x9.json`), whose Y maps
   to registered y with r=1.0000; registered coordinates are rotated, so the
   layer cannot be found by thresholding registered y.

2. **The `registered_jsons/` alignment is not exact — it carries a ~0.8% x-scale
   error**, ~5.6 voxels of drift across the specimen, against struts only ~4
   voxels thick. Symptom: flagged fraction climbing with x. Measured for the
   sphere method at N=2, far-x band against the rest: 27.6% vs ~0.9% at r=4,
   9.6% at r=5, 2.2% at r=6, 0.87% at r=7, 0.80% vs 0.33–0.58% at r=8. **The
   radius must be ≥7.** The alternative fix, still unused, is refitting an
   affine (mean node offset 2.13 → 0.87 vox).

3. **Never sample at a strut's endpoints.** Twelve struts meet at a junction and
   the blob stays bright when any one is absent, so an endpoint probe answers a
   question about the junction. The cylinder detector needed `--end-trim` for
   this (untrimmed it found 1 strut in 17460); the bisection points are interior
   by construction.

4. **Junction material reaches ~12 voxels along an absent strut's own axis**
   (33% of probes lit at 6 voxels, 2.3% at 10, 0% at 14). This is why a probe
   near a strut's end reports on its junction, and it is the quantity
   `cap_voxels` exists to clear.

   It used to be a hard ceiling on the *bisection count*. With a sphere the same
   radius had to absorb the drift (needs r ≥ 7) and stay out of the junctions;
   N=2's outer points sat 14.0 voxels out and would have needed r < 2, so **no
   radius did both** and the count was pinned to 1. That conclusion was correct
   about spheres and is **not** a law about probes in general — it was the
   motivation for the cap, which trims along the axis while the radius keeps
   working perpendicular to it. Both demands are now satisfiable at once, and
   the sweep above measures the cap that does it.

**This scan detects absence over the scored span, not integrity.** Two things
read as `present`, and a clean result is not evidence against either:

- **Thin struts.** A maximum saturates regardless of the probe's shape. Not
  detected, and not something the retired cylinder's cross-section was ever
  validated to catch.
- **A break inside an end cap.** The caps are scored by nothing. This is the
  price of clearing the junction bleed, and the reason `cap_voxels` should be
  the smallest value that works rather than a generous one.

Breaks in the scored span *are* now visible, as `partial`. That class is real at
the shipped setting — check `n_segments_per_strut` is 3 — but it is evidence
rather than a verdict: junction bleed from too small a cap populates it too.

Statistics that do *not* work, both verified on real data during the cylinder
era and still worth not re-trying: the cylinder **mean intensity** (a
solid/background mixture, so a healthy strut averages near the volume Otsu
itself — flags 35% at r=3, 50% at r=5), and **Otsu on a derived per-strut
distribution** (modes are ~3%/97%, so it lands mid-population and flags 48%).
Cut against the full-volume Otsu, never against a statistic derived from the
lattice.
