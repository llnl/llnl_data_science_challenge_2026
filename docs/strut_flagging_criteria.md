# Strut flagging criteria

How `src/strut_scan.py` decides that a nominal strut is missing or partial, and
what the two classes are worth. This replaces the cylinder/cross-section
detector that previously lived in `src/strut_cylinder_segmentation.py`; the
comparison between the two is recorded at the bottom.

## The criterion

Sample each strut at points placed by recursive bisection of its
junction-to-junction span, take the brightest voxel within `radius` of each, and
call a point **dark** when that maximum falls below the **full-volume Otsu
threshold**. Then:

| Dark points | Class |
|---|---|
| all | `missing` |
| some but not all | `partial` |
| none | `present` |

That is the whole rule. There is no derived quantity, no per-strut normalization,
and no cut fitted to the data — the threshold is the same full-volume Otsu the
junction detector uses, measured where the balanced-bimodal assumption actually
holds (11.3% material on this scan, threshold 40049).

### Sample point placement

`n_bisections` halvings leave `2**n - 1` points at the fractions `k / 2**n`:

```
n=1:        ●                  1/2
n=2:    ●   ●   ●              1/4, 1/2, 3/4
n=3:  ● ● ● ● ● ● ●            eighths
```

**The MCP tool `scan_lattice_struts` fixes `n_bisections = 1`** — the midpoint
alone — for the reason worked out below. At one sample point the `partial` class
cannot arise, so the operative rule reduces to: a strut is `missing` when its
midpoint is dark.

**The endpoints are never sampled.** Twelve struts meet at a junction of this
lattice and the resulting blob stays bright when any one of them is absent, so a
probe at or near an endpoint answers a question about the junction, not the
strut. This is the same problem the cylinder detector solved with `--end-trim`,
and it is equally non-optional: an untrimmed full-span cylinder found 1 flagged
strut in 17460.

## The two free parameters

| Parameter | Default | Effect |
|---|---|---|
| `radius` | 8 | Sampling sphere radius in voxels. |
| `n_bisections` | 1 | Halvings of the span; `2**n - 1` sample points. Pinned to 1 by the MCP tool. |

Nothing else. In particular there is no area cut, no end-trim, and no
specimen-specific exclusion — `strut_scan` reports every flagged strut and
suppresses nothing.

### Why the radius has to be about 8, and what that costs

The radius is pulled in two directions on this scan, and the two demands do not
both fit.

**Drift pushes it up.** The `registered_jsons/` alignment carries a ~0.8%
x-scale error, ~5.6 voxels of drift across the specimen, against struts only ~4
voxels thick. A radius too small to cover it reads the drift as absence, and the
symptom is a flagged fraction that climbs with x. Measured, at two bisections,
as the flagged fraction (missing or partial) in four equal-width x bands:

| radius | flagged | x band 0 | x band 1 | x band 2 | x band 3 |
|---|---|---|---|---|---|
| 4 | 1293 | 0.98% | 0.83% | 0.96% | **27.61%** |
| 5 | 485 | 0.63% | 0.69% | 0.47% | **9.57%** |
| 6 | 160 | 0.58% | 0.61% | 0.38% | 2.15% |
| 7 | 105 | 0.58% | 0.61% | 0.36% | 0.87% |
| 8 | 99 | 0.58% | 0.56% | 0.33% | 0.80% |

The gradient is gone by radius 7–8.

**Junction reach pushes it down.** Walking outward from a junction along a
*known-absent* strut's own axis, any material found belongs to the junction and
its other eleven struts. Measured over the 87 struts an independent detector
found wholly empty, as the fraction of probes above Otsu:

| distance out | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|
| probes lit | 77.0% | 59.8% | 33.3% | 11.5% | 2.3% | 0.6% | 0.0% |

Junction-local material reaches about 12 voxels. This is what decides the
bisection count.

## Why the bisection count is 1

Against a 55.8-voxel strut, the two demands above are compatible at one
bisection and incompatible at two:

| bisections | outer point sits | radius-8 sphere spans | clears the 12-voxel reach? |
|---|---|---|---|
| 1 | 27.9 vox from either junction | 19.9 – 35.9 | yes, by ~8 voxels at both ends |
| 2 | 14.0 vox from a junction | 6.0 – 22.0 | no — would need `radius < 2` |

At two bisections the outer sample point on a genuinely absent strut is often
lit by the junction it points at, and the strut reads `partial` rather than
`missing`. Over the same 87 known-empty struts, the dark-point patterns:

| radius | (1,1,1) all dark | one outer point bright | midpoint dark |
|---|---|---|---|
| 4 | 84 | 3 | 87 |
| 5 | 72 | 14 | 87 |
| 6 | 52 | 34 | 87 |
| 7 | 40 | 46 | 87 |
| 8 | 21 | 60 | 87 |

**The midpoint was dark on all 87 struts at every radius.** That is the column
that settles it: the midpoint alone carries the whole detection, while the outer
points only add a class that junction bleed and a real break populate
indistinguishably. So the tool samples the midpoint and reports one class.

Raising `n_bisections` is left in `scan_struts` as a module parameter, but on
these specimens it cannot be used without the bleed. Recovering partial-strut
sensitivity needs a refit registration that permits a narrower probe — not a
different threshold or a different cut.

## What this statistic cannot see

Both of these read as `present`, and a clean result must not be reported as
evidence against either:

- **A thin strut.** A maximum over a sphere saturates, so a strut printed
  under-thickness still has a bright voxel at its midpoint. The cylinder
  detector's cross-section could in principle have caught these — 8 of its 95
  flagged struts were flagged but not empty — but that capability was never
  validated, and it cost three free parameters.
- **A partial strut.** A break that does not cover the midpoint leaves the
  single sample point bright. The scan bounds *absence*, not integrity.

The summary JSON still carries `n_partial`, and at one sample point it is
structurally zero. Read `n_points_per_strut` before drawing anything from it.

## Comparison against the cylinder detector

Both run on the `0point5dash1` scan with the machined-off bottom face excluded
(1008 of 18468 struts), leaving 17460 scored. The cylinder configuration is the
validated one: `--radii 8 --end-trim 0.25`, cut at 25% of the nominal
cross-section.

| | flagged | fully empty / missing |
|---|---|---|
| cylinder, r=8, trim 0.25 | 95 | 87 |
| **sphere, N=1, r=8 (shipped)** | **97** | **97** |
| sphere, N=2, r=8 | 99 | 21 missing + 78 partial |

- **The shipped configuration recovers all 87 cylinder-empty struts** and adds
  10 more, so it clears the agreed bar of ≥ 87 with no misses.
- **It also contains all 95 struts the cylinder flagged, disagreeing on 2 of
  17460 — both sphere-only, none cylinder-only.** The sphere flagged set is a
  strict superset of the cylinder's. (At N=2 the same holds with 4
  disagreements, so the relationship does not depend on the bisection count.)
- Both bisection counts flag **all 24 struts incident to the two confirmed
  missing junctions** (merged ids 513 and 2682).
- The x-band flagged fractions are flat at r=8 (0.33–0.80%), so no residual
  drift gradient survives.

Reproduce with:

```python
from strut_scan import scan_struts, summarize_strut_scan
result = scan_struts(volume, registered_json_path, radius=8)   # n_bisections=1
```

or through the MCP tool `scan_lattice_struts`, which pins `n_bisections` to 1.
