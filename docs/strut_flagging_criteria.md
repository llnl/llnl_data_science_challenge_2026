# Strut flagging criteria

How `src/strut_scan.py` decides that a nominal strut is missing or partial, and
what the two classes are worth. This replaces the sphere-at-points detector that
previously lived in the same module; the comparison between the two is recorded
at the bottom.

## The criterion

Wrap a cylinder of `radius` around each strut's junction-to-junction axis, cut
`cap_voxels` off each end, tile what remains into abutting segments, and take
the brightest voxel in each. A segment is **dark** when that maximum falls below
the **full-volume Otsu threshold**.

```
n_bisections = 1:   -[==========]-        1 segment
n_bisections = 2:   -[===|===|===]-       3 segments
n_bisections = 3:   -[=|=|=|=|=|=|=]-     7 segments

-  = end cap, excluded from every segment
[] = the scored span
```

| Dark segments | Class |
|---|---|
| all | `missing` |
| some but not all | `partial` |
| none | `present` |

`n` bisections leave `2**n - 1` equal segments. **The MCP tool
`scan_lattice_struts` fixes `n_bisections = 2`**, so three segments, leaving
`radius` and `cap_voxels` as the caller's two parameters.

There is no derived quantity, no per-strut normalization, and no cut fitted to
the data — the threshold is the same full-volume Otsu the junction detector
uses, measured where the balanced-bimodal assumption actually holds (11.3%
material on this scan, threshold 40049).

## The two free parameters

| Parameter | Default | Axis it acts on | What it answers |
|---|---|---|---|
| `radius` | 8 | perpendicular to the strut | residual registration drift |
| `cap_voxels` | 14 | along the strut | the endpoint junctions' own material |

**That separation is the entire reason for the shape.** The predecessor sampled
*points* along the span with a sphere, which forced one radius to answer both
questions at once:

- **Drift pushes the radius up.** The `registered_jsons/` alignment carries a
  ~0.8% x-scale error, ~5.6 voxels of drift across the specimen, against struts
  only ~4 voxels thick. Below radius 7 the residual reads as absence and the
  flagged fraction climbs with x (27.6% in the far-x band at radius 4, 0.80% at
  radius 8).
- **Junction reach pushes it down.** Twelve struts meet at a junction and the
  blob stays bright when any one is absent, so a probe reaching into it reports
  on the junction. Walking outward along a *known-absent* strut's own axis, the
  fraction of probes above Otsu: 77.0% at 2 voxels, 59.8% at 4, 33.3% at 6,
  11.5% at 8, 2.3% at 10, 0.6% at 12, 0.0% at 14.

With a sphere those collide. At two bisections the outer points sit 14.0 voxels
from a junction and would need `radius < 2`; the drift needs `radius ≥ 7`. No
value satisfied both, so the bisection count was pinned to 1 and the `partial`
class was structurally unreachable. The cap dissolves the conflict: it trims
along the axis, where only the junction reach lives, while the radius keeps
working perpendicular to it.

### Measuring the cap directly

The reference scan has 24 struts incident to the two confirmed missing junctions
(merged ids 513 and 2682). All 24 are genuinely absent, so all 24 *should* read
`missing`; any that read `partial` are being lit by their own junction. That
makes them a direct readout of how far the bleed reaches. At `radius = 8`:

| `cap_voxels` | 6 | 8 | 10 | 12 | **14** | 16 | 20 |
|---|---|---|---|---|---|---|---|
| confirmed struts reading `missing` | 3/24 | 7/24 | 13/24 | 21/24 | **24/24** | 24/24 | 24/24 |
| total `missing` | 305 | 323 | 362 | 397 | **417** | 422 | 427 |
| total `partial` | 128 | 112 | 74 | 38 | **23** | 18 | 10 |

The bleed is gone at 14 and not before, which is exactly where the probe-lit
fraction above reaches zero. **14 is the smallest cap that clears it**, and it
is the default for that reason. Going further is not free: every voxel of cap is
span no longer scored, and a break inside a cap is invisible.

### Measuring the radius

At `cap_voxels = 14`:

| `radius` | 6 | 7 | **8** | 10 |
|---|---|---|---|---|
| confirmed struts reading `missing` | 24/24 | 24/24 | **24/24** | 21/24 |
| total `missing` | 424 | 419 | **417** | 400 |
| total `partial` | 58 | 26 | **23** | 34 |
| `missing` not seen by the sphere baseline | 3 | 0 | **0** | 0 |

Radius 6 is too narrow: the drift leaves material outside the cylinder, which
shows up both as extra partials (58) and as 3 struts flagged `missing` that a
sphere at the same radius found material in. Radius 10 is too wide in the other
direction — it reaches sideways far enough to pick up neighbouring material and
loses 3 of the 24 confirmed struts. Radius 8 is the junction phase's radius and
sits in the flat middle.

x-band `missing` fractions at r=8, cap=14 are 2.24 / 2.35 / 1.82 / 2.64%, so no
drift gradient survives.

## What this statistic cannot see

- **A thin strut.** A maximum saturates, so a strut printed under-thickness
  still has a bright voxel in every segment. Separate defect class, not detected.
- **A break inside a cap.** The caps are scored by nothing. This is the price of
  clearing the junction bleed and the reason `cap_voxels` should be the smallest
  value that works rather than a generous one.

A clean result bounds *absence over the scored span*, and must not be reported
as evidence against either of the above.

## Comparison against the sphere-at-midpoint detector

Both on the full `0point5dash1` lattice (18468 struts, no exclusions), at
`radius = 8`. The sphere configuration is the retired shipped one: a single
sample point at the midpoint.

| | flagged | missing | partial |
|---|---|---|---|
| sphere, midpoint, r=8 (retired) | 429 | 429 | — (unreachable) |
| **cylinder, r=8, cap=14, 3 segments (shipped)** | **440** | **417** | **23** |

- **The cylinder flags every strut the sphere flagged — 429 of 429 — at every
  `(radius, cap)` pair in the sweep.** Nothing the old detector found is lost at
  any setting, which is what makes the cap safe to tune.
- **The cylinder's `missing` set is a strict subset of the sphere's.** At r=8 no
  strut is called `missing` by the cylinder and `present` by the sphere. The 12
  struts the sphere called `missing` and the cylinder calls `partial` have
  material somewhere off the midpoint — precisely what a single sample point
  could not see.
- **11 further struts read `partial` with a bright midpoint**, so the sphere
  called them `present` outright. These are the new sensitivity.
- All **24 struts incident to the confirmed missing junctions** read `missing`.
- The machined bottom face is rediscovered unprompted: **328 of the 417 missing
  struts** have an endpoint in the 171-junction dark component.

A `partial` is evidence, not a verdict. Two things produce it — an end segment
lit by junction bleed (cap too short) and a genuine mid-span break — and they
are told apart by looking, with `visualize_lattice_element`. Raising the cap
converts the first kind to `missing` and leaves the second alone.

Reproduce with:

```python
from strut_scan import scan_struts, summarize_strut_scan
result = scan_struts(volume, registered_json_path, radius=8, cap_voxels=14)
```

or through the MCP tool `scan_lattice_struts`, which pins `n_bisections` to 2.
