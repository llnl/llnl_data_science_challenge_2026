"""Score every lattice strut against the CT volume with a capped cylinder probe
and classify each strut as present, partial or missing.

The statistic is still the junction detector's -- the brightest voxel in a
region, cut against the *full-volume* Otsu threshold -- but the region is now a
cylinder wrapped around the strut's own axis rather than a sphere at a point on
it. Two ends of the cylinder are cut off by ``cap_voxels``, and what remains is
tiled into abutting segments, each scored by its own maximum::

    n_bisections = 1:   -[==========]-        1 segment
    n_bisections = 2:   -[===|===|===]-       3 segments
    n_bisections = 3:   -[=|=|=|=|=|=|=]-     7 segments

``-`` is the end cap, excluded from every segment; ``[...]`` is the scored span.
``n`` bisections leave ``2**n - 1`` equal segments, so the count matches the
sphere probe this replaces and reduces to one whole-cylinder segment at ``n=1``.

Classification follows directly from how many segments are dark:

- **missing** -- every segment dark. No material anywhere along the scored span.
- **partial** -- some but not all segments dark. Material over part of the span
  and none over the rest, which is what a broken or partly-printed strut looks
  like. At one bisection there is a single segment, so this class cannot arise.
- **present** -- no segment dark.

Why the cap exists
------------------
Twelve struts meet at a junction of this lattice and the resulting blob stays
bright when any one of them is absent, so a probe near an endpoint answers a
question about the junction rather than the strut. The predecessor of this
module handled that by sampling *points* placed by recursive bisection, which
made the sampling radius carry two incompatible jobs at once:

- **Drift pushes the radius up.** The ``registered_jsons/`` alignment carries a
  ~0.8% x-scale error, about 5.6 voxels of drift across the specimen, against
  struts only ~4 voxels thick. Below radius 7 the residual reads as absence and
  the flagged fraction climbs across the specimen (27.6% in the far-x band at
  radius 4, 0.80% at radius 8).
- **Junction reach pushes it down.** Material belonging to a junction and its
  other eleven struts extends about 12 voxels along an absent strut's own axis
  (33% of probes still lit at 6 voxels, 2.3% at 10, none at 14). A sphere whose
  centre sits closer than that to a junction gets lit by the junction.

With a sphere those demands collide: at two bisections the outer points sit 14.0
voxels from a junction and would need a radius below about 2, while the drift
needs 7 or more. No radius satisfied both, so the bisection count was pinned to
one and the ``partial`` class was structurally unreachable.

The cap breaks the collision by giving each demand its own parameter. ``radius``
is measured *perpendicular* to the strut axis, where only the drift matters;
``cap_voxels`` is measured *along* it, where only the junction reach matters.
Setting ``cap_voxels`` past the junction's reach clears the bleed at any radius,
so the span can be subdivided as finely as the geometry allows and a break that
misses the strut's midpoint becomes visible.

What this statistic still cannot see
------------------------------------
A maximum saturates: a strut that is present but *thin* still has a bright voxel
in every segment and reads as present. Thin struts are a separate defect class
and this module does not detect them.

A break confined to a cap also reads as present, since the caps are scored by
nothing. That is the price of clearing the junction bleed, and it is why
``cap_voxels`` should be set to the smallest value that clears it rather than
generously.

Nothing here knows anything about a particular specimen. Each strut carries the
merged junction ids of its two endpoints, which is how a caller attributes a
flagged strut to a systematic region -- a machined face, an unprinted corner --
that the junction scan identified. Nothing is suppressed on that basis here.

Coordinate conventions follow the rest of the pipeline: JSON ``position`` fields
are [x, y, z] and index the volume's (z, y, x) axes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu

from junction_scan import band_stats
from strut_center_intensity_histogram import load_lattice

# Perpendicular to the strut axis, so this absorbs the residual registration
# drift and nothing else. Matches the junction scan's default for the same
# reason: it is the smallest radius at which the reference scan's drift stops
# producing a gradient in the flagged fraction across the specimen.
DEFAULT_RADIUS_VOXELS = 8

# Along the strut axis, so this clears the junction's own material and nothing
# else. Junction-local material reached 0% of probes at 14 voxels on the
# reference scan; see the module docstring.
DEFAULT_CAP_VOXELS = 14

DEFAULT_BISECTIONS = 2

STATUS_PRESENT = "present"
STATUS_PARTIAL = "partial"
STATUS_MISSING = "missing"


@dataclass(frozen=True)
class StrutScanResult:
    """Per-strut scoring of one lattice against one volume.

    Struts are indexed as the JSON lists them; junction ids are *merged* ids and
    do not index the JSON's ``junctions`` list.

    Attributes:
        strut_junction_ids: (S, 2) merged junction ids joined by each strut.
        positions_xyz: (N, 3) merged junction positions as [x, y, z].
        segment_centers_xyz: (S, P, 3) centre of each scored segment, unrounded.
        intensities: (S, P) brightest voxel inside each segment.
        segment_dark: (S, P) True where a segment's maximum fell below
            ``threshold``.
        n_dark_segments: (S,) how many of the P segments are dark.
        missing: (S,) True where every segment is dark.
        partial: (S,) True where some but not all segments are dark.
        threshold: The full-volume Otsu threshold that darkness was cut against.
        radius: Cylinder radius in voxels, perpendicular to the strut axis.
        cap_voxels: Length trimmed from each end of the strut before scoring.
        n_bisections: Number of halvings; P is ``2**n_bisections - 1``.
    """

    strut_junction_ids: np.ndarray
    positions_xyz: np.ndarray
    segment_centers_xyz: np.ndarray
    intensities: np.ndarray
    segment_dark: np.ndarray
    n_dark_segments: np.ndarray
    missing: np.ndarray
    partial: np.ndarray
    threshold: float
    radius: int
    cap_voxels: float
    n_bisections: int

    @property
    def n_struts(self) -> int:
        return len(self.strut_junction_ids)

    @property
    def n_segments(self) -> int:
        return self.segment_centers_xyz.shape[1]

    @property
    def centers_xyz(self) -> np.ndarray:
        """(S, 3) midpoint of each strut, which the segments are symmetric about."""
        start = self.positions_xyz[self.strut_junction_ids[:, 0]]
        end = self.positions_xyz[self.strut_junction_ids[:, 1]]
        return (start + end) / 2.0

    @property
    def status(self) -> np.ndarray:
        """(S,) array of ``"present"`` / ``"partial"`` / ``"missing"`` strings."""
        labels = np.full(self.n_struts, STATUS_PRESENT, dtype=object)
        labels[self.partial] = STATUS_PARTIAL
        labels[self.missing] = STATUS_MISSING
        return labels


def segment_bounds(
    lengths: np.ndarray, cap_voxels: float, n_bisections: int
) -> tuple[np.ndarray, np.ndarray]:
    """Axial start and end of each scored segment, measured from the start junction.

    The scored span runs from ``cap_voxels`` to ``length - cap_voxels`` and is
    tiled into ``2**n_bisections - 1`` equal abutting segments.

    Args:
        lengths: (S,) junction-to-junction length of each strut in voxels.
        cap_voxels: Length excluded at each end.
        n_bisections: Number of halvings, at least 1.

    Returns:
        (lower, upper), each (S, 2**n_bisections - 1), in voxels along the axis.

    Raises:
        ValueError: If ``n_bisections`` is less than 1, ``cap_voxels`` is
            negative, or any strut is too short to leave a scored span.
    """
    if n_bisections < 1:
        raise ValueError(f"n_bisections must be at least 1, got {n_bisections}")
    if cap_voxels < 0:
        raise ValueError(f"cap_voxels must not be negative, got {cap_voxels}")

    spans = lengths - 2.0 * cap_voxels
    if np.any(spans <= 0):
        shortest = int(np.argmin(lengths))
        raise ValueError(
            f"cap_voxels={cap_voxels} leaves no span to score on strut {shortest}, "
            f"which is {lengths[shortest]:.1f} voxels long; cap_voxels must be "
            f"below half the shortest strut ({lengths.min() / 2:.1f})"
        )

    n_segments = 2**n_bisections - 1
    edges = np.arange(n_segments + 1, dtype=float) / n_segments
    starts = cap_voxels + spans[:, None] * edges[None, :-1]
    ends = cap_voxels + spans[:, None] * edges[None, 1:]
    return starts, ends


def segment_maxima(
    volume: np.ndarray,
    start_xyz: np.ndarray,
    unit_xyz: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    radius: int,
    strut_id: int,
) -> np.ndarray:
    """Brightest voxel inside each segment of one strut's capped cylinder.

    A voxel belongs to segment ``k`` when its axial coordinate along the strut
    falls in ``[lower[k], upper[k])`` and its perpendicular distance from the
    axis is at most ``radius``. Only the cylinder's padded bounding box is
    examined, which is what keeps this affordable per strut.

    Raises:
        ValueError: If a segment contains no voxels, which means the radius or
            the cap has collapsed the region being measured.
    """
    z_size, y_size, x_size = volume.shape
    axis_ends = np.stack(
        [start_xyz + lower[0] * unit_xyz, start_xyz + upper[-1] * unit_xyz]
    )
    shape_xyz = np.array([x_size, y_size, z_size])
    lo = np.clip(np.floor(axis_ends.min(axis=0) - radius), 0, shape_xyz - 1)
    hi = np.clip(np.ceil(axis_ends.max(axis=0) + radius) + 1, 1, shape_xyz)
    (x0, y0, z0), (x1, y1, z1) = lo.astype(int), hi.astype(int)

    # Axial and radial coordinates are separable in x, y and z, so broadcasting
    # three 1-D offset arrays covers the box without materializing a coordinate
    # array per voxel.
    dx = np.arange(x0, x1, dtype=float)[None, None, :] - start_xyz[0]
    dy = np.arange(y0, y1, dtype=float)[None, :, None] - start_xyz[1]
    dz = np.arange(z0, z1, dtype=float)[:, None, None] - start_xyz[2]

    axial = dx * unit_xyz[0] + dy * unit_xyz[1] + dz * unit_xyz[2]
    perpendicular_squared = (dx * dx + dy * dy + dz * dz) - axial * axial

    box = volume[z0:z1, y0:y1, x0:x1]
    within_radius = perpendicular_squared <= float(radius) ** 2

    maxima = np.empty(len(lower), dtype=volume.dtype)
    for segment, (low, high) in enumerate(zip(lower, upper)):
        # The last segment takes its upper edge inclusively so the cylinder's far
        # end is not left unscored by a floating-point hair.
        above = axial >= low
        below = axial <= high if segment == len(lower) - 1 else axial < high
        inside = within_radius & above & below
        if not inside.any():
            raise ValueError(
                f"segment {segment} of strut {strut_id} contains no voxels at "
                f"radius {radius}; the radius or cap_voxels has collapsed the "
                f"region being measured"
            )
        maxima[segment] = box[inside].max()
    return maxima


def score_strut(
    volume: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    radius: int = DEFAULT_RADIUS_VOXELS,
    cap_voxels: float = DEFAULT_CAP_VOXELS,
    n_bisections: int = DEFAULT_BISECTIONS,
    strut_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score one strut's capped cylinder, without touching the rest of the lattice.

    This is what an inspection tool needs: re-running a whole scan to look at a
    single candidate costs minutes and produces numbers already on disk.

    Args:
        volume: 3D CT volume in (z, y, x) order.
        start_xyz: First endpoint junction's [x, y, z] position.
        end_xyz: Second endpoint junction's [x, y, z] position.
        radius: Cylinder radius in voxels, perpendicular to the strut axis.
        cap_voxels: Length excluded at each end, along the axis.
        n_bisections: Number of halvings of the scored span.
        strut_id: Only used to name the strut in error messages.

    Returns:
        (intensities, lower, upper), each of length ``2**n_bisections - 1``: the
        brightest voxel in each segment and the segment's axial bounds in voxels
        from the start junction.
    """
    if radius < 1:
        raise ValueError(f"radius must be a positive number of voxels, got {radius}")

    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length == 0:
        raise ValueError(f"strut {strut_id} has coincident endpoints")

    lower, upper = segment_bounds(np.array([length]), cap_voxels, n_bisections)
    intensities = segment_maxima(
        volume, start, direction / length, lower[0], upper[0], radius, strut_id
    )
    return intensities, lower[0], upper[0]


def scan_struts(
    volume: np.ndarray,
    registered_json_path: str | Path,
    radius: int = DEFAULT_RADIUS_VOXELS,
    cap_voxels: float = DEFAULT_CAP_VOXELS,
    n_bisections: int = DEFAULT_BISECTIONS,
) -> StrutScanResult:
    """Score every strut's capped cylinder segment by segment and classify it.

    Args:
        volume: 3D CT volume in (z, y, x) order.
        registered_json_path: Lattice JSON already registered to this volume's
            voxel coordinates. Un-registered/nominal JSONs do not line up.
        radius: Cylinder radius in voxels, perpendicular to the strut axis. Wide
            enough to absorb residual registration drift.
        cap_voxels: Length excluded at each end of the strut, along its axis.
            Large enough to clear the endpoint junctions' own material.
        n_bisections: Number of halvings of the scored span; it is divided into
            ``2**n_bisections - 1`` segments.

    Returns:
        A ``StrutScanResult``.

    Raises:
        ValueError: If ``volume`` is not 3D, ``radius`` is not positive,
            ``n_bisections`` is less than 1, ``cap_voxels`` is negative or
            leaves no span on some strut, or a segment ends up empty.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, found {volume.ndim} dimensions")
    if radius < 1:
        raise ValueError(f"radius must be a positive number of voxels, got {radius}")

    threshold = float(threshold_otsu(volume))

    positions_xyz, strut_junction_ids, _ = load_lattice(Path(registered_json_path))
    start_xyz = positions_xyz[strut_junction_ids[:, 0]]
    end_xyz = positions_xyz[strut_junction_ids[:, 1]]

    directions = end_xyz - start_xyz
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        raise ValueError("degenerate strut with coincident endpoints in lattice JSON")
    unit_xyz = directions / lengths[:, None]

    lower, upper = segment_bounds(lengths, cap_voxels, n_bisections)
    segment_centers_xyz = (
        start_xyz[:, None, :] + ((lower + upper) / 2.0)[:, :, None] * unit_xyz[:, None, :]
    )

    intensities = np.empty(lower.shape, dtype=volume.dtype)
    for strut in range(len(strut_junction_ids)):
        intensities[strut] = segment_maxima(
            volume,
            start_xyz[strut],
            unit_xyz[strut],
            lower[strut],
            upper[strut],
            radius,
            strut,
        )

    segment_dark = intensities < threshold
    n_dark_segments = segment_dark.sum(axis=1)
    missing = n_dark_segments == segment_dark.shape[1]

    return StrutScanResult(
        strut_junction_ids=strut_junction_ids,
        positions_xyz=positions_xyz,
        segment_centers_xyz=segment_centers_xyz,
        intensities=intensities,
        segment_dark=segment_dark,
        n_dark_segments=n_dark_segments,
        missing=missing,
        partial=(n_dark_segments > 0) & ~missing,
        threshold=threshold,
        radius=int(radius),
        cap_voxels=float(cap_voxels),
        n_bisections=int(n_bisections),
    )


def summarize_strut_scan(result: StrutScanResult, n_bands: int = 4) -> dict:
    """Reduce a strut scan to a JSON-serializable summary aimed at triage.

    Every missing or partial strut is a *candidate*; the summary says how many
    there are and where they sit, and the caller decides which of them belong to
    a systematic region by looking up their endpoint junctions in the junction
    scan. Nothing is suppressed here.

    Args:
        result: A scan to summarize.
        n_bands: Number of equal-width bands per axis.

    Returns:
        A dict of counts, per-axis band fractions for the missing and the
        partial sets separately, and a per-candidate record. Every value is a
        plain Python type, so the dict survives ``json.dumps`` unchanged.
    """
    centers = result.centers_xyz
    flagged = result.missing | result.partial
    candidates = [
        {
            "strut_id": int(s),
            "junction0": int(result.strut_junction_ids[s, 0]),
            "junction1": int(result.strut_junction_ids[s, 1]),
            "x": int(round(float(centers[s, 0]))),
            "y": int(round(float(centers[s, 1]))),
            "z": int(round(float(centers[s, 2]))),
            "status": STATUS_MISSING if result.missing[s] else STATUS_PARTIAL,
            "n_dark_segments": int(result.n_dark_segments[s]),
            "segment_dark": [bool(d) for d in result.segment_dark[s]],
            "segment_intensities": [float(v) for v in result.intensities[s]],
        }
        for s in np.flatnonzero(flagged)
    ]

    n_struts = result.n_struts
    n_missing = int(result.missing.sum())
    n_partial = int(result.partial.sum())
    return {
        "radius": result.radius,
        "cap_voxels": result.cap_voxels,
        "n_bisections": result.n_bisections,
        "n_segments_per_strut": result.n_segments,
        "otsu_threshold": result.threshold,
        "n_struts": int(n_struts),
        "n_junctions": int(len(result.positions_xyz)),
        "n_missing": n_missing,
        "missing_fraction": (n_missing / n_struts) if n_struts else None,
        "n_partial": n_partial,
        "partial_fraction": (n_partial / n_struts) if n_struts else None,
        "intensity": {
            "min": float(result.intensities.min()),
            "max": float(result.intensities.max()),
            "mean": float(result.intensities.mean()),
            "median": float(np.median(result.intensities)),
        },
        "missing_fraction_by_band": {
            name: band_stats(centers[:, column], result.missing, n_bands)
            for name, column in (("x", 0), ("y", 1), ("z", 2))
        },
        "partial_fraction_by_band": {
            name: band_stats(centers[:, column], result.partial, n_bands)
            for name, column in (("x", 0), ("y", 1), ("z", 2))
        },
        "candidates": candidates,
    }


CSV_HEADER = (
    "strut_id,junction0,junction1,x,y,z,n_dark_segments,n_segments,"
    "min_intensity,status"
)


def write_strut_csv(csv_path: str | Path, result: StrutScanResult) -> Path:
    """Write one row per strut, flagged or not.

    Every strut is written so a caller can do its own spatial reasoning -- band
    the specimen, group by endpoint junction, intersect the flagged set with a
    systematic region found at junction level -- without re-running the scan.
    """
    path = Path(csv_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    centers = result.centers_xyz
    status = result.status
    minimum = result.intensities.min(axis=1)

    with path.open("w") as f:
        f.write(CSV_HEADER + "\n")
        for strut in range(result.n_struts):
            x, y, z = (int(round(float(c))) for c in centers[strut])
            f.write(
                f"{strut},{result.strut_junction_ids[strut, 0]},"
                f"{result.strut_junction_ids[strut, 1]},{x},{y},{z},"
                f"{result.n_dark_segments[strut]},{result.n_segments},"
                f"{minimum[strut]},{status[strut]}\n"
            )
    return path
