"""Score every lattice strut against the CT volume with the same sphere probe
the junction scan uses, and classify each strut as present, partial or missing.

The measurement is deliberately the junction detector's, moved onto the strut:
take the brightest voxel within a sphere of ``radius`` around a point, and call
the point dark when that maximum falls below the *full-volume* Otsu threshold.
A strut is measured at several such points along its span, placed by recursive
bisection -- ``n_bisections`` halvings give ``2**n_bisections - 1`` points at the
fractions ``k / 2**n_bisections`` of the junction-to-junction line. One bisection
is the midpoint; two are the quarter, half and three-quarter points.

Classification follows directly from how many of those points are dark:

- **missing** -- every point dark. Nothing was found anywhere along the strut.
- **partial** -- some but not all points dark. Material is present over part of
  the span and absent over the rest, which is what a broken or partly-printed
  strut looks like. It is evidence, not a confirmed defect. At one bisection
  there is a single point, so this class cannot arise.
- **present** -- no point dark.

Why the default is one bisection
--------------------------------
Twelve struts meet at a junction of this lattice and the resulting blob stays
bright when any one of them is absent, so a probe near an endpoint answers a
question about the junction rather than the strut. Every bisection point is
strictly interior, but they are not equally safe: the outermost sit at
``1/2**n`` of the span, and the more of them there are the closer they crowd the
junctions.

On the reference scan that crowding is fatal at two bisections and comfortable
at one. Junction-local material reaches about 12 voxels out along an absent
strut's own axis (33% of probes still lit at 6 voxels, 2.3% at 10, none at 14),
while the registered lattice's residual x-drift needs a sampling radius of 7 or
more before the flagged fraction stops climbing across the specimen (28% in the
far-x band at radius 4, 0.80% at radius 8 against 0.33-0.58% elsewhere). Against
a 55.8-voxel strut:

- **one bisection** puts its single point 27.9 voxels from either junction, so a
  radius-8 sphere spans 19.9 to 35.9 along the strut and clears the junctions'
  12-voxel reach by about 8 voxels at both ends. Both demands are satisfied.
- **two bisections** put the outer points 14.0 voxels out, which would need a
  radius below about 2 to stay clear. No radius satisfies both demands.

The consequence at two bisections is that an outer point on a genuinely absent
strut is often lit by the junction it points at, so the strut reads *partial*
rather than *missing*. Measured on the 87 struts an independent cylinder
detector found wholly empty: at radius 4, 84 of 87 read all-dark; at radius 8,
only 21 do, while 60 read dark at the midpoint with exactly one outer point
bright. The midpoint was dark on all 87 at every radius -- which is the
measurement that makes one bisection the safe default.

What this statistic cannot see
------------------------------
A maximum over a sphere saturates: a strut that is present but *thin* still has
a bright voxel at every sample point and reads as present. Thin struts are a
separate defect class and this module does not detect them.

At one bisection the scan is also blind to a **partial** strut: a break that
does not cover the midpoint leaves the single sample point bright, and the strut
reads present. Recovering that sensitivity means raising ``n_bisections``, which
on this scan cannot be done without the junction bleed above. It needs a refit
registration that permits a narrower probe, not a different threshold.

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
from overlay_registered_nodes_histogram import sample_volume_at_nodes
from strut_center_intensity_histogram import load_lattice

# Matches the junction scan's default, and for the same reason: it is the
# smallest radius at which the reference scan's residual registration drift stops
# producing a gradient in the flagged fraction across the specimen. See the
# module docstring for what it costs at the outer sample points.
DEFAULT_RADIUS_VOXELS = 8
DEFAULT_BISECTIONS = 1

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
        sample_xyz: (S, P, 3) the P sample points along each strut, unrounded.
        intensities: (S, P) brightest voxel within ``radius`` of each point.
        point_dark: (S, P) True where a point's intensity fell below ``threshold``.
        n_dark_points: (S,) how many of the P points are dark.
        missing: (S,) True where every point is dark.
        partial: (S,) True where some but not all points are dark.
        threshold: The full-volume Otsu threshold that darkness was cut against.
        radius: Sampling sphere radius in voxels.
        n_bisections: Number of halvings; P is ``2**n_bisections - 1``.
    """

    strut_junction_ids: np.ndarray
    positions_xyz: np.ndarray
    sample_xyz: np.ndarray
    intensities: np.ndarray
    point_dark: np.ndarray
    n_dark_points: np.ndarray
    missing: np.ndarray
    partial: np.ndarray
    threshold: float
    radius: int
    n_bisections: int

    @property
    def n_struts(self) -> int:
        return len(self.strut_junction_ids)

    @property
    def n_points(self) -> int:
        return self.sample_xyz.shape[1]

    @property
    def centers_xyz(self) -> np.ndarray:
        """(S, 3) midpoint of each strut, which is always a sample point."""
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


def strut_sample_points(
    start_xyz: np.ndarray, end_xyz: np.ndarray, n_bisections: int
) -> np.ndarray:
    """Place ``2**n_bisections - 1`` points along each strut by recursive halving.

    One bisection splits the span once and leaves its midpoint; two split each
    half again and leave the quarter, half and three-quarter points; in general
    the points sit at ``k / 2**n_bisections`` for ``k = 1 .. 2**n - 1``. The
    endpoints are never sampled -- see the module docstring for why.

    Args:
        start_xyz: (S, 3) strut start points as [x, y, z].
        end_xyz: (S, 3) strut end points as [x, y, z].
        n_bisections: Number of halvings, at least 1.

    Returns:
        An (S, 2**n_bisections - 1, 3) array of sample points, ordered from the
        start point towards the end point.

    Raises:
        ValueError: If ``n_bisections`` is less than 1.
    """
    if n_bisections < 1:
        raise ValueError(f"n_bisections must be at least 1, got {n_bisections}")

    divisions = 2**n_bisections
    fractions = np.arange(1, divisions, dtype=float) / divisions
    return start_xyz[:, None, :] + fractions[None, :, None] * (
        end_xyz - start_xyz
    )[:, None, :]


def scan_struts(
    volume: np.ndarray,
    registered_json_path: str | Path,
    radius: int = DEFAULT_RADIUS_VOXELS,
    n_bisections: int = DEFAULT_BISECTIONS,
) -> StrutScanResult:
    """Sample every strut at its bisection points and classify each one.

    Args:
        volume: 3D CT volume in (z, y, x) order.
        registered_json_path: Lattice JSON already registered to this volume's
            voxel coordinates. Un-registered/nominal JSONs do not line up.
        radius: Sampling sphere radius in voxels. Wide enough to absorb residual
            registration drift, narrow enough that the outermost sample points
            do not reach the junction blobs.
        n_bisections: Number of halvings of each strut's span.

    Returns:
        A ``StrutScanResult``.

    Raises:
        ValueError: If ``volume`` is not 3D, ``radius`` is not positive, or
            ``n_bisections`` is less than 1.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, found {volume.ndim} dimensions")
    if radius < 1:
        raise ValueError(f"radius must be a positive number of voxels, got {radius}")

    threshold = float(threshold_otsu(volume))

    positions_xyz, strut_junction_ids, _ = load_lattice(Path(registered_json_path))
    start_xyz = positions_xyz[strut_junction_ids[:, 0]]
    end_xyz = positions_xyz[strut_junction_ids[:, 1]]
    sample_xyz = strut_sample_points(start_xyz, end_xyz, n_bisections)

    n_struts, n_points = sample_xyz.shape[:2]
    flat_intensities, _ = sample_volume_at_nodes(
        volume, sample_xyz.reshape(-1, 3), radius
    )
    intensities = flat_intensities.reshape(n_struts, n_points)

    point_dark = intensities < threshold
    n_dark_points = point_dark.sum(axis=1)
    missing = n_dark_points == n_points

    return StrutScanResult(
        strut_junction_ids=strut_junction_ids,
        positions_xyz=positions_xyz,
        sample_xyz=sample_xyz,
        intensities=intensities,
        point_dark=point_dark,
        n_dark_points=n_dark_points,
        missing=missing,
        partial=(n_dark_points > 0) & ~missing,
        threshold=threshold,
        radius=int(radius),
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
            "n_dark_points": int(result.n_dark_points[s]),
            "point_intensities": [float(v) for v in result.intensities[s]],
        }
        for s in np.flatnonzero(flagged)
    ]

    n_struts = result.n_struts
    n_missing = int(result.missing.sum())
    n_partial = int(result.partial.sum())
    return {
        "radius": result.radius,
        "n_bisections": result.n_bisections,
        "n_points_per_strut": result.n_points,
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
    "strut_id,junction0,junction1,x,y,z,n_dark_points,n_points,"
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
                f"{result.n_dark_points[strut]},{result.n_points},"
                f"{minimum[strut]},{status[strut]}\n"
            )
    return path
