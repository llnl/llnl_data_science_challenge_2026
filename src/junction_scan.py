"""Score every lattice junction against the CT volume and describe the result
well enough for a caller to tell a *systematic* failure from a *stochastic* one.

The measurement is the one validated in ``mark_junction_candidates_tiff``: merge
the JSON's co-located junction entries, take the brightest voxel within a sphere
of ``radius`` around each merged position, and call a junction dark when that
maximum falls below the *full-volume* Otsu threshold. What this module adds is
everything needed to interpret the dark set rather than just count it --
per-axis bands, octants, and connected components over the strut graph -- plus
overlays that draw the sampling sphere to scale so alignment can be judged by
eye.

Why the extra statistics exist
------------------------------
A dark junction is not by itself evidence of a defect. The same flag is produced
by a lattice that is misregistered against the scan, by a region of the specimen
that was never printed or was machined away, and by a genuinely absent junction.
Those three cases separate on *where* the dark junctions are and on *how they
respond to the radius*:

- misalignment spreads darkness broadly and shrinks sharply as the radius grows,
  because the probe eventually reaches the strut it had drifted off of;
- a missing region puts the dark junctions in one contiguous lump, which shows
  up as a large connected component confined to one band or octant;
- stochastic absence leaves isolated dark junctions -- component size one, no
  dark neighbors -- that stay dark at every radius.

``summarize_scan`` emits exactly those three discriminants. The caller decides;
this module does not classify.

A junction goes dark only when *every* incident strut is absent, and an interior
junction of this lattice has twelve. Missing single struts leave both endpoints
bright and must be found at strut level -- see ``strut_cylinder_segmentation``.

The design grid
---------------
Every scan is measured against two lattice files: the *registered* JSON, whose
coordinates are in this volume's voxel space, and the *nominal* JSON, the design
on a clean integer grid. The nominal file is required rather than optional
because registered coordinates are rotated, so no threshold on them recovers a
design layer, and a face-shaped or slab-shaped absence is only nameable in
design space. ``summarize_scan`` therefore reports a dark fraction per design
layer along each design axis, which surfaces *any* missing face on *any*
specimen -- no scan-specific geometry is built in. A caller that recognizes such
a layer as a real property of the part drops it by name through
``exclude_design_layers``.

Coordinate conventions follow the rest of the pipeline: JSON ``position`` fields
are [x, y, z] and index the volume's (z, y, x) axes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.filters import threshold_otsu

from mark_missing_junctions_tiff import mark_points, to_rgb_uint8
from overlay_registered_nodes_histogram import (
    MAX_SAMPLE_RADIUS_VOXELS,
    sample_volume_at_nodes,
)
from strut_center_intensity_histogram import load_lattice

CANDIDATE_COLOR = (255, 0, 0)

# Design coordinates are quantized before layers are compared, so that a grid
# written as 18 and one written as 17.999999 name the same layer.
_LAYER_DECIMALS = 6
_AXIS_COLUMN = {"x": 0, "y": 1, "z": 2}

# Axis 0 of the volume is z, axis 1 is y, axis 2 is x, while positions are
# [x, y, z]. Slicing along volume axis ``a`` therefore fixes position column
# ``_DEPTH_COLUMN[a]`` and leaves ``_PLOT_COLUMNS[a]`` as the image's
# (horizontal, vertical) axes.
_DEPTH_COLUMN = (2, 1, 0)
_PLOT_COLUMNS = ((0, 1), (0, 2), (1, 2))
_AXIS_LABELS = (("x", "y"), ("x", "z"), ("y", "z"))


@dataclass(frozen=True)
class JunctionScanResult:
    """Per-junction scoring of one lattice against one volume.

    Every array is indexed by *merged* junction id, which does not index the
    JSON's ``junctions`` list -- use ``entry_to_junction`` to get back to source
    entries.

    Attributes:
        positions_xyz: (N, 3) merged junction positions as [x, y, z], unrounded.
        design_xyz: (N, 3) the same junctions on the nominal design's integer
            grid, which is axis-aligned where ``positions_xyz`` is rotated.
        voxel_xyz: (N, 3) integer voxel coordinates actually sampled.
        intensities: (N,) brightest voxel within ``radius`` of each junction.
        degree: (N,) number of incident struts, 3-12 for this lattice.
        dark: (N,) True where ``intensities`` fell below ``threshold``.
        excluded: (N,) True for junctions the caller removed from scoring.
        strut_junction_ids: (S, 2) merged junction ids joined by each strut.
        entry_to_junction: (E,) merged junction id of each JSON entry.
        threshold: The full-volume Otsu threshold that ``dark`` was cut against.
        radius: Sampling sphere radius in voxels.
    """

    positions_xyz: np.ndarray
    design_xyz: np.ndarray
    voxel_xyz: np.ndarray
    intensities: np.ndarray
    degree: np.ndarray
    dark: np.ndarray
    excluded: np.ndarray
    strut_junction_ids: np.ndarray
    entry_to_junction: np.ndarray
    threshold: float
    radius: int

    @property
    def candidate(self) -> np.ndarray:
        """(N,) dark junctions that were not excluded -- the reportable flags."""
        return self.dark & ~self.excluded

    @property
    def n_junctions(self) -> int:
        return len(self.positions_xyz)


def design_positions(
    nominal_json_path: str | Path, entry_to_junction: np.ndarray, n_junctions: int
) -> np.ndarray:
    """Carry the nominal design's coordinates onto merged junctions.

    The nominal JSON is read per *entry*, matching the registered file's own
    entry list, and scattered onto merged junctions through ``entry_to_junction``.
    Co-located entries share a nominal position, so the scatter is unambiguous --
    and that is checked, because a nominal file paired with the wrong registered
    file would otherwise silently produce a plausible-looking design grid.

    Args:
        nominal_json_path: The design lattice JSON, on a clean integer grid.
        entry_to_junction: Entry-to-merged-junction map from ``load_lattice``.
        n_junctions: Number of merged junctions.

    Returns:
        (N, 3) design coordinates as [x, y, z], one row per merged junction.

    Raises:
        ValueError: If the nominal JSON does not describe the same lattice,
            entry for entry, as the registered one.
    """
    path = Path(nominal_json_path)
    with path.open() as f:
        data = json.load(f)

    junctions = data["junctions"]
    n_entries = len(entry_to_junction)
    if len(junctions) != n_entries:
        raise ValueError(
            f"{path.name} has {len(junctions)} junction entries but the registered "
            f"lattice has {n_entries}; the nominal and registered files must "
            "describe the same lattice entry for entry"
        )
    if [junction["id"] for junction in junctions] != list(range(n_entries)):
        raise ValueError(f"junction ids in {path.name} are not 0..N-1 in list order")

    entry_positions = np.array(
        [junction["position"] for junction in junctions], dtype=float
    )
    design = np.zeros((n_junctions, 3), dtype=float)
    design[entry_to_junction] = entry_positions
    if not np.allclose(design[entry_to_junction], entry_positions):
        raise ValueError(
            f"{path.name} gives different design positions to entries that the "
            "registered lattice places at one junction; the two files are not the "
            "same lattice"
        )
    return design


def design_layer_values(design_xyz: np.ndarray, axis: str) -> np.ndarray:
    """The distinct design-grid coordinates along one axis, ascending."""
    return np.unique(np.round(design_xyz[:, _AXIS_COLUMN[axis]], _LAYER_DECIMALS))


def design_layer_mask(design_xyz: np.ndarray, spec: str) -> np.ndarray:
    """Select one design-grid layer, named as ``"y=max"``, ``"y=min"`` or ``"y=18"``.

    Layers are named in design space rather than by voxel coordinate because the
    registered coordinates are rotated -- an axis-aligned face of the part is not
    an axis-aligned set of registered positions.

    Raises:
        ValueError: On a malformed spec, an unknown axis, or a numeric value
            that names no layer of this lattice.
    """
    axis, separator, value = spec.partition("=")
    axis = axis.strip().lower()
    value = value.strip().lower()
    if not separator or axis not in _AXIS_COLUMN or not value:
        raise ValueError(
            f"design layer {spec!r} must look like 'y=max', 'y=min' or 'y=18', "
            "with an axis of x, y or z"
        )

    layers = design_layer_values(design_xyz, axis)
    if value == "max":
        target = float(layers[-1])
    elif value == "min":
        target = float(layers[0])
    else:
        try:
            requested = float(value)
        except ValueError:
            raise ValueError(
                f"design layer {spec!r} must name a coordinate, 'max' or 'min', "
                f"not {value!r}"
            ) from None
        target = float(layers[np.abs(layers - requested).argmin()])
        # Silently excluding nothing would look identical to a clean result, so
        # a value that names no layer is an error rather than an empty mask.
        if abs(target - requested) > 0.5:
            shown = ", ".join(f"{layer:g}" for layer in layers[:12])
            raise ValueError(
                f"no design layer at {axis}={requested:g}; this lattice has "
                f"{len(layers)} layers on that axis ({shown}"
                f"{', ...' if len(layers) > 12 else ''})"
            )

    coordinate = np.round(design_xyz[:, _AXIS_COLUMN[axis]], _LAYER_DECIMALS)
    return coordinate == target


def scan_junctions(
    volume: np.ndarray,
    registered_json_path: str | Path,
    nominal_json_path: str | Path,
    radius: int = MAX_SAMPLE_RADIUS_VOXELS,
    exclude_design_layers: Iterable[str] = (),
    exclude_junction_ids: Iterable[int] = (),
) -> JunctionScanResult:
    """Sample the volume at every merged junction and flag the dark ones.

    Args:
        volume: 3D CT volume in (z, y, x) order.
        registered_json_path: Lattice JSON already registered to this volume's
            voxel coordinates. Un-registered/nominal JSONs do not line up.
        nominal_json_path: The same lattice as designed, on a clean integer
            grid. Required: it is what makes a missing face or slab nameable,
            and it must list the same entries in the same order.
        radius: Sampling sphere radius in voxels. This is the free parameter
            that absorbs residual registration drift; see the module docstring
            of ``mark_junction_candidates_tiff`` for what too small looks like.
        exclude_design_layers: Design-grid layers to keep out of the candidate
            list, each named as ``"y=max"``, ``"y=min"`` or ``"y=18"``. Use this
            for a face the caller has established is absent from the *part*, not
            from the scan. Excluded junctions are still scored and reported.
        exclude_junction_ids: Individual merged junction ids to exclude.

    Returns:
        A ``JunctionScanResult``.

    Raises:
        ValueError: If ``radius`` is not positive, if ``volume`` is not 3D, if
            the nominal lattice does not match the registered one, if a layer
            spec is malformed, or if an excluded id is out of range.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, found {volume.ndim} dimensions")
    if radius < 1:
        raise ValueError(f"radius must be a positive number of voxels, got {radius}")

    threshold = float(threshold_otsu(volume))

    positions_xyz, strut_junction_ids, entry_to_junction = load_lattice(
        Path(registered_json_path)
    )
    n_junctions = len(positions_xyz)
    degree = np.bincount(strut_junction_ids.ravel(), minlength=n_junctions)
    intensities, voxel_xyz = sample_volume_at_nodes(volume, positions_xyz, radius)
    design_xyz = design_positions(nominal_json_path, entry_to_junction, n_junctions)

    excluded = np.zeros(n_junctions, dtype=bool)
    for spec in exclude_design_layers:
        excluded |= design_layer_mask(design_xyz, spec)

    extra = np.asarray(list(exclude_junction_ids), dtype=int)
    if extra.size:
        if extra.min() < 0 or extra.max() >= n_junctions:
            raise ValueError(
                f"exclude_junction_ids must lie in 0..{n_junctions - 1}, got "
                f"{extra.min()}..{extra.max()}"
            )
        excluded[extra] = True

    return JunctionScanResult(
        positions_xyz=positions_xyz,
        design_xyz=design_xyz,
        voxel_xyz=voxel_xyz,
        intensities=intensities,
        degree=degree,
        dark=intensities < threshold,
        excluded=excluded,
        strut_junction_ids=strut_junction_ids,
        entry_to_junction=entry_to_junction,
        threshold=threshold,
        radius=int(radius),
    )


def dark_neighbor_counts(
    strut_junction_ids: np.ndarray, dark: np.ndarray, n_junctions: int
) -> np.ndarray:
    """Count, per junction, the incident struts whose far endpoint is dark.

    Zero for an isolated dark junction, high for one inside a dark region. This
    is the cheapest signal separating stochastic absence from a missing swath.
    """
    counts = np.zeros(n_junctions, dtype=int)
    first, second = strut_junction_ids[:, 0], strut_junction_ids[:, 1]
    np.add.at(counts, first, dark[second].astype(int))
    np.add.at(counts, second, dark[first].astype(int))
    return counts


def dark_component_sizes(
    strut_junction_ids: np.ndarray, dark: np.ndarray, n_junctions: int
) -> np.ndarray:
    """Size of each dark junction's connected component within the dark subgraph.

    Only struts with both endpoints dark are edges, so components never mix dark
    and bright junctions. Bright junctions get 0.

    A component of size 1 is a lone junction whose neighbors are all intact --
    the stochastic signature. A large component is a contiguous absent region: a
    machined face, an unprinted corner, or a lattice extending past the scan.
    """
    sizes = np.zeros(n_junctions, dtype=int)
    if not dark.any():
        return sizes

    both_dark = dark[strut_junction_ids[:, 0]] & dark[strut_junction_ids[:, 1]]
    edges = strut_junction_ids[both_dark]
    graph = coo_matrix(
        (
            np.ones(len(edges), dtype=np.int8),
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(n_junctions, n_junctions),
    )
    n_components, labels = connected_components(graph, directed=False)
    # Bright junctions are isolated in this graph, so each sits in a singleton
    # label and counting only dark members leaves their labels at zero.
    member_counts = np.bincount(labels[dark], minlength=n_components)
    sizes[dark] = member_counts[labels[dark]]
    return sizes


def _band_stats(
    coordinate: np.ndarray, scored: np.ndarray, flagged: np.ndarray, n_bands: int
) -> list[dict]:
    """Split scored junctions into equal-width bands along one coordinate.

    Bands are equal *width*, not equal count, because the questions being asked
    of them are geometric -- "is one side of the specimen dark" -- and because
    a lattice puts many junctions at identical coordinates, which collapses
    quantile edges.
    """
    values = coordinate[scored]
    if values.size == 0:
        return []

    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, n_bands + 1)
    # ``np.digitize`` puts the maximum in a band of its own; fold it back.
    index = np.clip(np.digitize(coordinate, edges[1:-1]), 0, n_bands - 1)

    bands = []
    for band in range(n_bands):
        in_band = scored & (index == band)
        n = int(in_band.sum())
        n_flagged = int((in_band & flagged).sum())
        bands.append(
            {
                "lo": float(edges[band]),
                "hi": float(edges[band + 1]),
                "n_scored": n,
                "n_dark": n_flagged,
                "dark_fraction": (n_flagged / n) if n else None,
            }
        )
    return bands


def _design_layer_stats(
    design_xyz: np.ndarray, axis: str, scored: np.ndarray, flagged: np.ndarray
) -> list[dict]:
    """One record per distinct design-grid layer along one design axis.

    Bands over registered coordinates answer "is one side of the volume dark";
    these answer the sharper question "is one *layer of the design* absent". A
    machined-off face, an unprinted end, or a slab of lattice that extends past
    the scanned field all appear here as a single layer at or near a dark
    fraction of 1, whichever face of whichever specimen they land on.
    """
    coordinate = np.round(design_xyz[:, _AXIS_COLUMN[axis]], _LAYER_DECIMALS)
    layers = []
    for value in np.unique(coordinate):
        in_layer = scored & (coordinate == value)
        n = int(in_layer.sum())
        n_dark = int((in_layer & flagged).sum())
        layers.append(
            {
                "layer": f"{axis}={value:g}",
                "value": float(value),
                "n_scored": n,
                "n_dark": n_dark,
                "dark_fraction": (n_dark / n) if n else None,
            }
        )
    return layers


def summarize_scan(result: JunctionScanResult, n_bands: int = 4) -> dict:
    """Reduce a scan to a JSON-serializable summary aimed at triage.

    The spatial fields all describe *non-excluded* junctions, with non-excluded
    dark junctions (``result.candidate``) as the numerator, so the same summary
    reads correctly before and after a systematic region is excluded.

    Args:
        result: A scan to summarize.
        n_bands: Number of equal-width bands per axis.

    Returns:
        A dict of counts, per-axis band fractions, per-design-layer fractions,
        octant fractions, dark connected-component sizes, and a per-candidate
        record. Every value is a plain Python type, so the dict survives
        ``json.dumps`` unchanged.
    """
    scored = ~result.excluded
    candidate = result.candidate
    neighbors = dark_neighbor_counts(
        result.strut_junction_ids, candidate, result.n_junctions
    )
    components = dark_component_sizes(
        result.strut_junction_ids, candidate, result.n_junctions
    )

    candidate_ids = np.flatnonzero(candidate)
    candidates = [
        {
            "junction_id": int(j),
            "x": int(result.voxel_xyz[j, 0]),
            "y": int(result.voxel_xyz[j, 1]),
            "z": int(result.voxel_xyz[j, 2]),
            "intensity": float(result.intensities[j]),
            "degree": int(result.degree[j]),
            "dark_neighbor_count": int(neighbors[j]),
            "component_size": int(components[j]),
        }
        for j in candidate_ids
    ]

    # ``components`` counts membership: a component of size k puts the value k
    # on each of its k junctions. Dividing back out turns membership counts into
    # component counts.
    # ``size_histogram`` is a list rather than a size-keyed dict because
    # ``json.dumps`` silently stringifies integer keys, so a dict would not
    # survive the round trip through the summary file an agent reads back.
    component_sizes = components[candidate]
    unique_sizes, member_counts = np.unique(component_sizes, return_counts=True)
    size_histogram = [
        {"size": int(size), "count": int(count // size)}
        for size, count in zip(unique_sizes, member_counts)
    ]

    centers = {}
    for axis_name, column in (("x", 0), ("y", 1), ("z", 2)):
        values = result.positions_xyz[scored, column]
        centers[axis_name] = (
            float((values.min() + values.max()) / 2.0) if values.size else 0.0
        )
    octant_index = sum(
        (result.positions_xyz[:, column] > centers[name]).astype(int) << bit
        for bit, (name, column) in enumerate((("x", 0), ("y", 1), ("z", 2)))
    )
    octants = []
    for cell in range(8):
        in_cell = scored & (octant_index == cell)
        n = int(in_cell.sum())
        n_dark = int((in_cell & candidate).sum())
        octants.append(
            {
                "octant": f"{'+' if cell & 1 else '-'}x"
                f"{'+' if cell & 2 else '-'}y"
                f"{'+' if cell & 4 else '-'}z",
                "n_scored": n,
                "n_dark": n_dark,
                "dark_fraction": (n_dark / n) if n else None,
            }
        )

    n_scored = int(scored.sum())
    return {
        "radius": result.radius,
        "otsu_threshold": result.threshold,
        "n_entries": int(len(result.entry_to_junction)),
        "n_junctions": int(result.n_junctions),
        "n_struts": int(len(result.strut_junction_ids)),
        "degree_min": int(result.degree.min()),
        "degree_max": int(result.degree.max()),
        "n_dark": int(result.dark.sum()),
        "n_excluded": int(result.excluded.sum()),
        "n_dark_excluded": int((result.dark & result.excluded).sum()),
        "n_scored": n_scored,
        "n_candidates": int(candidate.sum()),
        "candidate_fraction": (int(candidate.sum()) / n_scored) if n_scored else None,
        "intensity": {
            "min": float(result.intensities.min()),
            "max": float(result.intensities.max()),
            "mean": float(result.intensities.mean()),
            "median": float(np.median(result.intensities)),
        },
        "dark_fraction_by_band": {
            name: _band_stats(
                result.positions_xyz[:, column], scored, candidate, n_bands
            )
            for name, column in (("x", 0), ("y", 1), ("z", 2))
        },
        "dark_fraction_by_design_layer": {
            axis: _design_layer_stats(result.design_xyz, axis, scored, candidate)
            for axis in ("x", "y", "z")
        },
        "dark_fraction_by_octant": octants,
        "dark_components": {
            "n_components": int(sum(entry["count"] for entry in size_histogram)),
            "largest": int(component_sizes.max()) if candidate.any() else 0,
            "size_histogram": size_histogram,
        },
        "candidates": candidates,
    }


CSV_HEADER = (
    "junction_id,x,y,z,design_x,design_y,design_z,intensity,degree,dark,excluded,"
    "dark_neighbor_count,component_size,entry_ids"
)


def write_junction_csv(csv_path: str | Path, result: JunctionScanResult) -> Path:
    """Write one row per merged junction, flagged or not.

    Every junction is written, not just the candidates, so a caller can do its
    own spatial reasoning -- re-band the specimen, group by design layer,
    correlate with degree, check whether a neighborhood is uniformly dim --
    without re-running the scan.

    Merged ids do not index the JSON's ``junctions`` array, so each row also
    carries the source entry ids it was built from.
    """
    path = Path(csv_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    candidate = result.candidate
    neighbors = dark_neighbor_counts(
        result.strut_junction_ids, candidate, result.n_junctions
    )
    components = dark_component_sizes(
        result.strut_junction_ids, candidate, result.n_junctions
    )
    entries_by_junction: list[list[int]] = [[] for _ in range(result.n_junctions)]
    for entry, junction in enumerate(result.entry_to_junction):
        entries_by_junction[junction].append(entry)

    with path.open("w") as f:
        f.write(CSV_HEADER + "\n")
        for junction in range(result.n_junctions):
            x, y, z = result.voxel_xyz[junction]
            design = ",".join(f"{c:g}" for c in result.design_xyz[junction])
            entries = " ".join(str(e) for e in entries_by_junction[junction])
            f.write(
                f"{junction},{x},{y},{z},{design},{result.intensities[junction]},"
                f"{result.degree[junction]},{int(result.dark[junction])},"
                f"{int(result.excluded[junction])},{neighbors[junction]},"
                f"{components[junction]},{entries}\n"
            )
    return path


def save_slice_overlay_with_radius(
    volume: np.ndarray,
    result: JunctionScanResult,
    output_path: str | Path,
    axis: int = 0,
    index: int | None = None,
) -> int:
    """Draw one slice with each nearby junction's sampling sphere to scale.

    A junction ``d`` voxels off the slice plane intersects it in a circle of
    radius ``sqrt(radius^2 - d^2)``, and that is what is drawn -- so the picture
    shows exactly the voxels the flag was computed from. Circles are colored
    green for bright, red for a candidate, and orange for a dark junction that
    was excluded.

    Args:
        volume: 3D CT volume in (z, y, x) order.
        result: The scan whose junctions and radius are drawn.
        output_path: Destination image; the suffix picks the format.
        axis: Volume axis to slice along (0 = z, 1 = y, 2 = x).
        index: Slice index, or None for the median junction coordinate.

    Returns:
        The slice index used.

    Raises:
        ValueError: On an invalid axis or an out-of-range index.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    depth_column = _DEPTH_COLUMN[axis]
    if index is None:
        index = int(np.median(result.voxel_xyz[:, depth_column]))
    if index < 0 or index >= volume.shape[axis]:
        raise ValueError(
            f"index must be between 0 and {volume.shape[axis] - 1} for axis {axis}"
        )

    horizontal, vertical = _PLOT_COLUMNS[axis]
    depth = np.abs(result.positions_xyz[:, depth_column] - index)
    visible = depth <= result.radius
    in_plane_radius = np.sqrt(np.maximum(result.radius**2 - depth**2, 0.0))

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(10, 10))
    try:
        axes.imshow(np.take(volume, index, axis=axis), cmap="gray")
        groups = (
            (visible & ~result.dark, "lime", "junction present"),
            (visible & result.dark & result.excluded, "orange", "dark, excluded"),
            (visible & result.candidate, "red", "candidate"),
        )
        for mask, color, label in groups:
            for junction in np.flatnonzero(mask):
                axes.add_patch(
                    Circle(
                        (
                            result.positions_xyz[junction, horizontal],
                            result.positions_xyz[junction, vertical],
                        ),
                        float(in_plane_radius[junction]),
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.0,
                    )
                )
            if mask.any():
                axes.plot([], [], color=color, label=f"{label} ({int(mask.sum())})")
        if visible.any():
            axes.legend(loc="upper right")
        horizontal_label, vertical_label = _AXIS_LABELS[axis]
        axes.set_title(
            f"Slice {'zyx'[axis]}={index} with radius-{result.radius} "
            f"sampling spheres to scale"
        )
        axes.set_xlabel(f"{horizontal_label} (voxels)")
        axes.set_ylabel(f"{vertical_label} (voxels)")
        figure.tight_layout()
        figure.savefig(path, dpi=200)
    finally:
        plt.close(figure)
    return index


def save_mip_overlay_with_status(
    volume: np.ndarray,
    result: JunctionScanResult,
    output_path: str | Path,
    axis: int = 0,
) -> Path:
    """Project the volume and scatter every junction, highlighting the dark ones.

    This is the coarse-alignment view: if the faint cloud of junction markers
    does not sit on the projected lattice, the registered JSON does not belong
    to this scan and no per-junction number from it means anything.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    horizontal, vertical = _PLOT_COLUMNS[axis]
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(10, 10))
    try:
        axes.imshow(volume.max(axis=axis), cmap="gray")
        present = ~result.dark
        axes.scatter(
            result.positions_xyz[present, horizontal],
            result.positions_xyz[present, vertical],
            s=2,
            c="lime",
            alpha=0.4,
            linewidths=0,
            label=f"junction present ({int(present.sum())})",
        )
        for mask, color, label in (
            (result.dark & result.excluded, "orange", "dark, excluded"),
            (result.candidate, "red", "candidate"),
        ):
            if mask.any():
                axes.scatter(
                    result.positions_xyz[mask, horizontal],
                    result.positions_xyz[mask, vertical],
                    s=30,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1,
                    label=f"{label} ({int(mask.sum())})",
                )
        axes.legend(loc="upper right")
        horizontal_label, vertical_label = _AXIS_LABELS[axis]
        axes.set_title(
            f"{'zyx'[axis]} max-intensity projection with junctions overlaid "
            f"(radius {result.radius})"
        )
        axes.set_xlabel(f"{horizontal_label} (voxels)")
        axes.set_ylabel(f"{vertical_label} (voxels)")
        figure.tight_layout()
        figure.savefig(path, dpi=200)
    finally:
        plt.close(figure)
    return path


def save_marked_tiff(
    volume: np.ndarray, result: JunctionScanResult, output_path: str | Path
) -> Path:
    """Burn a red sphere into each candidate's location in an RGB TIFF stack.

    Markers are drawn at the sampling radius, so each painted sphere is exactly
    the neighborhood whose brightest voxel produced the flag. A marker that
    swallowed more than was measured would invite reading absence into voxels
    the detector never looked at.

    The RGB copy is three uint8 channels of a full volume, so this costs roughly
    three bytes per input voxel in peak memory.
    """
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb_volume = to_rgb_uint8(volume)
    mark_points(
        rgb_volume,
        result.voxel_xyz[result.candidate],
        result.radius,
        CANDIDATE_COLOR,
    )
    tifffile.imwrite(path, rgb_volume, photometric="rgb", bigtiff=True)
    return path


def parse_junction_ids(text: str) -> Sequence[int]:
    """Parse a comma- or whitespace-separated list of junction ids.

    Empty and whitespace-only input yields an empty sequence, so a caller can
    pass through an unset optional argument unchanged.
    """
    tokens = text.replace(",", " ").split()
    return [int(token) for token in tokens]


def parse_design_layers(text: str) -> Sequence[str]:
    """Parse a comma- or whitespace-separated list of design layer specs.

    Empty and whitespace-only input yields an empty sequence, so a caller can
    pass through an unset optional argument unchanged. Individual specs are not
    validated here -- ``design_layer_mask`` does that against a real lattice, so
    that an unknown layer is reported with the layers that do exist.
    """
    tokens = text.replace(",", " ").split()
    return [token for token in tokens if token]
