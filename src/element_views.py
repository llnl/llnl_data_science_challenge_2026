"""Render one junction or one strut as three orthogonal projections of the
voxels its flag was actually computed from.

The whole-slice overlays in ``junction_scan`` answer questions about the
specimen: is the lattice aligned, where does the dark set concentrate. They
cannot answer questions about a *single* element, because one plane through a
55-voxel strut shows a few voxels of it and a candidate can hide between slices.

This module crops the element's own probe region -- the sphere for a junction,
the capped cylinder for a strut -- pads it, and projects it along z, y and x.
Three images of a small box, with the probe drawn to scale on each, are enough
to see whether material is there and whether the probe was pointed at it. That
is the view the parameter sweep is judged from: at a good radius the cylinder's
band covers the strut despite the residual drift, and at a good cap the scored
span visibly stops short of both junction blobs.

Projections are maxima, matching the statistic being illustrated. A max
projection of a box containing material shows that material wherever it sits
along the projection axis, so an element reading dark in all three views is dark
through the whole box, not merely in one plane.

Coordinate conventions follow the rest of the pipeline: JSON ``position`` fields
are [x, y, z] and index the volume's (z, y, x) axes. Axes are drawn in the
volume's own voxel coordinates, not the crop's, so a position read off one of
these images can be handed straight back to another tool.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from junction_scan import AXIS_LABELS, PLOT_COLUMNS

# Room around the probe so the element is seen in context -- for a strut, enough
# to show both junction blobs the caps have to clear.
DEFAULT_PADDING_VOXELS = 6

DARK_COLOR = "red"
BRIGHT_COLOR = "lime"
AXIS_NAMES = ("z", "y", "x")


def _crop(
    volume: np.ndarray, lo_xyz: np.ndarray, hi_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip an [x, y, z] box to the volume and return it with its actual bounds."""
    shape_xyz = np.array(volume.shape[::-1])
    lo = np.clip(np.floor(lo_xyz), 0, shape_xyz - 1).astype(int)
    hi = np.clip(np.ceil(hi_xyz) + 1, 1, shape_xyz).astype(int)
    hi = np.maximum(hi, lo + 1)
    box = volume[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
    return box, lo, hi


def _projection_figure(
    box: np.ndarray, lo_xyz: np.ndarray, hi_xyz: np.ndarray, axis: int
):
    """Open a figure showing ``box`` projected along ``axis`` in volume coordinates.

    Returns (figure, axes). The caller draws the probe on top and saves.
    """
    horizontal, vertical = PLOT_COLUMNS[axis]
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.imshow(
        box.max(axis=axis),
        cmap="gray",
        extent=(
            lo_xyz[horizontal] - 0.5,
            hi_xyz[horizontal] - 0.5,
            hi_xyz[vertical] - 0.5,
            lo_xyz[vertical] - 0.5,
        ),
        interpolation="nearest",
    )
    horizontal_label, vertical_label = AXIS_LABELS[axis]
    axes.set_xlabel(f"{horizontal_label} (voxels)")
    axes.set_ylabel(f"{vertical_label} (voxels)")
    return figure, axes


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    # The crop is rarely square and ``imshow`` keeps voxels square, so without
    # trimming most of a strut view would be margin.
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_junction_views(
    volume: np.ndarray,
    position_xyz: np.ndarray,
    radius: int,
    output_dir: str | Path,
    stem: str,
    dark: bool | None = None,
    padding: int = DEFAULT_PADDING_VOXELS,
) -> list[Path]:
    """Save three projections of one junction's sampling sphere.

    A sphere projects to a circle of the same radius along every direction, so
    the drawn circle is exactly the probe in all three views.

    Args:
        volume: 3D CT volume in (z, y, x) order.
        position_xyz: The junction's [x, y, z] position, unrounded.
        radius: Sampling sphere radius in voxels.
        output_dir: Directory for the images.
        stem: Filename stem; each image gets a ``_proj_<axis>`` suffix.
        dark: Colors the circle red when True and green when False. None draws
            it in a neutral color, for when the junction has not been scored.
        padding: Extra voxels around the sphere, so the neighbourhood is visible.

    Returns:
        The three saved paths, in z, y, x projection order.
    """
    position = np.asarray(position_xyz, dtype=float)
    reach = radius + padding
    box, lo, hi = _crop(volume, position - reach, position + reach)

    color = BRIGHT_COLOR if dark is False else DARK_COLOR if dark else "deepskyblue"
    directory = Path(output_dir).expanduser()
    paths = []
    for axis in range(3):
        horizontal, vertical = PLOT_COLUMNS[axis]
        figure, axes = _projection_figure(box, lo, hi, axis)
        axes.add_patch(
            Circle(
                (position[horizontal], position[vertical]),
                float(radius),
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
            )
        )
        axes.set_title(
            f"{stem}: max projection along {AXIS_NAMES[axis]}, "
            f"sampling sphere r={radius}"
        )
        paths.append(_save(figure, directory / f"{stem}_proj_{AXIS_NAMES[axis]}.png"))
    return paths


def save_strut_views(
    volume: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    radius: int,
    cap_voxels: float,
    lower: np.ndarray,
    upper: np.ndarray,
    output_dir: str | Path,
    stem: str,
    segment_dark: np.ndarray | None = None,
    padding: int = DEFAULT_PADDING_VOXELS,
) -> list[Path]:
    """Save three projections of one strut's capped cylinder.

    The full junction-to-junction axis is drawn faintly and the *scored* span is
    drawn on top of it, one colored run per segment, with ticks at the segment
    boundaries. Both junction blobs stay inside the crop, which is the point: the
    picture answers whether the caps clear them.

    The radius is perpendicular to the strut axis in 3D, so the dashed band drawn
    at +/- radius about the projected axis is the probe's true width only when
    the axis lies in the projection plane, and an over-estimate otherwise. It is
    drawn for scale, and the two other projections cover the direction it
    foreshortens.

    Args:
        volume: 3D CT volume in (z, y, x) order.
        start_xyz: First endpoint junction's [x, y, z] position.
        end_xyz: Second endpoint junction's [x, y, z] position.
        radius: Cylinder radius in voxels.
        cap_voxels: Length excluded at each end, drawn as the gap between the
            axis end and the start of the scored span.
        lower: (P,) axial start of each segment, from ``segment_bounds``.
        upper: (P,) axial end of each segment.
        output_dir: Directory for the images.
        stem: Filename stem; each image gets a ``_proj_<axis>`` suffix.
        segment_dark: (P,) per-segment darkness, coloring each drawn run red or
            green. None draws every run in a neutral color.
        padding: Extra voxels around the strut's bounding box.

    Returns:
        The three saved paths, in z, y, x projection order.
    """
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length == 0:
        raise ValueError("strut endpoints are coincident")
    unit = direction / length

    reach = radius + padding
    box, lo, hi = _crop(
        volume, np.minimum(start, end) - reach, np.maximum(start, end) + reach
    )

    directory = Path(output_dir).expanduser()
    paths = []
    for axis in range(3):
        horizontal, vertical = PLOT_COLUMNS[axis]
        figure, axes = _projection_figure(box, lo, hi, axis)

        # The full nominal span, so the excluded caps are visible as the part of
        # this line no colored segment covers.
        axes.plot(
            [start[horizontal], end[horizontal]],
            [start[vertical], end[vertical]],
            color="deepskyblue",
            linewidth=0.8,
            linestyle=":",
        )

        in_plane = np.array([unit[horizontal], unit[vertical]])
        norm = float(np.linalg.norm(in_plane))
        if norm > 0:
            perpendicular = np.array([-in_plane[1], in_plane[0]]) / norm * radius
            for sign in (1, -1):
                axes.plot(
                    [
                        start[horizontal] + sign * perpendicular[0],
                        end[horizontal] + sign * perpendicular[0],
                    ],
                    [
                        start[vertical] + sign * perpendicular[1],
                        end[vertical] + sign * perpendicular[1],
                    ],
                    color="deepskyblue",
                    linewidth=0.6,
                    linestyle="--",
                    alpha=0.7,
                )

        for segment, (low, high) in enumerate(zip(lower, upper)):
            first = start + low * unit
            last = start + high * unit
            color = (
                "deepskyblue"
                if segment_dark is None
                else (DARK_COLOR if segment_dark[segment] else BRIGHT_COLOR)
            )
            axes.plot(
                [first[horizontal], last[horizontal]],
                [first[vertical], last[vertical]],
                color=color,
                linewidth=2.5,
                solid_capstyle="butt",
            )
            for edge in (first, last):
                axes.plot(
                    edge[horizontal],
                    edge[vertical],
                    marker="|" if abs(unit[horizontal]) > abs(unit[vertical]) else "_",
                    color="white",
                    markersize=8,
                    markeredgewidth=1.5,
                )

        axes.set_title(
            f"{stem}: max projection along {AXIS_NAMES[axis]}\n"
            f"r={radius}, cap={cap_voxels:g}, {len(lower)} segment(s) of "
            f"{length:.1f}-voxel span"
        )
        paths.append(_save(figure, directory / f"{stem}_proj_{AXIS_NAMES[axis]}.png"))
    return paths
