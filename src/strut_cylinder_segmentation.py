"""Detect missing struts by segmenting the CT volume and counting material
inside a cylinder drawn along each nominal strut.

The earlier midpoint probe (``strut_center_intensity_histogram``) samples a
single small neighborhood at each strut's center, which is too fragile for a
~4-voxel-thick strut. This module replaces the point probe with a cylinder
spanning the full junction-to-junction line.

The statistic matters as much as the geometry. A cylinder's *mean* intensity is
a mixture of solid (~50000) and background (~32400), so a healthy strut averages
near the full-volume Otsu threshold itself and the flagged fraction tracks the
chosen radius rather than the specimen. Segmenting first avoids this: the volume
is binarized once at the Otsu threshold, and each cylinder is scored by how many
segmented voxels it contains.

The headline number is that count normalized by strut length,

    area = n_segmented / L

which is the strut's mean cross-sectional area in voxels^2. Normalizing this way
makes the measurement comparable across radii and directly comparable to the
nominal design cross-section (see ``NOMINAL_CROSS_SECTION_VOXELS``).

Coordinate conventions match ``overlay_registered_nodes_histogram``: JSON
``position`` fields are [x, y, z] and index the TIFF's (z, y, x) axes. Cylinders
are built from the registered JSON as supplied, with no affine refit, so the
residual registration drift documented in that analysis still applies here.

Machined-off bottom face
------------------------
The specimen's bottom face was cut or sanded off after printing: inspecting any
mid-stack slice shows the lattice terminating in a sawtooth around y~720, with
no material at all where the nominal design's last junction row sits (y~760).
Those junctions exist in the design JSON but not in the physical part, so the
struts touching them are excluded by default (``--keep-bottom-layer`` disables
this). The face is identified in the *nominal* design JSON, whose Y axis maps to
the registered y with r = 1.0000 and which is on a clean integer grid; the
registered coordinates are rotated and so cannot be layered by thresholding y.

Before the exclusion this face accounted for 324 of 325 flagged struts and all
291 empty cylinders at radius 8, while every other layer flagged 0.00% -- i.e.
it was the entire signal, and none of it was a defect.
"""

from pathlib import Path

import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from overlay_registered_nodes_histogram import (
    NPY_PATH,
    REGISTERED_JSON_PATH,
    TIFF_PATH,
)
from strut_center_intensity_histogram import junction_missing_mask, load_lattice
from mark_missing_junctions_tiff import to_rgb_uint8
from volume_io import convert_tiff_to_numpy

OUTPUT_DIR = Path("outputs/missing_struts/strut_cylinders")

# The nominal design lattice, on a clean 0..18 integer grid. Its junction ids
# are in the same order as the registered JSON's, so a mask built here applies
# directly to the registered analysis.
NOMINAL_JSON_PATH = Path("data/missing_struts/octet_truss_9x9x9.json")

# Wider cylinders tolerate the residual junction-alignment error between the
# JSON and the TIFF, so the sweep runs well past the strut's own thickness.
DEFAULT_RADII = (2.0, 3.0, 5.0, 7.0, 8.0)
DEFAULT_TIFF_RADIUS = 8.0

# Struts are 0.1 design units thick and the registered lattice runs at
# ~39.5 voxels per design unit, giving a 3.95-voxel diameter. A fully printed
# strut should therefore present roughly this cross-sectional area.
NOMINAL_CROSS_SECTION_VOXELS = np.pi * (0.1 * 39.5 / 2) ** 2

# A strut counts as a missing candidate below this fraction of the nominal
# design cross-section. Otsu is unusable for this cut: the two modes are wildly
# unbalanced (~3% empty vs ~97% present), which drags the between-class variance
# optimum into the middle of the dominant mode and splits the healthy population
# roughly in half. Anchoring to the design geometry instead puts the cut in the
# measured trough of the area distribution (3.0-3.7 vox^2 at radius 3).
DEFAULT_AREA_FRACTION = 0.25

MISSING_COLOR = np.array([255, 0, 0], dtype=np.float32)
PRESENT_COLOR = np.array([0, 255, 0], dtype=np.float32)
MISSING_ALPHA = 0.85
PRESENT_ALPHA = 0.4


def cylinder_mask(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    shape: tuple[int, int, int],
    end_trim: float = 0.0,
) -> tuple[tuple[int, int, int, int, int, int], np.ndarray]:
    """Rasterize the cylinder of ``radius`` swept from ``p0`` to ``p1``.

    Args:
        p0: Start point as [x, y, z] in voxel coordinates.
        p1: End point as [x, y, z] in voxel coordinates.
        radius: Cylinder radius in voxels.
        shape: Volume shape as (z, y, x); the mask is clipped to it.
        end_trim: Fraction of the span dropped at each end. Twelve struts meet
            at a junction and the resulting blob stays bright when any one of
            them is absent, so an untrimmed cylinder cannot register an interior
            strut as empty. Trimming restricts the measurement to the midspan.

    Returns:
        ``((z0, z1, y0, y1, x0, x1), mask)`` such that
        ``volume[z0:z1, y0:y1, x0:x1][mask]`` selects the cylinder's voxels.
        A voxel is inside when its perpendicular distance to the axis is at
        most ``radius`` and its projection onto the axis lies within the span.
    """
    if not 0.0 <= end_trim < 0.5:
        raise ValueError(f"end_trim must be in [0, 0.5), got {end_trim}")

    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length == 0:
        raise ValueError("degenerate strut with coincident endpoints")
    unit = axis / length

    lower = np.floor(np.minimum(p0, p1) - radius).astype(int)
    upper = np.ceil(np.maximum(p0, p1) + radius).astype(int) + 1
    x0, y0, z0 = np.maximum(lower, 0)
    x1, y1, z1 = np.minimum(upper, np.array(shape)[::-1])

    zz, yy, xx = np.meshgrid(
        np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij"
    )
    offset = np.stack([xx - p0[0], yy - p0[1], zz - p0[2]], axis=-1)
    along = offset @ unit
    radial = np.sqrt(((offset - along[..., None] * unit) ** 2).sum(axis=-1))
    mask = (
        (radial <= radius)
        & (along >= end_trim * length)
        & (along <= (1.0 - end_trim) * length)
    )
    return (z0, z1, y0, y1, x0, x1), mask


def count_segmented_in_cylinders(
    segmentation: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    radius: float,
    end_trim: float = 0.0,
) -> dict[str, np.ndarray]:
    """Count segmented voxels inside each strut's cylinder.

    Returns a dict of per-strut arrays:
        ``n_segmented`` -- segmented voxels in the cylinder (primary statistic)
        ``area``        -- ``n_segmented / sampled_length``, cross-section in voxels^2
        ``fill``        -- ``n_segmented / n_cylinder``, occupied fraction
        ``n_cylinder``  -- cylinder size in voxels, for rasterizer sanity checks

    ``area`` divides by the length actually sampled, so it stays a cross-section
    in voxels^2 and remains comparable across ``end_trim`` settings.
    """
    n_struts = len(start_xyz)
    n_segmented = np.empty(n_struts, dtype=np.int64)
    n_cylinder = np.empty(n_struts, dtype=np.int64)
    sampled_lengths = np.linalg.norm(end_xyz - start_xyz, axis=1) * (1.0 - 2.0 * end_trim)

    for i in range(n_struts):
        (z0, z1, y0, y1, x0, x1), mask = cylinder_mask(
            start_xyz[i], end_xyz[i], radius, segmentation.shape, end_trim
        )
        n_cylinder[i] = mask.sum()
        n_segmented[i] = segmentation[z0:z1, y0:y1, x0:x1][mask].sum()

    return {
        "n_segmented": n_segmented,
        "area": n_segmented / sampled_lengths,
        "fill": n_segmented / np.maximum(n_cylinder, 1),
        "n_cylinder": n_cylinder,
    }


def bottom_layer_junctions(
    nominal_json_path: Path, entry_to_junction: np.ndarray, n_junctions: int
) -> np.ndarray:
    """Flag merged junctions on the machined-off bottom face of the specimen.

    The face is the maximum-Y layer of the *nominal* design lattice. Nominal Y
    corresponds to the registered volume's y axis (r = 1.0000), but the
    registered coordinates are rotated, so the layer cannot be recovered by
    thresholding registered y directly.

    The nominal JSON is read per *entry*, matching the registered file's own
    entry list, and the resulting layer is then scattered onto merged junctions
    through ``entry_to_junction``. Nominal y is constant within a group, so the
    scatter is unambiguous.

    Args:
        nominal_json_path: The nominal design JSON, on a clean integer grid.
        entry_to_junction: Entry-to-merged-junction map from ``load_lattice``.
        n_junctions: Number of merged junctions; the returned mask's length.

    Raises:
        ValueError: If the nominal JSON's entry count or id ordering does not
            match the registered lattice's entries, which would make the mask
            meaningless when applied to the registered analysis.
    """
    with nominal_json_path.open() as f:
        data = json.load(f)

    junctions = data["junctions"]
    n_entries = len(entry_to_junction)
    if len(junctions) != n_entries:
        raise ValueError(
            f"{nominal_json_path} has {len(junctions)} junctions, "
            f"but the registered lattice has {n_entries} entries"
        )
    if [j["id"] for j in junctions] != list(range(n_entries)):
        raise ValueError(f"junction ids in {nominal_json_path} are not 0..N-1 in list order")

    y = np.array([j["position"][1] for j in junctions], dtype=float)
    bottom = np.zeros(n_junctions, dtype=bool)
    bottom[entry_to_junction[y >= y.max() - 0.5]] = True
    return bottom


def save_area_histogram(
    area: np.ndarray,
    cut: float,
    radius: float,
    otsu_threshold: float,
    output_path: Path,
) -> None:
    """Histogram per-strut cross-sectional area with the cut and nominal design marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(area, bins=100, color="steelblue", edgecolor="black", linewidth=0.2)
    ax.axvline(
        cut,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Missing-candidate cut = {cut:.1f} vox²",
    )
    ax.axvline(
        NOMINAL_CROSS_SECTION_VOXELS,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Nominal design cross-section = {NOMINAL_CROSS_SECTION_VOXELS:.1f} vox²",
    )
    ax.legend()
    ax.set_title(
        f"Strut cross-sectional area, cylinder radius {radius:g} voxels\n"
        f"(segmented at full-volume Otsu = {otsu_threshold:.0f})"
    )
    ax.set_xlabel("Segmented voxels per unit strut length (vox²)")
    ax.set_ylabel("Strut count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_radius_sweep_plot(summary: list[dict], output_path: Path) -> None:
    """Plot flagged strut/junction fraction against cylinder radius.

    A statistic that measures the specimen rather than the sampling geometry
    should stay roughly flat across radii.
    """
    radii = [row["radius"] for row in summary]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(radii, [row["flagged_strut_fraction"] for row in summary], "o-", label="struts flagged")
    ax.plot(
        radii,
        [row["empty_cylinder_fraction"] for row in summary],
        "s--",
        label="cylinders with zero segmented voxels",
    )
    ax.plot(
        radii,
        [row["flagged_junction_fraction"] for row in summary],
        "^-",
        label="junctions fully unsupported",
    )
    ax.set_xlabel("Cylinder radius (voxels)")
    ax.set_ylabel("Fraction")
    ax.set_title("Missing-candidate fraction vs cylinder radius")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def paint_cylinders(
    rgb_volume: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    radius: float,
    color: np.ndarray,
    alpha: float,
    end_trim: float = 0.0,
) -> None:
    """Alpha-blend ``color`` into each strut's cylinder, in place.

    Blending rather than overwriting keeps the underlying CT visible, so a
    flagged tube can be checked against the grayscale beneath it.
    """
    for i in range(len(start_xyz)):
        (z0, z1, y0, y1, x0, x1), mask = cylinder_mask(
            start_xyz[i], end_xyz[i], radius, rgb_volume.shape[:3], end_trim
        )
        block = rgb_volume[z0:z1, y0:y1, x0:x1]
        block[mask] = (block[mask] * (1 - alpha) + color * alpha).astype(np.uint8)


def write_stats_csv(
    output_path: Path,
    strut_junction_ids: np.ndarray,
    centers_xyz: np.ndarray,
    stats: dict[str, np.ndarray],
    missing_mask: np.ndarray,
) -> None:
    """Write one row per strut with its cylinder statistics and classification."""
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strut_id", "junction0", "junction1", "x", "y", "z",
                "n_segmented", "area", "fill", "n_cylinder", "missing_candidate",
            ]
        )
        for i in range(len(strut_junction_ids)):
            writer.writerow(
                [
                    i,
                    strut_junction_ids[i, 0],
                    strut_junction_ids[i, 1],
                    round(centers_xyz[i, 0]),
                    round(centers_xyz[i, 1]),
                    round(centers_xyz[i, 2]),
                    stats["n_segmented"][i],
                    f"{stats['area'][i]:.4f}",
                    f"{stats['fill'][i]:.5f}",
                    stats["n_cylinder"][i],
                    int(missing_mask[i]),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect missing struts by counting segmented voxels in a "
        "cylinder along each nominal strut."
    )
    parser.add_argument(
        "--radii", type=float, nargs="+", default=list(DEFAULT_RADII),
        help="cylinder radii in voxels to sweep",
    )
    parser.add_argument(
        "--area-fraction", type=float, default=DEFAULT_AREA_FRACTION,
        help="missing-candidate cut as a fraction of the nominal design cross-section",
    )
    parser.add_argument(
        "--tiff-radius", type=float, default=DEFAULT_TIFF_RADIUS,
        help="radius whose cylinders are painted into the QC TIFF (~1.5 GB)",
    )
    parser.add_argument(
        "--end-trim", type=float, default=0.0,
        help="fraction of each strut's span dropped at both ends before sampling",
    )
    parser.add_argument(
        "--keep-bottom-layer", action="store_true",
        help="include struts touching the machined-off bottom face (excluded by default)",
    )
    parser.add_argument(
        "--no-tiff", action="store_true", help="skip writing the QC TIFF stack"
    )
    parser.add_argument(
        "--suffix", default="", help="string appended to every output filename"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NPY_PATH.exists():
        print(f"Converting {TIFF_PATH} to {NPY_PATH} ...")
        convert_tiff_to_numpy(TIFF_PATH, NPY_PATH)
    volume = np.load(NPY_PATH)
    print(f"Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    otsu_threshold = threshold_otsu(volume)
    segmentation = volume >= otsu_threshold
    print(
        f"Full-volume Otsu threshold: {otsu_threshold:.0f} "
        f"({segmentation.mean():.2%} of voxels segmented as material)"
    )

    positions_xyz, all_strut_junction_ids, entry_to_junction = load_lattice(
        REGISTERED_JSON_PATH
    )
    n_all_struts = len(all_strut_junction_ids)

    if args.keep_bottom_layer:
        strut_junction_ids = all_strut_junction_ids
        print("Keeping struts on the machined-off bottom face (--keep-bottom-layer)")
    else:
        bottom = bottom_layer_junctions(
            NOMINAL_JSON_PATH, entry_to_junction, len(positions_xyz)
        )
        touches_bottom = bottom[all_strut_junction_ids[:, 0]] | bottom[all_strut_junction_ids[:, 1]]
        strut_junction_ids = all_strut_junction_ids[~touches_bottom]
        print(
            f"Excluding the machined-off bottom face: {bottom.sum()} junctions, "
            f"{touches_bottom.sum()} struts ({touches_bottom.mean():.2%})"
        )

    start_xyz = positions_xyz[strut_junction_ids[:, 0]]
    end_xyz = positions_xyz[strut_junction_ids[:, 1]]
    centers_xyz = (start_xyz + end_xyz) / 2.0
    lengths = np.linalg.norm(end_xyz - start_xyz, axis=1)
    n_scored_junctions = len(np.unique(strut_junction_ids))
    print(
        f"Analyzing {len(strut_junction_ids)} of {n_all_struts} struts across "
        f"{n_scored_junctions} junctions "
        f"(length {lengths.min():.1f}-{lengths.max():.1f} voxels)"
    )
    if args.end_trim:
        print(f"Sampling the middle {1 - 2 * args.end_trim:.0%} of each strut's span")
    print(f"Nominal design cross-section: {NOMINAL_CROSS_SECTION_VOXELS:.1f} vox²")

    summary: list[dict] = []
    missing_masks: dict[float, np.ndarray] = {}

    for radius in args.radii:
        print(f"\n--- cylinder radius {radius:g} voxels ---")
        stats = count_segmented_in_cylinders(
            segmentation, start_xyz, end_xyz, radius, args.end_trim
        )
        area = stats["area"]

        cut = args.area_fraction * NOMINAL_CROSS_SECTION_VOXELS
        missing_mask = area < cut
        missing_masks[radius] = missing_mask
        empty_mask = stats["n_segmented"] == 0
        junction_mask = junction_missing_mask(
            strut_junction_ids, missing_mask, len(positions_xyz)
        )

        analytic = np.pi * radius**2 * lengths.mean() * (1.0 - 2.0 * args.end_trim)
        print(
            f"Cylinder size: mean {stats['n_cylinder'].mean():.0f} voxels "
            f"(analytic πr²L = {analytic:.0f})"
        )
        print(
            f"Cross-sectional area (vox²): median={np.median(area):.2f} "
            f"p1={np.percentile(area, 1):.2f} p5={np.percentile(area, 5):.2f} "
            f"max={area.max():.2f}"
        )
        print(
            f"Missing-candidate cut: {cut:.2f} vox² "
            f"({args.area_fraction:.0%} of nominal design cross-section)"
        )
        print(
            f"Struts flagged: {missing_mask.sum()} / {missing_mask.size} "
            f"({missing_mask.mean():.2%});  fully empty cylinders: "
            f"{empty_mask.sum()} ({empty_mask.mean():.2%})"
        )
        print(
            f"Junctions fully unsupported: {junction_mask.sum()} / {n_scored_junctions} "
            f"({junction_mask.sum() / n_scored_junctions:.2%})"
        )

        hist_path = OUTPUT_DIR / f"strut_cylinder_area_histogram_r{radius:g}{args.suffix}.png"
        save_area_histogram(area, cut, radius, otsu_threshold, hist_path)
        print(f"Saved {hist_path}")

        stats_path = OUTPUT_DIR / f"strut_cylinder_stats_r{radius:g}{args.suffix}.csv"
        write_stats_csv(stats_path, strut_junction_ids, centers_xyz, stats, missing_mask)
        print(f"Saved {stats_path}")

        if radius == args.tiff_radius:
            junction_csv = (
                OUTPUT_DIR / f"missing_junction_candidates_r{radius:g}{args.suffix}.csv"
            )
            np.savetxt(
                junction_csv,
                np.column_stack(
                    [
                        np.flatnonzero(junction_mask),
                        np.round(positions_xyz[junction_mask]).astype(int),
                    ]
                ),
                delimiter=",",
                header="junction_id,x,y,z",
                comments="",
                fmt="%d",
            )
            print(f"Saved {junction_csv}")

        summary.append(
            {
                "radius": radius,
                "median_area": float(np.median(area)),
                "p1_area": float(np.percentile(area, 1)),
                "p5_area": float(np.percentile(area, 5)),
                "cut": cut,
                "n_flagged_struts": int(missing_mask.sum()),
                "flagged_strut_fraction": float(missing_mask.mean()),
                "n_empty_cylinders": int(empty_mask.sum()),
                "empty_cylinder_fraction": float(empty_mask.mean()),
                "n_flagged_junctions": int(junction_mask.sum()),
                "flagged_junction_fraction": junction_mask.sum() / n_scored_junctions,
            }
        )

    summary_csv = OUTPUT_DIR / f"radius_sweep_summary{args.suffix}.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved {summary_csv}")

    sweep_path = OUTPUT_DIR / f"flagged_fraction_vs_radius{args.suffix}.png"
    save_radius_sweep_plot(summary, sweep_path)
    print(f"Saved {sweep_path}")

    if args.no_tiff:
        return
    if args.tiff_radius not in missing_masks:
        print(f"Skipping TIFF: radius {args.tiff_radius:g} was not in the sweep")
        return

    print(f"\nPainting QC TIFF at radius {args.tiff_radius:g} ...")
    missing_mask = missing_masks[args.tiff_radius]
    rgb_volume = to_rgb_uint8(volume)
    del volume, segmentation
    paint_cylinders(
        rgb_volume,
        start_xyz[~missing_mask], end_xyz[~missing_mask],
        args.tiff_radius, PRESENT_COLOR, PRESENT_ALPHA, args.end_trim,
    )
    # Painted last so flagged tubes win any overlap at a shared junction.
    paint_cylinders(
        rgb_volume,
        start_xyz[missing_mask], end_xyz[missing_mask],
        args.tiff_radius, MISSING_COLOR, MISSING_ALPHA, args.end_trim,
    )
    tiff_path = OUTPUT_DIR / f"strut_cylinders_marked_r{args.tiff_radius:g}{args.suffix}.tif"
    tifffile.imwrite(tiff_path, rgb_volume, photometric="rgb", bigtiff=True)
    print(f"Saved {tiff_path}")


if __name__ == "__main__":
    main()
