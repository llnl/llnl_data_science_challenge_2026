"""Sample CT intensity at the center of every nominal strut and histogram it.

Each strut in the registered lattice JSON is defined by the ids of the two
junctions it connects. Tiling the lattice therefore reduces to walking the
strut list, forming the unit vector between its two registered junction
positions, and stepping half the strut length along it:

    u = (p1 - p0) / |p1 - p0|
    center = p0 + u * (|p1 - p0| / 2)

Strut centers are the most diagnostic place to look for a missing strut: a
junction stays bright as long as *any* incident strut is printed, whereas the
midspan of an absent strut is pure background.

Intensities are sampled with the same neighborhood-max sampler used for
junctions (see ``overlay_registered_nodes_histogram``), which tolerates small
registration error. The histogram is annotated with the Otsu threshold of the
*whole* volume, and struts whose center falls below it are reported as missing
candidates. Junction-level candidates are junctions all of whose incident
struts are missing candidates.

Coordinate conventions match ``overlay_registered_nodes_histogram``: JSON
``position`` fields are [x, y, z] and index the TIFF's (z, y, x) axes.
"""

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from overlay_registered_nodes_histogram import (
    NPY_PATH,
    REGISTERED_JSON_PATH,
    TIFF_PATH,
    sample_volume_at_nodes,
)
from mark_missing_junctions_tiff import mark_points, to_rgb_uint8
from volume_io import convert_tiff_to_numpy

OUTPUT_DIR = Path("outputs/missing_struts/strut_centers")

# The struts are ~4 voxels across, so markers are kept small enough not to
# swallow the neighboring lattice in the QC stack.
MISSING_MARKER_RADIUS_VOXELS = 4
PRESENT_MARKER_RADIUS_VOXELS = 2
MISSING_MARKER_COLOR = (255, 0, 0)
PRESENT_MARKER_COLOR = (0, 255, 0)


def load_lattice(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return registered junction positions [x, y, z] and strut junction-id pairs.

    Returns:
        (positions_xyz, strut_junction_ids) with shapes (N_junctions, 3) and
        (N_struts, 2). Strut entries index rows of ``positions_xyz`` by list
        order, which the JSON keeps consistent with each junction's ``id``.
    """
    with json_path.open() as f:
        data = json.load(f)

    junction_ids = [j["id"] for j in data["junctions"]]
    if junction_ids != list(range(len(junction_ids))):
        raise ValueError(f"junction ids in {json_path} are not 0..N-1 in list order")

    positions_xyz = np.array([j["position"] for j in data["junctions"]], dtype=float)
    strut_junction_ids = np.array(
        [[s["junction0"], s["junction1"]] for s in data["struts"]], dtype=int
    )
    return positions_xyz, strut_junction_ids


def strut_centers_from_unit_vectors(
    positions_xyz: np.ndarray, strut_junction_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile the lattice by stepping half a strut length along each strut's unit vector.

    Returns:
        (centers_xyz, unit_vectors_xyz, lengths) for every strut, in strut order.
    """
    start = positions_xyz[strut_junction_ids[:, 0]]
    end = positions_xyz[strut_junction_ids[:, 1]]

    directions = end - start
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths == 0):
        raise ValueError("degenerate strut with coincident endpoints in lattice JSON")
    unit_vectors = directions / lengths[:, None]

    centers = start + unit_vectors * (lengths[:, None] / 2.0)
    return centers, unit_vectors, lengths


def junction_missing_mask(
    strut_junction_ids: np.ndarray, missing_strut_mask: np.ndarray, n_junctions: int
) -> np.ndarray:
    """Flag junctions whose every incident strut is a missing candidate.

    A junction is only unsupported once all struts meeting there are absent, so
    the per-junction test is a logical AND over incident struts.
    """
    incident_total = np.bincount(strut_junction_ids.ravel(), minlength=n_junctions)
    incident_missing = np.bincount(
        strut_junction_ids.ravel(),
        weights=np.repeat(missing_strut_mask.astype(float), 2),
        minlength=n_junctions,
    )
    return (incident_total > 0) & (incident_missing == incident_total)


def save_histogram(
    values: np.ndarray, volume_otsu_threshold: float, output_path: Path
) -> None:
    """Histogram strut-center intensities with the full-volume Otsu threshold marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=100, color="steelblue", edgecolor="black", linewidth=0.2)
    ax.axvline(
        volume_otsu_threshold,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Full-volume Otsu threshold = {volume_otsu_threshold:.0f}",
    )
    ax.legend()
    ax.set_title("CT intensity at nominal strut centers")
    ax.set_xlabel("Voxel intensity (uint16)")
    ax.set_ylabel("Strut count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_mip_overlay(
    volume: np.ndarray,
    centers_xyz: np.ndarray,
    missing_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a Z max-intensity projection with strut centers colored by classification."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(volume.max(axis=0), cmap="gray")
    ax.scatter(
        centers_xyz[~missing_mask, 0],
        centers_xyz[~missing_mask, 1],
        s=1,
        c="lime",
        alpha=0.3,
        linewidths=0,
        label="strut present",
    )
    ax.scatter(
        centers_xyz[missing_mask, 0],
        centers_xyz[missing_mask, 1],
        s=25,
        facecolors="none",
        edgecolors="red",
        linewidths=1,
        label="missing candidate",
    )
    ax.legend(loc="upper right")
    ax.set_title("Z max-intensity projection with strut centers overlaid")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NPY_PATH.exists():
        print(f"Converting {TIFF_PATH} to {NPY_PATH} ...")
        convert_tiff_to_numpy(TIFF_PATH, NPY_PATH)
    volume = np.load(NPY_PATH)
    print(f"Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    volume_otsu_threshold = threshold_otsu(volume)
    print(f"Full-volume Otsu threshold ({volume.size} voxels): {volume_otsu_threshold:.0f}")

    positions_xyz, strut_junction_ids = load_lattice(REGISTERED_JSON_PATH)
    centers_xyz, unit_vectors, lengths = strut_centers_from_unit_vectors(
        positions_xyz, strut_junction_ids
    )
    n_orientations = len(np.unique(np.round(np.abs(unit_vectors), 3), axis=0))
    print(
        f"Loaded {len(positions_xyz)} junctions and {len(centers_xyz)} struts "
        f"(length {lengths.min():.1f}-{lengths.max():.1f} voxels, "
        f"{n_orientations} distinct orientations)"
    )

    values, center_voxel_xyz = sample_volume_at_nodes(volume, centers_xyz)
    missing_mask = values < volume_otsu_threshold
    print(
        f"Strut-center intensity: min={values.min()}, max={values.max()}, "
        f"mean={values.mean():.1f}, median={np.median(values):.1f}"
    )
    print(
        f"Missing strut candidates: {missing_mask.sum()} / {missing_mask.size} "
        f"({missing_mask.mean():.2%} of struts)"
    )

    junction_mask = junction_missing_mask(
        strut_junction_ids, missing_mask, len(positions_xyz)
    )
    print(
        f"Missing junction candidates (all incident struts below threshold): "
        f"{junction_mask.sum()} / {junction_mask.size} ({junction_mask.mean():.2%} of junctions)"
    )

    hist_path = OUTPUT_DIR / "strut_center_intensity_histogram.png"
    save_histogram(values, volume_otsu_threshold, hist_path)
    print(f"Saved {hist_path}")

    mip_path = OUTPUT_DIR / "strut_center_overlay_mip.png"
    save_mip_overlay(volume, center_voxel_xyz, missing_mask, mip_path)
    print(f"Saved {mip_path}")

    values_csv = OUTPUT_DIR / "strut_center_intensity_values.csv"
    np.savetxt(
        values_csv,
        np.column_stack(
            [
                np.arange(len(values)),
                strut_junction_ids,
                center_voxel_xyz,
                values,
                missing_mask.astype(int),
            ]
        ),
        delimiter=",",
        header="strut_id,junction0,junction1,x,y,z,intensity,missing_candidate",
        comments="",
        fmt="%d",
    )
    print(f"Saved {values_csv}")

    junction_csv = OUTPUT_DIR / "missing_junction_candidates.csv"
    junction_voxel_xyz = np.round(positions_xyz[junction_mask]).astype(int)
    np.savetxt(
        junction_csv,
        np.column_stack([np.flatnonzero(junction_mask), junction_voxel_xyz]),
        delimiter=",",
        header="junction_id,x,y,z",
        comments="",
        fmt="%d",
    )
    print(f"Saved {junction_csv}")

    rgb_volume = to_rgb_uint8(volume)
    del volume
    mark_points(
        rgb_volume,
        center_voxel_xyz[~missing_mask],
        PRESENT_MARKER_RADIUS_VOXELS,
        PRESENT_MARKER_COLOR,
    )
    mark_points(
        rgb_volume,
        center_voxel_xyz[missing_mask],
        MISSING_MARKER_RADIUS_VOXELS,
        MISSING_MARKER_COLOR,
    )
    tiff_path = OUTPUT_DIR / "strut_centers_marked.tif"
    tifffile.imwrite(tiff_path, rgb_volume, photometric="rgb", bigtiff=True)
    print(f"Saved {tiff_path}")


if __name__ == "__main__":
    main()
