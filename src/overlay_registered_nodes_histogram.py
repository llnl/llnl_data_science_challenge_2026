"""Convert the 9x9x9 octet lattice CT volume, overlay registered lattice
nodes on it, and histogram the CT intensity sampled at each node.

The raw/nominal design JSONs in this repo are NOT aligned with the CT TIFF
coordinate system (see README.md). ``9x9x9_octet_lattice.tif`` is the same
scan as ``data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_
0point5dash1 1 Slices.tif`` (per data/9x9x9_octet_lattice/note.txt), and the
companion file in ``registered_jsons/`` is already registered to that scan's
voxel coordinates, so it is used here instead of the un-registered
``octet_truss_8x8x8.json`` (which also describes a different, smaller
8x8x8-unit-cell lattice than this 9x9x9 specimen).

Node ``position`` fields in the registered JSON are stored as
[x, y, z] and line up directly with the TIFF's own (z, y, x) axis order.
Each node is sampled as the brightest voxel within a 5x5x5-voxel spherical
neighborhood centered on ``volume[round(z), round(y), round(x)]``, rather
than that single nearest voxel, to tolerate small registration error.
"""

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

from volume_io import convert_tiff_to_numpy

TIFF_PATH = Path("data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif")
REGISTERED_JSON_PATH = Path(
    "data/missing_struts/registered_jsons/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
)
OUTPUT_DIR = Path("outputs/9x9x9_octet_lattice")
NPY_PATH = OUTPUT_DIR / "9x9x9_octet_lattice.npy"


def load_node_voxel_coords(json_path: Path) -> np.ndarray:
    """Return an (N, 3) array of registered node positions as [x, y, z]."""
    with json_path.open() as f:
        data = json.load(f)
    return np.array([j["position"] for j in data["junctions"]], dtype=float)


MAX_SAMPLE_RADIUS_VOXELS = 2  # radius 2 -> inscribed sphere in a 5x5x5 voxel box


def sample_volume_at_nodes(
    volume: np.ndarray, positions_xyz: np.ndarray, radius: int = MAX_SAMPLE_RADIUS_VOXELS
) -> tuple[np.ndarray, np.ndarray]:
    """Round node positions to voxel indices and sample the brightest voxel nearby.

    For each node, takes the maximum volume value within a sphere of the
    given ``radius`` (inscribed in a ``(2*radius+1)``-wide voxel box)
    centered on its rounded voxel position. This is more forgiving of small
    registration error than sampling the single nearest voxel.

    Returns (values, voxel_indices_xyz), where out-of-bounds nodes (if any)
    are clipped to the nearest valid voxel, and voxel_indices_xyz gives each
    neighborhood's center (not the location of the brightest voxel).
    """
    z_size, y_size, x_size = volume.shape
    voxel_xyz = np.round(positions_xyz).astype(int)
    voxel_xyz[:, 0] = np.clip(voxel_xyz[:, 0], 0, x_size - 1)
    voxel_xyz[:, 1] = np.clip(voxel_xyz[:, 1], 0, y_size - 1)
    voxel_xyz[:, 2] = np.clip(voxel_xyz[:, 2], 0, z_size - 1)

    offsets = np.arange(-radius, radius + 1)
    dz, dy, dx = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    sphere_mask = (dz**2 + dy**2 + dx**2) <= radius**2

    values = np.empty(len(voxel_xyz), dtype=volume.dtype)
    for i, (x, y, z) in enumerate(voxel_xyz):
        z0, z1 = max(z - radius, 0), min(z + radius + 1, z_size)
        y0, y1 = max(y - radius, 0), min(y + radius + 1, y_size)
        x0, x1 = max(x - radius, 0), min(x + radius + 1, x_size)
        local_mask = sphere_mask[
            z0 - (z - radius) : z1 - (z - radius),
            y0 - (y - radius) : y1 - (y - radius),
            x0 - (x - radius) : x1 - (x - radius),
        ]
        values[i] = volume[z0:z1, y0:y1, x0:x1][local_mask].max()
    return values, voxel_xyz


def save_mip_overlay(volume: np.ndarray, voxel_xyz: np.ndarray, output_path: Path) -> None:
    """Save a max-intensity Z-projection with all node (x, y) positions overlaid."""
    mip = volume.max(axis=0)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mip, cmap="gray")
    ax.scatter(voxel_xyz[:, 0], voxel_xyz[:, 1], s=3, c="red", alpha=0.5, linewidths=0)
    ax.set_title("Z max-intensity projection with registered nodes overlaid")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_slice_overlay(
    volume: np.ndarray, voxel_xyz: np.ndarray, z_index: int, output_path: Path, tolerance: int = 2
) -> None:
    """Save a single Z-slice with nearby nodes (within +/- tolerance voxels) overlaid."""
    nearby = np.abs(voxel_xyz[:, 2] - z_index) <= tolerance
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(volume[z_index], cmap="gray")
    ax.scatter(
        voxel_xyz[nearby, 0], voxel_xyz[nearby, 1], s=20, facecolors="none", edgecolors="red", linewidths=1
    )
    ax.set_title(f"Slice z={z_index} with nodes within +/-{tolerance} voxels overlaid")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_histogram(values: np.ndarray, volume_otsu_threshold: float, output_path: Path) -> float:
    node_otsu_threshold = threshold_otsu(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=100, color="steelblue", edgecolor="black", linewidth=0.2)
    ax.axvline(
        node_otsu_threshold, color="black", linestyle=":", linewidth=2,
        label=f"Node-sampled Otsu threshold = {node_otsu_threshold:.0f}",
    )
    ax.axvline(
        volume_otsu_threshold, color="crimson", linestyle="--", linewidth=2,
        label=f"Full-volume Otsu threshold = {volume_otsu_threshold:.0f}",
    )
    ax.legend()
    ax.set_title("CT intensity at registered lattice node locations")
    ax.set_xlabel("Voxel intensity (uint16)")
    ax.set_ylabel("Node count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return node_otsu_threshold


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Converting {TIFF_PATH} to {NPY_PATH} ...")
    convert_tiff_to_numpy(TIFF_PATH, NPY_PATH)
    volume = np.load(NPY_PATH)
    print(f"Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    volume_otsu_threshold = threshold_otsu(volume)
    print(f"Full-volume Otsu threshold ({volume.size} voxels): {volume_otsu_threshold:.0f}")

    positions_xyz = load_node_voxel_coords(REGISTERED_JSON_PATH)
    print(f"Loaded {len(positions_xyz)} registered junctions")

    values, voxel_xyz = sample_volume_at_nodes(volume, positions_xyz)

    mip_path = OUTPUT_DIR / "node_overlay_mip.png"
    save_mip_overlay(volume, voxel_xyz, mip_path)
    print(f"Saved {mip_path}")

    mid_z = int(np.median(voxel_xyz[:, 2]))
    slice_path = OUTPUT_DIR / f"node_overlay_slice_z{mid_z:03d}.png"
    save_slice_overlay(volume, voxel_xyz, mid_z, slice_path)
    print(f"Saved {slice_path}")

    hist_path = OUTPUT_DIR / "node_intensity_histogram.png"
    node_otsu_threshold = save_histogram(values, volume_otsu_threshold, hist_path)
    print(
        f"Saved {hist_path} "
        f"(node-sampled Otsu = {node_otsu_threshold:.0f}, full-volume Otsu = {volume_otsu_threshold:.0f})"
    )

    csv_path = OUTPUT_DIR / "node_intensity_values.csv"
    np.savetxt(
        csv_path,
        np.column_stack([voxel_xyz, values]),
        delimiter=",",
        header="x,y,z,intensity",
        comments="",
        fmt="%d",
    )
    print(f"Saved {csv_path}")

    print(
        f"Node intensity stats: min={values.min()}, max={values.max()}, "
        f"mean={values.mean():.1f}, median={np.median(values):.1f}"
    )


if __name__ == "__main__":
    main()
