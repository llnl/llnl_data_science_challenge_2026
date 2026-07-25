"""Experiment: overlay octet_truss_8x8x8.json nodes on the 9x9x9 octet lattice
CT volume and histogram the sampled intensities.

CAVEAT: octet_truss_8x8x8.json is the nominal (as-designed) node/strut
definition for a *different, smaller* 8x8x8-unit-cell lattice than the
9x9x9-unit-cell specimen actually imaged in 9x9x9_octet_lattice.tif, and it
is not registered to the TIFF's voxel coordinate system (see README.md and
data/9x9x9_octet_lattice/note.txt). Its ``position`` values only span
[0, 16] on each axis (design units), so there is no principled way to place
them in the volume's (761, 815, 837)-voxel space.

This script does the crude thing anyway, purely to see what it looks like:
each axis's positions are independently min-max scaled from their own
[0, 16] range onto the volume's corresponding voxel index range. This is
NOT a real registration -- it ignores build-plate placement, physical
voxel spacing, and the lattice size mismatch -- so misalignment is
expected. Compare against overlay_registered_nodes_histogram.py, which uses
the properly registered JSON for this exact specimen.
"""

from pathlib import Path

import json
import numpy as np
from skimage.filters import threshold_otsu

from overlay_registered_nodes_histogram import (
    save_histogram,
    save_mip_overlay,
    save_slice_overlay,
    sample_volume_at_nodes,
)

JSON_PATH = Path("data/octet_truss_8x8x8/octet_truss_8x8x8.json")
NPY_PATH = Path("outputs/9x9x9_octet_lattice/9x9x9_octet_lattice.npy")
OUTPUT_DIR = Path("outputs/9x9x9_octet_lattice/octet_truss_8x8x8_unregistered_experiment")


def load_scaled_node_voxel_coords(json_path: Path, volume_shape: tuple[int, int, int]) -> np.ndarray:
    """Min-max scale octet_truss_8x8x8.json node positions onto the volume's voxel index range.

    Returns an (N, 3) array of [x, y, z] voxel coordinates (unrounded).
    """
    with json_path.open() as f:
        data = json.load(f)
    positions_xyz = np.array([j["position"] for j in data["junctions"]], dtype=float)

    z_size, y_size, x_size = volume_shape
    voxel_bounds = np.array([x_size - 1, y_size - 1, z_size - 1], dtype=float)

    pos_min = positions_xyz.min(axis=0)
    pos_max = positions_xyz.max(axis=0)
    normalized = (positions_xyz - pos_min) / (pos_max - pos_min)
    return normalized * voxel_bounds


def main() -> None:
    if not NPY_PATH.exists():
        raise FileNotFoundError(
            f"{NPY_PATH} not found -- run overlay_registered_nodes_histogram.py first "
            "to convert the TIFF to a .npy array."
        )
    volume = np.load(NPY_PATH)
    print(f"Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    volume_otsu_threshold = threshold_otsu(volume)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    positions_xyz = load_scaled_node_voxel_coords(JSON_PATH, volume.shape)
    print(f"Loaded {len(positions_xyz)} octet_truss_8x8x8.json junctions (linearly scaled, unregistered)")

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

    print(
        f"Node intensity stats: min={values.min()}, max={values.max()}, "
        f"mean={values.mean():.1f}, median={np.median(values):.1f}"
    )


if __name__ == "__main__":
    main()
