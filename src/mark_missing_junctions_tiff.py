"""Flag junctions whose CT intensity falls below the full-volume Otsu
threshold as potential missing/disconnected junctions, and burn a red
marker into each one's location in an RGB TIFF stack for visual QC.

Reuses the volume load, registered-JSON node loading, and voxel sampling
from overlay_registered_nodes_histogram.py. See that module's docstring for
why the registered JSON (not the un-registered octet_truss_8x8x8.json) is
used to locate junctions in this volume's voxel coordinate system.

Why the volume's threshold and not the nodes' own
-------------------------------------------------
Otsu maximizes w0*w1*(m0-m1)^2, which assumes two comparably sized classes.
Only a few hundred of the ~10000 nodes are dark, so that product penalizes
any cut isolating them and the objective goes flat: between 34000 and 48000
it varies by 5%, and its argmax lands at 43583 -- past the one real gap in
the distribution (empty from 35500 to 39500) and well inside the material
mode. Worse, the answer depends on which nodes are scored, so it degrades as
the node set is cleaned up: excluding the machined-off bottom face and the
far-x registration-drift band leaves ~0.17% of nodes genuinely dark, and
node-sampled Otsu responds by returning 52952 and flagging 21.6% of them.

The volume's own Otsu has neither problem. It is measured where the balanced
bimodal assumption actually holds (11.3% material against 88.7% background),
it falls mid-trough for the node distribution, and it is independent of the
node set -- so it does not move when junctions are excluded. It is also what
strut_center_intensity_histogram and strut_scan already cut against.
"""

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from overlay_registered_nodes_histogram import (
    NPY_PATH,
    OUTPUT_DIR,
    REGISTERED_JSON_PATH,
    load_node_voxel_coords,
    merge_colocated_junctions,
    sample_volume_at_nodes,
)

MARKER_RADIUS_VOXELS = 5
MARKER_COLOR = (255, 0, 0)

TIFF_OUTPUT_PATH = OUTPUT_DIR / "missing_junction_candidates.tif"
CSV_OUTPUT_PATH = OUTPUT_DIR / "missing_junction_candidates.csv"


def to_rgb_uint8(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize a grayscale volume to uint8 and stack it into 3 channels."""
    vmin, vmax = volume.min(), volume.max()
    gray = ((volume.astype(np.float32) - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def mark_points(
    rgb_volume: np.ndarray, voxel_xyz: np.ndarray, radius: int, color: tuple[int, int, int]
) -> None:
    """Paint a filled sphere of ``color`` at each (x, y, z) voxel coordinate, in place."""
    z_size, y_size, x_size, _ = rgb_volume.shape
    color_array = np.array(color, dtype=np.uint8)
    offsets = np.arange(-radius, radius + 1)
    dz, dy, dx = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    sphere_mask = (dz**2 + dy**2 + dx**2) <= radius**2

    for x, y, z in voxel_xyz:
        z0, z1 = max(z - radius, 0), min(z + radius + 1, z_size)
        y0, y1 = max(y - radius, 0), min(y + radius + 1, y_size)
        x0, x1 = max(x - radius, 0), min(x + radius + 1, x_size)
        local_mask = sphere_mask[
            z0 - (z - radius) : z1 - (z - radius),
            y0 - (y - radius) : y1 - (y - radius),
            x0 - (x - radius) : x1 - (x - radius),
        ]
        rgb_volume[z0:z1, y0:y1, x0:x1][local_mask] = color_array


def main() -> None:
    if not NPY_PATH.exists():
        raise FileNotFoundError(
            f"{NPY_PATH} not found -- run overlay_registered_nodes_histogram.py first."
        )
    volume = np.load(NPY_PATH)
    print(f"Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    threshold = threshold_otsu(volume)
    print(f"Full-volume Otsu threshold ({volume.size} voxels): {threshold:.0f}")

    entry_positions_xyz = load_node_voxel_coords(REGISTERED_JSON_PATH)
    positions_xyz, entry_to_junction = merge_colocated_junctions(entry_positions_xyz)
    values, voxel_xyz = sample_volume_at_nodes(volume, positions_xyz)
    print(
        f"Loaded {len(positions_xyz)} registered junctions "
        f"(merged from {len(entry_to_junction)} JSON entries)"
    )

    missing_mask = values < threshold
    missing_voxel_xyz = voxel_xyz[missing_mask]
    missing_values = values[missing_mask]
    print(
        f"{missing_mask.sum()} potential missing junctions "
        f"({missing_mask.mean():.1%}), {(~missing_mask).sum()} present "
        f"({(~missing_mask).mean():.1%})"
    )

    rgb_volume = to_rgb_uint8(volume)
    del volume
    mark_points(rgb_volume, missing_voxel_xyz, MARKER_RADIUS_VOXELS, MARKER_COLOR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(TIFF_OUTPUT_PATH, rgb_volume, photometric="rgb", bigtiff=True)
    print(f"Saved {TIFF_OUTPUT_PATH}")

    np.savetxt(
        CSV_OUTPUT_PATH,
        np.column_stack([missing_voxel_xyz, missing_values]),
        delimiter=",",
        header="x,y,z,intensity",
        comments="",
        fmt="%d",
    )
    print(f"Saved {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
