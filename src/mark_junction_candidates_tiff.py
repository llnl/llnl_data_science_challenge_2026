"""Mark the surviving missing-junction candidates in an RGB TIFF stack.

``mark_missing_junctions_tiff`` flags every junction whose sampled intensity
falls below the full-volume Otsu threshold. Most of those belong to the
machined-off bottom face -- real absence, but a machining artifact rather than a
print defect -- so this module excludes that face and paints only what is left,
producing a stack that can be stepped through without hunting for a handful of
markers among hundreds.

What a hit here can and cannot mean
-----------------------------------
A junction goes dark only when *every* incident strut is absent, and an interior
junction of this lattice has twelve. At the 0.5% nominal missing-strut rate the
expected yield is 0.005^12, which is zero for any practical purpose; even the
sparsest boundary junctions have three struts. So this detector has essentially
no sensitivity to the defect population in this specimen, and a hit is evidence
of something else: residual registration error, a second machined face, or a
gross printing failure. Missing *struts* leave both their junctions bright and
must be found at strut level, by ``strut_cylinder_segmentation``.

Two things this depends on getting right
----------------------------------------
Junction entries are merged by position before anything is sampled. The JSON
lists a physical junction up to eight times, splitting its struts across the
duplicates, so unmerged degrees run 1-8 instead of 3-12 and every count is
inflated threefold. See ``merge_colocated_junctions``.

The sampling radius has to absorb the registered lattice's residual drift. At
radius 2 this script flagged 377 junctions beyond x = 700 -- 53% of that band,
against 0.2% elsewhere -- purely because the JSON sat off the struts. At the
current ``MAX_SAMPLE_RADIUS_VOXELS`` that band flags nothing, which is what
retired the explicit x cutoff this module used to carry.
"""

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from overlay_registered_nodes_histogram import (
    MAX_SAMPLE_RADIUS_VOXELS,
    NPY_PATH,
    OUTPUT_DIR,
    REGISTERED_JSON_PATH,
    sample_volume_at_nodes,
)
from mark_missing_junctions_tiff import mark_points, to_rgb_uint8
from strut_center_intensity_histogram import load_lattice
from strut_cylinder_segmentation import NOMINAL_JSON_PATH, bottom_layer_junctions

TIFF_OUTPUT_PATH = OUTPUT_DIR / "junction_candidates_marked.tif"
CSV_OUTPUT_PATH = OUTPUT_DIR / "junction_candidates.csv"

# Markers are drawn at the sampling radius, so each painted sphere is exactly
# the neighborhood whose brightest voxel produced the flag. A marker that
# swallowed more than was measured would invite reading absence into voxels the
# detector never looked at.
MARKER_RADIUS_VOXELS = MAX_SAMPLE_RADIUS_VOXELS
CANDIDATE_COLOR = (255, 0, 0)


def main() -> None:
    if not NPY_PATH.exists():
        raise FileNotFoundError(
            f"{NPY_PATH} not found -- run overlay_registered_nodes_histogram.py first."
        )
    volume = np.load(NPY_PATH)
    print(f"Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    threshold = threshold_otsu(volume)
    print(f"Full-volume Otsu threshold ({volume.size} voxels): {threshold:.0f}")

    positions_xyz, strut_junction_ids, entry_to_junction = load_lattice(
        REGISTERED_JSON_PATH
    )
    n_junctions = len(positions_xyz)
    degree = np.bincount(strut_junction_ids.ravel(), minlength=n_junctions)
    values, voxel_xyz = sample_volume_at_nodes(volume, positions_xyz)
    print(
        f"Loaded {n_junctions} junctions (merged from {len(entry_to_junction)} "
        f"JSON entries), degree {degree.min()}-{degree.max()}"
    )

    dark = values < threshold
    bottom = bottom_layer_junctions(NOMINAL_JSON_PATH, entry_to_junction, n_junctions)
    candidate = dark & ~bottom

    print(
        f"Dark junctions: {dark.sum()}, of which {(dark & bottom).sum()} lie on "
        f"the machined-off bottom face"
    )
    print(
        f"Candidates: {candidate.sum()} of {(~bottom).sum()} scored junctions "
        f"({candidate.sum() / (~bottom).sum():.2%})"
    )
    for junction in np.flatnonzero(candidate):
        x, y, z = voxel_xyz[junction]
        print(
            f"  junction {junction:5d}  x={x:3d} y={y:3d} z={z:3d}  "
            f"intensity={values[junction]:6d}  degree={degree[junction]:2d}"
        )

    rgb_volume = to_rgb_uint8(volume)
    del volume
    mark_points(rgb_volume, voxel_xyz[candidate], MARKER_RADIUS_VOXELS, CANDIDATE_COLOR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(TIFF_OUTPUT_PATH, rgb_volume, photometric="rgb", bigtiff=True)
    print(f"Saved {TIFF_OUTPUT_PATH}")

    # Merged ids do not index the JSON's junctions array, so each row also
    # carries the entry ids it was built from, for cross-referencing the source.
    with CSV_OUTPUT_PATH.open("w") as f:
        f.write("junction_id,x,y,z,intensity,degree,entry_ids\n")
        for junction in np.flatnonzero(candidate):
            x, y, z = voxel_xyz[junction]
            entries = " ".join(str(e) for e in np.flatnonzero(entry_to_junction == junction))
            f.write(
                f"{junction},{x},{y},{z},{values[junction]},"
                f"{degree[junction]},{entries}\n"
            )
    print(f"Saved {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
