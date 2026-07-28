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
must be found at strut level, by ``strut_scan``.

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

from dataclasses import replace
from pathlib import Path

import json
import numpy as np

from junction_scan import save_marked_tiff, scan_junctions
from overlay_registered_nodes_histogram import (
    MAX_SAMPLE_RADIUS_VOXELS,
    NPY_PATH,
    OUTPUT_DIR,
    REGISTERED_JSON_PATH,
)

TIFF_OUTPUT_PATH = OUTPUT_DIR / "junction_candidates_marked.tif"
CSV_OUTPUT_PATH = OUTPUT_DIR / "junction_candidates.csv"

# The nominal design lattice, on a clean 0..18 integer grid. Its junction entries
# are in the same order as the registered JSON's, so a mask built from it applies
# directly to the registered analysis.
NOMINAL_JSON_PATH = Path("data/missing_struts/octet_truss_9x9x9.json")

# Markers are drawn at the sampling radius, so each painted sphere is exactly
# the neighborhood whose brightest voxel produced the flag. A marker that
# swallowed more than was measured would invite reading absence into voxels the
# detector never looked at.
MARKER_RADIUS_VOXELS = MAX_SAMPLE_RADIUS_VOXELS


def bottom_layer_junctions(
    nominal_json_path: Path, entry_to_junction: np.ndarray, n_junctions: int
) -> np.ndarray:
    """Flag merged junctions on the machined-off bottom face of the specimen.

    The specimen's bottom face was cut or sanded off after printing: any
    mid-stack slice shows the lattice terminating in a sawtooth around y~720,
    with no material where the nominal design's last junction row sits (y~760).
    Those junctions exist in the design JSON but not in the physical part.

    The face is the maximum-Y layer of the *nominal* design lattice. Nominal Y
    corresponds to the registered volume's y axis (r = 1.0000), but the
    registered coordinates are rotated, so the layer cannot be recovered by
    thresholding registered y directly.

    This lives here, in a script hardcoded to one specimen, and not in the
    detectors: the other eight scans in this dataset will not share this
    machining artifact, and a detector that knew about it would go looking for
    it in parts that do not have one. An agent working from ``scan_junctions``
    reaches the same set by reading the component sizes and the overlays.

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
        raise ValueError(
            f"junction ids in {nominal_json_path} are not 0..N-1 in list order"
        )

    y = np.array([j["position"][1] for j in junctions], dtype=float)
    bottom = np.zeros(n_junctions, dtype=bool)
    bottom[entry_to_junction[y >= y.max() - 0.5]] = True
    return bottom


def main() -> None:
    if not NPY_PATH.exists():
        raise FileNotFoundError(
            f"{NPY_PATH} not found -- run overlay_registered_nodes_histogram.py first."
        )
    volume = np.load(NPY_PATH)
    print(f"Volume shape (z, y, x): {volume.shape}, dtype: {volume.dtype}")

    scan = scan_junctions(volume, REGISTERED_JSON_PATH, radius=MARKER_RADIUS_VOXELS)
    print(f"Full-volume Otsu threshold ({volume.size} voxels): {scan.threshold:.0f}")
    print(
        f"Loaded {scan.n_junctions} junctions (merged from "
        f"{len(scan.entry_to_junction)} JSON entries), degree "
        f"{scan.degree.min()}-{scan.degree.max()}"
    )

    # The machined-off face is a property of *this* specimen, so it is dropped
    # here rather than inside the detector, which knows nothing about any
    # particular part. Clearing those junctions' ``dark`` flags takes them out of
    # everything downstream -- the printed list and the painted markers. An agent
    # working from ``scan_junctions`` reaches the same set by reading the CSV and
    # the overlays.
    face = bottom_layer_junctions(
        NOMINAL_JSON_PATH, scan.entry_to_junction, scan.n_junctions
    )
    result = replace(scan, dark=scan.dark & ~face)
    candidate = result.dark
    n_scored = int((~face).sum())
    print(
        f"Dark junctions: {scan.dark.sum()}, of which "
        f"{(scan.dark & face).sum()} lie on the machined-off bottom face"
    )
    print(
        f"Candidates: {candidate.sum()} of {n_scored} scored "
        f"junctions ({candidate.sum() / n_scored:.2%})"
    )
    for junction in np.flatnonzero(candidate):
        x, y, z = result.voxel_xyz[junction]
        print(
            f"  junction {junction:5d}  x={x:3d} y={y:3d} z={z:3d}  "
            f"intensity={result.intensities[junction]:6d}  "
            f"degree={result.degree[junction]:2d}"
        )

    save_marked_tiff(volume, result, TIFF_OUTPUT_PATH)
    print(f"Saved {TIFF_OUTPUT_PATH}")

    # Merged ids do not index the JSON's junctions array, so each row also
    # carries the entry ids it was built from, for cross-referencing the source.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT_PATH.open("w") as f:
        f.write("junction_id,x,y,z,intensity,degree,entry_ids\n")
        for junction in np.flatnonzero(candidate):
            x, y, z = result.voxel_xyz[junction]
            entries = " ".join(
                str(e) for e in np.flatnonzero(result.entry_to_junction == junction)
            )
            f.write(
                f"{junction},{x},{y},{z},{result.intensities[junction]},"
                f"{result.degree[junction]},{entries}\n"
            )
    print(f"Saved {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
