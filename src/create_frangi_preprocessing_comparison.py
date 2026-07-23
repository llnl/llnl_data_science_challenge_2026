"""Create compact Frangi/no-filter segmentation artifacts for Crosshair review.

The full CT volume is too large for an interactive first-pass filter comparison.
This script samples a centered cubic region, creates binary masks with and
without a Frangi vesselness filter, and writes point-cloud CSVs plus skeleton
wireframe CSVs for a 2-by-2 Crosshair view.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import frangi, sato, threshold_otsu
from skimage.morphology import skeletonize

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "skan_numba_cache")
)

from skan.csr import skeleton_to_csgraph


INPUT_PATH = Path("data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif")
OUTPUT_DIR = Path("outputs/frangi_preprocessing_comparison")
CROP_SIZE = 256
SAMPLE_STRIDE = 4
MAX_VOXEL_POINTS = 60_000
MAX_WIREFRAME_EDGES = 40_000


def _centered_sample(volume: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return a centered, regularly sampled cubic region and its z/y/x origin."""
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, found shape {volume.shape}")
    starts = tuple((size - CROP_SIZE) // 2 for size in volume.shape)
    if any(start < 0 for start in starts):
        raise ValueError(f"volume is smaller than the {CROP_SIZE}-voxel test crop")
    z_start, y_start, x_start = starts
    cropped = volume[
        z_start : z_start + CROP_SIZE : SAMPLE_STRIDE,
        y_start : y_start + CROP_SIZE : SAMPLE_STRIDE,
        x_start : x_start + CROP_SIZE : SAMPLE_STRIDE,
    ]
    return cropped, starts


def _write_voxel_points(
    mask: np.ndarray, output_path: Path, origin_zyx: tuple[int, int, int]
) -> int:
    """Write a bounded, spatially even sample of foreground mask voxels."""
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("segmentation mask has no foreground voxels")
    step = max(1, int(np.ceil(len(coordinates) / MAX_VOXEL_POINTS)))
    sampled = coordinates[::step]
    offsets = np.asarray(origin_zyx, dtype=np.int32)
    source_coordinates = sampled * SAMPLE_STRIDE + offsets
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["x", "y", "z"])
        writer.writerows(source_coordinates[:, [2, 1, 0]].tolist())
    return len(sampled)


def _write_wireframe(
    skeleton: np.ndarray, output_path: Path, origin_zyx: tuple[int, int, int]
) -> int:
    """Write bounded NaN-separated skeleton-edge segments for Plotly lines."""
    graph, coordinates = skeleton_to_csgraph(skeleton)
    edges = graph.tocoo()
    keep = edges.row < edges.col
    source_ids = edges.row[keep]
    target_ids = edges.col[keep]
    step = max(1, int(np.ceil(len(source_ids) / MAX_WIREFRAME_EDGES)))
    source_ids = source_ids[::step]
    target_ids = target_ids[::step]
    z, y, x = (np.asarray(axis) for axis in coordinates)
    offsets = np.asarray(origin_zyx, dtype=np.float32)
    source = np.column_stack((x[source_ids], y[source_ids], z[source_ids]))
    target = np.column_stack((x[target_ids], y[target_ids], z[target_ids]))
    points = np.empty((len(source_ids) * 3, 3), dtype=np.float32)
    points[0::3] = source * SAMPLE_STRIDE + offsets[[2, 1, 0]]
    points[1::3] = target * SAMPLE_STRIDE + offsets[[2, 1, 0]]
    points[2::3] = np.nan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points.tolist())
    return len(source_ids)


def _process_variant(
    name: str,
    mask: np.ndarray,
    origin: tuple[int, int, int],
    output_dir: Path,
) -> dict[str, int | str]:
    """Save a mask, skeleton, point cloud, and wireframe for one variant."""
    skeleton = skeletonize(mask)
    np.save(output_dir / f"{name}_mask.npy", mask, allow_pickle=False)
    np.save(output_dir / f"{name}_skeleton.npy", skeleton, allow_pickle=False)
    voxel_count = _write_voxel_points(
        mask, output_dir / f"{name}_voxels.csv", origin
    )
    edge_count = _write_wireframe(
        skeleton, output_dir / f"{name}_wireframe.csv", origin
    )
    return {
        "name": name,
        "foreground_voxels": int(mask.sum()),
        "skeleton_voxels": int(skeleton.sum()),
        "displayed_voxel_points": voxel_count,
        "displayed_wireframe_edges": edge_count,
    }


def main() -> None:
    """Run the compact no-filter and Frangi preprocessing comparison."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sigmas",
        default="1,2",
        help="Comma-separated filter scales in sampled voxels (default: 1,2).",
    )
    parser.add_argument(
        "--filter",
        choices=("frangi", "sato"),
        default="frangi",
        help="Tubular-structure filter to apply (default: frangi).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Destination directory (default: {OUTPUT_DIR}).",
    )
    arguments = parser.parse_args()
    sigmas = tuple(float(value) for value in arguments.sigmas.split(","))
    if not sigmas or any(sigma <= 0 for sigma in sigmas):
        raise ValueError("sigmas must contain one or more positive values")

    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = tifffile.memmap(INPUT_PATH)
    sampled, origin = _centered_sample(volume)
    normalized = sampled.astype(np.float32) / np.iinfo(sampled.dtype).max

    unfiltered_threshold = float(threshold_otsu(normalized))
    unfiltered_mask = normalized >= unfiltered_threshold

    filter_function = frangi if arguments.filter == "frangi" else sato
    filter_response = filter_function(normalized, sigmas=sigmas, black_ridges=False)
    filter_threshold = float(threshold_otsu(filter_response))
    filter_mask = filter_response >= filter_threshold

    variants = [
        _process_variant("unfiltered", unfiltered_mask, origin, output_dir),
        _process_variant(arguments.filter, filter_mask, origin, output_dir),
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "input": str(INPUT_PATH),
                "source_shape_zyx": list(volume.shape),
                "crop_origin_zyx": list(origin),
                "crop_size": CROP_SIZE,
                "sample_stride": SAMPLE_STRIDE,
                "working_shape_zyx": list(sampled.shape),
                "unfiltered_otsu_threshold_normalized": unfiltered_threshold,
                "filter": arguments.filter,
                "sigmas": list(sigmas),
                "filter_otsu_threshold": filter_threshold,
                "variants": variants,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
