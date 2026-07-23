"""Create a scaled 3D skeleton wireframe from a thresholded CT volume."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "skan_numba_cache")
)
from skan import Skeleton


def create_threshold_wireframe(
    volume_path: Path,
    output_skeleton_path: Path,
    output_wireframe_path: Path,
    threshold: int,
    downsample_factor: int,
) -> tuple[tuple[int, int, int], int, int]:
    """Threshold, skeletonize, and save connected branch paths at original scale.

    The volume is sampled every ``downsample_factor`` voxels before thinning to
    bound memory use. Coordinates in the Parquet wireframe are multiplied back
    to original voxel units.
    """
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be at least 1")
    volume = np.load(volume_path, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D CT volume, found shape {volume.shape}")

    mask = volume[::downsample_factor, ::downsample_factor, ::downsample_factor] > threshold
    skeleton_image = skeletonize(mask)
    output_skeleton_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_skeleton_path, skeleton_image)

    skeleton = Skeleton(skeleton_image)
    paths = skeleton.paths_list()
    rows: list[np.ndarray] = []
    for node_ids in paths:
        coordinates_zyx = skeleton.coordinates[np.asarray(node_ids)] * downsample_factor
        rows.append(coordinates_zyx[:, [2, 1, 0]].astype(np.float32, copy=False))
        rows.append(np.full((1, 3), np.nan, dtype=np.float32))
    line_data = np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.float32)
    output_wireframe_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(line_data, columns=["x", "y", "z"]).to_parquet(
        output_wireframe_path, index=False
    )
    return tuple(int(size) for size in skeleton_image.shape), int(skeleton_image.sum()), len(paths)


def main() -> None:
    """Run the threshold-to-wireframe conversion from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_volume", type=Path)
    parser.add_argument("output_skeleton", type=Path)
    parser.add_argument("output_wireframe", type=Path)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--downsample-factor", type=int, default=2)
    arguments = parser.parse_args()
    shape, voxel_count, branch_count = create_threshold_wireframe(
        arguments.input_volume,
        arguments.output_skeleton,
        arguments.output_wireframe,
        arguments.threshold,
        arguments.downsample_factor,
    )
    print(
        f"skeleton shape={shape}, skeleton_voxels={voxel_count}, branch_paths={branch_count}"
    )


if __name__ == "__main__":
    main()
