"""Threshold and skeletonize a cropped 3D TIFF volume."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage.morphology import skeletonize


def skeletonize_tiff(input_path: Path, output_path: Path, threshold: int) -> tuple[tuple[int, ...], int]:
    """Save a boolean 3D skeleton for TIFF voxels at or above ``threshold``."""
    volume = tifffile.memmap(input_path)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D TIFF, found shape {volume.shape}")
    skeleton = skeletonize(volume >= threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, skeleton)
    return skeleton.shape, int(skeleton.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("output_skeleton", type=Path)
    parser.add_argument("--threshold", type=int, required=True)
    arguments = parser.parse_args()
    shape, voxel_count = skeletonize_tiff(
        arguments.input_tiff, arguments.output_skeleton, arguments.threshold
    )
    print(f"skeleton shape={shape}, nonzero_voxels={voxel_count}")


if __name__ == "__main__":
    main()
