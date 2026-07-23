"""Crop an existing skeleton using bounds recorded by crop_tiff_to_foreground."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def crop_skeleton(skeleton_path: Path, metadata_path: Path, output_path: Path) -> tuple[int, int, int]:
    """Write the metadata-defined subvolume page-by-page without loading it all."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    z_start, y_start, x_start = metadata["crop_start_zyx"]
    z_stop, y_stop, x_stop = metadata["crop_stop_zyx"]
    skeleton = np.load(skeleton_path, mmap_mode="r", allow_pickle=False)
    if tuple(skeleton.shape) != tuple(metadata["input_shape_zyx"]):
        raise ValueError("skeleton shape does not match crop metadata input shape")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.bool_,
        shape=(z_stop - z_start, y_stop - y_start, x_stop - x_start),
    )
    for target_z, source_z in enumerate(range(z_start, z_stop)):
        cropped[target_z] = skeleton[source_z, y_start:y_stop, x_start:x_stop]
    cropped.flush()
    return cropped.shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_skeleton", type=Path)
    parser.add_argument("crop_metadata", type=Path)
    parser.add_argument("output_skeleton", type=Path)
    arguments = parser.parse_args()
    print(f"cropped skeleton shape={crop_skeleton(arguments.input_skeleton, arguments.crop_metadata, arguments.output_skeleton)}")


if __name__ == "__main__":
    main()
