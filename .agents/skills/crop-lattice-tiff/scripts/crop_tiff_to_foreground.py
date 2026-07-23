"""Crop a 3D TIFF to the threshold-defined foreground bounding box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def foreground_bounds(input_path: Path, threshold: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return half-open Z, Y, X bounds containing every voxel >= threshold."""
    z_start: int | None = None
    z_stop = 0
    y_start: int | None = None
    y_stop = 0
    x_start: int | None = None
    x_stop = 0

    with tifffile.TiffFile(input_path) as source:
        if source.series[0].ndim != 3:
            raise ValueError(f"expected a 3D TIFF, found shape {source.series[0].shape}")
        for z_index, page in enumerate(source.pages):
            foreground = page.asarray() >= threshold
            if not foreground.any():
                continue
            y_indices, x_indices = np.nonzero(foreground)
            z_start = z_index if z_start is None else z_start
            z_stop = z_index + 1
            y_start = int(y_indices.min()) if y_start is None else min(y_start, int(y_indices.min()))
            y_stop = max(y_stop, int(y_indices.max()) + 1)
            x_start = int(x_indices.min()) if x_start is None else min(x_start, int(x_indices.min()))
            x_stop = max(x_stop, int(x_indices.max()) + 1)

    if z_start is None or y_start is None or x_start is None:
        raise ValueError(f"no voxels met threshold {threshold} in {input_path}")
    return (z_start, z_stop), (y_start, y_stop), (x_start, x_stop)


def crop_tiff(
    input_path: Path,
    output_path: Path,
    threshold: int,
    metadata_path: Path,
    z_start_override: int | None = None,
    z_stop_override: int | None = None,
) -> dict[str, object]:
    """Write a cropped TIFF and machine-readable metadata without altering input."""
    z_bounds, y_bounds, x_bounds = foreground_bounds(input_path, threshold)
    if z_start_override is not None:
        if z_start_override < 0 or z_start_override >= z_bounds[1]:
            raise ValueError("z_start_override must be inside the detected volume")
        z_bounds = (max(z_bounds[0], z_start_override), z_bounds[1])
    if z_stop_override is not None:
        if z_stop_override <= z_bounds[0] or z_stop_override > z_bounds[1]:
            raise ValueError("z_stop_override must be inside the detected volume")
        z_bounds = (z_bounds[0], min(z_bounds[1], z_stop_override))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(input_path) as source:
        input_shape = tuple(int(value) for value in source.series[0].shape)
        dtype = source.series[0].dtype
        with tifffile.TiffWriter(output_path, bigtiff=False) as destination:
            for z_index in range(*z_bounds):
                cropped_page = source.pages[z_index].asarray()[slice(*y_bounds), slice(*x_bounds)]
                destination.write(cropped_page, photometric="minisblack", contiguous=True)

    metadata = {
        "source_tiff": str(input_path),
        "output_tiff": str(output_path),
        "threshold": threshold,
        "z_start_override": z_start_override,
        "z_stop_override": z_stop_override,
        "input_shape_zyx": input_shape,
        "crop_start_zyx": [z_bounds[0], y_bounds[0], x_bounds[0]],
        "crop_stop_zyx": [z_bounds[1], y_bounds[1], x_bounds[1]],
        "output_shape_zyx": [z_bounds[1] - z_bounds[0], y_bounds[1] - y_bounds[0], x_bounds[1] - x_bounds[0]],
        "dtype": str(dtype),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("output_tiff", type=Path)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--z-start", type=int, default=None)
    parser.add_argument("--z-stop", type=int, default=None)
    arguments = parser.parse_args()
    print(
        json.dumps(
            crop_tiff(
                arguments.input_tiff,
                arguments.output_tiff,
                arguments.threshold,
                arguments.metadata,
                arguments.z_start,
                arguments.z_stop,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
