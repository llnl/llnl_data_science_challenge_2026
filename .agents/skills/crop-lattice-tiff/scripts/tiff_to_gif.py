"""Create a contrast-normalized axial GIF animation from a 3D TIFF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def write_animation(
    input_path: Path, output_path: Path, frame_stride: int, max_dimension: int
) -> int:
    """Write every ``frame_stride`` axial slice as a GIF frame.

    Grayscale stacks are contrast-stretched to the 1st-99.5th percentile. An
    RGB stack is passed through untouched: it is already uint8 and carries
    marked-up overlays whose colors a per-channel stretch would distort.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    with tifffile.TiffFile(input_path) as source:
        frames = [source.pages[index].asarray() for index in range(0, len(source.pages), frame_stride)]
    is_rgb = frames[0].ndim == 3 and frames[0].shape[-1] == 3
    lower, upper = 0.0, 1.0
    if not is_rgb:
        lower, upper = np.percentile(np.stack(frames), (1, 99.5))
        if upper <= lower:
            upper = lower + 1
    images = []
    for frame in frames:
        if is_rgb:
            image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        else:
            image = Image.fromarray(
                np.round(np.clip((frame.astype(np.float32) - lower) / (upper - lower), 0, 1) * 255).astype(np.uint8),
                mode="L",
            )
        if max(image.size) > max_dimension:
            scale = max_dimension / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
            )
        images.append(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=110, loop=0, optimize=True)
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("output_gif", type=Path)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-dimension", type=int, default=256)
    arguments = parser.parse_args()
    print(
        f"saved {write_animation(arguments.input_tiff, arguments.output_gif, arguments.frame_stride, arguments.max_dimension)} frames"
    )


if __name__ == "__main__":
    main()
