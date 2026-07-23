"""Create a compact Plotly image-animation specification from a 3D TIFF."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def _encode_frame(frame: np.ndarray, lower: float, upper: float, max_dimension: int) -> str:
    image = Image.fromarray(
        np.round(np.clip((frame.astype(np.float32) - lower) / (upper - lower), 0, 1) * 255).astype(np.uint8),
        mode="L",
    )
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_spec(input_path: Path, frame_stride: int, max_dimension: int) -> dict[str, object]:
    """Build a portable Plotly spec with image frames, buttons, and a slider."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    with tifffile.TiffFile(input_path) as source:
        indices = list(range(0, len(source.pages), frame_stride))
        slices = [source.pages[index].asarray() for index in indices]
    lower, upper = np.percentile(np.stack(slices), (1, 99.5))
    if upper <= lower:
        upper = lower + 1
    labels = [str(index) for index in indices]
    sources = [_encode_frame(frame, lower, upper, max_dimension) for frame in slices]
    return {
        "data": [{"type": "image", "source": sources[0]}],
        "frames": [
            {"name": label, "data": [{"type": "image", "source": source}]}
            for label, source in zip(labels, sources, strict=True)
        ],
        "layout": {
            "margin": {"l": 20, "r": 20, "t": 20, "b": 65},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False, "scaleanchor": "x"},
            "sliders": [{
                "active": 0,
                "currentvalue": {"prefix": "Cropped Z slice: "},
                "pad": {"t": 35},
                "steps": [
                    {"label": label, "method": "animate", "args": [[label], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
                    for label in labels
                ],
            }],
            "updatemenus": [{
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 110, "redraw": True}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
                ],
            }],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("output_spec", type=Path)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-dimension", type=int, default=128)
    arguments = parser.parse_args()
    spec = build_spec(arguments.input_tiff, arguments.frame_stride, arguments.max_dimension)
    arguments.output_spec.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_spec.write_text(json.dumps(spec), encoding="utf-8")
    print(json.dumps(spec))


if __name__ == "__main__":
    main()
