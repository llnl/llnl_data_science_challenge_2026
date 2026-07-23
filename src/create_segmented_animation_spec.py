"""Build a Plotly animation spec for an axial binary segmentation volume."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _encode_mask(mask: np.ndarray, max_dimension: int) -> str:
    """Return a compact black-and-white JPEG data URI for one mask slice."""
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.NEAREST,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_spec(
    mask_path: Path, frame_stride: int, max_dimension: int, threshold: int | None = None
) -> dict[str, object]:
    """Create image frames, a slider, and controls for a mask or CT NPY."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if mask.ndim != 3:
        raise ValueError(f"expected a 3D mask, found shape {mask.shape}")

    indices = list(range(0, mask.shape[0], frame_stride))
    labels = [str(index) for index in indices]
    sources = [
        _encode_mask(mask[index] > threshold if threshold is not None else mask[index], max_dimension)
        for index in indices
    ]
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
                "currentvalue": {"prefix": "Z slice: "},
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
    """Write the animation spec to a JSON file and stdout for Crosshair use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mask", type=Path)
    parser.add_argument("output_spec", type=Path)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-dimension", type=int, default=128)
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Apply a strict greater-than segmentation threshold to a raw CT volume.",
    )
    arguments = parser.parse_args()
    spec = build_spec(
        arguments.input_mask,
        arguments.frame_stride,
        arguments.max_dimension,
        arguments.threshold,
    )
    arguments.output_spec.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_spec.write_text(json.dumps(spec), encoding="utf-8")
    print(json.dumps(spec))


if __name__ == "__main__":
    main()
