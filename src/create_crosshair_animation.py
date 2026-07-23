"""Create a Crosshair animation from the 9x9x9 octet-lattice TIFF stack.

The animation uses every tenth Z slice, preserving the source in-plane
resolution while JPEG-compressing individual frames for efficient browser
transfer.  Start the Crosshair UI after running this script and select the
``codex_9x9x9_octet`` view.
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


SOURCE_TIFF = Path("data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif")
VIEW_NAME = "codex_9x9x9_octet"
PANEL_ID = "codex_9x9x9_octet_animation"
FRAME_STRIDE = 10
JPEG_QUALITY = 75


def encode_frame(frame: np.ndarray, lower: float, upper: float) -> str:
    """Encode one uint16 TIFF slice as a contrast-normalized JPEG data URI."""
    scaled = np.clip((frame.astype(np.float32) - lower) / (upper - lower), 0, 1)
    image = Image.fromarray(np.round(scaled * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_animation_spec(volume: np.ndarray) -> dict:
    """Build a Plotly image animation spec from every tenth Z slice."""
    sampled = volume[::FRAME_STRIDE]
    lower, upper = np.percentile(sampled, (1, 99.5))
    if upper <= lower:
        upper = lower + 1

    sources = [encode_frame(frame, lower, upper) for frame in sampled]
    frame_labels = [str(index * FRAME_STRIDE) for index in range(len(sources))]
    frames = [
        {
            "name": label,
            "data": [{"type": "image", "source": source}],
        }
        for label, source in zip(frame_labels, sources, strict=True)
    ]

    return {
        "data": [{"type": "image", "source": sources[0]}],
        "frames": frames,
        "layout": {
            "margin": {"l": 50, "r": 20, "t": 20, "b": 65},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False, "scaleanchor": "x"},
            "sliders": [
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Z slice: "},
                    "pad": {"t": 35},
                    "steps": [
                        {
                            "label": label,
                            "method": "animate",
                            "args": [[label], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                        }
                        for label in frame_labels
                    ],
                }
            ],
            "updatemenus": [
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0,
                    "y": 1.08,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"fromcurrent": True, "frame": {"duration": 110, "redraw": True}}],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                        },
                    ],
                }
            ],
        },
    }


async def publish_animation(spec: dict) -> str:
    """Create or replace the dedicated Crosshair view and return its URL."""
    from crosshair import client

    url = await client.ensure()
    await client.call("create_view", name=VIEW_NAME, rows=1, cols=1)
    await client.call(
        "upsert_panel",
        view=VIEW_NAME,
        panel_id=PANEL_ID,
        title="9x9x9 octet lattice — every 10th Z slice",
        type="plotly",
        spec=spec,
        row=1,
        col=1,
        base_dir=str(Path.cwd()),
    )
    await client.call(
        "add_note",
        text="Created codex_9x9x9_octet using 77 frames: Z slices 0 through 760 at a stride of 10.",
    )
    return url


def main() -> None:
    """Load the TIFF and publish the animation to Crosshair."""
    if not SOURCE_TIFF.is_file():
        raise FileNotFoundError(f"Missing TIFF stack: {SOURCE_TIFF}")
    volume = tifffile.imread(SOURCE_TIFF)
    print(asyncio.run(publish_animation(build_animation_spec(volume))))


if __name__ == "__main__":
    main()
