"""Export vertical (constant-x) slices of the unitcell CT volume as PNGs.

The volume is stored as a (z, y, x) stack, with z as the build/vertical axis.
A "vertical slice" here means a constant-x cross-section, i.e. volume[:, :, x],
which shows the z axis running down the image rather than the usual
constant-z (top-down) slice.
"""

from pathlib import Path

import numpy as np
from PIL import Image

INPUT_PATH = Path("data/unitcell/unitcell.npy")
OUTPUT_DIR = Path("outputs/unitcell_vertical_slices")
NUM_SLICES = 5


def main() -> None:
    volume = np.load(INPUT_PATH)
    vmin, vmax = volume.min(), volume.max()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_size = volume.shape[2]
    x_indices = np.linspace(0, x_size - 1, NUM_SLICES + 2, dtype=int)[1:-1]

    for x_index in x_indices:
        vertical_slice = volume[:, :, x_index]
        normalized = (vertical_slice - vmin) / (vmax - vmin)
        image_array = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        output_path = OUTPUT_DIR / f"unitcell_vertical_slice_x{x_index:03d}.png"
        Image.fromarray(image_array, mode="L").save(output_path)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
