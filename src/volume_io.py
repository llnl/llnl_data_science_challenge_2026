"""Input/output helpers for volumetric CT datasets."""

from pathlib import Path

import numpy as np
import tifffile


def convert_tiff_to_numpy(
    input_filepath: str | Path, output_filepath: str | Path
) -> Path:
    """Convert a 3D TIFF CT volume to an unmodified NumPy ``.npy`` array.

    Args:
        input_filepath: Existing TIFF file with a ``.tif`` or ``.tiff`` suffix.
        output_filepath: Destination NumPy file. Its parent directories are created.

    Returns:
        The destination path after the array has been saved.

    Raises:
        ValueError: If an input/output suffix is invalid or the TIFF is not 3D.
        FileNotFoundError: If the input file does not exist.
    """
    input_path = Path(input_filepath).expanduser()
    output_path = Path(output_filepath).expanduser()

    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"input file must have a .tif or .tiff extension: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if output_path.suffix.lower() != ".npy":
        raise ValueError(f"output file must have a .npy extension: {output_path}")

    volume = tifffile.imread(input_path)
    if volume.ndim != 3:
        raise ValueError(
            f"expected a 3D TIFF volume in {input_path}, found {volume.ndim} dimensions"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, volume, allow_pickle=False)
    return output_path
