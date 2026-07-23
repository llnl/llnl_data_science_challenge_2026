"""Skeletonization routines for three-dimensional binary masks."""

from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

def skeletonize_mask(file_path: str | Path, output_path: str | Path) -> np.ndarray:
    """Create and save a 3D boolean skeleton from a nonzero-valued ``.npy`` mask.

    This routine deliberately does not print progress messages: it is called by
    the stdio MCP server, whose standard output is reserved for JSON-RPC.

    Args:
        file_path: Existing NumPy file containing a three-dimensional mask.
        output_path: Destination NumPy path for the boolean skeleton.

    Returns:
        The saved boolean skeleton array.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the input array is not three-dimensional.
    """
    input_path = Path(file_path).expanduser()
    destination_path = Path(output_path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"mask input file not found: {input_path}")

    mask = np.load(input_path, allow_pickle=False)
    if mask.ndim != 3:
        raise ValueError(
            f"expected a 3D mask in {input_path}, found {mask.ndim} dimensions"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton = skeletonize(mask.astype(bool, copy=False))
    np.save(destination_path, skeleton, allow_pickle=False)
    return skeleton

if __name__ == "__main__":
    # Hardcoded parameters for testing
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "unitcell", "unitcell.npy"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "octet_truss_unit_cell_skeleton.npy"))
    
    # Create the data directory if it doesn't exist
    skeletonize_mask(
        file_path=file_path, 
        output_path=output_path
    )
