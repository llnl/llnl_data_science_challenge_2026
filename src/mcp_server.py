from contextlib import redirect_stdout
import io
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastmcp import FastMCP

from skeletonization import skeletonize_mask

mcp = FastMCP("CT Segmentation")


@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """Segment a 3D NumPy CT volume using a density threshold."""
    try:
        data = np.load(input_filepath)
        mask = (data >= threshold).astype(np.uint8)
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_filepath, mask)
        return f"Saved segmentation mask to {output_filepath}. Shape: {mask.shape}"
    except Exception as error:
        return f"Segmentation failed: {error}"


@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """Save a visualization of one slice from a 3D NumPy volume."""
    try:
        data = np.load(input_filepath)
        if data.ndim != 3:
            return f"Visualization failed: expected a 3D array, got {data.ndim}D."
        if axis not in (0, 1, 2):
            return "Visualization failed: axis must be 0, 1, or 2."
        if not 0 <= slice_index < data.shape[axis]:
            return f"Visualization failed: slice_index must be between 0 and {data.shape[axis] - 1} for axis {axis}."
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(np.take(data, slice_index, axis=axis), cmap="gray")
        plt.title(f"Slice {slice_index} along axis {axis}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_filepath, dpi=150)
        plt.close()
        return f"Saved slice visualization to {output_filepath}."
    except Exception as error:
        return f"Visualization failed: {error}"


@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """Create a skeleton from a 3D segmentation mask via skeletonize_mask."""
    try:
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with redirect_stdout(io.StringIO()):
            skeleton = skeletonize_mask(input_filepath, output_filepath)
        if skeleton is None:
            return "Skeletonization failed: no skeleton was created."
        return f"Saved skeleton to {output_filepath}. Shape: {skeleton.shape}; skeleton voxels: {np.count_nonzero(skeleton)}."
    except Exception as error:
        return f"Skeletonization failed: {error}"


if __name__ == "__main__":
    mcp.run()
