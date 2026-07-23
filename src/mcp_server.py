import math
import os
import tempfile
from pathlib import Path

import numpy as np
from fastmcp import FastMCP

from skeleton_graph import create_skeleton_graph
from skeletonization import skeletonize_mask
from volume_io import convert_tiff_to_numpy

# MCP communicates over standard output. Configure Matplotlib before importing
# pyplot so it does not emit first-run cache warnings in restricted runtimes.
matplotlib_cache_dir = Path(tempfile.gettempdir()) / "ct_segmentation_matplotlib"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")


@mcp.tool()
def convert_tiff_volume(input_filepath: str, output_filepath: str) -> str:
    """Convert a 3D TIFF CT scan to a dtype-preserving NumPy ``.npy`` volume.

    Args:
        input_filepath: Path to an existing 3D ``.tif`` or ``.tiff`` CT volume.
        output_filepath: Destination ``.npy`` file. Missing parent directories are made.

    Returns:
        A status message with the saved NumPy volume location, or an error message.
    """
    try:
        output_path = convert_tiff_to_numpy(input_filepath, output_filepath)
        return f"NumPy volume saved to {output_path}"
    except Exception as exc:
        return f"Error converting TIFF volume: {exc}"


@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path = _validate_npy_input(input_filepath)
        output_path = _validate_npy_output(output_filepath)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be a finite number")

        ct_data = np.load(input_path, allow_pickle=False)
        _validate_3d_array(ct_data, input_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        segmentation = (ct_data >= threshold).astype(np.uint8)
        np.save(output_path, segmentation)
        return f"Segmentation saved to {output_path}"
    except Exception as exc:
        return f"Error segmenting CT dataset: {exc}"

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy file and saves a visualization of a specific slice to an image file.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    figure = None
    try:
        input_path = _validate_npy_input(input_filepath)
        output_path = Path(output_filepath).expanduser()
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")

        ct_data = np.load(input_path, allow_pickle=False)
        _validate_3d_array(ct_data, input_path)
        if slice_index < 0 or slice_index >= ct_data.shape[axis]:
            raise ValueError(
                f"slice_index must be between 0 and {ct_data.shape[axis] - 1} "
                f"for axis {axis}"
            )

        slice_data = np.take(ct_data, slice_index, axis=axis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots()
        axes.imshow(slice_data, cmap="gray")
        axes.axis("off")
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
        return f"Slice visualization saved to {output_path}"
    except Exception as exc:
        return f"Error visualizing CT slice: {exc}"
    finally:
        if figure is not None:
            plt.close(figure)

@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path = _validate_npy_input(input_filepath)
        output_path = _validate_npy_output(output_filepath)

        mask = np.load(input_path, mmap_mode="r", allow_pickle=False)
        _validate_3d_array(mask, input_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = skeletonize_mask(str(input_path), str(output_path))
        if result is None or not output_path.is_file():
            raise RuntimeError("skeletonization did not create an output file")
        return f"Skeleton saved to {output_path}"
    except Exception as exc:
        return f"Error skeletonizing mask: {exc}"


@mcp.tool()
def graph_skeleton(
    input_filepath: str, graph_output_filepath: str, histogram_output_filepath: str
) -> str:
    """Create a Skan voxel-neighbor graph and node-degree histogram from a skeleton.

    Args:
        input_filepath: Existing 3D skeleton stored as a nonzero-valued ``.npy`` file.
        graph_output_filepath: Destination ``.npz`` file for the sparse adjacency graph.
            A ``.coordinates.npy`` sidecar is also created; graph row IDs index this array.
        histogram_output_filepath: Destination CSV containing ``degree`` and ``node_count``.

    Returns:
        A status message listing the graph, coordinate mapping, histogram, and counts.
    """
    try:
        result = create_skeleton_graph(
            input_filepath, graph_output_filepath, histogram_output_filepath
        )
        return (
            f"Skeleton graph saved to {result.graph_path} with {result.node_count:,} "
            f"nodes and {result.edge_count:,} edges; coordinates saved to "
            f"{result.coordinates_path}; degree histogram saved to {result.histogram_path}"
        )
    except Exception as exc:
        return f"Error creating skeleton graph: {exc}"


def _validate_npy_input(filepath: str) -> Path:
    """Return a validated path to an existing NumPy array file."""
    path = Path(filepath).expanduser()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"input file must have a .npy extension: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    return path


def _validate_npy_output(filepath: str) -> Path:
    """Return a validated NumPy output path."""
    path = Path(filepath).expanduser()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"output file must have a .npy extension: {path}")
    return path


def _validate_3d_array(array: np.ndarray, filepath: Path) -> None:
    """Raise a useful error when an input array is not volumetric."""
    if array.ndim != 3:
        raise ValueError(
            f"expected a 3D array in {filepath}, found {array.ndim} dimensions"
        )

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
