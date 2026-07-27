import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile
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

# Imported after the Agg block: junction_scan pulls in pyplot transitively, and
# a default backend selected first could try to reach a display.
from junction_scan import (
    save_marked_tiff,
    save_mip_overlay_with_status,
    save_slice_overlay_with_radius,
    scan_junctions,
    summarize_scan,
    write_junction_csv,
)

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


@mcp.tool()
def scan_lattice_junctions(
    volume_path: str,
    registered_json_path: str,
    output_dir: str,
    radius: int = 8,
    write_marked_tiff: bool = False,
) -> str:
    """Score every lattice junction against a CT volume and report the dark ones.

    Merges the JSON's co-located junction entries, samples the brightest voxel
    within ``radius`` of each merged position, and flags junctions falling below
    the full-volume Otsu threshold. Writes a per-junction CSV and a summary JSON
    carrying the spatial statistics needed to tell misregistration from a
    missing region from genuinely absent junctions.

    A junction reads dark only when *every* incident strut is absent, so a clean
    result here does not mean the specimen has no missing struts.

    Args:
        volume_path: Existing 3D CT volume as ``.npy``, ``.tif``, or ``.tiff``.
        registered_json_path: Lattice JSON registered to this volume's voxel
            coordinates. Nominal or raw design JSONs are not aligned and will
            flag nearly everything.
        output_dir: Directory for the CSV and summary JSON. Both filenames carry
            the radius, so repeated calls at different radii do not overwrite.
        radius: Sampling sphere radius in voxels. This absorbs residual
            registration drift; 8 is the validated starting point for the 9x9x9
            octet specimens.
        write_marked_tiff: Also write an RGB TIFF with candidates painted. Costs
            roughly three bytes per input voxel in memory and on disk.

    Returns:
        A status message with the counts, the strongest spatial signals, the
        candidate ids, and the output paths, or an error message.
    """
    try:
        volume_input = _validate_volume_input(volume_path)
        registered_path = _validate_json_input(registered_json_path, "registered_json_path")
        output_directory = Path(output_dir).expanduser()

        volume = _load_volume(volume_input)
        result = scan_junctions(volume, registered_path, radius=radius)
        summary = summarize_scan(result)

        output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = write_junction_csv(
            output_directory / f"junction_scan_r{result.radius}.csv", result
        )
        summary_path = output_directory / f"junction_scan_r{result.radius}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        lines = [
            f"Scanned {registered_path.name} against {volume_input.name} at radius "
            f"{result.radius}.",
            f"{summary['n_junctions']} junctions (merged from {summary['n_entries']} "
            f"JSON entries, degree {summary['degree_min']}-{summary['degree_max']}); "
            f"full-volume Otsu threshold {summary['otsu_threshold']:.0f}.",
            f"{summary['n_dark']} dark of {summary['n_junctions']} junctions "
            f"({_format_fraction(summary['dark_fraction'])}).",
            f"Largest dark component: {summary['dark_components']['largest']} "
            f"junction(s) across {summary['dark_components']['n_components']} "
            f"component(s).",
            _describe_worst_band(summary),
            _describe_candidates(summary),
        ]

        if write_marked_tiff:
            tiff_path = save_marked_tiff(
                volume,
                result,
                output_directory / f"junction_scan_r{result.radius}_marked.tif",
            )
            lines.append(f"Marked TIFF: {tiff_path}.")
        lines.append(f"Per-junction CSV: {csv_path}; summary JSON: {summary_path}.")
        return " ".join(lines)
    except Exception as exc:
        return f"Error scanning lattice junctions: {exc}"


@mcp.tool()
def visualize_junction_overlay(
    volume_path: str,
    registered_json_path: str,
    output_path: str,
    mode: str = "slice",
    axis: int = 0,
    slice_index: int = -1,
    radius: int = 8,
) -> str:
    """Render the lattice over the CT volume, colored by junction status.

    Use ``mode="mip"`` first to judge coarse alignment: if the junction markers
    do not sit on the projected lattice, the JSON does not belong to this scan
    and no per-junction number derived from it means anything. Use
    ``mode="slice"`` to inspect a specific candidate, where each junction's
    sampling sphere is drawn to scale as its cross-section in that plane, so the
    picture shows exactly the voxels a flag was computed from.

    Junctions are green when bright and red when flagged.

    Args:
        volume_path: Existing 3D CT volume as ``.npy``, ``.tif``, or ``.tiff``.
        registered_json_path: Lattice JSON registered to this volume.
        output_path: Destination image; the suffix picks the format (e.g. .png).
        mode: ``"slice"`` for a single plane, ``"mip"`` for a maximum-intensity
            projection along ``axis``.
        axis: Volume axis to slice or project along (0 = z, 1 = y, 2 = x).
        slice_index: Slice index for ``mode="slice"``; -1 selects the median
            junction coordinate along ``axis``. Ignored for ``mode="mip"``.
        radius: Sampling sphere radius in voxels, matching the scan being checked.

    Returns:
        A status message with the saved image location, or an error message.
    """
    try:
        if mode not in ("slice", "mip"):
            raise ValueError(f'mode must be "slice" or "mip", got {mode!r}')

        volume_input = _validate_volume_input(volume_path)
        registered_path = _validate_json_input(registered_json_path, "registered_json_path")
        destination = Path(output_path).expanduser()

        volume = _load_volume(volume_input)
        result = scan_junctions(volume, registered_path, radius=radius)

        if mode == "mip":
            save_mip_overlay_with_status(volume, result, destination, axis=axis)
            return (
                f"Junction overlay saved to {destination} "
                f"({'zyx'[axis]} max-intensity projection, radius {result.radius}, "
                f"{int(result.dark.sum())} candidates of {result.n_junctions} "
                f"junctions)"
            )

        index = save_slice_overlay_with_radius(
            volume,
            result,
            destination,
            axis=axis,
            index=None if slice_index < 0 else slice_index,
        )
        return (
            f"Junction overlay saved to {destination} (slice {'zyx'[axis]}={index}, "
            f"radius {result.radius}, {int(result.dark.sum())} candidates of "
            f"{result.n_junctions} junctions)"
        )
    except Exception as exc:
        return f"Error visualizing junction overlay: {exc}"


def _format_fraction(fraction: float | None) -> str:
    """Render an optional fraction as a percentage, or "n/a" when undefined."""
    return "n/a" if fraction is None else f"{fraction:.2%}"


def _describe_worst_band(summary: dict) -> str:
    """Name the single band with the highest dark fraction across all three axes.

    A dark set concentrated in one band is the cheapest systematic signal there
    is: drift shows up as a gradient along one axis, a missing region as one hot
    band, and stochastic absence as no band standing out at all.
    """
    ranked = [
        (band["dark_fraction"], name, band)
        for name, bands in summary["dark_fraction_by_band"].items()
        for band in bands
        if band["dark_fraction"] is not None
    ]
    if not ranked:
        return "No junctions to band."
    fraction, name, band = max(ranked, key=lambda item: item[0])
    return (
        f"Highest dark fraction in any band: {fraction:.2%} "
        f"({name} {band['lo']:.0f}-{band['hi']:.0f}, {band['n_dark']}/"
        f"{band['n_junctions']})."
    )


CANDIDATE_ID_LIMIT = 20


def _describe_candidates(summary: dict) -> str:
    """List candidate ids, truncating to keep the status message readable."""
    ids = [candidate["junction_id"] for candidate in summary["candidates"]]
    if not ids:
        return "No candidate junctions."
    if len(ids) > CANDIDATE_ID_LIMIT:
        shown = ", ".join(str(i) for i in ids[:CANDIDATE_ID_LIMIT])
        return f"First {CANDIDATE_ID_LIMIT} candidate ids: {shown} (see CSV for all)."
    return f"Candidate junction ids: {', '.join(str(i) for i in ids)}."


def _validate_json_input(filepath: str, parameter: str) -> Path:
    """Return a validated path to an existing JSON lattice description."""
    path = Path(filepath).expanduser()
    if path.suffix.lower() != ".json":
        raise ValueError(f"{parameter} must have a .json extension: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{parameter} not found: {path}")
    return path


def _validate_volume_input(filepath: str) -> Path:
    """Return a validated path to an existing NumPy or TIFF volume."""
    path = Path(filepath).expanduser()
    if path.suffix.lower() not in {".npy", ".tif", ".tiff"}:
        raise ValueError(
            f"volume file must have a .npy, .tif, or .tiff extension: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"volume file not found: {path}")
    return path


def _load_volume(path: Path) -> np.ndarray:
    """Read a validated volume path as a 3D array, from NumPy or TIFF."""
    if path.suffix.lower() == ".npy":
        volume = np.load(path, allow_pickle=False)
    else:
        volume = tifffile.imread(path)
    _validate_3d_array(volume, path)
    return volume


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
