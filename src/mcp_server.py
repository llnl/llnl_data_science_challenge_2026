import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from fastmcp import FastMCP
from skimage.filters import threshold_otsu

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
from element_views import save_junction_views, save_strut_views
from overlay_registered_nodes_histogram import sample_volume_at_nodes
from strut_center_intensity_histogram import load_lattice
from strut_scan import (
    STATUS_MISSING,
    STATUS_PARTIAL,
    STATUS_PRESENT,
    scan_struts,
    score_strut,
    summarize_strut_scan,
    write_strut_csv,
)

# Two bisections cut the scored span into three segments, which is the coarsest
# division that can distinguish a strut absent along its whole length from one
# broken over part of it. Fixed rather than exposed: the caller already has two
# parameters shaping the same region, and a third that only subdivides it would
# be tuned against the same evidence as the other two.
STRUT_BISECTIONS = 2

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
            _describe_worst_band(summary["dark_fraction_by_band"], "junctions"),
            _describe_candidates(summary["candidates"], "junction_id", "junction"),
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
def scan_lattice_struts(
    volume_path: str,
    registered_json_path: str,
    output_dir: str,
    radius: int = 8,
    cap_voxels: float = 14.0,
) -> str:
    """Score every lattice strut against a CT volume with a capped cylinder probe.

    Wraps a cylinder of ``radius`` around each strut's axis, cuts ``cap_voxels``
    off each end, divides what remains into **three** abutting segments, and
    takes the brightest voxel in each. A segment is dark when that maximum falls
    below the full-volume Otsu threshold::

        -[===|===|===]-      - = cap (not scored)   [...] = scored span

    - **missing** -- all three segments dark. No material along the strut.
    - **partial** -- some but not all dark. Material over part of the span only,
      which is what a break looks like.
    - **present** -- none dark.

    The two parameters do separate jobs, which is the point of the shape.
    ``radius`` is measured perpendicular to the axis and absorbs residual
    registration drift. ``cap_voxels`` is measured along the axis and excludes
    the endpoint junctions' own material -- twelve struts meet at a junction and
    the blob stays bright when any one is absent, so a probe reaching into it
    reports on the junction rather than the strut. Set the cap past that reach
    (about 12 voxels on the 9x9x9 octet specimens) or genuinely absent struts get
    demoted to **partial** by junction bleed. Set it too high and the scored span
    shrinks until a real break can hide inside a cap.

    What this still cannot see, and a clean result does not rule out: a **thin**
    strut. The statistic is a maximum, so an under-thickness strut has a bright
    voxel in every segment and reads as present.

    Run the junction scan first: this tool knows nothing about the specimen and
    suppresses nothing, so struts belonging to a systematic region are separated
    out by looking their endpoint junctions up in the junction scan's results.
    Use ``visualize_lattice_element`` to see what a given ``radius``/``cap_voxels``
    pair actually covers before trusting its counts.

    Args:
        volume_path: Existing 3D CT volume as ``.npy``, ``.tif``, or ``.tiff``.
        registered_json_path: Lattice JSON registered to this volume's voxel
            coordinates. Nominal or raw design JSONs are not aligned and will
            flag nearly everything.
        output_dir: Directory for the CSV and summary JSON. Both filenames carry
            the radius and the cap, so a sweep does not overwrite itself.
        radius: Cylinder radius in voxels, perpendicular to the strut axis. Use
            the radius settled on for the junction scan of the same volume.
        cap_voxels: Voxels trimmed from each end of the strut before scoring.
            Must be below half the shortest strut's length.

    Returns:
        A status message with the counts, the strongest spatial signal, the
        candidate strut ids, and the output paths, or an error message.
    """
    try:
        volume_input = _validate_volume_input(volume_path)
        registered_path = _validate_json_input(registered_json_path, "registered_json_path")
        output_directory = Path(output_dir).expanduser()

        volume = _load_volume(volume_input)
        result = scan_struts(
            volume,
            registered_path,
            radius=radius,
            cap_voxels=cap_voxels,
            n_bisections=STRUT_BISECTIONS,
        )
        summary = summarize_strut_scan(result)

        output_directory.mkdir(parents=True, exist_ok=True)
        stem = f"strut_scan_r{result.radius}_c{_format_cap(result.cap_voxels)}"
        csv_path = write_strut_csv(output_directory / f"{stem}.csv", result)
        summary_path = output_directory / f"{stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        lines = [
            f"Scanned {registered_path.name} against {volume_input.name} at radius "
            f"{result.radius}, cap {result.cap_voxels:g}, "
            f"{result.n_segments} segment(s) per strut.",
            f"{summary['n_struts']} struts across {summary['n_junctions']} merged "
            f"junctions; full-volume Otsu threshold "
            f"{summary['otsu_threshold']:.0f}.",
            f"{summary['n_missing']} missing "
            f"({_format_fraction(summary['missing_fraction'])}), "
            f"{summary['n_partial']} partial "
            f"({_format_fraction(summary['partial_fraction'])}).",
            _describe_worst_band(summary["missing_fraction_by_band"], "struts"),
            _describe_candidates(summary["candidates"], "strut_id", "strut"),
            f"Per-strut CSV: {csv_path}; summary JSON: {summary_path}.",
        ]
        return " ".join(lines)
    except Exception as exc:
        return f"Error scanning lattice struts: {exc}"


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


@mcp.tool()
def visualize_lattice_element(
    volume_path: str,
    registered_json_path: str,
    element: str,
    element_id: int,
    output_dir: str,
    radius: int = 8,
    cap_voxels: float = 14.0,
) -> str:
    """Render one junction or one strut as three orthogonal projections.

    This is the view for judging a *single* element, which the whole-slice
    overlays cannot do: one plane through a 55-voxel strut shows a few voxels of
    it, and a candidate can hide between slices. Here the element's own probe
    region is cropped, padded, and projected along z, y and x, with the probe
    drawn to scale on each image.

    Use it for two things:

    - **Choosing ``radius`` and ``cap_voxels``.** At a good radius the cylinder's
      band covers the strut despite the residual drift; at a good cap the scored
      span visibly stops short of both junction blobs. Sweep the parameters and
      look, rather than picking the pair that produces an agreeable count.
    - **Verifying a candidate.** A flagged element should show empty space inside
      the probe while its surroundings show material.

    The element is scored on its own here, so this does not re-run a whole scan.
    Segments are drawn red when dark and green when bright; a junction's circle
    is colored the same way.

    Args:
        volume_path: Existing 3D CT volume as ``.npy``, ``.tif``, or ``.tiff``.
        registered_json_path: Lattice JSON registered to this volume.
        element: ``"junction"`` or ``"strut"``.
        element_id: For ``"junction"``, a **merged** junction id as reported by
            ``scan_lattice_junctions`` -- not a JSON entry id. For ``"strut"``,
            a strut id, which is the JSON's own.
        output_dir: Directory for the three images.
        radius: Sphere radius for a junction, cylinder radius for a strut.
        cap_voxels: Voxels excluded at each end of a strut. Ignored for a
            junction.

    Returns:
        A status message with the element's measured intensities, its status,
        and the three image paths, or an error message.
    """
    try:
        if element not in ("junction", "strut"):
            raise ValueError(
                f'element must be "junction" or "strut", got {element!r}'
            )
        if radius < 1:
            raise ValueError(
                f"radius must be a positive number of voxels, got {radius}"
            )

        volume_input = _validate_volume_input(volume_path)
        registered_path = _validate_json_input(registered_json_path, "registered_json_path")
        output_directory = Path(output_dir).expanduser()

        volume = _load_volume(volume_input)
        threshold = float(threshold_otsu(volume))
        positions_xyz, strut_junction_ids, _ = load_lattice(registered_path)

        if element == "junction":
            if not 0 <= element_id < len(positions_xyz):
                raise ValueError(
                    f"junction id must be between 0 and {len(positions_xyz) - 1}, "
                    f"got {element_id}"
                )
            position = positions_xyz[element_id]
            intensity, _ = sample_volume_at_nodes(volume, position[None, :], radius)
            dark = bool(intensity[0] < threshold)
            stem = f"junction_{element_id}_r{radius}"
            paths = save_junction_views(
                volume, position, radius, output_directory, stem, dark=dark
            )
            return (
                f"Junction {element_id} at "
                f"({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f}), "
                f"radius {radius}: brightest voxel {float(intensity[0]):.0f} "
                f"against Otsu {threshold:.0f} -- "
                f"{'DARK (candidate)' if dark else 'bright (present)'}. "
                f"Images: {', '.join(str(p) for p in paths)}."
            )

        if not 0 <= element_id < len(strut_junction_ids):
            raise ValueError(
                f"strut id must be between 0 and {len(strut_junction_ids) - 1}, "
                f"got {element_id}"
            )
        first, second = strut_junction_ids[element_id]
        start_xyz, end_xyz = positions_xyz[first], positions_xyz[second]
        intensities, lower, upper = score_strut(
            volume,
            start_xyz,
            end_xyz,
            radius=radius,
            cap_voxels=cap_voxels,
            n_bisections=STRUT_BISECTIONS,
            strut_id=element_id,
        )
        segment_dark = intensities < threshold
        status = (
            STATUS_MISSING
            if segment_dark.all()
            else STATUS_PARTIAL
            if segment_dark.any()
            else STATUS_PRESENT
        )
        stem = f"strut_{element_id}_r{radius}_c{_format_cap(cap_voxels)}"
        paths = save_strut_views(
            volume,
            start_xyz,
            end_xyz,
            radius,
            cap_voxels,
            lower,
            upper,
            output_directory,
            stem,
            segment_dark=segment_dark,
        )
        readings = ", ".join(
            f"{float(v):.0f}{'*' if d else ''}"
            for v, d in zip(intensities, segment_dark)
        )
        return (
            f"Strut {element_id} joins junctions {first} and {second}, "
            f"radius {radius}, cap {cap_voxels:g}, {len(intensities)} segments. "
            f"Segment maxima (start to end, * = dark): {readings} against Otsu "
            f"{threshold:.0f} -- {status.upper()}. "
            f"Images: {', '.join(str(p) for p in paths)}."
        )
    except Exception as exc:
        return f"Error visualizing lattice element: {exc}"


def _format_cap(cap_voxels: float) -> str:
    """Render a cap for a filename, dropping a trailing ``.0`` so 14.0 reads 14."""
    return f"{cap_voxels:g}".replace(".", "p")


def _format_fraction(fraction: float | None) -> str:
    """Render an optional fraction as a percentage, or "n/a" when undefined."""
    return "n/a" if fraction is None else f"{fraction:.2%}"


def _describe_worst_band(bands_by_axis: dict, noun: str) -> str:
    """Name the single band with the highest flagged fraction across all axes.

    A flagged set concentrated in one band is the cheapest systematic signal
    there is: drift shows up as a gradient along one axis, a missing region as
    one hot band, and stochastic absence as no band standing out at all.
    """
    ranked = [
        (band["flagged_fraction"], name, band)
        for name, bands in bands_by_axis.items()
        for band in bands
        if band["flagged_fraction"] is not None
    ]
    if not ranked:
        return f"No {noun} to band."
    fraction, name, band = max(ranked, key=lambda item: item[0])
    return (
        f"Highest flagged fraction in any band: {fraction:.2%} "
        f"({name} {band['lo']:.0f}-{band['hi']:.0f}, {band['n_flagged']}/"
        f"{band['n_total']})."
    )


CANDIDATE_ID_LIMIT = 20


def _describe_candidates(candidates: list, id_key: str, noun: str) -> str:
    """List candidate ids, truncating to keep the status message readable."""
    ids = [candidate[id_key] for candidate in candidates]
    if not ids:
        return f"No candidate {noun}."
    if len(ids) > CANDIDATE_ID_LIMIT:
        shown = ", ".join(str(i) for i in ids[:CANDIDATE_ID_LIMIT])
        return f"First {CANDIDATE_ID_LIMIT} candidate ids: {shown} (see CSV for all)."
    return f"Candidate {noun} ids: {', '.join(str(i) for i in ids)}."


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
