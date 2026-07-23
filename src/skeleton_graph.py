"""Create a voxel-neighbor graph and degree summary from a 3D skeleton."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz

# Skan uses Numba with on-disk caching. Choose a writable cache directory when
# the Python environment itself is read-only (as it is in some MCP deployments).
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "skan_numba_cache")
)

from skan.csr import skeleton_to_csgraph


@dataclass(frozen=True)
class SkeletonGraphResult:
    """Locations and basic counts describing a graph created from a skeleton."""

    graph_path: Path
    coordinates_path: Path
    histogram_path: Path
    node_count: int
    edge_count: int


def create_skeleton_graph(
    skeleton_path: str | Path,
    graph_path: str | Path,
    histogram_path: str | Path,
) -> SkeletonGraphResult:
    """Convert a 3D skeleton array into a sparse voxel-neighbor graph.

    The adjacency graph is saved in SciPy ``.npz`` sparse-matrix format. Its
    row and column IDs correspond to the voxel coordinates written to the
    ``.coordinates.npy`` sidecar in the same order. The CSV histogram contains
    the number of graph nodes for each node degree.

    Args:
        skeleton_path: Existing 3D ``.npy`` array containing a nonzero skeleton.
        graph_path: Destination ``.npz`` path for the Skan CSR adjacency graph.
        histogram_path: Destination ``.csv`` path for degree and node-count rows.

    Returns:
        Paths and counts for the saved graph artifacts.
    """
    input_path = _validate_input_path(skeleton_path)
    output_graph_path = _validate_output_path(graph_path, ".npz")
    output_histogram_path = _validate_output_path(histogram_path, ".csv")

    skeleton = np.load(input_path, mmap_mode="r", allow_pickle=False)
    if skeleton.ndim != 3:
        raise ValueError(
            f"expected a 3D skeleton in {input_path}, found {skeleton.ndim} dimensions"
        )
    if not np.any(skeleton):
        raise ValueError(f"skeleton contains no nonzero voxels: {input_path}")

    graph, coordinates = skeleton_to_csgraph(skeleton)
    degrees = np.diff(graph.indptr)
    degree_values, node_counts = np.unique(degrees, return_counts=True)

    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    output_histogram_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_graph_path, graph, compressed=True)

    coordinates_path = output_graph_path.with_suffix(".coordinates.npy")
    np.save(coordinates_path, np.column_stack(coordinates))
    _write_degree_histogram(output_histogram_path, degree_values, node_counts)

    return SkeletonGraphResult(
        graph_path=output_graph_path,
        coordinates_path=coordinates_path,
        histogram_path=output_histogram_path,
        node_count=graph.shape[0],
        edge_count=graph.nnz // 2,
    )


def _write_degree_histogram(
    path: Path, degrees: np.ndarray, node_counts: np.ndarray
) -> None:
    """Write node-degree frequencies in a Crosshair-readable CSV format."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["degree", "node_count"])
        writer.writerows(zip(degrees.tolist(), node_counts.tolist(), strict=True))


def _validate_input_path(path_value: str | Path) -> Path:
    """Validate a readable NumPy skeleton path."""
    path = Path(path_value).expanduser()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"skeleton input must have a .npy extension: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"skeleton input not found: {path}")
    return path


def _validate_output_path(path_value: str | Path, suffix: str) -> Path:
    """Validate an output artifact path with the expected file extension."""
    path = Path(path_value).expanduser()
    if path.suffix.lower() != suffix:
        raise ValueError(f"output path must have a {suffix} extension: {path}")
    return path
