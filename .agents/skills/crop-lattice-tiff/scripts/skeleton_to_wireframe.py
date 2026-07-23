"""Convert a skeleton NPY into sampled 3D line segments for Crosshair."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "skan_numba_cache"))
from skan.csr import skeleton_to_csgraph


def write_wireframe(skeleton_path: Path, output_path: Path, edge_stride: int) -> int:
    """Save every nth undirected Skan graph edge as a NaN-separated line segment."""
    if edge_stride < 1:
        raise ValueError("edge_stride must be at least 1")
    skeleton = np.load(skeleton_path, mmap_mode="r", allow_pickle=False)
    if skeleton.ndim != 3:
        raise ValueError(f"expected a 3D skeleton, found shape {skeleton.shape}")
    graph, coordinates = skeleton_to_csgraph(skeleton)
    upper_triangle = graph.tocoo()
    keep = upper_triangle.row < upper_triangle.col
    source_ids = upper_triangle.row[keep][::edge_stride]
    target_ids = upper_triangle.col[keep][::edge_stride]
    z, y, x = (np.asarray(axis) for axis in coordinates)
    rows = np.empty((source_ids.size * 3, 3), dtype=np.float32)
    rows[0::3] = np.column_stack((x[source_ids], y[source_ids], z[source_ids]))
    rows[1::3] = np.column_stack((x[target_ids], y[target_ids], z[target_ids]))
    rows[2::3] = np.nan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["x", "y", "z"]).to_parquet(output_path, index=False)
    return source_ids.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_skeleton", type=Path)
    parser.add_argument("output_parquet", type=Path)
    parser.add_argument("--edge-stride", type=int, default=8)
    arguments = parser.parse_args()
    print(
        f"saved {write_wireframe(arguments.input_skeleton, arguments.output_parquet, arguments.edge_stride)} "
        "line segments"
    )


if __name__ == "__main__":
    main()
