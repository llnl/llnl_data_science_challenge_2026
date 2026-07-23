"""Convert a skeleton into NaN-separated, continuous Skan branch paths."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "skan_numba_cache"))
from skan import Skeleton


def write_branch_wireframe(skeleton_path: Path, output_path: Path) -> tuple[int, int]:
    """Save each skeleton branch as a connected, NaN-separated 3D line path."""
    image = np.load(skeleton_path, mmap_mode="r", allow_pickle=False)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D skeleton, found shape {image.shape}")
    skeleton = Skeleton(image)
    paths = skeleton.paths_list()
    rows: list[np.ndarray] = []
    for node_ids in paths:
        coordinates_zyx = skeleton.coordinates[np.asarray(node_ids)]
        rows.append(coordinates_zyx[:, [2, 1, 0]].astype(np.float32, copy=False))
        rows.append(np.full((1, 3), np.nan, dtype=np.float32))
    line_data = np.concatenate(rows) if rows else np.empty((0, 3), dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(line_data, columns=["x", "y", "z"]).to_parquet(output_path, index=False)
    return len(paths), len(line_data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_skeleton", type=Path)
    parser.add_argument("output_parquet", type=Path)
    arguments = parser.parse_args()
    path_count, row_count = write_branch_wireframe(
        arguments.input_skeleton, arguments.output_parquet
    )
    print(f"saved {path_count} connected branch paths in {row_count} rows")


if __name__ == "__main__":
    main()
