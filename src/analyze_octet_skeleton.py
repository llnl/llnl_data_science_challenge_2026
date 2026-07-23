"""Segment a cropped octet CT TIFF, skeletonize it, and analyse it with Skan.

The saved graph is a voxel-neighbour graph: its rows/columns correspond to the
coordinates in the companion ``.coordinates.npy`` file.  Branch lengths are
the Skan path lengths, measured in voxels because no voxel spacing was supplied.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy.sparse import save_npz
from skimage.morphology import skeletonize

# Skan's Numba cache needs a writable location in the base environment.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "skan_numba_cache"))

from skan.csr import Skeleton, skeleton_to_csgraph, summarize


def analyse_cropped_tiff(
    input_tiff: Path, output_dir: Path, threshold: int, downsample_factor: int = 1
) -> dict[str, int]:
    """Create segmentation, skeleton, Skan graph, and histogram-source CSVs.

    Args:
        input_tiff: Cropped, three-dimensional uint16 CT TIFF.
        output_dir: New directory for all generated artifacts.
        threshold: Material voxels satisfy ``intensity >= threshold``.
        downsample_factor: Stride applied equally on all axes before
            skeletonization. A value above one reduces memory use; Skan branch
            lengths are scaled back to the original voxel units.

    Returns:
        Counts that summarize the created graph and Skan branches.
    """
    volume = tifffile.memmap(input_tiff)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D TIFF, found shape {volume.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask = volume >= threshold
    foreground_voxels = int(mask.sum())
    np.save(output_dir / "segmented_mask.npy", mask)

    skeleton_input = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
    np.save(output_dir / "skeleton_input_mask.npy", skeleton_input)
    skeleton = skeletonize(skeleton_input)
    np.save(output_dir / "skeleton.npy", skeleton)
    del mask

    graph, coordinates = skeleton_to_csgraph(skeleton)
    save_npz(output_dir / "skeleton_voxel_graph.npz", graph, compressed=True)
    np.save(output_dir / "skeleton_voxel_graph.coordinates.npy", np.column_stack(coordinates))

    degrees = np.diff(graph.indptr)
    unique_degrees, degree_counts = np.unique(degrees, return_counts=True)
    _write_rows(
        output_dir / "node_degree_histogram.csv",
        ("edges_per_node", "node_count"),
        zip(unique_degrees.tolist(), degree_counts.tolist(), strict=True),
    )

    branch_data = summarize(
        Skeleton(skeleton, spacing=(downsample_factor,) * 3, keep_images=False)
    )
    lengths = branch_data["branch-distance"].to_numpy(dtype=float)
    np.savetxt(
        output_dir / "branch_lengths.csv",
        lengths,
        delimiter=",",
        header="branch_length_voxels",
        comments="",
    )
    _write_summary(
        output_dir / "analysis_summary.csv",
        threshold, volume.shape, skeleton.shape, foreground_voxels, downsample_factor,
        int(skeleton.sum()), graph.shape[0], graph.nnz // 2, len(lengths), lengths,
    )
    return {
        "skeleton_voxels": int(skeleton.sum()),
        "graph_nodes": int(graph.shape[0]),
        "graph_edges": int(graph.nnz // 2),
        "skan_branches": int(len(lengths)),
    }


def _write_rows(path: Path, header: tuple[str, str], rows: object) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _write_summary(
    path: Path,
    threshold: int,
    volume_shape: tuple[int, ...],
    skeleton_shape: tuple[int, ...],
    foreground_voxels: int,
    downsample_factor: int,
    skeleton_voxels: int,
    graph_nodes: int,
    graph_edges: int,
    branch_count: int,
    lengths: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ("threshold", threshold),
            ("volume_shape_zyx", "x".join(map(str, volume_shape))),
            ("skeleton_shape_zyx", "x".join(map(str, skeleton_shape))),
            ("foreground_voxels", foreground_voxels),
            ("skeleton_downsample_factor", downsample_factor),
            ("skeleton_voxels", skeleton_voxels),
            ("graph_nodes", graph_nodes),
            ("graph_edges", graph_edges),
            ("skan_branch_count", branch_count),
            ("branch_length_min_voxels", float(lengths.min()) if len(lengths) else 0),
            ("branch_length_max_voxels", float(lengths.max()) if len(lengths) else 0),
            ("branch_length_mean_voxels", float(lengths.mean()) if len(lengths) else 0),
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=int, default=50000)
    parser.add_argument("--downsample-factor", type=int, default=1)
    args = parser.parse_args()
    if args.downsample_factor < 1:
        parser.error("--downsample-factor must be at least 1")
    print(
        analyse_cropped_tiff(
            args.input_tiff, args.output_dir, args.threshold, args.downsample_factor
        )
    )


if __name__ == "__main__":
    main()
