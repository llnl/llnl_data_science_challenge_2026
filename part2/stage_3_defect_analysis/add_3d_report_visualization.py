#!/usr/bin/env python3
"""Add representative 3-D defect-strut visualizations to an existing report.

This utility intentionally reads only the existing per-strut CSV and registered
graph. It does not open the TIFF or rerun Stage 3 feature extraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


PART2 = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
DEFAULT_REGISTRATION = PART2 / "registration/alignment_check/candidate_affine_registered.json"
DEFAULT_SCAN = PART2 / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
REFERENCE_VISUALIZER = Path("/home/hannahdc/llnl_data_science_challenge_2026/stage3_visualization/strut_visualizer.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--threshold", type=float, default=40081.0)
    parser.add_argument("--nominal-radius-voxels", type=float, default=6.0)
    parser.add_argument("--downsample", type=int, default=2, help="render-only voxel downsampling for the four-panel figure")
    parser.add_argument("--samples-per-class", type=int, default=2)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        raise ValueError("samples-per-class must be positive")
    if args.downsample < 1:
        raise ValueError("downsample must be positive")

    report_path = args.output / "defect_analysis_report.md"
    csv_path = args.output / "defect_analysis_by_strut.csv"
    if not report_path.exists() or not csv_path.exists() or not args.registration.exists() or not args.scan.exists():
        raise FileNotFoundError("Expected existing report, per-strut CSV, registration graph, and TIFF scan.")

    if not REFERENCE_VISUALIZER.exists():
        raise FileNotFoundError(REFERENCE_VISUALIZER)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_visualization-mpl")
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for 3-D report visualizations") from exc

    import importlib.util
    spec = importlib.util.spec_from_file_location("reference_strut_visualizer", REFERENCE_VISUALIZER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {REFERENCE_VISUALIZER}")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    registration = json.loads(args.registration.read_text())
    junctions = np.asarray([item["position"] for item in registration["junctions"]], dtype=float)
    struts = {int(item["id"]): item for item in registration["struts"]}

    score_for_class = {
        "Thin": "thin_score",
        "Inflated": "inflated_score",
        "Bent": "bend_score",
        "Broken": "broken_score",
    }
    classes = ["Thin", "Inflated", "Bent", "Broken"]
    colors = {"Thin": "#377eb8", "Inflated": "#e41a1c", "Bent": "#984ea3", "Broken": "#ff7f00"}
    selected: list[dict[str, str]] = []
    for label in classes:
        candidates = [row for row in rows if row["primary_defect"] == label]
        score_name = score_for_class[label]
        candidates = [row for row in candidates if np.isfinite(float(row[score_name]))]
        selected.extend(sorted(candidates, key=lambda row: float(row[score_name]), reverse=True)[: args.samples_per_class])

    def local_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reference = np.zeros(3, dtype=float)
        reference[int(np.argmin(np.abs(direction)))] = 1.0
        first = np.cross(direction, reference)
        first /= max(np.linalg.norm(first), 1e-8)
        second = np.cross(direction, first)
        second /= max(np.linalg.norm(second), 1e-8)
        return first, second

    def crop_voxels(row: dict[str, str]) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
        """Use the reference visualizer's crop and return its display mask."""
        crop, metadata = reference.extract_strut(args.scan, args.registration, int(row["strut_id"]), margin_voxels=10.0)
        step = args.downsample
        display_xyz = (crop >= args.threshold)[::step, ::step, ::step].transpose(2, 1, 0)
        return display_xyz, metadata, crop

    def baseline_rings(metadata: dict[str, object]) -> np.ndarray:
        p0 = np.asarray(metadata["endpoint0_xyz_source_voxels"], dtype=float)
        p1 = np.asarray(metadata["endpoint1_xyz_source_voxels"], dtype=float)
        vector = p1 - p0
        length = float(np.linalg.norm(vector))
        direction = vector / max(length, 1e-8)
        basis_a, basis_b = local_basis(direction)
        origin = np.asarray(metadata["clipped_bounds_xyz_source_voxels_half_open"][0], dtype=float)
        angles = np.linspace(0, 2 * np.pi, 80)
        rings = []
        for fraction in np.linspace(0.08, 0.92, 9):
            center = p0 + fraction * vector
            rings.append(center[None, :] + args.nominal_radius_voxels * (np.cos(angles)[:, None] * basis_a + np.sin(angles)[:, None] * basis_b))
        return (np.concatenate(rings, axis=0) - origin) / args.downsample

    figure = plt.figure(figsize=(16, 12), constrained_layout=True)
    axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    for axis, label in zip(axes, classes):
        samples = [row for row in selected if row["primary_defect"] == label][:1]
        for row in samples:
            display_xyz, metadata, crop = crop_voxels(row)
            origin = np.asarray(metadata["clipped_bounds_xyz_source_voxels_half_open"][0], dtype=float)
            if display_xyz.any():
                voxel_artists = axis.voxels(display_xyz, facecolors=colors[label], edgecolor="none", alpha=0.28)
                for artist in voxel_artists.values():
                    artist.set_zorder(1)
            else:
                axis.text2D(0.05, 0.95, "No voxels above threshold", transform=axis.transAxes)
            p0 = (np.asarray(metadata["endpoint0_xyz_source_voxels"], dtype=float) - origin) / args.downsample
            p1 = (np.asarray(metadata["endpoint1_xyz_source_voxels"], dtype=float) - origin) / args.downsample
            axis.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#08306b", linewidth=4.0, label="registered centerline", zorder=10)
            axis.scatter([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#08306b", edgecolors="white", linewidths=0.8, s=34, depthshade=False, zorder=11)
            rings = baseline_rings(metadata)
            for start in range(0, len(rings), 80):
                ring = rings[start:start + 80]
                axis.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="gray", linewidth=0.8, alpha=0.55, zorder=5)
            axis.text2D(0.05, 0.90, f"id {row['strut_id']} — score {float(row[score_for_class[label]]):.1f}\n{int(display_xyz.sum()):,} material voxels", transform=axis.transAxes, fontsize=8)
        axis.set_title(f"{label} — reference TIFF voxel rendering")
        axis.set_xlabel("X crop source voxels")
        axis.set_ylabel("Y crop source voxels")
        axis.set_zlabel("Z crop source voxels")
        axis.legend(loc="upper left", fontsize=7)
        axis.view_init(elev=22, azim=-58)

    image_name = "defect_analysis_3d_samples.png"
    figure.savefig(args.output / image_name, dpi=160)
    plt.close(figure)

    report = report_path.read_text()
    marker = "## 3-D representative defect struts"
    section = (
        f"{marker}\n\n"
        "The following 3-D views show thresholded TIFF voxels for one highest-scoring representative of each Stage 3 material-defect class. "
        "The colored voxel faces use the reference renderer's `ax.voxels` approach at alpha 0.28, the blue line is the registered centerline used for scoring, "
        "and the gray rings are the nominal-radius baseline (6 voxels). Coordinates use the reference renderer's XYZ crop convention after the TIFF ZYX-to-XYZ transpose.\n\n"
        f"![3-D representative defect struts]({image_name})\n\n"
        f"The figure was generated from the existing `defect_analysis_by_strut.csv`, registered graph, and small TIFF bounding-box crops around four samples; "
        f"the reference voxel renderer was used with render-only downsampling={args.downsample}; no full-dataset reprocessing was performed.\n"
    )
    if marker in report:
        report = report[: report.index(marker)].rstrip() + "\n\n" + section
    else:
        report = report.rstrip() + "\n\n" + section
    report_path.write_text(report)
    print(f"Wrote {args.output / image_name}")
    print(f"Updated {report_path}")


if __name__ == "__main__":
    main()
