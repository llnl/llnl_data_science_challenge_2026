"""Quantitative lattice measurements from the existing Stage 2 outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "part2" / "developer" / "output" / "anomaly_summary.json"
MASK = ROOT / "part2" / "developer" / "output" / "downsampled_material_mask.uint8.memmap"
CANDIDATES = ROOT / "part2" / "interagent_feedback" / "anomaly_candidates.json"
VALIDATED = ROOT / "part2" / "interagent_feedback" / "validated_candidates_v2.json"
REPORT = ROOT / "part2" / "reporting" / "analysis_report.md"
OUT = ROOT / "part2" / "reporting"
VOXEL_UM = np.array([58.09, 58.09, 58.09], dtype=float)
DESIGN_THICKNESS_UM = 350.0


def _relative_density(mask: np.ndarray):
    points = np.argwhere(mask > 0)
    if not len(points):
        return {"whole_bbox": None, "interior_trimmed_1_voxel": None}
    lo = points.min(axis=0)
    hi = points.max(axis=0) + 1
    whole = mask[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
    trim_lo = lo + 1
    trim_hi = hi - 1
    if np.any(trim_hi <= trim_lo):
        interior = None
    else:
        interior = mask[tuple(slice(int(a), int(b)) for a, b in zip(trim_lo, trim_hi))]
    result = {
        "whole_bbox_zyx_voxels": [int(v) for v in (hi - lo)],
        "whole_bbox_material_relative_density": float(whole.mean()),
        "interior_trim_rule": "remove one pooled-mask voxel from each bbox face where possible",
        "interior_trimmed_1_voxel_relative_density": None if interior is None else float(interior.mean()),
    }
    return result


def _thickness(mask: np.ndarray):
    # The mask is z,y,x and is pooled at the Stage 2a factor. Convert the
    # distance-transform sampling to physical microns.
    dist_vox = distance_transform_edt(mask > 0)
    values_um = (2.0 * dist_vox[dist_vox > 0] * VOXEL_UM.mean())
    if not len(values_um):
        return {"count": 0}
    quantiles = np.percentile(values_um, [5, 25, 50, 75, 95])
    return {
        "count": int(values_um.size),
        "min_um": float(values_um.min()), "mean_um": float(values_um.mean()),
        "median_um": float(np.median(values_um)), "max_um": float(values_um.max()),
        "quantiles_um": {f"p{p}": float(v) for p, v in zip((5, 25, 50, 75, 95), quantiles)},
        "design_thickness_um": DESIGN_THICKNESS_UM,
        "median_to_design_ratio": float(np.median(values_um) / DESIGN_THICKNESS_UM),
        "note": "Local thickness is twice the distance-to-background radius; values are limited by the pooled-mask resolution.",
    }, values_um


def _spatial(records, shape_zyx):
    coords = np.array([r["location"]["voxel_xyz"] for r in records], dtype=float)
    shape_xyz = np.array([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=float)
    norm = coords / shape_xyz
    edge = np.any((norm < 0.2) | (norm > 0.8), axis=1)
    center = ~edge
    # Monte-Carlo CSR reference in normalized scan coordinates.
    if len(norm) > 1:
        diffs = norm[:, None, :] - norm[None, :, :]
        d = np.sqrt((diffs * diffs).sum(axis=2))
        d[d == 0] = np.inf
        observed_nn = d.min(axis=1)
        rng = np.random.default_rng(20260722)
        random_means = []
        for _ in range(32):
            sample = rng.random((len(norm), 3))
            dd = sample[:, None, :] - sample[None, :, :]
            dist = np.sqrt((dd * dd).sum(axis=2))
            dist[dist == 0] = np.inf
            random_means.append(float(dist.min(axis=1).mean()))
        random_nn = float(np.mean(random_means))
        nn_ratio = float(observed_nn.mean() / random_nn) if random_nn else None
    else:
        observed_nn, random_nn, nn_ratio = np.array([]), None, None
    classification = "clustered" if nn_ratio is not None and nn_ratio < 0.8 else (
        "regular/repelled" if nn_ratio is not None and nn_ratio > 1.2 else "consistent_with_random_scattering")
    return {
        "candidate_count": len(records),
        "edge_definition": "any normalized coordinate below 0.2 or above 0.8",
        "edge_count": int(edge.sum()), "center_count": int(center.sum()),
        "edge_fraction": float(edge.mean()) if len(edge) else None,
        "observed_mean_nearest_neighbor_normalized": None if not len(observed_nn) else float(observed_nn.mean()),
        "random_reference_mean_nearest_neighbor_normalized": random_nn,
        "nearest_neighbor_ratio_observed_over_random": nn_ratio,
        "spatial_classification": classification,
    }, norm, edge


def _plot_thickness(values_um):
    path = OUT / "thickness_distribution.png"
    plt.figure(figsize=(8, 4.5))
    plt.hist(values_um, bins=50, color="#4477aa", alpha=0.85)
    plt.axvline(DESIGN_THICKNESS_UM, color="crimson", linestyle="--", label="350 µm design thickness")
    plt.xlabel("Local estimated strut thickness (µm)")
    plt.ylabel("Material voxel count")
    plt.title("Stage 2a pooled-mask local thickness distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path


def _plot_spatial(records, shape_zyx):
    path = OUT / "spatial_distribution.png"
    coords = np.array([r["location"]["voxel_xyz"] for r in records], dtype=float)
    shape_xyz = np.array([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=float)
    n = coords / shape_xyz
    edge = np.any((n < 0.2) | (n > 0.8), axis=1)
    plt.figure(figsize=(7, 6))
    ax = plt.axes(projection="3d")
    ax.scatter(n[~edge, 0], n[~edge, 1], n[~edge, 2], s=10, alpha=.45, label="center")
    ax.scatter(n[edge, 0], n[edge, 1], n[edge, 2], s=16, alpha=.8, label="edge")
    ax.set(xlabel="x / scan width", ylabel="y / scan height", zlabel="z / scan depth", title="Stage 2a candidate spatial distribution")
    ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    validated = json.loads(VALIDATED.read_text(encoding="utf-8"))
    mask_shape = tuple(summary["downsampled_mask_shape_zyx"])
    mask = np.memmap(MASK, dtype=np.uint8, mode="r", shape=mask_shape)
    density = _relative_density(mask)
    thickness, values_um = _thickness(mask)
    spatial_all, norm_all, edge_all = _spatial(candidates, tuple(summary["shape_zyx"]))
    spatial_reviewed, _, _ = _spatial(validated, tuple(summary["shape_zyx"]))
    thickness_plot = _plot_thickness(values_um)
    spatial_plot = _plot_spatial(candidates, tuple(summary["shape_zyx"]))

    rows = []
    for r, n, edge in zip(candidates, norm_all, edge_all):
        rows.append({"ID": r["ID"], "x": r["location"]["voxel_xyz"][0], "y": r["location"]["voxel_xyz"][1], "z": r["location"]["voxel_xyz"][2], "x_norm": n[0], "y_norm": n[1], "z_norm": n[2], "edge_or_center": "edge" if edge else "center", "confidence": r["confidence"], "type": r["type"]})
    with (OUT / "candidate_spatial_measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    result = {"voxel_size_um_xyz": VOXEL_UM.tolist(), "design_thickness_um": DESIGN_THICKNESS_UM,
              "source_summary": str(SUMMARY.relative_to(ROOT)), "density": density,
              "thickness": thickness, "spatial_stage2a": spatial_all,
              "spatial_stage2b_reviewed": spatial_reviewed,
              "plots": [str(thickness_plot.relative_to(ROOT)), str(spatial_plot.relative_to(ROOT))],
              "limitations": ["Measurements use the Stage 2a factor-4 max-pooled mask.", "Outer-skin exclusion is an inward one-voxel bbox trim, not a CAD-derived wall mask.", "Distance-transform thickness is an apparent segmented thickness and is resolution-limited.", "Spatial edge/center and clustering thresholds are descriptive, not defect-certification criteria."]}
    (OUT / "lattice_measurements.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n\n## Quantitative lattice measurements\n\n")
        f.write(f"Using voxel size **58.09 × 58.09 × 58.09 µm**, the Stage 2a pooled mask has whole-bbox relative density **{density['whole_bbox_material_relative_density']:.4f}** and inward-trimmed relative density **{density['interior_trimmed_1_voxel_relative_density']:.4f}** where available. The median distance-transform thickness estimate is **{thickness['median_um']:.1f} µm** versus the **350 µm** design thickness (ratio **{thickness['median_to_design_ratio']:.3f}**).\n\n")
        f.write(f"Of {spatial_all['candidate_count']} Stage 2a candidates, **{spatial_all['edge_count']}** are in the descriptive edge region and **{spatial_all['center_count']}** in the center region. The nearest-neighbor ratio is **{spatial_all['nearest_neighbor_ratio_observed_over_random']:.3f}**, classified as **{spatial_all['spatial_classification']}** under the Monte-Carlo reference. For the 20 Stage 2b-reviewed candidates, the corresponding edge/center counts are {spatial_reviewed['edge_count']}/{spatial_reviewed['center_count']}.\n\n")
        f.write(f"Plots: [thickness distribution]({thickness_plot.relative_to(ROOT).as_posix()}) and [spatial distribution]({spatial_plot.relative_to(ROOT).as_posix()}).\n\n")
        f.write("Limitations: these values use a factor-4 max-pooled mask (effective voxel size about 232.36 µm), so thin-strut thickness and local density are resolution-limited. The inward trim is only a proxy for excluding outer skins/walls; a registered CAD wall mask would be preferable. Spatial categories and clustering are descriptive, and candidate locations remain possible defects rather than confirmed manufacturing defects.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
