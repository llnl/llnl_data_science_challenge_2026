#!/usr/bin/env python3
"""Quantify and visualize unusually thin and thick strut screening candidates.

The inventory contains tube occupancy, not direct cross-sectional metrology.
This script therefore uses the same effective-diameter proxy as the existing
post-analysis and defines unusualness relative to the nominal inventory class.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
INVENTORY = ROOT / "part2/stage_2a_developer_output/all_struts_inventory.csv"
EXCESS = ROOT / "post_analysis/excess_material/excess_by_strut.csv"
VOXEL_SIZE_UM = 58.09
NOMINAL_RADIUS_VOXELS = 6.0
REFERENCE_DIAMETER_UM = 2 * NOMINAL_RADIUS_VOXELS * VOXEL_SIZE_UM


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    occupancy = np.asarray([float(row["ct_material_occupancy"]) for row in rows])
    nominal = occupancy[np.asarray([row["classification"] == "Nominal" for row in rows])]
    low_occ = percentile(nominal, 5)
    high_occ = percentile(nominal, 95)
    thickness = REFERENCE_DIAMETER_UM * np.sqrt(np.clip(occupancy, 0, 1))
    low_um = REFERENCE_DIAMETER_UM * np.sqrt(low_occ)
    high_um = REFERENCE_DIAMETER_UM * np.sqrt(high_occ)
    status = np.where(occupancy < low_occ, "overly_thin",
                      np.where(occupancy > high_occ, "overly_thick", "within_reference"))

    excess_by_id = {}
    if EXCESS.exists():
        with EXCESS.open(newline="") as handle:
            for row in csv.DictReader(handle):
                excess_by_id[row["strut_id"]] = row

    output_rows = []
    for i, row in enumerate(rows):
        excess = excess_by_id.get(row["strut_id"], {})
        output_rows.append({
            "strut_id": row["strut_id"],
            "classification": row["classification"],
            "screening_status": status[i],
            "ct_material_occupancy": f"{occupancy[i]:.9f}",
            "thickness_proxy_um": f"{thickness[i]:.6f}",
            "distance_from_nominal_median_um": f"{thickness[i] - REFERENCE_DIAMETER_UM * np.sqrt(np.median(nominal)):.6f}",
            "centroid_x_um": row.get("centroid_x_um", ""),
            "centroid_y_um": row.get("centroid_y_um", ""),
            "centroid_z_um": row.get("centroid_z_um", ""),
            "excess_voxels": excess.get("excess_voxels", ""),
            "excess_fraction_of_mask": excess.get("excess_fraction_of_mask", ""),
        })

    fields = list(output_rows[0])
    with (OUT / "per_strut_thickness_thinness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    candidate_rows = [row for row in output_rows if row["screening_status"] != "within_reference"]
    with (OUT / "candidate_struts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    class_counts = {}
    for classification in sorted(set(row["classification"] for row in rows)):
        mask = np.asarray([row["classification"] == classification for row in rows])
        class_counts[classification] = {
            "count": int(mask.sum()),
            "overly_thin": int(np.sum(mask & (status == "overly_thin"))),
            "overly_thick": int(np.sum(mask & (status == "overly_thick"))),
            "median_thickness_proxy_um": float(np.median(thickness[mask])),
        }

    summary = {
        "source_inventory": str(INVENTORY.relative_to(ROOT)),
        "strut_count": len(rows),
        "measure": "occupancy-derived effective diameter proxy; not direct cross-sectional metrology",
        "formula": "2 * nominal_radius_um * sqrt(ct_material_occupancy)",
        "voxel_size_um": VOXEL_SIZE_UM,
        "nominal_radius_voxels": NOMINAL_RADIUS_VOXELS,
        "reference_diameter_um": REFERENCE_DIAMETER_UM,
        "reference_population": "classification == Nominal",
        "threshold_rule": "overly thin < nominal-class 5th percentile; overly thick > nominal-class 95th percentile",
        "nominal_reference": {
            "count": int(len(nominal)),
            "occupancy_median": float(np.median(nominal)),
            "occupancy_p05": low_occ,
            "occupancy_p95": high_occ,
            "thickness_proxy_median_um": float(np.median(thickness[np.asarray([r["classification"] == "Nominal" for r in rows])])),
            "thin_cutoff_um": low_um,
            "thick_cutoff_um": high_um,
        },
        "screening_counts": {
            "overly_thin": int(np.sum(status == "overly_thin")),
            "overly_thick": int(np.sum(status == "overly_thick")),
            "within_reference": int(np.sum(status == "within_reference")),
        },
        "screening_percentages": {
            "overly_thin": float(np.mean(status == "overly_thin") * 100),
            "overly_thick": float(np.mean(status == "overly_thick") * 100),
        },
        "by_classification": class_counts,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = {"overly_thin": "#2c7fb8", "within_reference": "#bdbdbd", "overly_thick": "#d95f0e"}
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, REFERENCE_DIAMETER_UM * 1.02, 42)
    for name in ("within_reference", "overly_thin", "overly_thick"):
        vals = thickness[status == name]
        if len(vals):
            ax.hist(vals, bins=bins, color=colors[name], alpha=0.78,
                    label=f"{name.replace('_', ' ').title()} (n={len(vals):,})")
    ax.axvline(low_um, color=colors["overly_thin"], linestyle="--", linewidth=2,
               label=f"Thin cutoff = {low_um:.1f} µm")
    ax.axvline(high_um, color=colors["overly_thick"], linestyle="--", linewidth=2,
               label=f"Thick cutoff = {high_um:.1f} µm")
    ax.set(xlabel="Occupancy-derived effective diameter proxy (µm)", ylabel="Strut count",
           title="Screening distribution for overly thin and overly thick struts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "thickness_thinness_distribution.png", dpi=180)
    plt.close(fig)

    classifications = sorted(class_counts)
    x = np.arange(len(classifications))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    thin_counts = [class_counts[c]["overly_thin"] for c in classifications]
    thick_counts = [class_counts[c]["overly_thick"] for c in classifications]
    ax.bar(x - width / 2, thin_counts, width, color=colors["overly_thin"], label="Overly thin")
    ax.bar(x + width / 2, thick_counts, width, color=colors["overly_thick"], label="Overly thick")
    ax.set_xticks(x, [c.replace("_", " ") for c in classifications], rotation=20, ha="right")
    ax.set(ylabel="Strut count", title="Thickness/thinness screening candidates by inventory classification")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "candidates_by_classification.png", dpi=180)
    plt.close(fig)

    # A spatial view helps identify whether candidates cluster in a region.
    x_um = np.asarray([float(r.get("centroid_x_um", 0) or 0) for r in rows])
    y_um = np.asarray([float(r.get("centroid_y_um", 0) or 0) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x_um[status == "within_reference"], y_um[status == "within_reference"],
               s=2, c="#d9d9d9", alpha=0.18, linewidths=0, label="Within reference")
    for name in ("overly_thin", "overly_thick"):
        ax.scatter(x_um[status == name], y_um[status == name], s=9, c=colors[name],
                   alpha=0.7, linewidths=0, label=f"{name.replace('_', ' ').title()} (n={np.sum(status == name):,})")
    ax.set(xlabel="Centroid X (µm)", ylabel="Centroid Y (µm)", title="XY locations of thickness screening candidates")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "candidate_locations_xy.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
