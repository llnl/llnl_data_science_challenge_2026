#!/usr/bin/env python3
"""Create a scale-aware per-strut thickness-proxy histogram."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "strut_thickness"
INVENTORY = ROOT / "part2/stage_2a_developer_output/all_struts_inventory.csv"
VOXEL_SIZE_UM = 58.09
NOMINAL_RADIUS_VOXELS = 6.0
CAD_PARAMETER_THICKNESS = 0.1
NOMINAL_RADIUS_UM = NOMINAL_RADIUS_VOXELS * VOXEL_SIZE_UM
REFERENCE_DIAMETER_UM = 2.0 * NOMINAL_RADIUS_UM


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []
    thicknesses = []
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        occupancy = float(row["ct_material_occupancy"])
        thickness_um = REFERENCE_DIAMETER_UM * math.sqrt(np.clip(occupancy, 0.0, 1.0))
        thicknesses.append(thickness_um)
        by_class[row["classification"]].append(thickness_um)
        output_rows.append(
            {
                "strut_id": row["strut_id"],
                "classification": row["classification"],
                "ct_material_occupancy": f"{occupancy:.9f}",
                "thickness_proxy_um": f"{thickness_um:.6f}",
                "cad_thickness_parameter": row.get("thickness", "0.1"),
                "voxel_size_um": f"{VOXEL_SIZE_UM:.2f}",
            }
        )

    with (OUT / "per_strut_thickness.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    values = np.asarray(thicknesses)
    class_summary = {
        name: {
            "count": len(items),
            "mean_thickness_proxy_um": float(np.mean(items)),
            "median_thickness_proxy_um": float(np.median(items)),
        }
        for name, items in sorted(by_class.items())
    }
    summary = {
        "source_inventory": str(INVENTORY.relative_to(ROOT)),
        "strut_count": len(rows),
        "measure": "occupancy-derived effective diameter proxy",
        "formula": "2 * nominal_radius_um * sqrt(ct_material_occupancy)",
        "cad_model_thickness_parameter": CAD_PARAMETER_THICKNESS,
        "nominal_radius_voxels": NOMINAL_RADIUS_VOXELS,
        "voxel_size_um": VOXEL_SIZE_UM,
        "nominal_radius_um": NOMINAL_RADIUS_UM,
        "cad_reference_diameter_um": REFERENCE_DIAMETER_UM,
        "all_struts": {
            "mean_thickness_proxy_um": float(np.mean(values)),
            "median_thickness_proxy_um": float(np.median(values)),
            "std_thickness_proxy_um": float(np.std(values, ddof=1)),
            "min_thickness_proxy_um": float(np.min(values)),
            "max_thickness_proxy_um": float(np.max(values)),
            "p05_thickness_proxy_um": float(np.percentile(values, 5)),
            "p95_thickness_proxy_um": float(np.percentile(values, 95)),
        },
        "by_classification": class_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, REFERENCE_DIAMETER_UM * 1.05, 36)
    ax.hist(values, bins=bins, color="#7aa6c2", edgecolor="white", linewidth=0.5)
    ax.axvline(REFERENCE_DIAMETER_UM, color="#c0392b", linestyle="--", linewidth=2,
               label=f"CAD reference diameter = {REFERENCE_DIAMETER_UM:.1f} µm")
    ax.axvline(float(np.mean(values)), color="#2c7a4b", linestyle="-", linewidth=2,
               label=f"All-strut mean = {np.mean(values):.1f} µm")
    ax.axvline(float(np.median(values)), color="#6c3483", linestyle=":", linewidth=2,
               label=f"All-strut median = {np.median(values):.1f} µm")
    ax.set(
        xlabel="Average strut thickness proxy (µm)",
        ylabel="Strut count",
        title="Per-strut CT thickness-proxy distribution",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "strut_thickness_histogram.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
