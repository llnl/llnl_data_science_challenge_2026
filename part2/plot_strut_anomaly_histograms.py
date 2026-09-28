#!/usr/bin/env python3
"""Create lightweight inventory histograms for Stage 2a review."""
from pathlib import Path
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-strut-histograms")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/hannahdc/llnl_data_science_challenge_2026")
OUT = ROOT / "part2/stage_2a_developer_output"
CSV_PATH = OUT / "all_struts_inventory.csv"
FIG_PATH = OUT / "strut_anomaly_histograms.png"


def main():
    with CSV_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    classes = np.asarray([row["classification"] for row in rows])
    occupancy = np.asarray([float(row["ct_material_occupancy"]) for row in rows])
    gap = np.asarray([float(row["longest_gap_um"]) for row in rows])
    missing_volume = np.asarray([float(row["estimated_missing_volume_mm3"]) for row in rows])
    intensity = np.asarray([float(row["mean_intensity"]) for row in rows])
    length = np.asarray([float(row["trimmed_length_um"]) for row in rows])
    nominal_volume = np.asarray([float(row["nominal_sampled_volume_mm3"]) for row in rows])

    # The pipeline uses a nominal 348.54 um tube radius. Assuming occupancy
    # scales with filled cross-sectional area gives a useful radius proxy,
    # not a direct geometric thickness measurement.
    nominal_radius_um = np.sqrt(
        np.maximum(nominal_volume, 0) * 1e9 / np.maximum(np.pi * length, 1)
    )
    effective_radius_um = nominal_radius_um * np.sqrt(np.clip(occupancy, 0, 1))
    anomalous = classes != "Nominal"
    intentional = classes == "Missing_Intentional"
    unintentional = classes == "Missing_Unintentional"
    expected_present = classes == "Expected_Missing_But_Material_Present"

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Stage 2a strut occupancy and anomaly diagnostics", fontsize=15)

    # Occupancy is the most direct per-strut material signal.
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 41)
    ax.hist(occupancy, bins=bins, color="#bdbdbd", label="All struts")
    ax.hist(occupancy[anomalous], bins=bins, histtype="step", linewidth=2,
            color="#e15759", label="Non-nominal")
    ax.hist(occupancy[intentional], bins=bins, histtype="step", linewidth=2,
            color="#59a14f", label="Expected missing")
    ax.axvline(0.08, color="#333333", linestyle="--", label="0.08 occupancy reference")
    ax.set(xlabel="CT material occupancy", ylabel="Strut count", title="Material occupancy")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    bins = np.linspace(0, max(np.percentile(gap, 99.5), 1), 40)
    ax.hist(gap, bins=bins, color="#bdbdbd", label="All struts")
    ax.hist(gap[anomalous], bins=bins, histtype="step", linewidth=2,
            color="#e15759", label="Non-nominal")
    ax.set(xlabel="Longest low-material gap (µm)", ylabel="Strut count",
           title="Longitudinal gap proxy")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bins = np.linspace(0, max(np.percentile(effective_radius_um, 99.5), 1), 40)
    ax.hist(effective_radius_um, bins=bins, color="#bdbdbd", label="All struts")
    ax.hist(effective_radius_um[anomalous], bins=bins, histtype="step", linewidth=2,
            color="#e15759", label="Non-nominal")
    ax.axvline(float(np.median(nominal_radius_um)), color="#333333", linestyle="--",
               label=f"Nominal radius ≈ {np.median(nominal_radius_um):.0f} µm")
    ax.set(xlabel="Occupancy-derived effective radius (µm)", ylabel="Strut count",
           title="Thickness proxy (not direct diameter)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    positive = missing_volume > 0
    bins = np.geomspace(max(np.min(missing_volume[positive]), 1e-5),
                        max(np.percentile(missing_volume[positive], 99.5), 1e-4), 40)
    ax.hist(missing_volume[positive], bins=bins, color="#bdbdbd", label="Positive-volume rows")
    ax.hist(missing_volume[anomalous & positive], bins=bins, histtype="step", linewidth=2,
            color="#e15759", label="Non-nominal")
    ax.axvline(0.0052925856, color="#333333", linestyle="--", label="3×3×3 voxel floor")
    ax.set_xscale("log")
    ax.set(xlabel="Estimated missing volume (mm³)", ylabel="Strut count",
           title="Missing-volume diagnostic")
    ax.legend(fontsize=8)

    fig.text(0.01, 0.01,
             f"n={len(rows):,}; nominal={np.count_nonzero(~anomalous):,}; "
             f"intentional={np.count_nonzero(intentional):,}; "
             f"unintentional={np.count_nonzero(unintentional):,}; "
             f"expected-but-present={np.count_nonzero(expected_present):,}",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIG_PATH, dpi=170)
    plt.close(fig)
    print(FIG_PATH)


if __name__ == "__main__":
    main()
