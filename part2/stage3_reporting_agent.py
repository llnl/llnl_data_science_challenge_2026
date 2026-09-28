#!/usr/bin/env python3
"""Create post-analysis metrology figures and a Markdown report for Stage 2a.

The Stage 2a mask is a defect/candidate-void mask in TIFF array order (Z, Y, X).
Accordingly, the spatial "density" values reported here are mask occupancy fractions,
not calibrated material volume fractions.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "stage2a_output" / "defect_summary.csv"
MASK_PATH = ROOT / "stage2a_output" / "mask_output.tif"
OUT_DIR = ROOT / "stage3_output"
TOP_DIR = OUT_DIR / "top_anomalies"
N_BINS = 30
TOP_N = 5


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"defect summary is missing required columns: {missing}")


def save_density_figure(mask: np.ndarray, output: Path) -> tuple[np.ndarray, np.ndarray]:
    """Plot Z-slab occupancy and a 9x9 XY occupancy map."""
    z_edges = np.linspace(0, mask.shape[0], N_BINS + 1, dtype=int)
    z_density = np.array([
        mask[z_edges[i]:z_edges[i + 1]].mean() for i in range(N_BINS)
    ])
    # The source lattice is a 9x9x9 octet truss; use a 9x9 XY grid for comparison.
    y_edges = np.linspace(0, mask.shape[1], 10, dtype=int)
    x_edges = np.linspace(0, mask.shape[2], 10, dtype=int)
    xy_density = np.empty((9, 9), dtype=float)
    for yi in range(9):
        for xi in range(9):
            xy_density[yi, xi] = mask[:, y_edges[yi]:y_edges[yi + 1],
                                  x_edges[xi]:x_edges[xi + 1]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    centers = (z_edges[:-1] + z_edges[1:]) / 2
    axes[0].bar(centers, z_density * 100, width=np.diff(z_edges), color="#367588")
    axes[0].set(title="Z-slab defect occupancy", xlabel="Z voxel coordinate",
                ylabel="Defect-mask occupancy (%)")
    image = axes[1].imshow(xy_density * 100, origin="lower", cmap="magma", aspect="auto")
    axes[1].set(title="XY-cell defect occupancy", xlabel="X cell index", ylabel="Y cell index")
    fig.colorbar(image, ax=axes[1], label="Defect-mask occupancy (%)")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return z_density, xy_density


def save_histograms(defects: pd.DataFrame) -> None:
    radius = defects["equivalent_diameter_voxels"] / 2.0
    plots = [
        (radius, "Equivalent defect radius / thickness distribution",
         "Equivalent radius (voxels)", OUT_DIR / "defect_equivalent_radius_histogram.png"),
        (defects["voxel_count"], "Defect voxel-volume frequency distribution",
         "Defect volume (voxels)", OUT_DIR / "defect_voxel_volume_histogram.png"),
    ]
    for values, title, xlabel, path in plots:
        fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
        ax.hist(values, bins=N_BINS, color="#5c4b8a", edgecolor="white")
        ax.set(title=title, xlabel=xlabel, ylabel="Defect components")
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(path, dpi=180)
        plt.close(fig)


def save_top_anomaly_charts(top: pd.DataFrame) -> None:
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        extents = [row["bbox_x"], row["bbox_y"], row["bbox_z"]]
        fig, ax = plt.subplots(figsize=(5.8, 4.1), constrained_layout=True)
        bars = ax.bar(["X", "Y", "Z"], extents, color=["#3c7cb5", "#52a675", "#e08b45"])
        ax.bar_label(bars, fmt="%.0f voxels", padding=3)
        ax.set(title=f"Rank {rank}: component {int(row['component'])} bounding-box extent",
               ylabel="Extent (voxels)", ylim=(0, max(extents) * 1.18))
        ax.text(0.02, 0.96, f"Defect volume: {int(row['voxel_count']):,} voxels",
                transform=ax.transAxes, va="top")
        fig.savefig(TOP_DIR / f"rank_{rank:02d}_component_{int(row['component'])}_extent.png", dpi=180)
        plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate extra."""
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.2f}")
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        rows.append(cells)
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_report(defects: pd.DataFrame, mask: np.ndarray, z_density: np.ndarray,
                 xy_density: np.ndarray, components_in_mask: int) -> None:
    total_voxels = int(mask.size)
    void_voxels = int(mask.sum())
    void_fraction = void_voxels / total_voxels
    top = defects.nlargest(TOP_N, "voxel_count")
    median_radius = float((defects["equivalent_diameter_voxels"] / 2).median())
    top_share = float(top["voxel_count"].sum() / defects["voxel_count"].sum())
    peak_slab = int(np.argmax(z_density))
    peak_xy = np.unravel_index(np.argmax(xy_density), xy_density.shape)
    table = top[["component", "voxel_count", "equivalent_diameter_voxels", "centroid_x", "centroid_y", "centroid_z", "bbox_x", "bbox_y", "bbox_z"]].copy()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.columns = ["Rank", "Component", "Voxels", "Eq. diameter (vox)", "Centroid X", "Centroid Y", "Centroid Z", "BBox X", "BBox Y", "BBox Z"]
    lines = [
        "# Stage 3 Post-analysis Defect Report",
        "",
        "## Scope and interpretation",
        "",
        "This report summarizes Stage 2a candidate defect components. TIFF arrays use `(Z, Y, X)` order. Spatial-density values are **defect-mask occupancy fractions**; without a calibrated material/strut mask they should not be interpreted as physical strut volume fractions.",
        "",
        "## Global metrics",
        "",
        f"- Mask volume: `{total_voxels:,}` voxels (`{tuple(mask.shape)}` in Z, Y, X).",
        f"- Candidate void voxels: `{void_voxels:,}`; global defect-mask void fraction: `{void_fraction:.4%}`.",
        f"- Missing-strut/candidate-defect component count (CSV): `{len(defects):,}`.",
        f"- Connected components measured directly from the binary mask: `{components_in_mask:,}`.",
        f"- Median equivalent defect radius: `{median_radius:.2f}` voxels.",
        "",
        "## Spatial density and distributions",
        "",
        f"The highest Z-slab occupancy is slab {peak_slab + 1}/{N_BINS} at `{z_density[peak_slab]:.4%}`. The highest 9x9 XY-cell occupancy is cell `(Y={peak_xy[0]}, X={peak_xy[1]})` at `{xy_density[peak_xy]:.4%}`.",
        "",
        "![Spatial strut-density proxy](spatial_strut_density_distribution.png)",
        "",
        "![Equivalent radius distribution](defect_equivalent_radius_histogram.png)",
        "",
        "![Voxel-volume distribution](defect_voxel_volume_histogram.png)",
        "",
        "## Top severe anomalies",
        "",
        f"The five largest components account for `{top_share:.2%}` of all CSV-listed defect voxels. Severity is ranked by voxel count.",
        "",
        dataframe_to_markdown(table),
        "",
        "Per-anomaly extent charts:",
        "",
    ]
    lines.extend(f"- [Rank {rank} extent chart](top_anomalies/rank_{rank:02d}_component_{int(row.component)}_extent.png)" for rank, (_, row) in enumerate(top.iterrows(), start=1))
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not CSV_PATH.exists() or not MASK_PATH.exists():
        raise FileNotFoundError("Expected Stage 2a CSV and TIFF mask under part2/stage2a_output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    defects = pd.read_csv(CSV_PATH)
    # Normalize the documented name (voxel_count) to the actual Stage 2a CSV name.
    if "voxel_count" not in defects and "voxels" in defects:
        defects = defects.rename(columns={"voxels": "voxel_count"})
    require_columns(defects, ["component", "voxel_count", "equivalent_diameter_voxels",
                              "bbox_x", "bbox_y", "bbox_z"])
    mask = np.asarray(tifffile.imread(MASK_PATH) > 0, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3-D mask TIFF, got shape {mask.shape}")
    # scipy supplies an independent connected-component count for the report.
    _, components_in_mask = ndimage.label(mask)
    z_density, xy_density = save_density_figure(mask, OUT_DIR / "spatial_strut_density_distribution.png")
    save_histograms(defects)
    save_top_anomaly_charts(defects.nlargest(TOP_N, "voxel_count"))
    write_report(defects, mask, z_density, xy_density, components_in_mask)
    print(f"Wrote {OUT_DIR / 'report.md'} and visualizations to {OUT_DIR}")


if __name__ == "__main__":
    main()
