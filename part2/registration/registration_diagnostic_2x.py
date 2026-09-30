#!/usr/bin/env python3
"""Memory-bounded slice-wise registration diagnostic at 2x downsampling."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/registration-diagnostic-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from stage2a_missing_strut_pipeline import graph_arrays, rasterize_graph


ROOT = Path(__file__).resolve().parent
PART2 = ROOT
SCAN = PART2 / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
REGISTRATION = PART2 / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
OUT = PART2 / "registration_diagnostic_2x"
DS = 2
THRESHOLD = 40000.0
SLAB_Z = 8                 # downsampled slices, 16 source slices
MAX_SHIFT = 6              # +/-12 source voxels in X/Y
RADIUS_SOURCE = 4.0        # same tube radius used by Stage 2a


def shift_score(cad: np.ndarray, material: np.ndarray, dx: int, dy: int) -> float:
    """Overlap of CAD and material after translating material in displayed XY."""
    # Compare only overlapping array regions; no copies are made here.
    cy0, cy1 = max(0, -dy), min(cad.shape[1], cad.shape[1] - dy)
    cx0, cx1 = max(0, -dx), min(cad.shape[2], cad.shape[2] - dx)
    my0, my1 = cy0 + dy, cy1 + dy
    mx0, mx1 = cx0 + dx, cx1 + dx
    c = cad[:, cy0:cy1, cx0:cx1]
    m = material[:, my0:my1, mx0:mx1]
    denom = int(np.count_nonzero(c))
    return float(np.count_nonzero(c & m) / max(denom, 1))


def search_shifts(cad: np.ndarray, material: np.ndarray) -> tuple[int, int, float, float]:
    candidates = []
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            candidates.append((shift_score(cad, material, dx, dy), dx, dy))
    candidates.sort(reverse=True)
    best_score, dx, dy = candidates[0]
    zero = next(score for score, x, y in candidates if x == 0 and y == 0)
    return int(dx), int(dy), float(best_score), float(zero)


def linear_fit(z: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(z, values, 1)
    predicted = slope * z + intercept
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with REGISTRATION.open() as handle:
        registration = json.load(handle)
    junctions, _, edges, *_ = graph_arrays(registration, json.load(open(PART2 / "data/missing_struts/octet_truss_9x9x9.json")))

    scan = tifffile.memmap(SCAN, mode="r")
    scan_small = np.asarray(scan[::DS, ::DS, ::DS])
    material = scan_small >= THRESHOLD
    cad = rasterize_graph(scan_small.shape, junctions, edges, DS, RADIUS_SOURCE)

    rows = []
    for z0 in range(0, scan_small.shape[0], SLAB_Z):
        z1 = min(scan_small.shape[0], z0 + SLAB_Z)
        slab_cad = cad[z0:z1]
        slab_material = material[z0:z1]
        dx, dy, best, zero = search_shifts(slab_cad, slab_material)
        rows.append({
            "z0_downsampled": z0,
            "z1_downsampled_exclusive": z1,
            "z_center_source_voxels": ((z0 + z1 - 1) / 2) * DS,
            "best_shift_x_downsampled": dx,
            "best_shift_y_downsampled": dy,
            "best_shift_x_source_voxels": dx * DS,
            "best_shift_y_source_voxels": dy * DS,
            "best_overlap_fraction": best,
            "zero_shift_overlap_fraction": zero,
            "overlap_gain": best - zero,
        })

    z = np.array([r["z_center_source_voxels"] for r in rows])
    sx = np.array([r["best_shift_x_source_voxels"] for r in rows], dtype=float)
    sy = np.array([r["best_shift_y_source_voxels"] for r in rows], dtype=float)
    slope_x, intercept_x, r2_x = linear_fit(z, sx)
    slope_y, intercept_y, r2_y = linear_fit(z, sy)
    # Ignore the first/last edge slabs and slabs with almost no material overlap
    # when looking for layer-wise behavior.  Edge slabs can select arbitrary
    # shifts because only a small fraction of the lattice is present there.
    core = (z >= 40) & (z <= scan.shape[0] - 40) & np.array([
        r["zero_shift_overlap_fraction"] >= 0.15 for r in rows
    ])
    core_slope_x, _, core_r2_x = linear_fit(z[core], sx[core])
    core_slope_y, _, core_r2_y = linear_fit(z[core], sy[core])
    first = core & (z <= np.percentile(z[core], 40))
    last = core & (z >= np.percentile(z[core], 60))
    first_shift = (float(np.median(sx[first])), float(np.median(sy[first])))
    last_shift = (float(np.median(sx[last])), float(np.median(sy[last])))
    piecewise_delta = float(np.hypot(last_shift[0] - first_shift[0], last_shift[1] - first_shift[1]))
    max_gain = max(r["overlap_gain"] for r in rows)
    mean_gain = float(np.mean([r["overlap_gain"] for r in rows]))
    trend_mag = float(np.hypot(slope_x, slope_y))

    with (OUT / "slab_shift_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Mid-volume overlay at 2x, with registered CAD boundary over CT.
    zmid = int(np.argmax(cad.sum(axis=(1, 2))))
    image = scan_small[zmid].astype(np.float32)
    lo, hi = np.percentile(image, [1, 99.5])
    image = np.clip((image - lo) / max(hi - lo, 1), 0, 1)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(image, cmap="gray")
    ax.contour(cad[zmid], levels=[0.5], colors="#00ffff", linewidths=0.5)
    ax.set_title(f"2x registration overlay — source Z ≈ {zmid * DS}")
    ax.axis("off")
    fig.savefig(OUT / "registration_overlay_2x.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(z, sx, "o-", label="best X shift")
    axes[0].plot(z, sy, "o-", label="best Y shift")
    axes[0].axhline(0, color="k", linewidth=0.7)
    axes[0].set_ylabel("Best shift (source voxels)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(z, [r["zero_shift_overlap_fraction"] for r in rows], "o-", label="zero shift")
    axes[1].plot(z, [r["best_overlap_fraction"] for r in rows], "o-", label="best XY shift")
    axes[1].set_xlabel("Source Z coordinate (voxels)")
    axes[1].set_ylabel("CAD/material overlap")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Slice-wise XY registration diagnostic (2x downsampled)")
    fig.savefig(OUT / "slice_wise_shift_diagnostic.png", dpi=180)
    plt.close(fig)

    summary = {
        "downsample": DS,
        "threshold": THRESHOLD,
        "scan_shape_zyx_source": list(scan.shape),
        "scan_shape_zyx_downsampled": list(scan_small.shape),
        "slab_width_source_voxels": SLAB_Z * DS,
        "search_range_source_voxels": [-MAX_SHIFT * DS, MAX_SHIFT * DS],
        "best_shift_trend_x_source_voxels_per_z": slope_x,
        "best_shift_trend_y_source_voxels_per_z": slope_y,
        "best_shift_trend_magnitude_source_voxels_per_z": trend_mag,
        "trend_r2_x": r2_x,
        "trend_r2_y": r2_y,
        "core_trend_x_source_voxels_per_z": core_slope_x,
        "core_trend_y_source_voxels_per_z": core_slope_y,
        "core_trend_r2_x": core_r2_x,
        "core_trend_r2_y": core_r2_y,
        "core_lower_z_median_shift_source_voxels": list(first_shift),
        "core_upper_z_median_shift_source_voxels": list(last_shift),
        "core_shift_change_magnitude_source_voxels": piecewise_delta,
        "mean_overlap_gain_from_local_shift": mean_gain,
        "max_overlap_gain_from_local_shift": max_gain,
        "interpretation_note": "A linear shift trend with Z is evidence of tilt/shear; piecewise jumps suggest layer shifts. Overlap gains can also reflect missing material and threshold effects.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    verdict = "No strong systematic Z-dependent shift detected."
    if piecewise_delta >= 2.0:
        verdict = "A piecewise Z-dependent XY displacement is suggested: the upper volume prefers a different shift than the lower volume."
    elif trend_mag >= 0.05 and (r2_x >= 0.5 or r2_y >= 0.5):
        verdict = "Systematic Z-dependent XY shift detected; inspect the fitted tilt/shear trend."
    elif mean_gain >= 0.02:
        verdict = "Local XY shifts improve overlap, but the pattern is not a clear linear Z trend."
    report = f"""# 2x Slice-wise Registration Diagnostic

## Result

**{verdict}**

The source TIFF was processed by read-only memmap and downsampled 2x in Z, Y,
and X. The registered graph was rasterized on the same grid. For each {SLAB_Z * DS}-source-voxel Z slab, the diagnostic searched XY translations from
{-MAX_SHIFT * DS} to {MAX_SHIFT * DS} source voxels and selected the shift that
maximized registered-CAD/CT-material overlap.

## Measured trends

| Quantity | Value |
|---|---:|
| X shift trend | {slope_x:.4f} source voxels per Z voxel (R²={r2_x:.3f}) |
| Y shift trend | {slope_y:.4f} source voxels per Z voxel (R²={r2_y:.3f}) |
| Trend magnitude | {trend_mag:.4f} source voxels per Z voxel |
| Core lower-Z median shift | ({first_shift[0]:.1f}, {first_shift[1]:.1f}) source voxels |
| Core upper-Z median shift | ({last_shift[0]:.1f}, {last_shift[1]:.1f}) source voxels |
| Lower-to-upper shift change | {piecewise_delta:.2f} source voxels |
| Mean overlap gain after local shift | {mean_gain:.4f} |
| Maximum overlap gain | {max_gain:.4f} |

The core-slab analysis excludes the outer 40 source-voxel margins and slabs with
less than 0.15 zero-shift overlap. In the retained region, the preferred shift
changes from approximately `({first_shift[0]:.0f}, {first_shift[1]:.0f})` in the lower-Z
part to `({last_shift[0]:.0f}, {last_shift[1]:.0f})` in the upper-Z part. This is
consistent with a possible layer-wise or piecewise registration displacement,
but it is not proof by itself because the search is quantized at 2 source voxels.
Missing struts, thresholding,
partial-volume effects, and nearby lattice material can produce local overlap
improvements even without registration error. A smooth linear shift with Z is
the signature expected from tilt/shear; abrupt step changes are more consistent
with layer-wise displacement.

## Files

- [Slice-wise shift results](slab_shift_results.csv)
- [Shift and overlap plots](slice_wise_shift_diagnostic.png)
- [2x registration overlay](registration_overlay_2x.png)
- [Machine-readable summary](summary.json)
"""
    (OUT / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
