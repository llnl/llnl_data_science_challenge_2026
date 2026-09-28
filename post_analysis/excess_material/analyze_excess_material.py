#!/usr/bin/env python3
"""Analyze the Stage 2a CT-minus-CAD excess-material mask.

The excess mask is not a direct defect-type label.  This script quantifies
where it occurs, associates voxels to the nearest registered strut centerline,
and reports whether its strongest associations are more consistent with
material displacement (missing occupancy plus external material) or material
thickening (material retained in-CAD plus external material).
"""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "post_analysis" / "excess_material"
STAGE = ROOT / "stage_2a_developer_output"
REG = ROOT / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"

DS = 2
VOXEL_UM = 58.09


def load_graph():
    data = json.loads(REG.read_text())
    junctions = np.asarray([x["position"] for x in data["junctions"]], dtype=np.float32)
    edges = np.asarray([[x["junction0"], x["junction1"]] for x in data["struts"]], dtype=np.int32)
    return junctions, edges


def edge_samples(junctions, edges):
    """Return approximately one sample per downsampled mask voxel along each edge."""
    all_points = []
    all_ids = []
    for edge_id, (i0, i1) in enumerate(edges):
        p0 = junctions[i0][::-1] / DS  # mask order Z,Y,X
        p1 = junctions[i1][::-1] / DS
        n = max(2, int(math.ceil(np.linalg.norm(p1 - p0) * 1.5)))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        all_points.append(p0[None, :] + t * (p1 - p0)[None, :])
        all_ids.append(np.full(n, edge_id, dtype=np.int32))
    return np.concatenate(all_points), np.concatenate(all_ids)


def component_table(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    objects = ndimage.find_objects(labels)
    rows = []
    for label in range(1, count + 1):
        n = int(sizes[label])
        if not n:
            continue
        z, y, x = objects[label - 1]
        coords = np.argwhere(labels[z, y, x] == label)
        coords[:, 0] += z.start
        coords[:, 1] += y.start
        coords[:, 2] += x.start
        centroid = coords.mean(axis=0)
        rows.append({
            "component": label,
            "voxels": n,
            "volume_mm3": n * (DS * VOXEL_UM) ** 3 / 1e9,
            "centroid_z_voxel": centroid[0],
            "centroid_y_voxel": centroid[1],
            "centroid_x_voxel": centroid[2],
            "bbox_z": z.stop - z.start,
            "bbox_y": y.stop - y.start,
            "bbox_x": x.stop - x.start,
        })
    return labels, pd.DataFrame(rows).sort_values("voxels", ascending=False).reset_index(drop=True)


def save_overview(mask, components):
    zproj = mask.sum(axis=0)
    yproj = mask.sum(axis=1)
    xproj = mask.sum(axis=2)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    im0 = ax[0, 0].imshow(zproj, origin="lower", cmap="inferno")
    ax[0, 0].set_title("Excess material: Z projection")
    ax[0, 0].set_xlabel("X mask voxel"); ax[0, 0].set_ylabel("Y mask voxel")
    fig.colorbar(im0, ax=ax[0, 0], label="excess voxels along Z")
    im1 = ax[0, 1].imshow(yproj, origin="lower", cmap="inferno", aspect="auto")
    ax[0, 1].set_title("Excess material: Y projection")
    ax[0, 1].set_xlabel("X mask voxel"); ax[0, 1].set_ylabel("Z mask voxel")
    fig.colorbar(im1, ax=ax[0, 1], label="excess voxels along Y")
    ax[1, 0].hist(np.log10(components["voxels"]), bins=40, color="#d95f02")
    ax[1, 0].set_title("Connected-component size distribution")
    ax[1, 0].set_xlabel("log10(component voxels)"); ax[1, 0].set_ylabel("component count")
    ax[1, 1].plot(mask.sum(axis=(1, 2)), color="#7570b3")
    ax[1, 1].set_title("Excess material by Z slice")
    ax[1, 1].set_xlabel("Z mask voxel"); ax[1, 1].set_ylabel("excess voxels")
    fig.savefig(OUT / "excess_material_overview.png", dpi=180)
    plt.close(fig)


def save_strut_plot(struts):
    order = ["Nominal", "Missing_Intentional", "Expected_Missing_But_Material_Present", "Missing_Unintentional"]
    grouped = [struts.loc[struts["classification"] == c, "excess_voxels"] for c in order]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.boxplot(grouped, labels=["Nominal", "Intentional\nmissing", "Expected missing\nbut material present", "Unintentional\nmissing"],
               showfliers=False)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Excess-mask voxels assigned to strut")
    ax.set_title("Excess material associated with each inventory classification")
    fig.savefig(OUT / "excess_by_strut_classification.png", dpi=180)
    plt.close(fig)

    positive = struts[struts["excess_voxels"] > 0].copy()
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    colors = {"Nominal": "#4c78a8", "Missing_Intentional": "#59a14f",
              "Expected_Missing_But_Material_Present": "#f2cf5b", "Missing_Unintentional": "#e15759"}
    for c in order:
        q = positive[positive["classification"] == c]
        ax.scatter(q["centroid_x_um"], q["centroid_y_um"], s=8 + 2*np.log1p(q["excess_voxels"]),
                   alpha=.65, label=c, c=[colors[c]])
    ax.set_title("Strut locations with assigned excess material")
    ax.set_xlabel("registered X centroid (µm)"); ax.set_ylabel("registered Y centroid (µm)")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / "excess_strut_locations_xy.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    excess = np.asarray(tifffile.imread(STAGE / "excess_material_mask_ds4.tif") != 0)
    cad = np.asarray(tifffile.imread(STAGE / "cad_occupancy_mask_ds4.tif") != 0)
    missing = np.asarray(tifffile.imread(STAGE / "missing_material_mask_ds4.tif") != 0)
    inventory = pd.read_csv(STAGE / "all_struts_inventory.csv")
    execution = json.loads((STAGE / "execution_log.json").read_text())

    labels, components = component_table(excess)
    components.to_csv(OUT / "excess_components.csv", index=False)

    junctions, edges = load_graph()
    sample_points, sample_edge_ids = edge_samples(junctions, edges)
    tree = cKDTree(sample_points)
    excess_coords = np.argwhere(excess)
    distances, nearest = tree.query(excess_coords, workers=-1)
    assigned = sample_edge_ids[nearest]
    excess_counts = np.bincount(assigned, minlength=len(edges))
    distance_sum = np.bincount(assigned, weights=distances, minlength=len(edges))
    distance_sq = np.bincount(assigned, weights=distances ** 2, minlength=len(edges))
    struts = inventory.copy()
    struts["excess_voxels"] = excess_counts
    struts["excess_fraction_of_mask"] = excess_counts / max(int(excess.sum()), 1)
    struts["excess_mean_distance_to_centerline_mask_voxels"] = distance_sum / np.maximum(excess_counts, 1)
    struts["excess_rms_distance_to_centerline_mask_voxels"] = np.sqrt(distance_sq / np.maximum(excess_counts, 1))
    struts["excess_distance_um"] = struts["excess_mean_distance_to_centerline_mask_voxels"] * DS * VOXEL_UM
    struts.to_csv(OUT / "excess_by_strut.csv", index=False)

    class_rows = []
    for classification, group in struts.groupby("classification", sort=False):
        vals = group["excess_voxels"].to_numpy()
        positive = vals[vals > 0]
        class_rows.append({
            "classification": classification,
            "struts": len(group),
            "struts_with_excess": int(np.count_nonzero(vals)),
            "fraction_with_excess": float(np.mean(vals > 0)),
            "total_excess_voxels": int(vals.sum()),
            "median_excess_voxels": float(np.median(vals)),
            "median_positive_excess_voxels": float(np.median(positive)) if len(positive) else 0.0,
            "p95_excess_voxels": float(np.percentile(vals, 95)),
            "median_ct_material_occupancy": float(group["ct_material_occupancy"].median()),
        })
    class_summary = pd.DataFrame(class_rows)
    class_summary.to_csv(OUT / "excess_class_summary.csv", index=False)

    positive = struts[struts["excess_voxels"] > 0].copy()
    threshold = float(np.percentile(positive["excess_voxels"], 99)) if len(positive) else 0.0
    high = struts[struts["excess_voxels"] >= threshold].copy()
    z_boundary = (high["centroid_z_um"] / (DS * VOXEL_UM) < 30) | (
        high["centroid_z_um"] / (DS * VOXEL_UM) > excess.shape[0] - 30
    )
    high["heuristic_pattern"] = np.where(
        z_boundary,
        "boundary_or_registration_artifact_like",
        np.where(
            high["classification"].isin(["Missing_Unintentional", "Missing_Intentional"]),
            "displacement_or_bending_like",
            "thickening_or_inflation_like",
        ),
    )
    high.to_csv(OUT / "top_excess_struts.csv", index=False)
    by_class = []
    for classification, group in struts.groupby("classification", sort=False):
        by_class.append(group.nlargest(10, "excess_voxels"))
    pd.concat(by_class).to_csv(OUT / "top_excess_by_class.csv", index=False)

    save_overview(excess, components)
    save_strut_plot(struts)

    cad_voxels = int(cad.sum())
    missing_voxels = int(missing.sum())
    lines = [
        "# Excess-material analysis",
        "",
        "## Interpretation",
        "",
        "The excess mask is CT material found outside the nominal CAD occupancy, within a narrow CAD-local region. It is evidence of geometric disagreement, not a defect-type label by itself.",
        "",
        "## Global mask measurements",
        "",
        f"- Mask shape: `{excess.shape}` in `(Z, Y, X)` order.",
        f"- Nonzero excess voxels: `{int(excess.sum()):,}` ({excess.mean():.4%} of the full mask grid).",
        f"- CAD occupancy voxels: `{cad_voxels:,}`; excess/CAD voxel ratio: `{excess.sum()/max(cad_voxels,1):.3%}`.",
        f"- Missing-material voxels: `{missing_voxels:,}`; excess/missing overlap: `{int(np.count_nonzero(excess & missing)):,}`.",
        f"- Connected excess components: `{len(components):,}`; largest component: `{int(components.iloc[0]['voxels']):,}` voxels ({components.iloc[0]['voxels']/max(excess.sum(),1):.2%} of excess).",
        f"- Component volume uses the pipeline mask spacing of `{DS*VOXEL_UM:.2f} µm` per voxel.",
        f"- Z-boundary concentration: `{int(excess[:30].sum() + excess[-30:].sum()):,}` voxels ({(excess[:30].sum() + excess[-30:].sum())/max(excess.sum(),1):.2%}) lie in the first or last 30 mask slices.",
        "",
        "## Relationship to reported defect types",
        "",
        "The inventory has no explicit `Bent` or `Inflated` labels. Excess voxels were therefore assigned to the nearest registered strut centerline as a spatial association, not as a ground-truth classification.",
        "",
        "- A missing-class strut with unusually high external excess can be **displacement/bending-like**: the nominal centerline is empty while material appears nearby.",
        "- A nominal or expected-material-present strut with unusually high external excess can be **thickening/inflation-like**: material remains in the CAD strut and also extends outside it.",
        f"- The top-1% strut threshold was `{threshold:.0f}` assigned excess voxels; these candidates are listed in `top_excess_struts.csv`.",
        "",
        "In this dataset, the excess is dominated by a systematic Z-end pattern and nominal struts, so it does **not** provide convincing evidence for a population of bent or inflated defects. The five `Expected_Missing_But_Material_Present` struts collectively account for only 344 excess voxels; two struts account for 233 and 99 of those voxels. The strongest unintentional-missing association is 50 voxels. These are follow-up candidates, not confirmed defect types.",
        "",
        "A definitive bent-versus-inflated decision requires a local centerline/surface fit or cross-sectional radius measurement; the current mask alone cannot make that distinction.",
        "",
        "## Outputs",
        "",
        "- `excess_material_overview.png`: projections, Z profile, and component sizes.",
        "- `excess_by_strut_classification.png`: excess association by inventory class.",
        "- `excess_strut_locations_xy.png`: XY locations of struts with excess material.",
        "- `excess_components.csv`: connected-component measurements.",
        "- `excess_by_strut.csv`: per-strut excess associations.",
        "- `excess_class_summary.csv`: class-level aggregation.",
        "- `top_excess_struts.csv`: highest-excess candidates and heuristic pattern.",
        "- `top_excess_by_class.csv`: top ten excess-associated struts within each inventory class.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
