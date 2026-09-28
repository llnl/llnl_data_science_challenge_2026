#!/usr/bin/env python3
"""Refine and compare CT/graph registration models without loading the full TIFF."""
from __future__ import annotations

import csv
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/registration-alignment-check-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from stage2a_missing_strut_pipeline import (
    longest_run,
    rasterize_graph,
    sample_struts,
)


PART2 = Path(__file__).resolve().parent
SCAN = PART2 / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
REGISTRATION = PART2 / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
INVENTORY = PART2 / "stage_2a_developer_output/all_struts_inventory.csv"
OUT = PART2 / "registratiion_alignment_check"
THRESHOLD = 40081.0
TUBE_RADIUS = 6.0
EMPTY_STATION_FRACTION = 0.05
EXPECTED_SUPPORTED_CUTOFF = 0.04297328814864163
SEARCH_RADIUS = 4
CONTROL_QUANTILE = 0.75
FIT_STATIONS = 13
FIT_TRIM = 0.20
Z_SLAB = 64
XY_BINS = 3
MIN_BLOCK_POINTS = 120
VOXEL_UM = 58.09


def load_inputs():
    registration = json.loads(REGISTRATION.read_text())
    junctions = np.asarray([j["position"] for j in registration["junctions"]], dtype=np.float32)
    edges = np.asarray(
        [[s["junction0"], s["junction1"]] for s in registration["struts"]],
        dtype=np.int32,
    )
    with INVENTORY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return registration, junctions, edges, rows


def station_points(junctions, edges, ids, stations=FIT_STATIONS):
    t = np.linspace(FIT_TRIM, 1.0 - FIT_TRIM, stations, dtype=np.float32)
    selected = edges[np.asarray(ids, dtype=np.int32)]
    p0 = junctions[selected[:, 0]]
    p1 = junctions[selected[:, 1]]
    points = p0[:, None, :] + t[None, :, None] * (p1 - p0)[:, None, :]
    return points.reshape(-1, 3), np.repeat(np.asarray(ids, dtype=np.int32), stations)


FIT_OFFSETS = np.asarray(
    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int16,
)


def point_score(scan, points, dx, dy, offsets=FIT_OFFSETS):
    """Mean threshold occupancy near centerline points at a candidate XY shift."""
    shifted = points.copy()
    shifted[:, 0] += dx
    shifted[:, 1] += dy
    xyz = np.rint(shifted).astype(np.int32)
    total = hit = 0
    shape_xyz = np.asarray(scan.shape[::-1])
    for offset in offsets:
        q = xyz + offset
        valid = np.all((q >= 0) & (q < shape_xyz), axis=1)
        q = q[valid]
        if len(q):
            hit += int(np.count_nonzero(scan[q[:, 2], q[:, 1], q[:, 0]] >= THRESHOLD))
            total += len(q)
    return hit / max(total, 1)


def collect_observations(scan, points):
    x_edges = np.linspace(points[:, 0].min(), points[:, 0].max(), XY_BINS + 1)
    y_edges = np.linspace(points[:, 1].min(), points[:, 1].max(), XY_BINS + 1)
    z_edges = np.arange(32, scan.shape[0] - 31 + Z_SLAB, Z_SLAB)
    observations = []
    for zi in range(len(z_edges) - 1):
        for yi in range(XY_BINS):
            for xi in range(XY_BINS):
                keep = (
                    (points[:, 2] >= z_edges[zi])
                    & (points[:, 2] < z_edges[zi + 1])
                    & (points[:, 0] >= x_edges[xi])
                    & (points[:, 0] < x_edges[xi + 1])
                    & (points[:, 1] >= y_edges[yi])
                    & (points[:, 1] < y_edges[yi + 1])
                )
                block = points[keep]
                if len(block) < MIN_BLOCK_POINTS:
                    continue
                scores = []
                for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                    for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
                        scores.append((point_score(scan, block, dx, dy), dx, dy))
                scores.sort(reverse=True)
                best, dx, dy = scores[0]
                zero = next(s for s, x, y in scores if x == 0 and y == 0)
                second = scores[1][0]
                observations.append(
                    {
                        "x_center": float(np.mean(block[:, 0])),
                        "y_center": float(np.mean(block[:, 1])),
                        "z_center": float(np.mean(block[:, 2])),
                        "n_points": int(len(block)),
                        "best_dx": int(dx),
                        "best_dy": int(dy),
                        "best_score": float(best),
                        "zero_score": float(zero),
                        "score_gain": float(best - zero),
                        "peak_margin": float(best - second),
                    }
                )
    return observations


def design(points):
    return np.column_stack((np.ones(len(points)), points[:, 0], points[:, 1], points[:, 2]))


def weighted_fit(a, target, weights):
    root = np.sqrt(np.maximum(weights, 1.0))[:, None]
    return np.linalg.lstsq(a * root, target * root[:, 0], rcond=None)[0]


def fit_models(observations):
    centers = np.asarray([[o["x_center"], o["y_center"], o["z_center"]] for o in observations])
    dx = np.asarray([o["best_dx"] for o in observations], dtype=float)
    dy = np.asarray([o["best_dy"] for o in observations], dtype=float)
    weights = np.asarray([o["n_points"] * max(o["score_gain"], 0.002) for o in observations])
    global_shift = np.asarray([np.average(dx, weights=weights), np.average(dy, weights=weights)])

    a = design(centers)
    affine_x = weighted_fit(a, dx, weights)
    affine_y = weighted_fit(a, dy, weights)

    candidate_splits = sorted(set(centers[:, 2]))
    best_piecewise = None
    for split in candidate_splits[2:-2]:
        low = centers[:, 2] <= split
        high = ~low
        low_shift = np.asarray(
            [np.average(dx[low], weights=weights[low]), np.average(dy[low], weights=weights[low])]
        )
        high_shift = np.asarray(
            [np.average(dx[high], weights=weights[high]), np.average(dy[high], weights=weights[high])]
        )
        predicted = np.where(low[:, None], low_shift, high_shift)
        residual = np.column_stack((dx, dy)) - predicted
        loss = float(np.sum(weights[:, None] * residual**2))
        if best_piecewise is None or loss < best_piecewise["loss"]:
            best_piecewise = {
                "split_z": float(split),
                "low_shift": low_shift,
                "high_shift": high_shift,
                "loss": loss,
            }
    return {
        "baseline": {},
        "global_translation": {"shift": global_shift},
        "affine": {"x_coef": affine_x, "y_coef": affine_y},
        "piecewise_z": best_piecewise,
    }


def model_shift(points, name, model):
    if name == "baseline":
        return np.zeros((len(points), 2), dtype=float)
    if name == "global_translation":
        return np.repeat(model["shift"][None, :], len(points), axis=0)
    if name == "affine":
        a = design(points)
        return np.column_stack((a @ model["x_coef"], a @ model["y_coef"]))
    low = points[:, 2] <= model["split_z"]
    return np.where(low[:, None], model["low_shift"], model["high_shift"])


def transform_points(points, name, model):
    result = np.asarray(points, dtype=float).copy()
    result[:, :2] += model_shift(result, name, model)
    return result


def evaluate_models(scan, points, models):
    results = {}
    for name, model in models.items():
        shifted = transform_points(points, name, model)
        results[name] = point_score(scan, shifted, 0, 0)
    return results


def classify_model(scan, junctions, edges, rows, name, model):
    transformed = transform_points(junctions, name, model).astype(np.float32)
    occupancy, stations, _, _, _ = sample_struts(
        scan,
        transformed,
        edges,
        THRESHOLD,
        analysis_ds=2,
        tube_radius_full=TUBE_RADIUS,
        trim_fraction=0.20,
        stations=21,
    )
    expected = np.asarray([r["expected_by_0point5_cad"].lower() == "true" for r in rows])
    # Hold the recorded Stage 2a decision thresholds fixed so the unmodified
    # baseline reproduces the published inventory before comparing models.
    expected_cutoff = EXPECTED_SUPPORTED_CUTOFF
    empty = stations < EMPTY_STATION_FRACTION
    runs = np.asarray([longest_run(row) for row in empty])
    missing = (occupancy <= expected_cutoff) & (runs >= math.ceil(0.75 * stations.shape[1]))
    labels = np.full(len(rows), "Nominal", dtype=object)
    labels[expected & missing] = "Missing_Intentional"
    labels[expected & ~missing] = "Expected_Missing_But_Material_Present"
    labels[~expected & missing] = "Missing_Unintentional"
    return occupancy, labels, expected_cutoff


def save_observations(observations):
    with (OUT / "alignment_observations_1x.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)


def save_model_plot(observations, models):
    z = np.asarray([o["z_center"] for o in observations])
    dx = np.asarray([o["best_dx"] for o in observations])
    dy = np.asarray([o["best_dy"] for o in observations])
    gain = np.asarray([o["score_gain"] for o in observations])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    size = 25 + 500 * np.maximum(gain, 0)
    axes[0].scatter(z, dx, s=size, alpha=0.65, label="local 1x observations")
    axes[1].scatter(z, dy, s=size, alpha=0.65, label="local 1x observations")
    zline = np.linspace(z.min(), z.max(), 300)
    center = np.column_stack(
        (
            np.full_like(zline, np.median([o["x_center"] for o in observations])),
            np.full_like(zline, np.median([o["y_center"] for o in observations])),
            zline,
        )
    )
    for name in ("global_translation", "affine", "piecewise_z"):
        shifts = model_shift(center, name, models[name])
        axes[0].plot(zline, shifts[:, 0], linewidth=2, label=name.replace("_", " "))
        axes[1].plot(zline, shifts[:, 1], linewidth=2, label=name.replace("_", " "))
    axes[0].set_ylabel("X correction (source voxels)")
    axes[1].set_ylabel("Y correction (source voxels)")
    axes[1].set_xlabel("Source Z coordinate (voxels)")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(alpha=0.2)
        ax.legend(ncol=2)
    fig.suptitle("1x local alignment observations and fitted models")
    fig.savefig(OUT / "01_alignment_models_1x.png", dpi=180)
    plt.close(fig)


def save_score_plot(scores):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    names = list(scores)
    values = [scores[n] for n in names]
    bars = ax.bar([n.replace("_", "\n") for n in names], values, color=["#777777", "#4c78a8", "#f58518", "#54a24b"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(ylabel="Held-out nominal centerline material score", title="Registration model comparison")
    ax.set_ylim(max(0, min(values) - 0.03), min(1, max(values) + 0.03))
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "02_model_score_comparison.png", dpi=180)
    plt.close(fig)


def save_whole_volume_overlays(scan, junctions, edges, selected_name, selected_model):
    small = np.asarray(scan[::2, ::2, ::2])
    baseline = rasterize_graph(small.shape, junctions, edges, 2, TUBE_RADIUS)
    corrected_junctions = transform_points(junctions, selected_name, selected_model).astype(np.float32)
    corrected = rasterize_graph(small.shape, corrected_junctions, edges, 2, TUBE_RADIUS)
    counts = baseline.sum(axis=(1, 2))
    valid = np.flatnonzero(counts > np.percentile(counts[counts > 0], 40))
    zs = [int(valid[len(valid) // 6]), int(valid[len(valid) // 2]), int(valid[5 * len(valid) // 6])]
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)
    for row, z in enumerate(zs):
        image = small[z].astype(np.float32)
        lo, hi = np.percentile(image, [1, 99.5])
        image = np.clip((image - lo) / max(hi - lo, 1), 0, 1)
        for col, (mask, title, color) in enumerate(
            ((baseline, "baseline", "#00ffff"), (corrected, selected_name.replace("_", " "), "#ff4da6"))
        ):
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].contour(mask[z], levels=[0.5], colors=color, linewidths=0.45)
            axes[row, col].set_title(f"{title}: source Z ≈ {z * 2}")
            axes[row, col].axis("off")
    fig.suptitle("Intermediate 2x overlays across the volume")
    fig.savefig(OUT / "03_lower_middle_upper_overlays_2x.png", dpi=180)
    plt.close(fig)


def save_local_1x_examples(scan, junctions, edges, control_ids, selected_name, selected_model):
    mids = (junctions[edges[control_ids, 0]] + junctions[edges[control_ids, 1]]) / 2
    targets = np.percentile(mids[:, 2], [20, 50, 80])
    chosen = [control_ids[np.argmin(np.abs(mids[:, 2] - target))] for target in targets]
    corrected = transform_points(junctions, selected_name, selected_model)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, sid in zip(axes, chosen):
        edge = edges[sid]
        base = junctions[edge]
        corr = corrected[edge]
        allp = np.vstack((base, corr))
        lo = np.floor(allp.min(axis=0) - 8).astype(int)
        hi = np.ceil(allp.max(axis=0) + 8).astype(int) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(scan.shape[::-1]))
        crop = np.asarray(scan[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]])
        projection = crop.max(axis=0).astype(np.float32)
        p1, p99 = np.percentile(projection, [1, 99.5])
        projection = np.clip((projection - p1) / max(p99 - p1, 1), 0, 1)
        ax.imshow(projection, cmap="gray", origin="upper")
        ax.plot(base[:, 0] - lo[0], base[:, 1] - lo[1], color="#00ffff", linewidth=2, label="baseline")
        ax.plot(corr[:, 0] - lo[0], corr[:, 1] - lo[1], color="#ff4da6", linewidth=2, label=selected_name.replace("_", " "))
        ax.scatter(base[:, 0] - lo[0], base[:, 1] - lo[1], color="#00ffff", s=15)
        ax.set_title(f"1x local MIP: strut {sid}, Z≈{mids[control_ids == sid, 2][0]:.0f}")
        ax.axis("off")
    axes[0].legend(loc="upper left")
    fig.suptitle("Representative nominal-strut checks at full source resolution")
    fig.savefig(OUT / "04_local_nominal_struts_1x.png", dpi=180)
    plt.close(fig)


def save_classification_outputs(labels_by_model, occupancy_by_model, cutoffs):
    baseline = labels_by_model["baseline"]
    rows = []
    for name, labels in labels_by_model.items():
        changed = int(np.count_nonzero(labels != baseline))
        counts = Counter(labels)
        rows.append(
            {
                "model": name,
                "changed_vs_recomputed_baseline": changed,
                "Nominal": counts["Nominal"],
                "Missing_Unintentional": counts["Missing_Unintentional"],
                "Missing_Intentional": counts["Missing_Intentional"],
                "Expected_Missing_But_Material_Present": counts["Expected_Missing_But_Material_Present"],
                "expected_supported_cutoff": cutoffs[name],
                "mean_occupancy": float(np.mean(occupancy_by_model[name])),
            }
        )
    with (OUT / "classification_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    names = [r["model"] for r in rows]
    changes = [r["changed_vs_recomputed_baseline"] for r in rows]
    bars = ax.bar([n.replace("_", "\n") for n in names], changes, color="#b279a2")
    ax.bar_label(bars, padding=3)
    ax.set(ylabel="Strut labels changed vs baseline", title="Classification sensitivity to registration model")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "05_classification_sensitivity.png", dpi=180)
    plt.close(fig)
    return rows


def save_selected_changes(rows, labels_by_model, occupancy_by_model, selected_name):
    baseline = labels_by_model["baseline"]
    selected = labels_by_model[selected_name]
    changed_rows = []
    for index in np.flatnonzero(selected != baseline):
        changed_rows.append(
            {
                "strut_id": int(rows[index]["strut_id"]),
                "unit_cell_indices": rows[index]["unit_cell_indices"],
                "baseline_classification": baseline[index],
                "candidate_classification": selected[index],
                "baseline_occupancy": float(occupancy_by_model["baseline"][index]),
                "candidate_occupancy": float(occupancy_by_model[selected_name][index]),
                "occupancy_change": float(
                    occupancy_by_model[selected_name][index] - occupancy_by_model["baseline"][index]
                ),
            }
        )
    with (OUT / "selected_model_classification_changes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changed_rows[0]))
        writer.writeheader()
        writer.writerows(changed_rows)
    return changed_rows


def serializable_models(models):
    result = {}
    for name, model in models.items():
        result[name] = {}
        for key, value in model.items():
            result[name][key] = value.tolist() if isinstance(value, np.ndarray) else value
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registration, junctions, edges, rows = load_inputs()
    scan = tifffile.memmap(SCAN, mode="r")
    nominal_occ = np.asarray(
        [float(r["ct_material_occupancy"]) for r in rows if r["classification"] == "Nominal"]
    )
    control_cutoff = float(np.quantile(nominal_occ, CONTROL_QUANTILE))
    control_ids = np.asarray(
        [
            int(r["strut_id"])
            for r in rows
            if r["classification"] == "Nominal" and float(r["ct_material_occupancy"]) >= control_cutoff
        ],
        dtype=np.int32,
    )
    train_ids = control_ids[control_ids % 5 != 0]
    test_ids = control_ids[control_ids % 5 == 0]
    train_points, _ = station_points(junctions, edges, train_ids)
    test_points, _ = station_points(junctions, edges, test_ids)

    observations = collect_observations(scan, train_points)
    models = fit_models(observations)
    scores = evaluate_models(scan, test_points, models)
    selected_name = max(scores, key=scores.get)
    selected_model = models[selected_name]

    save_observations(observations)
    save_model_plot(observations, models)
    save_score_plot(scores)
    save_whole_volume_overlays(scan, junctions, edges, selected_name, selected_model)
    save_local_1x_examples(scan, junctions, edges, control_ids, selected_name, selected_model)

    occupancy_by_model = {}
    labels_by_model = {}
    cutoffs = {}
    for name, model in models.items():
        occupancy, labels, cutoff = classify_model(scan, junctions, edges, rows, name, model)
        occupancy_by_model[name] = occupancy
        labels_by_model[name] = labels
        cutoffs[name] = cutoff
    sensitivity = save_classification_outputs(labels_by_model, occupancy_by_model, cutoffs)
    selected_changes = save_selected_changes(
        rows, labels_by_model, occupancy_by_model, selected_name
    )

    baseline_score = scores["baseline"]
    selected_gain = scores[selected_name] - baseline_score
    candidate_registration = copy.deepcopy(registration)
    candidate_junctions = transform_points(junctions, selected_name, selected_model)
    for item, position in zip(candidate_registration["junctions"], candidate_junctions):
        item["position"] = position.tolist()
    candidate_registration["_registration_alignment_check"] = {
        "status": "candidate_for_manual_validation_not_ground_truth",
        "source_registration": str(REGISTRATION),
        "model": selected_name,
        "model_parameters": serializable_models({selected_name: selected_model})[selected_name],
        "held_out_score_gain_over_baseline": selected_gain,
        "source_registration_was_not_modified": True,
    }
    (OUT / "candidate_affine_registered.json").write_text(
        json.dumps(candidate_registration, indent=2) + "\n"
    )

    piece = models["piecewise_z"]
    recommendation = (
        "Do not replace the registration JSON automatically. The best model does not improve held-out nominal material support enough to justify a correction."
        if selected_gain < 0.005
        else f"Validate the {selected_name.replace('_', ' ')} model on manually reviewed struts before promoting the separate candidate JSON."
    )
    summary = {
        "scan_access": "read-only tifffile memmap",
        "whole_volume_downsample": 2,
        "local_refinement_resolution": "1x source voxels",
        "threshold": THRESHOLD,
        "tube_radius_source_voxels": TUBE_RADIUS,
        "empty_station_fraction": EMPTY_STATION_FRACTION,
        "fixed_expected_supported_cutoff": EXPECTED_SUPPORTED_CUTOFF,
        "control_occupancy_cutoff": control_cutoff,
        "control_struts": int(len(control_ids)),
        "training_control_struts": int(len(train_ids)),
        "held_out_control_struts": int(len(test_ids)),
        "alignment_observations": int(len(observations)),
        "models": serializable_models(models),
        "held_out_scores": scores,
        "selected_model": selected_name,
        "selected_score_gain_over_baseline": selected_gain,
        "classification_sensitivity": sensitivity,
        "selected_model_classification_changes": int(len(selected_changes)),
        "candidate_registration": "candidate_affine_registered.json",
        "recommendation": recommendation,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    model_lines = "\n".join(
        f"| {name.replace('_', ' ')} | {score:.5f} | {score - baseline_score:+.5f} |"
        for name, score in scores.items()
    )
    sens_lines = "\n".join(
        f"| {r['model'].replace('_', ' ')} | {r['changed_vs_recomputed_baseline']} | "
        f"{r['Missing_Unintentional']} | {r['Missing_Intentional']} |"
        for r in sensitivity
    )
    report = f"""# Registration Alignment Check

## Outcome

The alignment was refined with full-resolution (1x) centerline samples from
{len(control_ids):,} high-occupancy nominal control struts. Whole-volume overlays
remain at 2x to bound memory. The best held-out model was **{selected_name.replace('_', ' ')}**,
with a material-support gain of **{selected_gain:+.5f}** over the unmodified registration.

{recommendation}

## Method

- The source TIFF remained a read-only memmap.
- Controls were nominal struts in the upper occupancy quartile
  (`ct_material_occupancy >= {control_cutoff:.4f}`).
- Controls were split deterministically into training and held-out sets by strut ID.
- Local XY corrections were searched at integer 1x resolution from
  `-{SEARCH_RADIUS}` to `+{SEARCH_RADIUS}` source voxels.
- Four models were compared: baseline, global XY translation, affine XY displacement
  as a function of X/Y/Z, and a two-block piecewise-Z translation.
- All four models were then passed through the Stage 2a tube sampler to measure
  classification sensitivity, using the recorded execution parameters
  (6-voxel tube, threshold 40081, empty-station fraction 0.05, and fixed
  expected-supported cutoff 0.0429733).

## Model comparison

| Model | Held-out material score | Gain vs baseline |
|---|---:|---:|
{model_lines}

The fitted piecewise breakpoint is source Z **{piece['split_z']:.1f}**. Its lower
shift is `({piece['low_shift'][0]:.2f}, {piece['low_shift'][1]:.2f})` voxels and
its upper shift is `({piece['high_shift'][0]:.2f}, {piece['high_shift'][1]:.2f})`
voxels.

![1x observations and models](01_alignment_models_1x.png)

![Held-out model scores](02_model_score_comparison.png)

## Intermediate visual checks

The first figure compares baseline and selected-model contours at lower, middle,
and upper Z. The second uses full-resolution local maximum-intensity projections
for representative high-occupancy nominal struts.

![Lower, middle, and upper overlays](03_lower_middle_upper_overlays_2x.png)

![Local nominal controls at 1x](04_local_nominal_struts_1x.png)

## Classification sensitivity

| Model | Labels changed | Missing unintentional | Missing intentional |
|---|---:|---:|---:|
{sens_lines}

![Classification sensitivity](05_classification_sensitivity.png)

Classification changes are sensitivity indicators, not corrected ground truth.
The recorded Stage 2a cutoff is held fixed across models so changes reflect
registration alone rather than a moving decision boundary.

## Interpretation and recommended use

The 1x held-out score is the primary model-selection evidence. A model that only
fits training-block shifts, without improving held-out nominal struts, is likely
capturing threshold, partial-volume, or missing-material effects rather than a
true registration error. Preserve the supplied registration JSON unless a model
shows a meaningful held-out gain and its corrected overlays are consistently
better across the volume.

The earlier 3D voxel-render half-voxel display offset is separate from this
registration test; it should be corrected in the renderer but does not alter
the CT sampling performed here.

## Data files

- [Local 1x observations](alignment_observations_1x.csv)
- [Classification sensitivity table](classification_sensitivity.csv)
- [Selected-model per-strut changes](selected_model_classification_changes.csv)
- [Candidate corrected registration](candidate_affine_registered.json)
- [Machine-readable summary](summary.json)
"""
    (OUT / "report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
