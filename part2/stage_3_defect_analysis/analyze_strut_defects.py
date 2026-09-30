#!/usr/bin/env python3
"""Memory-bounded cross-sectional screening for registered lattice struts.

The Stage 2a inventory is retained as the authoritative missing-material
classification.  This program adds explainable screening measurements for
struts that contain material: cross-sectional radius, outside-tube material,
and centerline displacement/curvature.  It never builds a full-volume mask;
the TIFF is memory mapped and sampled in bounded strut batches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


PART2 = Path(__file__).resolve().parents[1]
DEFAULT_SCAN = PART2 / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
DEFAULT_REGISTRATION = PART2 / "registration/alignment_check/candidate_affine_registered.json"
DEFAULT_INVENTORY = PART2 / "registration/alignment_check/stage2a_candidate_registration_output/all_struts_inventory.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=40081.0)
    parser.add_argument("--voxel-size-um", type=float, default=58.09)
    parser.add_argument("--nominal-radius-voxels", type=float, default=6.0)
    parser.add_argument("--outer-radius-factor", type=float, default=1.5)
    parser.add_argument("--trim-fraction", type=float, default=0.20)
    parser.add_argument("--stations", type=int, default=21)
    parser.add_argument("--empty-station-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--score-threshold", type=float, default=3.0)
    parser.add_argument("--min-broken-stations", type=int, default=3)
    parser.add_argument("--write-stations", action="store_true")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--max-struts", type=int, help="Process only the first N struts (smoke testing).")
    return parser.parse_args()


def load_inventory(path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {int(row["strut_id"]): row for row in rows}
    return rows, by_id


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    junctions = np.asarray([row["position"] for row in payload["junctions"]], dtype=np.float32)
    struts = sorted(payload["struts"], key=lambda row: int(row["id"]))
    ids = np.asarray([int(row["id"]) for row in struts], dtype=np.int32)
    if not np.array_equal(ids, np.arange(len(struts), dtype=np.int32)):
        raise ValueError("Registration strut IDs must be dense and ordered.")
    edges = np.asarray([[row["junction0"], row["junction1"]] for row in struts], dtype=np.int32)
    return junctions, edges


def disk_offsets(outer_radius: float) -> tuple[np.ndarray, np.ndarray]:
    limit = int(math.ceil(outer_radius))
    yy, xx = np.mgrid[-limit : limit + 1, -limit : limit + 1]
    offsets = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    radial = np.linalg.norm(offsets, axis=1)
    return offsets[radial <= outer_radius + 1e-6], radial[radial <= outer_radius + 1e-6]


def bases_for_directions(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to each XYZ direction."""
    reference = np.zeros_like(directions)
    reference[np.arange(len(directions)), np.argmin(np.abs(directions), axis=1)] = 1.0
    first = np.cross(directions, reference)
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-8)
    second = np.cross(directions, first)
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-8)
    return first, second


def longest_runs(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[0], dtype=np.int16)
    current = np.zeros(values.shape[0], dtype=np.int16)
    for column in values.T:
        current = np.where(column, current + 1, 0)
        result = np.maximum(result, current)
    return result


def row_nanmax(values: np.ndarray) -> np.ndarray:
    """Maximum by row without warnings for fully absent-material struts."""
    finite = np.isfinite(values)
    result = np.max(np.where(finite, values, -np.inf), axis=1)
    return np.where(finite.any(axis=1), result, np.nan)


def row_nanrms(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    mean_square = np.nansum(values**2, axis=1) / np.maximum(count, 1)
    return np.where(count > 0, np.sqrt(mean_square), np.nan)


def robust_location_scale(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = values[mask & np.isfinite(values)]
    if not len(selected):
        selected = values[np.isfinite(values)]
    median = float(np.nanmedian(selected))
    mad = float(np.nanmedian(np.abs(selected - median)))
    # The small floor prevents an unrealistically tight nominal distribution
    # from exploding scores due to quantization at source-voxel resolution.
    return median, max(1.4826 * mad, 1e-6)


def positive_z(values: np.ndarray, median: float, scale: float) -> np.ndarray:
    return np.maximum((values - median) / scale, 0.0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_parquet(path: Path, rows: list[dict[str, object]]) -> str | None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return "pyarrow is not installed; CSV output was written instead."
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return None


def generate_report(output: Path, rows: list[dict[str, object]], provenance: dict[str, object]) -> str | None:
    """Write a reviewable Markdown report and a compact visual summary figure."""
    labels = sorted({str(row["primary_defect"]) for row in rows})
    preferred = ["Nominal", "Thin", "Inflated", "Bent", "Broken", "Missing_Intentional", "Missing_Unintentional"]
    labels = [label for label in preferred if label in labels] + [label for label in labels if label not in preferred]
    ranking_score = {"Thin": "thin_score", "Inflated": "inflated_score", "Bent": "bend_score", "Broken": "broken_score"}
    visualization_warning = None
    figure_name = "defect_analysis_visualizations.png"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        visualization_warning = "matplotlib is not installed; the Markdown report was written without visualizations."
    else:
        colors = {label: plt.cm.tab10(index % 10) for index, label in enumerate(labels)}
        counts = [sum(str(row["primary_defect"]) == label for row in rows) for label in labels]
        score_names = ["thin_score", "inflated_score", "bend_score", "broken_score"]
        score_titles = ["Thin", "Inflated", "Bent", "Broken"]

        figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
        axes[0, 0].bar(labels, counts, color=[colors[label] for label in labels])
        axes[0, 0].set_title("Primary classification counts")
        axes[0, 0].set_ylabel("Struts")
        axes[0, 0].tick_params(axis="x", rotation=35)
        for index, count in enumerate(counts):
            axes[0, 0].text(index, count, str(count), ha="center", va="bottom", fontsize=8)

        for label in labels:
            values = [float(row["overall_deviation_score"]) for row in rows if str(row["primary_defect"]) == label and np.isfinite(float(row["overall_deviation_score"]))]
            if values:
                axes[0, 1].hist(values, bins=30, alpha=0.55, label=f"{label} (n={len(values)})", color=colors[label])
        axes[0, 1].axvline(0, color="black", linewidth=0.8)
        axes[0, 1].set_title("Overall deviation score by primary classification")
        axes[0, 1].set_xlabel("Overall score")
        axes[0, 1].set_ylabel("Struts")
        axes[0, 1].legend(fontsize=8)

        for label in labels:
            selected = [row for row in rows if str(row["primary_defect"]) == label and np.isfinite(float(row["sampled_occupancy"])) and np.isfinite(float(row["median_cross_section_radius_um"]))]
            axes[1, 0].scatter(
                [float(row["sampled_occupancy"]) for row in selected],
                [float(row["median_cross_section_radius_um"]) for row in selected],
                s=10, alpha=0.45, label=label, color=colors[label], edgecolors="none",
            )
        axes[1, 0].set_title("Cross-sectional occupancy versus radius")
        axes[1, 0].set_xlabel("Sampled occupancy")
        axes[1, 0].set_ylabel("Median radius (um)")
        axes[1, 0].legend(fontsize=8)

        sample_rows = []
        for label in labels:
            candidates = [row for row in rows if str(row["primary_defect"]) == label]
            score_name = ranking_score.get(label, "overall_deviation_score")
            candidates = [row for row in candidates if np.isfinite(float(row[score_name]))]
            sample_rows.extend(sorted(candidates, key=lambda row: float(row[score_name]), reverse=True)[:3])
        sample_rows = sample_rows[:30]
        heat = np.asarray([[float(row[name]) for name in score_names] for row in sample_rows], dtype=float)
        if len(sample_rows):
            finite_heat = heat[np.isfinite(heat)]
            heat_max = max(3.0, float(np.max(finite_heat))) if len(finite_heat) else 3.0
            image = axes[1, 1].imshow(heat, aspect="auto", cmap="magma", vmin=0, vmax=heat_max)
            axes[1, 1].set_yticks(np.arange(len(sample_rows)))
            axes[1, 1].set_yticklabels([f"{row['strut_id']} {row['primary_defect']}" for row in sample_rows], fontsize=7)
            figure.colorbar(image, ax=axes[1, 1], label="Robust deviation score")
        axes[1, 1].set_xticks(np.arange(len(score_titles)))
        axes[1, 1].set_xticklabels(score_titles)
        axes[1, 1].set_title("Representative highest-scoring samples by class")
        axes[1, 1].set_xlabel("Signal")
        figure.savefig(output / figure_name, dpi=160)
        plt.close(figure)

    counts = provenance["primary_defect_counts"]
    total = len(rows)
    lines = [
        "# Stage 3 Defect Analysis Summary",
        "",
        f"Generated: `{provenance['created_utc']}`  ",
        f"Processed struts: **{total:,}**  ",
        "",
        "## Executive summary",
        "",
        "This report summarizes the Stage 3 cross-sectional screening run. Stage 2a missing-material classifications are preserved as authoritative labels; the additional Thin, Inflated, Bent, and Broken labels are screening calls calibrated against robust nominal behavior.",
        "",
        f"- Nominal: **{counts.get('Nominal', 0):,}** ({counts.get('Nominal', 0) / max(total, 1):.1%})",
        f"- Screening defect labels: **{sum(counts.get(label, 0) for label in ('Thin', 'Inflated', 'Bent', 'Broken')):,}** primary classifications",
        f"- Stage 2 missing labels retained: **{sum(counts.get(label, 0) for label in ('Missing_Intentional', 'Missing_Unintentional')):,}**",
        f"- Rows needing review: **{sum(bool(row['needs_review']) for row in rows):,}** ({sum(bool(row['needs_review']) for row in rows) / max(total, 1):.1%})",
        "",
        "## Inputs and run configuration",
        "",
        f"- Scan: `{provenance['scan']}`",
        f"- Registration graph: `{provenance['registration']}`",
        f"- Stage 2a inventory: `{provenance['inventory']}`",
        f"- Scan shape and dtype: `{provenance.get('scan_shape', 'not recorded')}`, `{provenance.get('scan_dtype', 'not recorded')}`",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]
    for key, value in provenance["parameters"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += ["", "## Primary classifications", "", "| Classification | Count | Fraction |", "|---|---:|---:|"]
    for label in labels:
        count = counts.get(label, 0)
        lines.append(f"| {label} | {count:,} | {count / max(total, 1):.2%} |")

    lines += ["", "## Feature and score summaries", "", "The table below reports median and 10th/90th percentile values across all processed struts. Scores are non-negative robust deviations; values at or above the configured threshold are candidates for that signal.", "", "| Metric | Median | P10 | P90 |", "|---|---:|---:|---:|"]
    report_metrics = [
        ("overall_deviation_score", "Overall score"),
        ("sampled_occupancy", "Sampled occupancy"),
        ("median_cross_section_radius_um", "Median radius (um)"),
        ("median_outside_nominal_fraction", "Outside nominal fraction"),
        ("max_centerline_offset_um", "Max centerline offset (um)"),
        ("bend_curvature_um", "Bend curvature (um)"),
    ]
    for key, title in report_metrics:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        lines.append(f"| {title} | {np.nanmedian(values):.4g} | {np.nanpercentile(values, 10):.4g} | {np.nanpercentile(values, 90):.4g} |")

    lines += ["", "## Representative defect samples", "", "The visualization selects up to three highest overall-deviation struts per primary classification. These are prioritization examples, not ground-truth exemplars; inspect them in the source volume before making a quality decision.", "", "| Strut ID | Primary class | Overall | Thin | Inflated | Bent | Broken | Review |", "|---:|---|---:|---:|---:|---:|---:|:---:|"]
    for label in labels:
        score_name = ranking_score.get(label, "overall_deviation_score")
        candidates = [row for row in rows if str(row["primary_defect"]) == label and np.isfinite(float(row[score_name]))]
        candidates = sorted(candidates, key=lambda row: float(row[score_name]), reverse=True)[:3]
        for row in candidates:
            lines.append(f"| {row['strut_id']} | {label} | {float(row['overall_deviation_score']):.2f} | {float(row['thin_score']):.2f} | {float(row['inflated_score']):.2f} | {float(row['bend_score']):.2f} | {float(row['broken_score']):.2f} | {'yes' if row['needs_review'] else 'no'} |")
    if visualization_warning:
        lines += ["", f"> Note: {visualization_warning}"]
    else:
        lines += ["", f"![Stage 3 defect visualizations]({figure_name})", "", f"The figure includes class counts, overall-score distributions, occupancy/radius scatter, and a score heatmap for representative samples ranked by their primary signal."]
    lines += ["", "## Interpretation and limitations", "", "- These are screening labels, not validated ground-truth defect annotations.", "- Stage 2a Missing_Intentional and Missing_Unintentional labels are preserved and are not reclassified from TIFF evidence.", "- Inflated material is measured as an outer-annulus proxy; attachedness and dross identity require local connected-component analysis.", "- A registration error can resemble bending or centerline offset.", "- Representative samples should be reviewed against the source TIFF and registration before downstream decisions.", ""]
    (output / "defect_analysis_report.md").write_text("\n".join(lines))
    return visualization_warning


def main() -> None:
    args = parse_args()
    if args.stations < 5:
        raise ValueError("Use at least five longitudinal stations for curvature screening.")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.max_struts is not None and args.max_struts < 1:
        raise ValueError("max-struts must be positive.")
    for path in (args.scan, args.registration, args.inventory):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=True)
    _, inventory = load_inventory(args.inventory)
    junctions, edges = load_graph(args.registration)
    ids = np.arange(len(edges), dtype=np.int32)
    if args.max_struts is not None:
        ids = ids[: args.max_struts]
    if any(int(strut_id) not in inventory for strut_id in ids):
        raise ValueError("Inventory does not contain every requested registered strut.")

    scan = tifffile.memmap(args.scan, mode="r")
    if scan.ndim != 3:
        raise ValueError(f"Expected a 3-D CT scan, received shape {scan.shape}.")
    shape_zyx = np.asarray(scan.shape, dtype=np.int32)
    outer_radius = args.nominal_radius_voxels * args.outer_radius_factor
    offsets, radial = disk_offsets(outer_radius)
    inner = radial <= args.nominal_radius_voxels + 1e-6
    outer = ~inner
    if not np.any(outer):
        raise ValueError("outer-radius-factor must produce an annulus outside the nominal radius.")
    t = np.linspace(args.trim_fraction, 1.0 - args.trim_fraction, args.stations, dtype=np.float32)

    count = len(ids)
    feature = {name: np.full(count, np.nan, dtype=np.float32) for name in (
        "sampled_occupancy", "sampled_mean_intensity", "median_radius_um", "p10_radius_um",
        "radius_cv", "median_excess_fraction", "max_centerline_offset_um",
        "rms_centerline_offset_um", "bend_curvature_um", "valid_station_fraction",
    )}
    longest_low_run = np.zeros(count, dtype=np.int16)
    station_writer = None
    station_handle = None
    if args.write_stations:
        station_path = args.output / "defect_analysis_by_station.csv"
        station_handle = station_path.open("w", newline="")
        station_writer = csv.DictWriter(station_handle, fieldnames=[
            "strut_id", "station_id", "position_fraction", "position_x_um", "position_y_um", "position_z_um",
            "material_occupancy", "mean_intensity", "equivalent_radius_um", "centroid_offset_um",
            "outside_nominal_fraction", "valid_sample_fraction",
        ])
        station_writer.writeheader()

    try:
        for start in range(0, count, args.batch_size):
            batch_ids = ids[start : start + args.batch_size]
            edge_block = edges[batch_ids]
            p0 = junctions[edge_block[:, 0]]
            p1 = junctions[edge_block[:, 1]]
            vector = p1 - p0
            length = np.linalg.norm(vector, axis=1)
            direction = vector / np.maximum(length[:, None], 1e-8)
            basis_a, basis_b = bases_for_directions(direction)
            centers = p0[:, None, :] + t[None, :, None] * vector[:, None, :]
            points = (
                centers[:, :, None, :]
                + offsets[None, None, :, 0, None] * basis_a[:, None, None, :]
                + offsets[None, None, :, 1, None] * basis_b[:, None, None, :]
            )
            indices = np.rint(points).astype(np.int32)
            valid = (
                (indices[..., 0] >= 0) & (indices[..., 0] < shape_zyx[2])
                & (indices[..., 1] >= 0) & (indices[..., 1] < shape_zyx[1])
                & (indices[..., 2] >= 0) & (indices[..., 2] < shape_zyx[0])
            )
            clipped = indices.copy()
            clipped[..., 0] = np.clip(clipped[..., 0], 0, shape_zyx[2] - 1)
            clipped[..., 1] = np.clip(clipped[..., 1], 0, shape_zyx[1] - 1)
            clipped[..., 2] = np.clip(clipped[..., 2], 0, shape_zyx[0] - 1)
            values = scan[clipped[..., 2], clipped[..., 1], clipped[..., 0]]
            material = (values >= args.threshold) & valid

            inner_valid = valid[:, :, inner]
            inner_material = material[:, :, inner]
            inner_values = values[:, :, inner]
            inner_count = inner_valid.sum(axis=2)
            material_count = inner_material.sum(axis=2)
            occupancy = material_count / np.maximum(inner_count, 1)
            station_intensity = (inner_values * inner_valid).sum(axis=2) / np.maximum(inner_count, 1)
            radius = np.sqrt(material_count / math.pi) * args.voxel_size_um
            outer_valid = valid[:, :, outer]
            outer_material = material[:, :, outer]
            excess = outer_material.sum(axis=2) / np.maximum(outer_valid.sum(axis=2), 1)
            valid_station = inner_count > 0

            weights = inner_material.astype(np.float32)
            inner_points = points[:, :, inner, :]
            centroid = (inner_points * weights[..., None]).sum(axis=2) / np.maximum(material_count[..., None], 1)
            displacement = centroid - centers
            centroid_offset = np.linalg.norm(displacement, axis=2) * args.voxel_size_um
            centroid_offset[material_count == 0] = np.nan
            displacement[material_count == 0] = np.nan
            second_diff = displacement[:, 2:] - 2 * displacement[:, 1:-1] + displacement[:, :-2]
            curvature = row_nanmax(np.linalg.norm(second_diff, axis=2)) * args.voxel_size_um
            low = occupancy < args.empty_station_fraction

            block = slice(start, start + len(batch_ids))
            feature["sampled_occupancy"][block] = material_count.sum(axis=1) / np.maximum(inner_count.sum(axis=1), 1)
            feature["sampled_mean_intensity"][block] = (inner_values * inner_valid).sum(axis=(1, 2)) / np.maximum(inner_count.sum(axis=1), 1)
            feature["median_radius_um"][block] = np.nanmedian(radius, axis=1)
            feature["p10_radius_um"][block] = np.nanpercentile(radius, 10, axis=1)
            feature["radius_cv"][block] = np.nanstd(radius, axis=1) / np.maximum(np.nanmean(radius, axis=1), 1e-6)
            feature["median_excess_fraction"][block] = np.nanmedian(excess, axis=1)
            feature["max_centerline_offset_um"][block] = row_nanmax(centroid_offset)
            feature["rms_centerline_offset_um"][block] = row_nanrms(centroid_offset)
            feature["bend_curvature_um"][block] = curvature
            feature["valid_station_fraction"][block] = valid_station.mean(axis=1)
            longest_low_run[block] = longest_runs(low)

            if station_writer is not None:
                for local_id, strut_id in enumerate(batch_ids):
                    for station_id in range(args.stations):
                        center = centers[local_id, station_id] * args.voxel_size_um
                        station_writer.writerow({
                            "strut_id": int(strut_id),
                            "station_id": station_id,
                            "position_fraction": float(t[station_id]),
                            "position_x_um": float(center[0]),
                            "position_y_um": float(center[1]),
                            "position_z_um": float(center[2]),
                            "material_occupancy": float(occupancy[local_id, station_id]),
                            "mean_intensity": float(station_intensity[local_id, station_id]),
                            "equivalent_radius_um": float(radius[local_id, station_id]),
                            "centroid_offset_um": float(centroid_offset[local_id, station_id]),
                            "outside_nominal_fraction": float(excess[local_id, station_id]),
                            "valid_sample_fraction": float(inner_count[local_id, station_id] / max(int(inner.sum()), 1)),
                        })
    finally:
        if station_handle is not None:
            station_handle.close()

    source_class = np.asarray([inventory[int(strut_id)]["classification"] for strut_id in ids])
    nominal = source_class == "Nominal"
    good_nominal = nominal & np.isfinite(feature["sampled_occupancy"])
    if np.any(good_nominal):
        cutoff = np.nanmedian(feature["sampled_occupancy"][good_nominal])
        high_quality = good_nominal & (feature["sampled_occupancy"] >= cutoff)
    else:
        high_quality = np.isfinite(feature["sampled_occupancy"])
    baselines = {name: robust_location_scale(values, high_quality) for name, values in feature.items()}
    thin_score = positive_z(-feature["median_radius_um"], -baselines["median_radius_um"][0], baselines["median_radius_um"][1])
    radius_high = positive_z(feature["median_radius_um"], *baselines["median_radius_um"])
    excess_high = positive_z(feature["median_excess_fraction"], *baselines["median_excess_fraction"])
    inflated_score = np.maximum(radius_high, excess_high)
    offset_high = positive_z(feature["max_centerline_offset_um"], *baselines["max_centerline_offset_um"])
    curvature_high = positive_z(feature["bend_curvature_um"], *baselines["bend_curvature_um"])
    bend_score = np.maximum(offset_high, curvature_high)
    run_median, run_scale = robust_location_scale(longest_low_run.astype(np.float32), high_quality)
    broken_score = positive_z(longest_low_run.astype(np.float32), run_median, run_scale)
    occupancy_low = positive_z(-feature["sampled_occupancy"], -baselines["sampled_occupancy"][0], baselines["sampled_occupancy"][1])
    missing_score = np.maximum(occupancy_low, broken_score)

    rows: list[dict[str, object]] = []
    for index, strut_id in enumerate(ids):
        source = inventory[int(strut_id)]
        source_label = source["classification"]
        is_stage2_missing = source_label in {"Missing_Intentional", "Missing_Unintentional"}
        labels: list[tuple[str, float]] = []
        if not is_stage2_missing:
            if broken_score[index] >= args.score_threshold and longest_low_run[index] >= args.min_broken_stations:
                labels.append(("Broken", float(broken_score[index])))
            if thin_score[index] >= args.score_threshold:
                labels.append(("Thin", float(thin_score[index])))
            if inflated_score[index] >= args.score_threshold:
                labels.append(("Inflated", float(inflated_score[index])))
            if bend_score[index] >= args.score_threshold:
                labels.append(("Bent", float(bend_score[index])))
        labels.sort(key=lambda item: item[1], reverse=True)
        if is_stage2_missing:
            primary = source_label
            secondary = ""
        elif labels:
            primary = labels[0][0]
            secondary = ";".join(label for label, _ in labels[1:])
        else:
            primary = "Nominal"
            secondary = ""
        active_scores = np.asarray([thin_score[index], inflated_score[index], bend_score[index], broken_score[index]], dtype=np.float32)
        overall = float(np.sqrt(np.sum(np.square(np.maximum(active_scores - args.score_threshold, 0.0)))))
        review = (
            feature["valid_station_fraction"][index] < 0.90
            or len(labels) > 1
            or (not is_stage2_missing and np.any((active_scores >= args.score_threshold - 0.5) & (active_scores < args.score_threshold)))
            or source_label == "Expected_Missing_But_Material_Present"
        )
        row: dict[str, object] = {
            "strut_id": int(strut_id),
            "junction0_id": source["junction0_id"],
            "junction1_id": source["junction1_id"],
            "junction0_degree": source["junction0_degree"],
            "junction1_degree": source["junction1_degree"],
            "unit_cell_ids": source["unit_cell_ids"],
            "unit_cell_indices": source["unit_cell_indices"],
            "stage2_classification": source_label,
            "expected_by_0point5_cad": source["expected_by_0point5_cad"],
            "primary_defect": primary,
            "secondary_defects": secondary,
            "overall_deviation_score": overall,
            "missing_score": float(missing_score[index]),
            "broken_score": float(broken_score[index]),
            "thin_score": float(thin_score[index]),
            "inflated_score": float(inflated_score[index]),
            "bend_score": float(bend_score[index]),
            "sampled_occupancy": float(feature["sampled_occupancy"][index]),
            "sampled_mean_intensity": float(feature["sampled_mean_intensity"][index]),
            "median_cross_section_radius_um": float(feature["median_radius_um"][index]),
            "p10_cross_section_radius_um": float(feature["p10_radius_um"][index]),
            "radius_cv": float(feature["radius_cv"][index]),
            "median_outside_nominal_fraction": float(feature["median_excess_fraction"][index]),
            "max_centerline_offset_um": float(feature["max_centerline_offset_um"][index]),
            "rms_centerline_offset_um": float(feature["rms_centerline_offset_um"][index]),
            "bend_curvature_um": float(feature["bend_curvature_um"][index]),
            "longest_low_material_run_stations": int(longest_low_run[index]),
            "valid_station_fraction": float(feature["valid_station_fraction"][index]),
            "confidence": "screening" if review else "nominal_reference_calibrated",
            "needs_review": bool(review),
        }
        rows.append(row)

    summary_path = args.output / "defect_analysis_by_strut.csv"
    write_csv(summary_path, rows)
    parquet_warning = maybe_write_parquet(args.output / "defect_analysis_by_strut.parquet", rows) if args.write_parquet else None
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["primary_defect"])] = counts.get(str(row["primary_defect"]), 0) + 1
    class_rows = [
        {"primary_defect": label, "strut_count": count, "fraction_of_processed_struts": count / len(rows)}
        for label, count in sorted(counts.items())
    ]
    write_csv(args.output / "defect_class_summary.csv", class_rows)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scan": str(args.scan),
        "registration": str(args.registration),
        "inventory": str(args.inventory),
        "processed_struts": len(rows),
        "scan_shape": list(map(int, scan.shape)),
        "scan_dtype": str(scan.dtype),
        "parameters": {
            "threshold": args.threshold,
            "voxel_size_um": args.voxel_size_um,
            "nominal_radius_voxels": args.nominal_radius_voxels,
            "outer_radius_factor": args.outer_radius_factor,
            "trim_fraction": args.trim_fraction,
            "stations": args.stations,
            "empty_station_fraction": args.empty_station_fraction,
            "batch_size": args.batch_size,
            "score_threshold": args.score_threshold,
        },
        "nominal_reference_population": int(high_quality.sum()),
        "feature_baselines": {name: {"median": median, "robust_scale": scale} for name, (median, scale) in baselines.items()},
        "primary_defect_counts": counts,
        "station_output_written": args.write_stations,
        "parquet_warning": parquet_warning,
        "limitations": [
            "Bent, thin, inflated, and broken labels are screening calls calibrated from robust nominal behavior, not manually validated ground truth.",
            "Stage 2a missing labels are preserved and are not reclassified by this program.",
            "Outside-nominal material is a cross-sectional excess proxy; attachedness requires local component analysis before a production dross/inflation claim.",
        ],
    }
    report_warning = generate_report(args.output, rows, provenance)
    provenance["report_warning"] = report_warning
    (args.output / "defect_analysis_summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(rows)} per-strut rows to {summary_path}")
    print(f"Wrote summary report to {args.output / 'defect_analysis_report.md'}")
    if args.write_stations:
        print(f"Wrote station-level evidence to {args.output / 'defect_analysis_by_station.csv'}")
    if parquet_warning:
        print(f"Parquet note: {parquet_warning}")


if __name__ == "__main__":
    main()
