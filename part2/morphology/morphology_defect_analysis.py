#!/usr/bin/env python3
"""Station-level morphology screening using existing Stage 2a artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("morphology_defect_thresholds.json")
MASK_NAMES = {"cad": "cad_occupancy_mask_ds4.tif", "missing": "missing_material_mask_ds4.tif", "excess": "excess_material_mask_ds4.tif", "expected_missing": "expected_missing_cad_mask_ds4.tif"}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("scan", "centerlines", "inventory", "masks_dir"):
        config[key] = str((ROOT / config[key]).resolve())
    return config


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def xyz_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array([float(row[f"{prefix}_x_vox"]), float(row[f"{prefix}_y_vox"]), float(row[f"{prefix}_z_vox"])], dtype=float)


def sample_nearest(volume, zyx: np.ndarray, scale: float = 1.0) -> int:
    q = np.rint(zyx / scale).astype(int)
    if np.any(q < 0) or np.any(q >= np.asarray(volume.shape)):
        return 0
    return int(volume[tuple(q)])


def disk_offsets(radius_um: float, spacing_xyz: np.ndarray, direction: np.ndarray) -> np.ndarray:
    lim = np.ceil(radius_um / spacing_xyz).astype(int)
    zz, yy, xx = np.mgrid[-lim[2]:lim[2] + 1, -lim[1]:lim[1] + 1, -lim[0]:lim[0] + 1]
    points = np.column_stack((xx.ravel() * spacing_xyz[0], yy.ravel() * spacing_xyz[1], zz.ravel() * spacing_xyz[2]))
    points = points[np.linalg.norm(points, axis=1) <= radius_um + 1e-6]
    return points - (points @ direction)[:, None] * direction


def longest_gap(values: np.ndarray, step_um: float) -> float:
    best = current = 0
    for value in values.astype(bool):
        current = current + 1 if not value else 0
        best = max(best, current)
    return float(best * step_um)


def analyze_strut(row: dict[str, str], scan, masks: dict[str, object], config: dict) -> tuple[dict, list[dict]]:
    spacing_zyx = np.asarray(config["voxel_spacing_um_zyx"], dtype=float)
    spacing_xyz = spacing_zyx[::-1]
    start, end = xyz_from_row(row, "start"), xyz_from_row(row, "end")
    delta = end - start
    length_um = float(np.linalg.norm(delta * spacing_xyz))
    if length_um <= 0:
        return {"strut_id": int(row["strut_id"]), "analysis_status": "invalid_centerline"}, []
    direction = delta * spacing_xyz / length_um
    step_um = float(config["station_step_um"])
    station_count = max(3, int(math.ceil(length_um / step_um)) + 1)
    stations = np.linspace(0.0, 1.0, station_count)
    centers = start[None, :] + stations[:, None] * delta[None, :]
    offsets = disk_offsets(float(config["transverse_radius_um"]), spacing_xyz, direction) / spacing_xyz
    threshold = float(config["thresholds"]["ct_intensity"])
    records, fractions, centroids = [], [], []
    for index, center in enumerate(centers):
        points = center[None, :] + offsets
        zyx = np.rint(points[:, ::-1]).astype(int)
        valid = np.all((zyx >= 0) & (zyx < np.asarray(scan.shape)), axis=1)
        material = np.zeros(len(points), dtype=bool)
        if np.any(valid):
            material[valid] = np.asarray(scan[zyx[valid, 0], zyx[valid, 1], zyx[valid, 2]]) >= threshold
        fraction = float(material.mean())
        fractions.append(fraction)
        centroids.append(points[material].mean(axis=0) if np.any(material) else np.full(3, np.nan))
        mask_values = {name: sample_nearest(volume, center[::-1], 4.0) for name, volume in masks.items()}
        records.append({"strut_id": int(row["strut_id"]), "station_index": index, "station_fraction": stations[index], "station_center_x_vox": float(center[0]), "station_center_y_vox": float(center[1]), "station_center_z_vox": float(center[2]), "material_fraction": fraction, "effective_diameter_um": 2.0 * float(config["transverse_radius_um"]) * math.sqrt(fraction / math.pi), "material_present": int(fraction >= float(config["thresholds"]["station_material_fraction"])), "cad_mask_value_ds4": mask_values["cad"], "missing_mask_value_ds4": mask_values["missing"], "excess_mask_value_ds4": mask_values["excess"], "expected_missing_mask_value_ds4": mask_values["expected_missing"]})
    present = np.asarray(fractions) >= float(config["thresholds"]["station_material_fraction"])
    diameters = np.asarray([r["effective_diameter_um"] for r in records])
    centroid_array = np.asarray(centroids)
    valid_centroids = np.isfinite(centroid_array).all(axis=1)
    if valid_centroids.sum() >= 2:
        q = centroid_array[valid_centroids]
        line = start + ((q - start) @ delta / max(float(delta @ delta), 1e-12))[:, None] * delta
        max_deviation = float(np.max(np.linalg.norm((q - line) * spacing_xyz, axis=1)))
        tortuosity = float(np.sum(np.linalg.norm(np.diff(q, axis=0) * spacing_xyz, axis=1)) / length_um)
    else:
        max_deviation = tortuosity = float("nan")
    nominal, thresholds = float(config["nominal_diameter_um"]), config["thresholds"]
    median_diameter = float(np.median(diameters[present])) if np.any(present) else float("nan")
    labels = []
    if np.isfinite(median_diameter) and median_diameter / nominal < float(thresholds["thin_diameter_ratio"]): labels.append("thin")
    if np.isfinite(median_diameter) and median_diameter / nominal > float(thresholds["thick_diameter_ratio"]): labels.append("thick")
    if longest_gap(present, step_um) >= float(thresholds["broken_gap_um"]): labels.append("broken")
    if np.isfinite(max_deviation) and (max_deviation >= float(thresholds["bent_max_deviation_um"]) or tortuosity >= float(thresholds["bent_tortuosity"])): labels.append("bent")
    return {"strut_id": int(row["strut_id"]), "analysis_status": "analyzed", "stage2a_classification": row.get("classification", ""), "stage2a_ct_material_occupancy": float(row.get("ct_material_occupancy", "nan")), "cad_length_um": length_um, "station_count": len(records), "station_material_fraction": float(np.mean(present)), "longest_material_gap_um": longest_gap(present, step_um), "median_effective_diameter_um": median_diameter, "diameter_ratio_to_nominal": median_diameter / nominal if np.isfinite(median_diameter) else float("nan"), "max_centerline_deviation_um": max_deviation, "tortuosity": tortuosity, "missing_mask_station_fraction": float(np.mean([r["missing_mask_value_ds4"] > 0 for r in records])), "excess_mask_station_fraction": float(np.mean([r["excess_mask_value_ds4"] > 0 for r in records])), "screening_labels": ";".join(labels)}, records


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run(config_path: Path, output_dir: Path, max_struts: int, overwrite: bool = False) -> dict:
    config = load_config(config_path.resolve())
    inventory = {int(r["strut_id"]): r for r in load_csv(Path(config["inventory"]))}
    centerlines = {int(r["strut_id"]): r for r in load_csv(Path(config["centerlines"]))}
    excluded = set(config["selection"]["exclude_classifications"])
    selected = [sid for sid in sorted(inventory) if inventory[sid].get("classification") not in excluded and sid in centerlines][:max_struts]
    scan = tifffile.memmap(config["scan"], mode="r")
    masks = {name: tifffile.memmap(Path(config["masks_dir"]) / filename, mode="r") for name, filename in MASK_NAMES.items()}
    summaries, stations = [], []
    for sid in selected:
        summary, station_rows = analyze_strut(centerlines[sid], scan, masks, config)
        summary["stage2a_classification"] = inventory[sid].get("classification", "")
        summary["stage2a_ct_material_occupancy"] = float(inventory[sid].get("ct_material_occupancy", "nan"))
        summaries.append(summary); stations.extend(station_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path, station_path = output_dir / "morphology_defect_test_summary.csv", output_dir / "morphology_defect_test_stations.csv"
    provenance_path = output_dir / "morphology_defect_test_provenance.json"
    for path in (summary_path, station_path, provenance_path):
        if path.exists() and not overwrite: raise FileExistsError(f"Refusing to replace existing new test output: {path}")
    write_csv(summary_path, summaries); write_csv(station_path, stations)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "selected_strut_ids": selected, "summary_csv": str(summary_path), "stations_csv": str(station_path), "config": config, "existing_outputs_read_only": [str(Path(config["inventory"])), str(Path(config["centerlines"]))] + [str(Path(config["masks_dir"]) / n) for n in MASK_NAMES.values()], "classification_policy": "Stage 2a classification is copied unchanged; excluded missingness classes are not relabeled."}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent); parser.add_argument("--max-struts", type=int, default=16); parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_dir.resolve(), args.max_struts, args.overwrite), indent=2))


if __name__ == "__main__": main()
