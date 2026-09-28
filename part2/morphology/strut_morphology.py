#!/usr/bin/env python3
"""Per-strut morphology screening for registered lattice CT data.

The source TIFF is memory-mapped and each strut is processed independently.
No full-volume mask or distance transform is created.  Morphology labels are
provisional screening labels; Stage 2a missingness is copied unchanged.
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
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("morphology_config.json")


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    required = {"scan", "registration", "inventory", "voxel_spacing_um_zyx",
                "nominal_diameter_um", "roi_radius_um", "node_exclusion_distance_um",
                "thresholds"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration missing keys: {sorted(missing)}")
    for key in ("scan", "registration", "inventory"):
        config[key] = str((ROOT / config[key]).resolve())
    return config


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    junctions = np.asarray([j["position"] for j in payload["junctions"]], dtype=np.float32)
    struts = sorted(payload["struts"], key=lambda x: int(x["id"]))
    ids = np.asarray([int(s["id"]) for s in struts])
    if not np.array_equal(ids, np.arange(len(struts))):
        raise ValueError("Registered graph strut IDs must be dense and ordered")
    edges = np.asarray([[int(s["junction0"]), int(s["junction1"])] for s in struts], dtype=np.int32)
    return junctions, edges


def load_inventory(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["strut_id"]): row for row in csv.DictReader(handle)}


def orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.zeros(3, dtype=np.float32)
    ref[int(np.argmin(np.abs(direction)))] = 1.0
    u = np.cross(direction, ref)
    u /= max(float(np.linalg.norm(u)), 1e-8)
    v = np.cross(direction, u)
    v /= max(float(np.linalg.norm(v)), 1e-8)
    return u, v


def disk_offsets(radius_um: float, spacing_xyz: np.ndarray) -> np.ndarray:
    """Physical xyz offsets for a transverse disk, with anisotropic sampling."""
    lim = np.ceil(radius_um / spacing_xyz).astype(int)
    zz, yy, xx = np.mgrid[-lim[2]:lim[2] + 1, -lim[1]:lim[1] + 1, -lim[0]:lim[0] + 1]
    # The returned points are expressed in xyz, while array indexing is zyx.
    xyz = np.column_stack((xx.ravel() * spacing_xyz[0], yy.ravel() * spacing_xyz[1], zz.ravel() * spacing_xyz[2]))
    return xyz[np.linalg.norm(xyz, axis=1) <= radius_um + 1e-6]


def longest_true_run(values: np.ndarray, step_um: float) -> float:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * step_um)


def _sample_mask(mask: np.ndarray, xyz_local: np.ndarray) -> np.ndarray:
    # xyz_local is N x 3; mask is z, y, x.
    zyx = np.rint(xyz_local[:, ::-1]).astype(np.int32)
    valid = np.all((zyx >= 0) & (zyx < np.asarray(mask.shape)), axis=1)
    result = np.zeros(len(zyx), dtype=bool)
    result[valid] = mask[zyx[valid, 0], zyx[valid, 1], zyx[valid, 2]]
    return result


def analyze_one_strut(scan, p0_xyz: np.ndarray, p1_xyz: np.ndarray, config: dict) -> dict:
    spacing = np.asarray(config["voxel_spacing_um_zyx"], dtype=float)[::-1]  # xyz
    nominal = float(config["nominal_diameter_um"])
    roi_radius = float(config["roi_radius_um"])
    exclusion = float(config["node_exclusion_distance_um"])
    threshold = float(config["thresholds"]["ct_intensity"])
    line = p1_xyz.astype(float) - p0_xyz.astype(float)
    length_um = float(np.linalg.norm(line * spacing))
    if length_um <= 2 * exclusion:
        return {"analysis_status": "skipped_short_after_node_exclusion", "cad_length_um": length_um}
    direction = line * spacing / length_um
    u, v = orthonormal_basis(direction)
    start = p0_xyz + line * (exclusion / length_um)
    end = p1_xyz - line * (exclusion / length_um)
    usable_um = length_um - 2 * exclusion
    step_um = min(float(np.min(spacing)), 30.0)
    n = max(9, int(math.ceil(usable_um / step_um)) + 1)
    stations = np.linspace(0.0, 1.0, n)
    center_xyz = start[None, :] + stations[:, None] * (end - start)[None, :]
    # Read only the axis-aligned ROI needed by this strut.
    margin_vox = np.ceil(roi_radius / spacing).astype(int) + 2
    lo_xyz = np.floor(np.minimum(p0_xyz, p1_xyz) - margin_vox).astype(int)
    hi_xyz = np.ceil(np.maximum(p0_xyz, p1_xyz) + margin_vox + 1).astype(int)
    shape_zyx = np.asarray(scan.shape)
    lo_xyz = np.maximum(lo_xyz, 0)
    hi_xyz = np.minimum(hi_xyz, shape_zyx[::-1])
    if np.any(hi_xyz <= lo_xyz):
        return {"analysis_status": "out_of_bounds", "cad_length_um": length_um}
    # Array is bounded by the one strut ROI, typically a few hundred KiB.
    block = np.asarray(scan[lo_xyz[2]:hi_xyz[2], lo_xyz[1]:hi_xyz[1], lo_xyz[0]:hi_xyz[0]]) >= threshold
    edt_um = ndimage.distance_transform_edt(block, sampling=tuple(spacing[::-1]))
    local = (center_xyz - lo_xyz)  # xyz voxel coordinates
    offsets_um = disk_offsets(roi_radius, spacing)
    offsets_um = offsets_um - np.sum(offsets_um * direction, axis=1, keepdims=True) * direction
    # Station coordinates are voxel xyz; convert physical offsets before
    # adding them to the local voxel coordinate.
    offsets = offsets_um / spacing
    support = np.zeros(n, dtype=float)
    diam = np.full(n, np.nan, dtype=float)
    scanned = np.full((n, 3), np.nan, dtype=float)
    for i, c in enumerate(local):
        points = c[None, :] + offsets
        material = _sample_mask(block, points)
        support[i] = float(material.mean())
        if material.any():
            # A centroid in a transverse station is a stable scan-derived
            # centerline proxy and does not connect across a real gap.
            transverse = points[material]
            scanned[i] = transverse.mean(axis=0)
            q = np.rint(scanned[i][::-1]).astype(int)
            if np.all(q >= 0) and np.all(q < np.asarray(edt_um.shape)):
                diam[i] = 2.0 * float(edt_um[q[0], q[1], q[2]])
    occupied = support >= float(config["thresholds"]["occupancy_station_fraction"])
    gap_um = longest_true_run(~occupied, step_um)
    valid = np.isfinite(diam)
    if valid.any():
        profile = diam[valid]
        median_d = float(np.median(profile))
        p10_d = float(np.percentile(profile, 10))
        p90_d = float(np.percentile(profile, 90))
        thin_ratio = float(np.mean(profile < nominal * float(config["thresholds"]["thin_ratio"])))
        thick_ratio = float(np.mean(profile > nominal * float(config["thresholds"]["thick_ratio"])))
    else:
        median_d = p10_d = p90_d = thin_ratio = thick_ratio = float("nan")
    finite_center = np.isfinite(scanned).all(axis=1)
    if finite_center.sum() >= 2:
        q = scanned[finite_center]
        expected = local[finite_center]
        rel = q - (p0_xyz + (q * 0))  # overwritten below; keep calculations explicit
        axis_origin = p0_xyz - lo_xyz
        axis_end = p1_xyz - lo_xyz
        axis = axis_end - axis_origin
        axis_norm2 = max(float(np.dot(axis, axis)), 1e-8)
        projection = axis_origin + ((q - axis_origin) @ axis / axis_norm2)[:, None] * axis
        deviations = np.linalg.norm((q - projection) * spacing, axis=1)
        max_dev = float(np.max(deviations))
        path_um = float(np.sum(np.linalg.norm(np.diff(q, axis=0) * spacing, axis=1)))
        tortuosity = path_um / usable_um
    else:
        max_dev = tortuosity = float("nan")
    labels = []
    th = config["thresholds"]
    if np.isfinite(median_d) and median_d / nominal < float(th["thin_ratio"]):
        labels.append("suspected_thin")
    if np.isfinite(median_d) and median_d / nominal > float(th["thick_ratio"]):
        labels.append("suspected_thick")
    if gap_um >= float(th["broken_gap_um"]):
        labels.append("suspected_broken")
    if np.isfinite(max_dev) and (max_dev >= float(th["bent_max_deviation_um"]) or tortuosity >= float(th["bent_tortuosity"])):
        labels.append("suspected_bent")
    return {
        "analysis_status": "analyzed", "cad_length_um": length_um, "usable_length_um": usable_um,
        "occupancy_fraction": float(np.mean(occupied)), "longest_gap_um": gap_um,
        "median_diameter_um": median_d, "p10_diameter_um": p10_d, "p90_diameter_um": p90_d,
        "diameter_ratio_median": median_d / nominal if np.isfinite(median_d) else float("nan"),
        "thin_profile_fraction": thin_ratio, "thick_profile_fraction": thick_ratio,
        "max_centerline_deviation_um": max_dev, "tortuosity": tortuosity,
        "valid_station_fraction": float(np.mean(valid)), "screening_labels": ";".join(labels),
    }


def output_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "strut_morphology_metrics.csv", output_dir / "strut_morphology_metrics.json"


def run(config: dict, output_dir: Path, max_struts: int | None, overwrite: bool) -> dict:
    csv_path, json_path = output_paths(output_dir)
    if (csv_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing morphology output in {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_path, reg_path, inv_path = Path(config["scan"]), Path(config["registration"]), Path(config["inventory"])
    if not scan_path.exists() or not reg_path.exists() or not inv_path.exists():
        raise FileNotFoundError(f"Missing configured input: {scan_path}, {reg_path}, or {inv_path}")
    scan = tifffile.memmap(scan_path, mode="r")
    junctions, edges = load_graph(reg_path)
    inventory = load_inventory(inv_path)
    missing_labels = {"Missing_Intentional", "Missing_Unintentional"}
    selected = [sid for sid in sorted(inventory) if inventory[sid]["classification"] not in missing_labels]
    if max_struts is not None:
        selected = selected[:max_struts]
    rows = []
    for sid in selected:
        row = inventory[sid]
        edge = edges[sid]
        metrics = analyze_one_strut(scan, junctions[edge[0]], junctions[edge[1]], config)
        out = {"strut_id": sid, "missingness_classification": row["classification"], "junction0_id": int(edge[0]), "junction1_id": int(edge[1])}
        out.update(metrics)
        rows.append({k: _jsonable(v) for k, v in out.items()})
    fields = list(rows[0]) if rows else ["strut_id", "missingness_classification", "analysis_status"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_type": "smoke_test" if max_struts else "full_lattice",
        "selected_struts": len(selected), "non_missing_inventory_struts": len([r for r in inventory.values() if r["classification"] not in missing_labels]),
        "scan_shape_zyx": list(scan.shape), "scan_dtype": str(scan.dtype), "configuration": config,
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
        "missingness_policy": "Stage 2a classification copied unchanged; Missing_Intentional and Missing_Unintentional are not morphologically reclassified.",
    }
    json_path.write_text(json.dumps({"provenance": provenance, "metrics": rows}, indent=2, default=_jsonable) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--max-struts", type=int, default=16, help="Deterministic smoke test count; omit with --full")
    parser.add_argument("--full", action="store_true", help="Analyze every non-missing strut")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    provenance = run(config, args.output_dir.resolve(), None if args.full else args.max_struts, args.overwrite)
    print(json.dumps(provenance, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
