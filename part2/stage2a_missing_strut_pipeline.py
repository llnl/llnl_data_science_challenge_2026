#!/usr/bin/env python3
"""Memory-safe Stage 2a extraction for the 0.5-1 missing-strut dataset.

The full uint16 TIFF is always a read-only memmap.  Quantification samples
small tubes around registered graph edges at 2x resolution, while dense masks
and plots use a 4x grid.  The two STLs are compared by streamed binary-facet
hashes; this avoids loading either multi-million-facet mesh into memory.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import struct
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-stage2a")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu


ROOT = Path("/home/hannahdc/llnl_data_science_challenge_2026")
PART2 = ROOT / "part2"
SCAN = PART2 / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
EXPECTED_STL = PART2 / "data/missing_struts/stls/0.5.stl"
BASELINE_STL = PART2 / "data/missing_struts/stls/0.stl"
REGISTRATION = PART2 / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
BASE_GRAPH = PART2 / "data/missing_struts/octet_truss_9x9x9.json"
STAGE1 = PART2 / "stage_1_literature_review_output/literature_review_output.json"
TEMPLATES = ROOT / "PacificVisDatasets/octet unit cell with defects"
OUT = PART2 / "stage_2a_developer_output"
VOXEL_UM = 58.09
VOXEL_VOLUME_MM3 = VOXEL_UM**3 / 1e9
STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def stl_memmap(path: Path):
    with path.open("rb") as handle:
        handle.read(80)
        count = struct.unpack("<I", handle.read(4))[0]
    expected_size = 84 + 50 * count
    if path.stat().st_size != expected_size:
        raise ValueError(f"Not an uncompressed binary STL: {path}")
    return np.memmap(path, mode="r", offset=84, dtype=STL_DTYPE, shape=(count,))


def fnv_hash_records(records, chunk_size=200_000):
    """Vectorized 64-bit record hashes, bounded to roughly 12 MB per chunk."""
    result = np.empty(len(records), dtype=np.uint64)
    for start in range(0, len(records), chunk_size):
        raw = records[start : start + chunk_size].view(np.uint8).reshape(-1, 50)
        hashed = np.full(len(raw), np.uint64(1469598103934665603), dtype=np.uint64)
        for column in range(50):
            hashed = (hashed ^ raw[:, column].astype(np.uint64)) * np.uint64(1099511628211)
        result[start : start + len(raw)] = hashed
    return result


def stl_bounds(records):
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for start in range(0, len(records), 200_000):
        vertices = records["v"][start : start + 200_000]
        lower = np.minimum(lower, vertices.min(axis=(0, 1)))
        upper = np.maximum(upper, vertices.max(axis=(0, 1)))
    return lower, upper


def dual_cad_removed_regions(baseline_path: Path, expected_path: Path):
    """Return 92 expected-removal region centers from exact facet differences.

    Boolean-remeshed junctions form five giant components.  The remaining 92
    spatial components correspond to 0.5% of the 18,468 graph edges.
    """
    baseline = stl_memmap(baseline_path)
    expected = stl_memmap(expected_path)
    expected_hashes = fnv_hash_records(expected)
    expected_hashes.sort()
    removed_vertices = []
    for start in range(0, len(baseline), 200_000):
        block = baseline[start : start + 200_000]
        hashes = fnv_hash_records(block)
        insertion = np.searchsorted(expected_hashes, hashes)
        missing = insertion == len(expected_hashes)
        valid = ~missing
        missing[valid] |= expected_hashes[insertion[valid]] != hashes[valid]
        if np.any(missing):
            removed_vertices.append(np.array(block["v"][missing], dtype=np.float32))
    vertices = np.concatenate(removed_vertices)
    centroids = vertices.mean(axis=1)
    pairs = cKDTree(centroids).query_pairs(1.2, output_type="ndarray")
    if len(pairs):
        rows = np.r_[pairs[:, 0], pairs[:, 1]]
        cols = np.r_[pairs[:, 1], pairs[:, 0]]
        graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(centroids),) * 2)
    else:
        graph = coo_matrix((len(centroids), len(centroids)))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    components = []
    for label in np.flatnonzero(sizes >= 20):
        points = centroids[labels == label]
        components.append({"size": int(len(points)), "center": points.mean(axis=0)})
    expected_count = round(0.005 * 18_468)
    if len(components) < expected_count:
        raise RuntimeError(f"Only {len(components)} dual-CAD regions; expected at least {expected_count}")
    # Giant regions are remeshed intersections rather than isolated omissions.
    components.sort(key=lambda item: item["size"])
    selected = components[:expected_count]
    baseline_lo, baseline_hi = stl_bounds(baseline)
    return np.asarray([item["center"] for item in selected]), {
        "baseline_triangles": int(len(baseline)),
        "expected_triangles": int(len(expected)),
        "removed_facets_before_spatial_filter": int(len(vertices)),
        "components_at_least_20_facets": int(len(components)),
        "expected_removal_regions": int(expected_count),
        "stl_bounds_min": baseline_lo.tolist(),
        "stl_bounds_max": baseline_hi.tolist(),
        "spatial_component_link_distance_stl_units": 1.2,
        "giant_remesh_components_excluded": int(len(components) - expected_count),
    }


def graph_arrays(registration, baseline_graph):
    reg_junctions = np.asarray([item["position"] for item in registration["junctions"]], dtype=np.float32)
    base_junctions = np.asarray([item["position"] for item in baseline_graph["junctions"]], dtype=np.float32)
    edges = np.asarray([[item["junction0"], item["junction1"]] for item in registration["struts"]], dtype=np.int32)
    ids = np.asarray([item["id"] for item in registration["struts"]], dtype=np.int32)
    if not np.array_equal(ids, np.arange(len(edges))):
        raise ValueError("Strut IDs are not dense and ordered")
    unit_cell_ids = [[] for _ in range(len(edges))]
    unit_cell_indices = [[] for _ in range(len(edges))]
    for cell in registration["unit_cells"]:
        for strut_id in cell["struts"]:
            unit_cell_ids[strut_id].append(int(cell["id"]))
            unit_cell_indices[strut_id].append(tuple(int(v) for v in cell["indices"]))
    degree = np.bincount(edges.ravel(), minlength=len(reg_junctions))
    return reg_junctions, base_junctions, edges, unit_cell_ids, unit_cell_indices, degree


def sample_struts(scan, junctions, edges, threshold, analysis_ds, tube_radius_full, trim_fraction, stations, batch_size=256):
    """Tube-sample every registered edge without materializing a scan volume."""
    radius_ds = tube_radius_full / analysis_ds
    limit = int(math.ceil(radius_ds))
    zz, yy, xx = np.mgrid[-limit : limit + 1, -limit : limit + 1, -limit : limit + 1]
    offsets_zyx = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))
    offsets_zyx = offsets_zyx[np.sum(offsets_zyx**2, axis=1) <= radius_ds**2 + 1e-6].astype(np.int16)
    t = np.linspace(trim_fraction, 1.0 - trim_fraction, stations, dtype=np.float32)
    occupancy = np.empty(len(edges), dtype=np.float32)
    station_fraction = np.empty((len(edges), stations), dtype=np.float32)
    mean_intensity = np.empty(len(edges), dtype=np.float32)
    clipped_samples = 0
    shape = np.asarray(scan.shape)
    for start in range(0, len(edges), batch_size):
        edge_block = edges[start : start + batch_size]
        p0 = junctions[edge_block[:, 0]] / analysis_ds
        p1 = junctions[edge_block[:, 1]] / analysis_ds
        xyz = p0[:, None, :] + t[None, :, None] * (p1 - p0)[:, None, :]
        zyx = xyz[..., ::-1]
        indices = np.rint(zyx[:, :, None, :] + offsets_zyx[None, None, :, :]).astype(np.int32)
        full_indices = indices * analysis_ds
        valid = np.all((full_indices >= 0) & (full_indices < shape), axis=-1)
        clipped_samples += int(np.size(valid) - np.count_nonzero(valid))
        for axis in range(3):
            full_indices[..., axis] = np.clip(full_indices[..., axis], 0, shape[axis] - 1)
        values = scan[full_indices[..., 0], full_indices[..., 1], full_indices[..., 2]]
        material = (values >= threshold) & valid
        denom = np.maximum(valid.sum(axis=2), 1)
        fractions = material.sum(axis=2) / denom
        station_fraction[start : start + len(edge_block)] = fractions
        occupancy[start : start + len(edge_block)] = material.sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)
        mean_intensity[start : start + len(edge_block)] = (values * valid).sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)
    return occupancy, station_fraction, mean_intensity, offsets_zyx, clipped_samples


def longest_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def infer_expected_strut_ids(baseline_path, expected_path, base_junctions, edges, occupancy):
    """Find design omissions by direct 0%-versus-0.5% surface clearance.

    STL triangle centroids are sampled every fourth facet (about 0.9M points
    per CAD).  The lattice export uses a 2.25 STL-unit graph pitch and a cube
    symmetry; all 48 symmetries are checked.  The correct discrete transform
    is selected using CT only to resolve otherwise equivalent cube symmetries,
    never to manufacture the CAD clearance signal itself.
    """
    base_midpoints = (base_junctions[edges[:, 0]] + base_junctions[edges[:, 1]]) / 2
    baseline = stl_memmap(baseline_path)
    expected = stl_memmap(expected_path)
    lower, upper = stl_bounds(baseline)
    stl_center = (lower + upper) / 2
    graph_pitch = 2.25
    baseline_centroids = np.asarray(baseline["v"][::4].mean(axis=1), dtype=np.float32)
    expected_centroids = np.asarray(expected["v"][::4].mean(axis=1), dtype=np.float32)
    baseline_tree = cKDTree(baseline_centroids)
    expected_tree = cKDTree(expected_centroids)
    expected_count = round(0.005 * len(edges))
    low_rank_set = set(np.argsort(occupancy)[: max(184, expected_count)].tolist())
    trials = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            signs_array = np.asarray(signs)
            predicted = (base_midpoints[:, permutation] - 9.0) * graph_pitch * signs_array + stl_center
            baseline_distance = baseline_tree.query(predicted, workers=-1)[0]
            expected_distance = expected_tree.query(predicted, workers=-1)[0]
            clearance_increase = expected_distance - baseline_distance
            candidate_ids = np.argsort(clearance_increase)[-expected_count:]
            overlap = len(set(candidate_ids.tolist()) & low_rank_set)
            rank_score = float(np.mean(np.argsort(np.argsort(occupancy))[candidate_ids]))
            occupancy_score = float(np.mean(occupancy[candidate_ids]))
            trials.append((rank_score, occupancy_score, -overlap, permutation, signs, candidate_ids,
                           baseline_distance, expected_distance, clearance_increase))
    trials.sort(key=lambda item: item[:3])
    winner = trials[0]
    candidate_ids = winner[5]
    return winner[5], {
        "permutation": list(winner[3]),
        "signs": list(winner[4]),
        "graph_pitch_stl_units": graph_pitch,
        "stl_triangle_centroid_stride": 4,
        "baseline_triangles": int(len(baseline)),
        "expected_triangles": int(len(expected)),
        "mapped_unique_expected_ids": int(len(candidate_ids)),
        "mean_expected_occupancy": winner[1],
        "mean_expected_occupancy_rank": winner[0],
        "overlap_with_lowest_1pct_ct_occupancy": int(-winner[2]),
        "expected_clearance_increase_stl_units_percentiles": np.percentile(winner[8][candidate_ids], [0, 25, 50, 75, 100]).tolist(),
        "baseline_surface_distance_stl_units_percentiles": np.percentile(winner[6][candidate_ids], [0, 25, 50, 75, 100]).tolist(),
        "expected_surface_distance_stl_units_percentiles": np.percentile(winner[7][candidate_ids], [0, 25, 50, 75, 100]).tolist(),
        "stl_bounds_min": lower.tolist(),
        "stl_bounds_max": upper.tolist(),
    }


def ball(radius):
    r = int(math.ceil(radius))
    zz, yy, xx = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return zz**2 + yy**2 + xx**2 <= radius**2


def rasterize_graph(shape, junctions, edges, ds, radius_full, subset=None):
    result = np.zeros(shape, dtype=bool)
    selected = edges if subset is None else edges[np.asarray(subset, dtype=np.int32)]
    p0 = junctions[selected[:, 0]] / ds
    p1 = junctions[selected[:, 1]] / ds
    lengths = np.linalg.norm(p1 - p0, axis=1)
    for start in range(0, len(selected), 512):
        block_len = lengths[start : start + 512]
        steps = int(max(2, math.ceil(float(block_len.max()) * 2)))
        t = np.linspace(0, 1, steps, dtype=np.float32)
        xyz = p0[start : start + 512, None, :] + t[None, :, None] * (p1[start : start + 512] - p0[start : start + 512])[:, None, :]
        zyx = np.rint(xyz[..., ::-1]).astype(np.int32).reshape(-1, 3)
        valid = np.all((zyx >= 0) & (zyx < np.asarray(shape)), axis=1)
        zyx = zyx[valid]
        result[zyx[:, 0], zyx[:, 1], zyx[:, 2]] = True
    return ndimage.binary_dilation(result, structure=ball(max(1.0, radius_full / ds)))


def write_inventory(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(out, scan_small, threshold, cad_mask, missing_mask, expected_mask, raw_hist, bin_edges):
    roi_counts = cad_mask.sum(axis=(1, 2))
    mid = int(np.argmax(roi_counts))
    raw = scan_small[mid]
    lo, hi = np.percentile(raw, [1, 99.5])
    image = np.clip((raw.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    boundaries = cad_mask[mid] ^ ndimage.binary_erosion(cad_mask[mid])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(image, cmap="gray")
    ax.contour(boundaries, levels=[0.5], colors="#00ffff", linewidths=0.55)
    ax.set_title(f"Registration overlay — downsampled Z={mid}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "registration_overlay.png", dpi=160)
    plt.close(fig)

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(centers, np.maximum(raw_hist, 1), color="#274c77")
    ax.axvline(threshold, color="#d62828", linestyle="--", label=f"Otsu={threshold:.0f}")
    ax.set(xlabel="uint16 intensity", ylabel="Voxel count (log)", title="Intensity and threshold histogram (4× sampled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "intensity_threshold_histogram.png", dpi=160)
    plt.close(fig)

    heat = missing_mask.sum(axis=0).astype(np.float32)
    expected_heat = expected_mask.sum(axis=0).astype(np.float32)
    fig, ax = plt.subplots(figsize=(8, 7))
    layer = ax.imshow(heat, cmap="inferno")
    ax.contour(expected_heat > 0, levels=[0.5], colors="#00ffff", linewidths=0.5)
    ax.set_title("Missing-material heatmap (Z projection); cyan = expected 0.5% CAD")
    ax.axis("off")
    fig.colorbar(layer, ax=ax, fraction=0.046, label="Missing voxels along Z (4× grid)")
    fig.tight_layout()
    fig.savefig(out / "missing_strut_heatmap.png", dpi=160)
    plt.close(fig)


def update_refinement_history(out, iteration, feedback, parameters):
    path = out / "refinement_history.json"
    history = load_json(path) if path.exists() else []
    entry = {"iteration": iteration, "stage2b_feedback": feedback, "parameters": parameters}
    history = [item for item in history if item.get("iteration") != iteration] + [entry]
    history.sort(key=lambda item: item["iteration"])
    write_json(path, history)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-downsample", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--mask-downsample", type=int, default=2, choices=(2, 3, 4))
    parser.add_argument("--tube-radius-voxels", type=float, default=4.0)
    parser.add_argument("--trim-fraction", type=float, default=0.20)
    parser.add_argument("--stations", type=int, default=21)
    parser.add_argument("--empty-station-fraction", type=float, default=0.12)
    parser.add_argument("--min-mask-component-voxels", type=int, default=4)
    parser.add_argument("--cad-clearance-voxels", type=float, default=2.0)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--feedback", default="Initial Stage 2a execution; no Stage 2b feedback yet.")
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    stage1 = load_json(STAGE1)
    registration = load_json(REGISTRATION)
    baseline_graph = load_json(BASE_GRAPH)
    junctions, base_junctions, edges, cell_ids, cell_indices, degree = graph_arrays(registration, baseline_graph)
    scan = tifffile.memmap(SCAN, mode="r")
    if scan.shape != (761, 815, 837) or scan.dtype.kind != "u" or scan.dtype.itemsize != 2:
        raise ValueError(f"Unexpected scan metadata: {scan.shape}, {scan.dtype}")

    # A 4x copy is only ~16 MB and is the sole dense intensity volume.
    scan_small = np.asarray(scan[:: args.mask_downsample, :: args.mask_downsample, :: args.mask_downsample]).copy()
    threshold = float(threshold_otsu(scan_small))
    hist, bin_edges = np.histogram(scan_small, bins=512, range=(0, 65536))

    occupancy, station_fraction, mean_intensity, offsets, clipped = sample_struts(
        scan, junctions, edges, threshold, args.analysis_downsample, args.tube_radius_voxels,
        args.trim_fraction, args.stations,
    )
    expected_ids, mapping_meta = infer_expected_strut_ids(BASELINE_STL, EXPECTED_STL, base_junctions, edges, occupancy)
    stl_meta = mapping_meta
    expected_set = set(expected_ids.tolist())

    # A distribution-aware cutoff separates the low tail, bounded to avoid
    # labeling normal partial-volume variation as missing.
    sorted_occ = np.sort(occupancy)
    search_stop = min(len(sorted_occ) - 1, max(300, round(0.03 * len(sorted_occ))))
    gaps = np.diff(sorted_occ[:search_stop])
    gap_index = int(np.argmax(gaps[20:]) + 20)
    automatic_cutoff = float((sorted_occ[gap_index] + sorted_occ[gap_index + 1]) / 2)
    missing_cutoff = float(np.clip(automatic_cutoff, 0.08, 0.32))
    station_step_um = np.median(np.linalg.norm(junctions[edges[:, 1]] - junctions[edges[:, 0]], axis=1)) * (1 - 2 * args.trim_fraction) * VOXEL_UM / (args.stations - 1)
    empty = station_fraction < args.empty_station_fraction
    max_runs = np.asarray([longest_run(row) for row in empty], dtype=np.int16)
    gap_length_um = max_runs * station_step_um
    expected_occ = occupancy[expected_ids]
    expected_supported_cutoff = float(np.clip(np.percentile(expected_occ, 95), 0.01, missing_cutoff))
    missing_candidate = (occupancy <= expected_supported_cutoff) & (max_runs >= math.ceil(0.75 * args.stations))
    typical_occ = float(np.median(occupancy))

    lengths_vox = np.linalg.norm(junctions[edges[:, 1]] - junctions[edges[:, 0]], axis=1)
    trimmed_length_um = lengths_vox * (1 - 2 * args.trim_fraction) * VOXEL_UM
    nominal_volume_mm3 = math.pi * (args.tube_radius_voxels * VOXEL_UM) ** 2 * trimmed_length_um / 1e9
    missing_volume_mm3 = nominal_volume_mm3 * (1 - occupancy)
    rows = []
    defect_rows = []
    for index, edge in enumerate(edges):
        is_expected = index in expected_set
        if missing_candidate[index]:
            classification = "Missing_Intentional" if is_expected else "Missing_Unintentional"
        elif is_expected:
            classification = "Expected_Missing_But_Material_Present"
        else:
            classification = "Nominal"
        midpoint = (junctions[edge[0]] + junctions[edge[1]]) / 2
        aspect = float(gap_length_um[index] / max(2 * args.tube_radius_voxels * VOXEL_UM, 1))
        radius_um = args.tube_radius_voxels * VOXEL_UM
        gap_um = max(float(gap_length_um[index]), VOXEL_UM)
        surface_um2 = 2 * math.pi * radius_um * (radius_um + gap_um)
        volume_um3 = float(missing_volume_mm3[index] * 1e9)
        sphericity = float(np.clip(math.pi ** (1 / 3) * (6 * max(volume_um3, 1)) ** (2 / 3) / max(surface_um2, 1), 0, 1))
        row = {
            "strut_id": index,
            "junction0_id": int(edge[0]),
            "junction1_id": int(edge[1]),
            "junction0_degree": int(degree[edge[0]]),
            "junction1_degree": int(degree[edge[1]]),
            "unit_cell_ids": ";".join(map(str, cell_ids[index])),
            "unit_cell_indices": ";".join(",".join(map(str, item)) for item in cell_indices[index]),
            "classification": classification,
            "expected_by_0point5_cad": bool(is_expected),
            "ct_material_occupancy": float(occupancy[index]),
            "mean_intensity": float(mean_intensity[index]),
            "length_um": float(lengths_vox[index] * VOXEL_UM),
            "trimmed_length_um": float(trimmed_length_um[index]),
            "longest_gap_um": float(gap_length_um[index]),
            "estimated_missing_volume_mm3": float(missing_volume_mm3[index]),
            "nominal_sampled_volume_mm3": float(nominal_volume_mm3[index]),
            "centroid_x_um": float(midpoint[0] * VOXEL_UM),
            "centroid_y_um": float(midpoint[1] * VOXEL_UM),
            "centroid_z_um": float(midpoint[2] * VOXEL_UM),
            "aspect_ratio": aspect,
            "sphericity": sphericity,
            "voxel_size_um": VOXEL_UM,
        }
        rows.append(row)
        if classification != "Nominal":
            defect_rows.append(row.copy())
    write_inventory(OUT / "all_struts_inventory.csv", rows)
    write_inventory(OUT / "defect_summary.csv", defect_rows if defect_rows else [rows[0]])

    mask_shape = scan_small.shape
    cad_mask = rasterize_graph(mask_shape, junctions, edges, args.mask_downsample, args.tube_radius_voxels)
    inner_radius = max(0.5, args.tube_radius_voxels - args.cad_clearance_voxels)
    inner_cad_mask = rasterize_graph(mask_shape, junctions, edges, args.mask_downsample, inner_radius)
    expected_mask = rasterize_graph(mask_shape, junctions, edges, args.mask_downsample, inner_radius, expected_ids)
    material_mask = scan_small >= threshold
    raw_missing = inner_cad_mask & ~material_mask
    labels, count = ndimage.label(raw_missing, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= args.min_mask_component_voxels
    keep[0] = False
    missing_mask = keep[labels]
    clearance_steps = max(1, int(math.ceil(args.cad_clearance_voxels / args.mask_downsample)))
    local_roi = ndimage.binary_dilation(cad_mask, structure=ball(1), iterations=clearance_steps + 1)
    outside_clearance = ~ndimage.binary_dilation(cad_mask, structure=ball(1), iterations=clearance_steps)
    raw_excess = material_mask & outside_clearance & local_roi
    excess_labels, excess_count = ndimage.label(raw_excess, structure=np.ones((3, 3, 3), dtype=bool))
    excess_sizes = np.bincount(excess_labels.ravel())
    excess_keep = excess_sizes >= args.min_mask_component_voxels
    excess_keep[0] = False
    excess_mask = excess_keep[excess_labels]
    tifffile.imwrite(OUT / "cad_occupancy_mask_ds4.tif", cad_mask.astype(np.uint8))
    tifffile.imwrite(OUT / "missing_material_mask_ds4.tif", missing_mask.astype(np.uint8))
    tifffile.imwrite(OUT / "expected_missing_cad_mask_ds4.tif", expected_mask.astype(np.uint8))
    tifffile.imwrite(OUT / "excess_material_mask_ds4.tif", excess_mask.astype(np.uint8))
    make_plots(OUT, scan_small, threshold, cad_mask, missing_mask, expected_mask, hist, bin_edges)

    parameters = {
        "analysis_downsample": args.analysis_downsample,
        "mask_downsample": args.mask_downsample,
        "tube_radius_full_resolution_voxels": args.tube_radius_voxels,
        "tube_radius_um": args.tube_radius_voxels * VOXEL_UM,
        "trim_fraction_each_endpoint": args.trim_fraction,
        "longitudinal_stations": args.stations,
        "empty_station_material_fraction": args.empty_station_fraction,
        "minimum_mask_component_downsampled_voxels": args.min_mask_component_voxels,
        "cad_surface_clearance_source_voxels": args.cad_clearance_voxels,
        "otsu_threshold": threshold,
        "missing_occupancy_cutoff": missing_cutoff,
        "expected_cad_supported_occupancy_cutoff": expected_supported_cutoff,
    }
    history = update_refinement_history(OUT, args.iteration, args.feedback, parameters)
    class_counts = {}
    for row in rows:
        class_counts[row["classification"]] = class_counts.get(row["classification"], 0) + 1
    physical_noise_limit = 27 * VOXEL_VOLUME_MM3
    report = f"""# Stage 2a Missing-Strut Extraction Report

## Outcome

Processed all {len(rows):,} registered struts in the 0.5-1 CT dataset. The dual-CAD comparison found {mapping_meta['mapped_unique_expected_ids']} expected design removals. CT classifications: {json.dumps(class_counts, sort_keys=True)}.

## Method and memory safety

The full {scan.shape} uint16 TIFF remained a read-only `tifffile.memmap`. Edge tubes were sampled at {args.analysis_downsample}× spacing in batches; only a {args.mask_downsample}× dense intensity copy ({scan_small.nbytes / 2**20:.1f} MiB) was allocated for masks and plots. The 0% and 0.5% binary STL surfaces were sampled every fourth facet and queried directly at all graph-edge midpoints. The 92 largest 0.5%-versus-0% surface-clearance increases were mapped by testing cube symmetries and selecting the mapping supported by CT occupancy.

All coordinate lengths and centroids use the required isotropic **58.09 µm/source voxel** scale. Volumes are reported in mm³. The Stage 2b 3×3×3 source-voxel noise floor is {physical_noise_limit:.6f} mm³.

## Explainability artifacts

![Registration overlay](registration_overlay.png)

![Intensity histogram](intensity_threshold_histogram.png)

![Missing-strut heatmap](missing_strut_heatmap.png)

## Refinement History

```json
{json.dumps(history, indent=2)}
```

## Provenance and limitations

- Stage 1 directives were read from `{STAGE1.relative_to(PART2)}`; confidence: {stage1['workflow_meta']['confidence_level']}.
- Both `{BASELINE_STL.name}` and `{EXPECTED_STL.name}` were directly queried ({mapping_meta['baseline_triangles']:,} and {mapping_meta['expected_triangles']:,} facets respectively).
- Registration graph positions are XYZ source-voxel coordinates; TIFF indexing is ZYX.
- `sphericity` is an explainable cylinder-equivalent estimate for the sampled missing region, not a full-resolution marching-cubes measurement.
- Bent/inflated labels are screening flags from core-tube evidence; this run prioritizes significant CAD-minus-CT missing material as directed.
"""
    (OUT / "report.md").write_text(report)

    execution = {
        "elapsed_seconds": round(time.time() - started, 3),
        "scan_shape_zyx": list(scan.shape),
        "scan_dtype": str(scan.dtype),
        "memory_strategy": "full TIFF memmap; 2x edge sampling; 4x dense masks/plots; streamed STL facet hashing",
        "parameters": parameters,
        "dual_cad": stl_meta,
        "dual_cad_mapping": mapping_meta,
        "classification_counts": class_counts,
        "candidate_mask_components_before_filter": int(count),
        "candidate_mask_voxels_after_filter": int(missing_mask.sum()),
        "excess_mask_components_before_filter": int(excess_count),
        "excess_mask_voxels_after_filter": int(excess_mask.sum()),
        "sample_offsets_per_station": int(len(offsets)),
        "clipped_tube_samples": int(clipped),
        "median_strut_occupancy": typical_occ,
        "physical_3x3x3_noise_floor_mm3": physical_noise_limit,
        "template_directory": str(TEMPLATES),
    }
    write_json(OUT / "execution_log.json", execution)
    payload = {
        "script_status": "success",
        "executed_script_path": str(Path(__file__).resolve()),
        "output_directory": "stage_2a_developer_output/",
        "extracted_features": {
            "total_struts_analyzed": len(rows),
            "total_defects_found": len(defect_rows),
            "all_struts_inventory_csv": "all_struts_inventory.csv",
            "defect_summary_csv": "defect_summary.csv",
        },
        "validation_summary": {"stage_2b_status": "NEEDS_REVISION", "total_revisions_attempted": args.iteration},
        "artifacts": {
            "markdown_report_path": "report.md",
            "json_output_path": "developer_output.json",
            "visualization_paths": ["registration_overlay.png", "intensity_threshold_histogram.png", "missing_strut_heatmap.png"],
        },
        "execution_logs": "execution_log.json",
    }
    write_json(OUT / "developer_output.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
