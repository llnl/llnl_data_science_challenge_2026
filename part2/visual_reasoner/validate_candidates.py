"""Stage 2b visual validation for anomaly candidates.

The validator intentionally reads only the small TIFF crop needed for each
candidate. It does not materialize the full scan in memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv
import tifffile


ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_PATH = ROOT / "part2" / "interagent_feedback" / "anomaly_candidates.json"
SUMMARY_PATH = ROOT / "part2" / "developer" / "output" / "anomaly_summary.json"
OUTPUT_ROOT = ROOT / "part2" / "visual_reasoner" / "outputs"
RESULT_PATH = ROOT / "part2" / "interagent_feedback" / "validated_candidates_v2.json"
MAX_CANDIDATES = 20
EDGE_MARGIN_VOXELS = 2


def _resolve_scan(summary: dict) -> Path:
    scan = Path(summary["input"])
    if not scan.is_absolute():
        scan = ROOT / scan
    return scan


def _crop_bounds(center_xyz: np.ndarray, size_xyz: np.ndarray, shape_zyx: tuple[int, int, int]):
    shape_xyz = np.array([shape_zyx[2], shape_zyx[1], shape_zyx[0]], dtype=int)
    center = np.rint(center_xyz).astype(int)
    size = np.maximum(np.rint(size_xyz).astype(int), 1)
    start = center - size // 2
    end = start + size
    clipped_start = np.maximum(start, 0)
    clipped_end = np.minimum(end, shape_xyz)
    clipped = bool(np.any(start < 0) or np.any(end > shape_xyz))
    near_edge = bool(np.any(center <= EDGE_MARGIN_VOXELS) or
                     np.any(center >= shape_xyz - 1 - EDGE_MARGIN_VOXELS))
    return clipped_start, clipped_end, clipped, near_edge


def _read_crop(tif: tifffile.TiffFile, start_xyz: np.ndarray, end_xyz: np.ndarray) -> np.ndarray:
    """Read [z, y, x] crop by loading only the required TIFF pages."""
    x0, y0, z0 = map(int, start_xyz)
    x1, y1, z1 = map(int, end_xyz)
    pages = []
    for z in range(z0, z1):
        page = tif.pages[z].asarray()
        pages.append(page[y0:y1, x0:x1])
    if not pages:
        return np.empty((0, max(0, y1 - y0), max(0, x1 - x0)), dtype=np.uint16)
    return np.stack(pages, axis=0)


def _make_grid(crop: np.ndarray, origin_xyz: np.ndarray, threshold: float):
    """Convert a zyx crop into a PyVista grid and extract a threshold surface."""
    if crop.size == 0 or crop.max() <= threshold or crop.min() >= threshold:
        return None
    grid = pv.ImageData(dimensions=(crop.shape[2], crop.shape[1], crop.shape[0]))
    grid.origin = tuple(float(v) for v in origin_xyz)
    grid.point_data["intensity"] = crop.astype(np.float32).ravel(order="F")
    return grid.contour([threshold], scalars="intensity")


def _render_views(candidate_id: str, surface, candidate_xyz: np.ndarray, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    center = np.asarray(candidate_xyz, dtype=float)
    bounds = surface.bounds if surface is not None else (
        center[0] - 1, center[0] + 1, center[1] - 1, center[1] + 1,
        center[2] - 1, center[2] + 1,
    )
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
    cameras = {
        "isometric": (center + np.array([1.7, 1.7, 1.4]) * span, center, (0, 0, 1)),
        "front": (center + np.array([0.0, 0.0, 2.4]) * span, center, (0, 1, 0)),
        "side": (center + np.array([2.4, 0.0, 0.0]) * span, center, (0, 0, 1)),
    }
    paths = []
    for name, camera in cameras.items():
        plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
        plotter.set_background("#111827")
        if surface is not None:
            plotter.add_mesh(surface, color="#9ecae1", opacity=0.78, smooth_shading=True)
        plotter.add_mesh(pv.Sphere(radius=max(span * 0.025, 0.5), center=center), color="tomato")
        plotter.add_axes(line_width=2)
        plotter.camera_position = camera
        path = output_dir / f"{name}.png"
        plotter.show(screenshot=str(path), auto_close=True)
        paths.append(str(path.relative_to(ROOT)))
    return paths


def validate():
    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))[:MAX_CANDIDATES]
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    scan_path = _resolve_scan(summary)
    shape_zyx = tuple(int(v) for v in summary["shape_zyx"])
    threshold = float(summary["threshold"])
    results = []

    with tifffile.TiffFile(scan_path) as tif:
        for candidate in candidates:
            candidate_id = str(candidate["ID"])
            center = np.asarray(candidate["location"]["voxel_xyz"], dtype=float)
            requested_size = np.asarray(candidate["crop_size"]["voxels_xyz"], dtype=int)
            start, end, clipped, near_edge = _crop_bounds(center, requested_size, shape_zyx)
            crop = _read_crop(tif, start, end)
            surface = _make_grid(crop, start, threshold)
            output_dir = OUTPUT_ROOT / candidate_id
            views = _render_views(candidate_id, surface, center, output_dir)
            material_fraction = float(np.mean(crop >= threshold)) if crop.size else 0.0
            boundary_issue = clipped or near_edge
            assessment = "inconclusive_boundary" if boundary_issue else (
                "confirmed_missing_or_broken_strut" if material_fraction < 0.35 else "manual_review_required"
            )
            result = dict(candidate)
            result["status"] = "validated_stage_2b"
            result["visual_validation"] = {
                "validator": "validation-agent",
                "method": "local TIFF crop rendered with PyVista off-screen",
                "threshold": threshold,
                "scan_shape_zyx": list(shape_zyx),
                "requested_crop_start_xyz": [int(v) for v in (np.rint(center).astype(int) - requested_size // 2)],
                "actual_crop_start_xyz": [int(v) for v in start],
                "actual_crop_end_xyz": [int(v) for v in end],
                "crop_was_clipped": clipped,
                "near_scan_boundary": near_edge,
                "material_fraction_in_crop": round(material_fraction, 6),
                "assessment": assessment,
                "views": views,
            }
            results.append(result)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    output = validate()
    print(json.dumps({"validated_count": len(output), "result": str(RESULT_PATH)}))
