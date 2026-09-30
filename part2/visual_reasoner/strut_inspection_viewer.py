"""Memory-bounded PyVista viewer for one Stage 2a strut.

The default ``focused`` mode keeps the expected endpoint-to-endpoint strut
yellow, shows CT material inside a narrow cylindrical corridor, and leaves
nearby context semi-transparent. TIFF pages are Z slices, so only pages
intersecting the selected local crop are read.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import tifffile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "part2/stage_2a_developer_output/all_struts_inventory.csv"
DEFAULT_GRAPH = ROOT / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
DEFAULT_SCAN = ROOT / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
DEFAULT_OUTPUT = ROOT / "part2/visual_reasoner/inspection_outputs"
DEFAULT_THRESHOLD = 40081.0
FOCUSED_MODE = "focused"
CONTEXT_MODE = "context"

CLASS_COLORS = {
    "Nominal": "#38bdf8", "Missing_Intentional": "#f43f5e",
    "Missing_Unintentional": "#fb923c", "Expected_Missing_But_Material_Present": "#facc15",
}


@dataclass(frozen=True)
class StrutRecord:
    strut_id: int
    junction0_id: int
    junction1_id: int
    classification: str
    inventory: dict[str, str]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _parse_strut_id(value: str) -> int:
    value = str(value).strip()
    if value.lower().startswith("strut-"):
        value = value[6:]
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid strut_id {value!r}; use an integer or strut-XXXXX") from exc


def load_strut(inventory_path: Path, strut_id: str) -> StrutRecord:
    wanted = _parse_strut_id(strut_id)
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["strut_id"]) == wanted:
                return StrutRecord(wanted, int(row["junction0_id"]), int(row["junction1_id"]), row["classification"], row)
    raise KeyError(f"strut_id {wanted} was not found in {inventory_path}")


def load_junctions(graph_path: Path) -> dict[int, np.ndarray]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    junctions = {}
    for junction in graph.get("junctions", []):
        position = np.asarray(junction["position"], dtype=np.float32)
        if position.shape != (3,):
            raise ValueError(f"Junction {junction.get('id')} does not have an XYZ position")
        junctions[int(junction["id"])] = position
    return junctions


def load_mapped_struts(graph_path: Path) -> list[tuple[int, int, int]]:
    """Return mapped strut ID and endpoint-junction IDs from the registered graph."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    return [
        (int(strut["id"]), int(strut["junction0"]), int(strut["junction1"]))
        for strut in graph.get("struts", [])
    ]


def crop_bounds(points_xyz: np.ndarray, padding: int, shape_zyx: tuple[int, int, int]):
    shape_xyz = np.asarray((shape_zyx[2], shape_zyx[1], shape_zyx[0]), dtype=int)
    start = np.floor(points_xyz.min(axis=0) - padding).astype(int)
    end = np.ceil(points_xyz.max(axis=0) + padding + 1).astype(int)
    return np.maximum(start, 0), np.minimum(end, shape_xyz)


def read_crop(tif: tifffile.TiffFile, start_xyz: np.ndarray, end_xyz: np.ndarray) -> np.ndarray:
    """Read only required TIFF pages and XY windows; return a ZYX array."""
    x0, y0, z0 = (int(v) for v in start_xyz)
    x1, y1, z1 = (int(v) for v in end_xyz)
    pages = [tif.pages[z].asarray()[y0:y1, x0:x1] for z in range(z0, z1)]
    if not pages:
        return np.empty((0, max(y1 - y0, 0), max(x1 - x0, 0)), dtype=np.uint16)
    return np.stack(pages, axis=0)


def make_material_surface(crop_zyx: np.ndarray, origin_xyz: np.ndarray, threshold: float,
                          material_mask: np.ndarray | None = None):
    """Contour a material mask, retaining the crop's source XYZ coordinates."""
    if material_mask is None:
        material_mask = crop_zyx >= threshold
    if crop_zyx.size == 0 or not np.any(material_mask):
        return None
    grid = pv.ImageData(dimensions=(crop_zyx.shape[2], crop_zyx.shape[1], crop_zyx.shape[0]))
    grid.origin = tuple(float(v) for v in origin_xyz)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.point_data["intensity"] = material_mask.astype(np.float32, copy=False).ravel(order="F")
    return grid.contour([float(threshold)], scalars="intensity")


def focus_material(crop_zyx: np.ndarray, origin_xyz: np.ndarray, a: np.ndarray, b: np.ndarray,
                   threshold: float, radius: float):
    """Split material into near-line focused colors and semi-transparent context."""
    material = crop_zyx >= threshold
    zyx = np.indices(crop_zyx.shape, dtype=np.float32)
    points = np.stack((zyx[2] + origin_xyz[0], zyx[1] + origin_xyz[1], zyx[0] + origin_xyz[2]), axis=-1)
    direction = b - a
    length_sq = float(np.dot(direction, direction))
    t = np.clip(np.sum((points - a) * direction, axis=-1) / max(length_sq, 1e-6), 0.0, 1.0)
    closest = a + t[..., None] * direction
    distance = np.linalg.norm(points - closest, axis=-1)
    focused = material & (distance <= float(radius))
    bins = max(int(np.ceil(np.sqrt(length_sq))), 1)
    occupied = np.array([
        np.any(focused & (t >= i / bins) & (t <= (i + 1) / bins if i == bins - 1
                                             else t < (i + 1) / bins))
        for i in range(bins)
    ], dtype=bool)
    runs = []
    i = 0
    while i < bins:
        if not occupied[i]:
            i += 1
            continue
        start = i
        while i < bins and occupied[i]:
            i += 1
        runs.append((start, i))
    dominant = max(runs, key=lambda run: run[1] - run[0], default=(0, 0))
    green = focused & (t >= dominant[0] / bins) & (
        t <= dominant[1] / bins if dominant[1] == bins else t < dominant[1] / bins
    )
    red = focused & ~green
    gaps = int(sum(1 for left, right in zip(runs, runs[1:]) if right[0] > left[1]))
    return material & ~focused, green, red, gaps


def clipping_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Choose a stable plane normal perpendicular to the selected strut."""
    direction = b - a
    trial = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction / max(np.linalg.norm(direction), 1e-6), trial))) > 0.9:
        trial = np.array([1.0, 0.0, 0.0])
    normal = np.cross(direction, trial)
    return normal / max(np.linalg.norm(normal), 1e-6)


def _tube(a: np.ndarray, b: np.ndarray, radius: float):
    return pv.Line(a, b, resolution=1).tube(radius=float(radius), n_sides=16, capping=True)


def _camera_positions(center: np.ndarray, span: float, zoom_out: float = 1.0):
    span *= float(zoom_out)
    return {
        "isometric": (center + np.array([1.65, 1.65, 1.35]) * span, center, (0, 0, 1)),
        "front": (center + np.array([0.0, 0.0, 2.5]) * span, center, (0, 1, 0)),
        "side": (center + np.array([2.5, 0.0, 0.0]) * span, center, (0, 0, 1)),
    }


def render_views(context_surface, green_surface, red_surface, a: np.ndarray, b: np.ndarray,
                 strut_id: int, classification: str, output_dir: Path, tube_radius: float,
                 window_size: tuple[int, int], gaps: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = _tube(a, b, tube_radius)
    center = (a + b) / 2.0
    span = max(float(np.linalg.norm(b - a)), 1.0)
    for surface in (context_surface, green_surface, red_surface):
        if surface is not None:
            bounds = surface.bounds
            span = max(span, bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    plane_normal = clipping_normal(a, b)
    plane_size = span * 1.15
    paths = {}
    for name, camera in _camera_positions(center, span).items():
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("#0f172a")
        if context_surface is not None:
            plotter.add_mesh(context_surface, color="#cbd5e1", opacity=0.20, smooth_shading=True)
        if green_surface is not None:
            plotter.add_mesh(green_surface, color="#22c55e", opacity=0.95, smooth_shading=True)
        if red_surface is not None:
            plotter.add_mesh(red_surface, color="#ef4444", opacity=0.95, smooth_shading=True)
        plotter.add_mesh(expected, color="#facc15", smooth_shading=True)
        clip_plane = pv.Plane(center=center, direction=plane_normal, i_size=plane_size, j_size=plane_size)
        plotter.add_mesh(clip_plane, color="#67e8f9", opacity=0.10, style="wireframe", line_width=1)
        plotter.add_mesh(pv.Sphere(radius=max(tube_radius * 1.25, 1.0), center=a), color="#f8fafc")
        plotter.add_mesh(pv.Sphere(radius=max(tube_radius * 1.25, 1.0), center=b), color="#f8fafc")
        status = f"gap bins: {gaps}" if gaps else "continuous"
        plotter.add_text(f"Strut {strut_id:05d} | focused | {status}", font_size=10, color="white")
        plotter.add_axes(line_width=2)
        plotter.camera_position = camera
        path = output_dir / f"{name}.png"
        plotter.show(screenshot=str(path), auto_close=True)
        paths[name] = str(path)
    return paths



def _combine_polydata(meshes) -> pv.PolyData:
    """Combine optional surface meshes into one ParaView-friendly PolyData."""
    valid = [mesh for mesh in meshes if mesh is not None and mesh.n_cells]
    if not valid:
        return pv.PolyData()
    return pv.MultiBlock(valid).combine(merge_points=False)


def export_interactive_geometry(output_dir: Path, material_surface, selected_strut,
                                neighboring_tubes) -> dict[str, str]:
    """Write the inspection layers as standalone VTP files for ParaView."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meshes = {
        "ct_material": material_surface if material_surface is not None else pv.PolyData(),
        "selected_strut": selected_strut,
        "neighboring_struts": _combine_polydata(neighboring_tubes),
    }
    paths = {}
    for name, mesh in meshes.items():
        path = output_dir / f"{name}.vtp"
        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.extract_surface(algorithm="dataset_surface")
        mesh.save(str(path))
        paths[name] = str(path)
    return paths



def export_glb(output_dir: Path, material_surface, supported_surface, gap_surface,
               selected_strut, neighboring_tubes, a: np.ndarray, b: np.ndarray,
               endpoint_radius: float) -> str:
    """Export a self-contained, layered inspection scene for browser/3D viewers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("#0f172a")
    if material_surface is not None and material_surface.n_cells:
        plotter.add_mesh(material_surface, name="ct_material", color="#9ca3af",
                         opacity=0.30, smooth_shading=True)
    if supported_surface is not None and supported_surface.n_cells:
        plotter.add_mesh(supported_surface, name="material_supported", color="#22c55e",
                         opacity=0.96, smooth_shading=True)
    if gap_surface is not None and gap_surface.n_cells:
        plotter.add_mesh(gap_surface, name="detected_gap_broken", color="#ef4444",
                         opacity=0.98, smooth_shading=True)
    plotter.add_mesh(selected_strut, name="selected_expected_strut", color="#facc15",
                     opacity=1.0, smooth_shading=True)
    for index, tube in enumerate(neighboring_tubes):
        plotter.add_mesh(tube, name=f"nearby_strut_{index:04d}", color="#6b7280",
                         opacity=0.24, smooth_shading=True)
    for name, center in (("endpoint_node_0", a), ("endpoint_node_1", b)):
        plotter.add_mesh(pv.Sphere(radius=endpoint_radius, center=center), name=name,
                         color="#ffffff", opacity=1.0, smooth_shading=True)
    path = output_dir / "inspection_model.glb"
    gltf_path = output_dir / "inspection_model.gltf"
    try:
        plotter.export_gltf(str(gltf_path))
    finally:
        plotter.close()
    document = json.loads(gltf_path.read_text(encoding="utf-8"))
    buffers = document.get("buffers", [])
    if not buffers:
        raise ValueError("PyVista GLTF export did not produce buffers")
    binary = bytearray()
    buffer_offsets = []
    for buffer in buffers:
        uri = buffer.get("uri", "")
        if not uri.startswith("data:") or ";base64," not in uri:
            raise ValueError("PyVista GLTF export produced a non-embedded buffer")
        binary.extend(b"\x00" * ((4 - len(binary) % 4) % 4))
        buffer_offsets.append(len(binary))
        binary.extend(base64.b64decode(uri.split(",", 1)[1]))
    for view in document.get("bufferViews", []):
        source_buffer = int(view.get("buffer", 0))
        view["buffer"] = 0
        view["byteOffset"] = buffer_offsets[source_buffer] + int(view.get("byteOffset", 0))
    document["buffers"] = [{"byteLength": len(binary)}]
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        handle.write(json_chunk)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)
    gltf_path.unlink()
    return str(path)


def surrounding_tubes(mapped_struts: list[tuple[int, int, int]], junctions: dict[int, np.ndarray],
                      selected_id: int, crop_start: np.ndarray, crop_end: np.ndarray,
                      radius: float):
    """Build faint gray tubes for mapped struts whose line boxes intersect the crop."""
    tubes = []
    for strut_id, junction0_id, junction1_id in mapped_struts:
        if strut_id == selected_id or junction0_id not in junctions or junction1_id not in junctions:
            continue
        start, end = junctions[junction0_id], junctions[junction1_id]
        line_min, line_max = np.minimum(start, end), np.maximum(start, end)
        if np.all(line_max >= crop_start) and np.all(line_min <= crop_end):
            tubes.append(_tube(start, end, radius))
    return tubes


def render_context_views(material_surface, neighboring_tubes, a: np.ndarray, b: np.ndarray,
                         strut_id: int, classification: str, output_dir: Path, tube_radius: float,
                         window_size: tuple[int, int], zoom_out: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _tube(a, b, tube_radius)
    center = (a + b) / 2.0
    span = max(float(np.linalg.norm(b - a)), 1.0)
    if material_surface is not None:
        bounds = material_surface.bounds
        span = max(span, bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    selected_color = CLASS_COLORS.get(classification, "#f8fafc")
    paths = {}
    for name, camera in _camera_positions(center, span, zoom_out).items():
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("#0f172a")
        if material_surface is not None:
            plotter.add_mesh(material_surface, color="#cbd5e1", opacity=0.34, smooth_shading=True)
        for tube in neighboring_tubes:
            plotter.add_mesh(tube, color="#cbd5e1", opacity=0.28, smooth_shading=True)
        plotter.add_mesh(selected, color=selected_color, opacity=1.0, smooth_shading=True)
        plotter.add_text(f"Strut {strut_id:05d} | context | {classification}",
                         font_size=10, color="white")
        plotter.add_axes(line_width=2)
        plotter.camera_position = camera
        path = output_dir / f"context_{name}.png"
        plotter.show(screenshot=str(path), auto_close=True)
        paths[name] = str(path)
    return paths


def inspect(args: argparse.Namespace) -> dict:
    if args.crop_radius_voxels <= 0 or args.zoom_out <= 0 or args.context_tube_radius <= 0:
        raise ValueError("crop radius, zoom-out, and context tube radius must be positive")
    inventory_path, graph_path, scan_path = map(_resolve, (args.inventory, args.graph, args.scan))
    record = load_strut(inventory_path, args.strut_id)
    junctions = load_junctions(graph_path)
    try:
        a, b = junctions[record.junction0_id], junctions[record.junction1_id]
    except KeyError as exc:
        raise KeyError(f"Missing registered junction {exc.args[0]} for strut {record.strut_id}") from exc

    with tifffile.TiffFile(scan_path) as tif:
        if not tif.pages or len(tif.pages[0].shape) != 2:
            raise ValueError(f"Expected a stack of 2-D TIFF pages: {scan_path}")
        shape_zyx = (len(tif.pages), int(tif.pages[0].shape[0]), int(tif.pages[0].shape[1]))
        padding = args.padding if args.mode == FOCUSED_MODE else args.crop_radius_voxels
        start, end = crop_bounds(np.stack((a, b)), padding, shape_zyx)
        crop = read_crop(tif, start, end)
    output_dir = _resolve(args.output_dir) if args.output_dir else DEFAULT_OUTPUT / f"strut-{record.strut_id:05d}"
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    junctions = load_junctions(graph_path)
    mapped_struts = load_mapped_struts(graph_path)
    neighboring_tubes = surrounding_tubes(mapped_struts, junctions, record.strut_id,
                                          start, end, args.context_tube_radius)
    neighboring_count = len(neighboring_tubes)
    if args.mode == FOCUSED_MODE:
        context_mask, green_mask, red_mask, gaps = focus_material(
            crop, start, a, b, args.threshold, args.focus_radius
        )
        context_surface = make_material_surface(crop, start, 0.5, context_mask)
        green_surface = make_material_surface(crop, start, 0.5, green_mask)
        red_surface = make_material_surface(crop, start, 0.5, red_mask)
        plane_origin = (a + b) / 2.0
        plane_normal = clipping_normal(a, b)
        # Keep one side of the plane through the selected strut for a readable inspection cut.
        context_surface = context_surface.clip(normal=plane_normal, origin=plane_origin) if context_surface is not None else None
        green_surface = green_surface.clip(normal=plane_normal, origin=plane_origin) if green_surface is not None else None
        red_surface = red_surface.clip(normal=plane_normal, origin=plane_origin) if red_surface is not None else None
        ct_material_surface = _combine_polydata((context_surface, green_surface, red_surface))
        views = render_views(context_surface, green_surface, red_surface, a, b, record.strut_id,
                             record.classification, output_dir, args.tube_radius,
                             tuple(args.window_size), gaps)
        focused_voxels = int(np.count_nonzero(green_mask | red_mask))
    else:
        material_surface = make_material_surface(crop, start, 0.5)
        ct_material_surface = material_surface
        views = render_context_views(material_surface, neighboring_tubes, a, b, record.strut_id,
                                     record.classification, output_dir, args.tube_radius,
                                     tuple(args.window_size), args.zoom_out)
        gaps = None
        focused_voxels = None

    glb_path = None
    if args.export_glb:
        # Derive semantic overlays in context mode as well, using the selected
        # strut's local CT corridor.
        if args.mode == FOCUSED_MODE:
            glb_material_surface = _combine_polydata((context_surface, green_surface, red_surface))
            glb_supported_surface, glb_gap_surface = green_surface, red_surface
        else:
            _, supported_mask, gap_mask, _ = focus_material(
                crop, start, a, b, args.threshold, args.focus_radius
            )
            glb_material_surface = ct_material_surface
            glb_supported_surface = make_material_surface(crop, start, 0.5, supported_mask)
            glb_gap_surface = make_material_surface(crop, start, 0.5, gap_mask)
        glb_path = export_glb(
            output_dir, glb_material_surface, glb_supported_surface, glb_gap_surface,
            _tube(a, b, args.tube_radius), neighboring_tubes, a, b,
            max(args.tube_radius * 1.25, 1.0),
        )

    vtp_paths = export_interactive_geometry(
        output_dir,
        ct_material_surface,
        _tube(a, b, args.tube_radius),
        neighboring_tubes,
    )

    metadata = {
        "strut_id": record.strut_id, "classification": record.classification,
        "junction_ids": [record.junction0_id, record.junction1_id],
        "endpoint_xyz_registered_voxels": [a.tolist(), b.tolist()],
        "scan": str(scan_path), "scan_shape_zyx": list(shape_zyx),
        "crop_start_xyz": start.tolist(), "crop_end_xyz": end.tolist(),
        "crop_shape_zyx": list(crop.shape), "threshold": args.threshold,
        "material_voxels": int(np.count_nonzero(crop >= args.threshold)),
        "focused_material_voxels": focused_voxels,
        "mode": args.mode, "focus_radius_voxels": args.focus_radius, "crop_radius_voxels": args.crop_radius_voxels,
        "zoom_out": args.zoom_out, "context_tube_count": neighboring_count,
        "gap_bins": gaps, "context_opacity": 0.34, "focused_opacity": 0.95,
        "clipping_plane_origin_xyz": ((a + b) / 2.0).tolist(),
        "clipping_plane_normal_xyz": clipping_normal(a, b).tolist(),
        "tube_radius_voxels": args.tube_radius, "views": views,
        "interactive_vtp": vtp_paths,
    }
    if glb_path is not None:
        metadata["interactive_glb"] = glb_path
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inspection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.interactive:
        plotter = pv.Plotter(window_size=tuple(args.window_size))
        plotter.set_background("#0f172a")
        if args.mode == FOCUSED_MODE:
            if context_surface is not None:
                plotter.add_mesh(context_surface, color="#cbd5e1", opacity=0.20, smooth_shading=True)
            if green_surface is not None:
                plotter.add_mesh(green_surface, color="#22c55e", opacity=0.95, smooth_shading=True)
            if red_surface is not None:
                plotter.add_mesh(red_surface, color="#ef4444", opacity=0.95, smooth_shading=True)
            plotter.add_mesh(_tube(a, b, args.tube_radius), color="#facc15")
            plotter.add_mesh(pv.Plane(center=(a + b) / 2.0, direction=clipping_normal(a, b),
                                      i_size=30, j_size=30), color="#67e8f9", opacity=0.10,
                             style="wireframe")
        else:
            if material_surface is not None:
                plotter.add_mesh(material_surface, color="#cbd5e1", opacity=0.34, smooth_shading=True)
            for tube in neighboring_tubes:
                plotter.add_mesh(tube, color="#cbd5e1", opacity=0.28, smooth_shading=True)
            plotter.add_mesh(_tube(a, b, args.tube_radius), color="#facc15", smooth_shading=True)
            endpoint_radius = max(args.tube_radius * 1.25, 1.0)
            plotter.add_mesh(pv.Sphere(radius=endpoint_radius, center=a), color="#facc15")
            plotter.add_mesh(pv.Sphere(radius=endpoint_radius, center=b), color="#facc15")
        plotter.add_axes()
        plotter.show(title=f"Strut {record.strut_id:05d} - {record.classification}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strut_id", help="Inventory ID, e.g. 1728 or strut-01728")
    parser.add_argument("--mode", choices=(FOCUSED_MODE, CONTEXT_MODE), default=FOCUSED_MODE,
                        help="focused for close inspection or context for defect finding")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="Registered Stage 2a junction JSON")
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--padding", type=int, default=18, help="XYZ voxel padding around endpoints in focused mode")
    parser.add_argument("--crop-radius-voxels", type=int, default=64,
                        help="Context-mode XYZ padding around the selected strut")
    parser.add_argument("--zoom-out", type=float, default=1.0,
                        help="Context camera distance multiplier")
    parser.add_argument("--context-tube-radius", type=float, default=2.0,
                        help="Radius of faint surrounding mapped-strut overlays")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--focus-radius", type=float, default=8.0,
                        help="Small cylinder radius around the endpoint-to-endpoint line")
    parser.add_argument("--tube-radius", type=float, default=6.0)
    parser.add_argument("--window-size", type=int, nargs=2, default=(900, 700), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--interactive", action="store_true", help="Open PyVista window after saving PNGs")
    parser.add_argument("--export-glb", action="store_true",
                        help="Export inspection_model.glb with color-coded inspection layers")
    return parser


if __name__ == "__main__":
    result = inspect(build_parser().parse_args())
    print(json.dumps({"strut_id": result["strut_id"], "views": result["views"]}, indent=2))





