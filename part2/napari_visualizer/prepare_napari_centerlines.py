#!/usr/bin/env python3
"""Precompute Napari-ready centerline geometry from the aligned Stage 2a data.

The output contains one row per strut.  Its numeric ``*_zyx_vox`` columns can
be stacked directly into Napari Vectors or Shapes arrays without looking up
junctions, transforming axes, calculating bounding boxes, or joining metadata
at viewer time.  The TIFF is opened only to read its ZYX shape; voxel data is
never loaded by this preparation script.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import tifffile


VOXEL_SIZE_UM = 58.09
DEFAULT_BBOX_MARGIN_VOX = 10.0
CLASS_COLORS = {
    "Nominal": ("#4daf4a", (0.302, 0.686, 0.290, 1.0)),
    "Missing_Unintentional": ("#e41a1c", (0.894, 0.102, 0.110, 1.0)),
    "Missing_Intentional": ("#377eb8", (0.216, 0.494, 0.722, 1.0)),
    "Expected_Missing_But_Material_Present": ("#ff7f00", (1.0, 0.498, 0.0, 1.0)),
}
DEFAULT_COLOR = ("#808080", (0.502, 0.502, 0.502, 1.0))


HERE = Path(__file__).resolve().parent
STAGE2_OUTPUT = HERE.parent
ALIGNMENT_CHECK = STAGE2_OUTPUT.parent
PART2 = ALIGNMENT_CHECK.parents[1]
SCAN_PATH = PART2 / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
REGISTRATION_PATH = ALIGNMENT_CHECK / "candidate_affine_registered.json"
INVENTORY_PATH = STAGE2_OUTPUT / "all_struts_inventory.csv"
OUTPUT_CSV = HERE / "napari_centerlines.csv"
OUTPUT_SUMMARY = HERE / "napari_centerlines_summary.json"


def read_inventory(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {int(row["strut_id"]): row for row in csv.DictReader(handle)}
    if len(rows) == 0:
        raise ValueError(f"Inventory CSV is empty: {path}")
    return rows


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def finite(value: float) -> float:
    """Keep CSV coordinate precision useful while avoiding scientific notation."""
    return round(float(value), 6)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    if not all(path.exists() for path in (SCAN_PATH, REGISTRATION_PATH, INVENTORY_PATH)):
        missing = [str(path) for path in (SCAN_PATH, REGISTRATION_PATH, INVENTORY_PATH) if not path.exists()]
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))

    # Reading TIFF metadata only: this avoids loading the approximately 1 GB CT volume.
    with tifffile.TiffFile(SCAN_PATH) as tiff:
        scan_shape_zyx = tuple(int(v) for v in tiff.series[0].shape)
    if len(scan_shape_zyx) != 3:
        raise ValueError(f"Expected a 3-D TIFF, received shape {scan_shape_zyx}")

    registration = json.loads(REGISTRATION_PATH.read_text())
    junctions = {
        int(item["id"]): tuple(float(v) for v in item["position"])
        for item in registration["junctions"]
    }
    inventory = read_inventory(INVENTORY_PATH)
    struts = sorted(registration["struts"], key=lambda item: int(item["id"]))
    if len(struts) != len(inventory):
        raise ValueError(
            f"Registration has {len(struts)} struts but inventory has {len(inventory)} rows"
        )

    fieldnames = [
        "strut_id",
        "junction0_id",
        "junction1_id",
        "napari_layer",
        "napari_color_hex",
        "napari_color_rgba_json",
        "coordinate_order",
        "start_z_vox",
        "start_y_vox",
        "start_x_vox",
        "end_z_vox",
        "end_y_vox",
        "end_x_vox",
        "direction_z_vox",
        "direction_y_vox",
        "direction_x_vox",
        "center_z_vox",
        "center_y_vox",
        "center_x_vox",
        "length_vox",
        "length_um_from_centerline",
        "start_z_um",
        "start_y_um",
        "start_x_um",
        "end_z_um",
        "end_y_um",
        "end_x_um",
        "direction_z_um",
        "direction_y_um",
        "direction_x_um",
        "center_z_um",
        "center_y_um",
        "center_x_um",
        "bbox_margin_vox",
        "bbox_z_min_inclusive",
        "bbox_y_min_inclusive",
        "bbox_x_min_inclusive",
        "bbox_z_max_exclusive",
        "bbox_y_max_exclusive",
        "bbox_x_max_exclusive",
        "napari_vector_zyx_json",
        "napari_line_zyx_json",
    ]
    inventory_columns = [
        f"inventory_{name}" for name in next(iter(inventory.values())).keys() if name != "strut_id"
    ]
    fieldnames.extend(inventory_columns)

    rows = []
    classes: dict[str, int] = {}
    shape_xyz = scan_shape_zyx[::-1]
    for strut in struts:
        strut_id = int(strut["id"])
        metadata = inventory.get(strut_id)
        if metadata is None:
            raise ValueError(f"Inventory does not contain strut {strut_id}")
        p0_xyz = junctions[int(strut["junction0"])]
        p1_xyz = junctions[int(strut["junction1"])]
        p0_zyx = p0_xyz[::-1]
        p1_zyx = p1_xyz[::-1]
        direction_zyx = tuple(b - a for a, b in zip(p0_zyx, p1_zyx))
        center_zyx = tuple((a + b) / 2 for a, b in zip(p0_zyx, p1_zyx))
        length_vox = sum(value * value for value in direction_zyx) ** 0.5
        classification = metadata["classification"]
        color_hex, rgba = CLASS_COLORS.get(classification, DEFAULT_COLOR)
        classes[classification] = classes.get(classification, 0) + 1

        lo_xyz = [int(min(a, b) // 1 - DEFAULT_BBOX_MARGIN_VOX) for a, b in zip(p0_xyz, p1_xyz)]
        hi_xyz = [int(-(-max(a, b) // 1) + DEFAULT_BBOX_MARGIN_VOX + 1) for a, b in zip(p0_xyz, p1_xyz)]
        lo_xyz = [clamp(value, 0, size) for value, size in zip(lo_xyz, shape_xyz)]
        hi_xyz = [clamp(value, 0, size) for value, size in zip(hi_xyz, shape_xyz)]

        row = {
            "strut_id": strut_id,
            "junction0_id": int(strut["junction0"]),
            "junction1_id": int(strut["junction1"]),
            "napari_layer": "defects" if classification != "Nominal" else "nominal",
            "napari_color_hex": color_hex,
            "napari_color_rgba_json": json.dumps(rgba),
            "coordinate_order": "ZYX (Napari/NumPy); source registration converted from XYZ",
            "start_z_vox": finite(p0_zyx[0]),
            "start_y_vox": finite(p0_zyx[1]),
            "start_x_vox": finite(p0_zyx[2]),
            "end_z_vox": finite(p1_zyx[0]),
            "end_y_vox": finite(p1_zyx[1]),
            "end_x_vox": finite(p1_zyx[2]),
            "direction_z_vox": finite(direction_zyx[0]),
            "direction_y_vox": finite(direction_zyx[1]),
            "direction_x_vox": finite(direction_zyx[2]),
            "center_z_vox": finite(center_zyx[0]),
            "center_y_vox": finite(center_zyx[1]),
            "center_x_vox": finite(center_zyx[2]),
            "length_vox": finite(length_vox),
            "length_um_from_centerline": finite(length_vox * VOXEL_SIZE_UM),
            "bbox_margin_vox": DEFAULT_BBOX_MARGIN_VOX,
            "bbox_z_min_inclusive": lo_xyz[2],
            "bbox_y_min_inclusive": lo_xyz[1],
            "bbox_x_min_inclusive": lo_xyz[0],
            "bbox_z_max_exclusive": hi_xyz[2],
            "bbox_y_max_exclusive": hi_xyz[1],
            "bbox_x_max_exclusive": hi_xyz[0],
            "napari_vector_zyx_json": json.dumps(
                [[finite(v) for v in p0_zyx], [finite(v) for v in direction_zyx]]
            ),
            "napari_line_zyx_json": json.dumps(
                [[finite(v) for v in p0_zyx], [finite(v) for v in p1_zyx]]
            ),
        }
        for prefix, values in (("start", p0_zyx), ("end", p1_zyx), ("direction", direction_zyx), ("center", center_zyx)):
            for axis, value in zip("zyx", values):
                row[f"{prefix}_{axis}_um"] = finite(value * VOXEL_SIZE_UM)
        for name, value in metadata.items():
            if name != "strut_id":
                row[f"inventory_{name}"] = value
        rows.append(row)

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_csv": str(OUTPUT_CSV),
        "rows": len(rows),
        "scan_shape_zyx": scan_shape_zyx,
        "scan_voxel_size_um_zyx": [VOXEL_SIZE_UM] * 3,
        "registration": str(REGISTRATION_PATH),
        "inventory": str(INVENTORY_PATH),
        "bbox_margin_vox": DEFAULT_BBOX_MARGIN_VOX,
        "class_counts": classes,
        "coordinate_convention": "All Napari geometry columns are ZYX; input JSON coordinates are XYZ.",
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
