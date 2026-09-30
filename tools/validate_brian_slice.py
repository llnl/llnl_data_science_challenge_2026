"""Validate one Brian CT slice against the defect masks and registered CAD.

Example:
    python tools/validate_brian_slice.py --slice 446
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
CT_PATH = ROOT / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
STAGE_DIR = ROOT / "part2/stage_2a_developer_output"
CAD_PATH = ROOT / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
OUT_DIR = ROOT / "outputs/brian_defect_overlay/slice_validation"


def display_gray(slice_2d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((slice_2d.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def mask_slice(mask_memmap: np.ndarray, z: int, shape: tuple[int, int]) -> np.ndarray:
    # Masks were written at 2x downsampled resolution. Read one source-aligned
    # mask plane and expand it in Y/X; no full mask stack is materialized.
    m = np.asarray(mask_memmap[min(z // 2, mask_memmap.shape[0] - 1)], dtype=bool)
    expanded = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)
    return expanded[: shape[0], : shape[1]]


def load_inventory() -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    with (STAGE_DIR / "all_struts_inventory.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            sid = int(row["strut_id"])
            out[sid] = {
                "classification": row["classification"],
                "centroid": np.array(
                    [float(row["centroid_x_um"]), float(row["centroid_y_um"]), float(row["centroid_z_um"])]
                ) / float(row["voxel_size_um"]),
            }
    return out


def make_composite(gray: np.ndarray, missing: np.ndarray, excess: np.ndarray, visible_struts, labels) -> Image.Image:
    h, w = gray.shape
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[missing] = (0.35 * rgb[missing] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    rgb[excess] = (0.35 * rgb[excess] + 0.65 * np.array([255, 220, 30])).astype(np.uint8)
    rgb[missing & excess] = np.array([255, 0, 255], dtype=np.uint8)
    overlay = Image.fromarray(rgb, "RGB")

    cad = Image.fromarray(np.repeat(gray[..., None], 3, axis=2), "RGB")
    draw = ImageDraw.Draw(cad)
    for sid, p0, p1, classification in visible_struts:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        color = {"Missing_Intentional": (60, 150, 255), "Missing_Unintentional": (255, 100, 0)}.get(
            classification, (80, 255, 80)
        )
        draw.line((x0, y0, x1, y1), fill=color, width=1)
        # Label every CAD strut visible in the plane. Candidate struts receive
        # a bright boxed label so they are easy to cross-reference in the CSV.
        mx, my = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
        text = str(sid)
        if sid in labels:
            draw.rectangle((mx - 2, my - 7, mx + 4 + 6 * len(text), my + 3), fill=(255, 255, 180))
            draw.text((mx, my - 6), text, fill=(0, 0, 0))
        elif 0 <= mx < w and 0 <= my < h:
            draw.text((mx, my - 5), text, fill=color)

    panel_w = 900
    panels = []
    for image in (Image.fromarray(gray, "L").convert("RGB"), overlay, cad):
        image.thumbnail((panel_w, 900))
        panels.append(image)
    scale = panels[0].width / w
    canvas = Image.new("RGB", (3 * panels[0].width, panels[0].height + 28), "white")
    for i, image in enumerate(panels):
        canvas.paste(image, (i * panels[0].width, 28))
    d = ImageDraw.Draw(canvas)
    d.text((8, 7), "Raw grayscale CT", fill="black")
    d.text((panels[0].width + 8, 7), "Existing missing/excess overlay", fill="black")
    d.text((2 * panels[0].width + 8, 7), "Registered CAD / expected struts", fill="black")
    d.text((8, canvas.height - 16), "red=missing  yellow=excess  purple=overlap", fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=int, default=446, dest="slice_index")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ct = tifffile.memmap(CT_PATH)
    z = args.slice_index
    if not 0 <= z < ct.shape[0]:
        raise ValueError(f"slice must be in [0, {ct.shape[0] - 1}]")
    raw = np.asarray(ct[z])
    sample = np.asarray(ct[::8, ::8, ::8], dtype=np.float32)
    lo, hi = np.percentile(sample, [1.0, 99.5])
    gray = display_gray(raw, lo, hi)

    missing_mm = tifffile.memmap(STAGE_DIR / "missing_material_mask_ds4.tif")
    excess_mm = tifffile.memmap(STAGE_DIR / "excess_material_mask_ds4.tif")
    missing = mask_slice(missing_mm, z, raw.shape)
    excess = mask_slice(excess_mm, z, raw.shape)
    region_color = np.full(raw.shape, "none", dtype=object)
    region_color[missing & ~excess] = "red"
    region_color[~missing & excess] = "yellow"
    region_color[missing & excess] = "purple"

    inventory = load_inventory()
    with CAD_PATH.open() as f:
        cad = json.load(f)
    junctions = {int(j["id"]): np.asarray(j["position"], dtype=float) for j in cad["junctions"]}
    visible = []
    for s in cad["struts"]:
        p0, p1 = junctions[int(s["junction0"])], junctions[int(s["junction1"])]
        if min(p0[2], p1[2]) - 4 <= z <= max(p0[2], p1[2]) + 4:
            sid = int(s["id"])
            visible.append((sid, p0, p1, str(inventory.get(sid, {}).get("classification", "Nominal"))))

    labels: set[int] = set()
    rows = []
    labeled_regions = ndimage.label(missing)[0]
    for rid in range(1, int(labeled_regions.max()) + 1):
        ys, xs = np.where(labeled_regions == rid)
        if len(xs) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        # Match in 3D using the region's source-slice z and inventory centroid.
        region_point = np.array([cx, cy, float(z)])
        candidates = []
        for sid, item in inventory.items():
            c = item["centroid"]
            dist = float(np.linalg.norm((c - region_point) * np.array([1.0, 1.0, 1.5])))
            if dist <= 30.0:
                candidates.append((dist, sid, item))
        candidates.sort(key=lambda x: x[0])
        # Keep nearby struts, with at least the nearest one for every region.
        for _, sid, item in candidates[:5] or [(0.0, -1, {"classification": "Unmatched", "centroid": region_point})]:
            labels.add(sid)
            rows.append(
                {
                    "slice_index": z,
                    "strut_id": sid,
                    "classification": item["classification"],
                    "centroid_x": round(cx, 3),
                    "centroid_y": round(cy, 3),
                    "centroid_z": z,
                    "region_color": str(region_color[ys[0], xs[0]]),
                }
            )

    image_path = OUT_DIR / f"brian_slice_{z:04d}_validation.png"
    make_composite(gray, missing, excess, visible, labels).save(image_path)
    csv_path = OUT_DIR / f"brian_slice_{z:04d}_defect_regions.csv"
    with csv_path.open("w", newline="") as f:
        fields = ["slice_index", "strut_id", "classification", "centroid_x", "centroid_y", "centroid_z", "region_color"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"slice={z} missing_pixels={int(missing.sum())} excess_pixels={int(excess.sum())}")
    print(f"visible_cad_struts={len(visible)} matched_rows={len(rows)}")
    print(f"image={image_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
