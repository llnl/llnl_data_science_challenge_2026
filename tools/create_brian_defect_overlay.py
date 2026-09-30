from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CT_PATH = ROOT / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
OUT_DIR = ROOT / "outputs/brian_defect_overlay"
OUT_TIF = OUT_DIR / "Brian_CT_defects_highlighted.tif"
OUT_PREVIEW = OUT_DIR / "Brian_CT_defects_preview.jpg"


def upsample_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    # The saved masks are downsampled by 2 in each axis; crop after repeat to
    # handle odd source dimensions (761, 815, 837).
    return np.repeat(np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1), 2, axis=2)[
        : shape[0], : shape[1], : shape[2]
    ].astype(bool, copy=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ct = tifffile.memmap(CT_PATH)
    shape = tuple(ct.shape)
    out_shape = shape + (3,)

    stage = ROOT / "part2/stage_2a_developer_output"
    missing = upsample_mask(tifffile.imread(stage / "missing_material_mask_ds4.tif"), shape)
    excess = upsample_mask(tifffile.imread(stage / "excess_material_mask_ds4.tif"), shape)

    # Global display window from a regularly sampled subset, preserving CT
    # contrast consistently across all slices.
    sample = np.asarray(ct[::8, ::8, ::8], dtype=np.float32)
    lo, hi = np.percentile(sample, [1.0, 99.5])

    preview_indices = np.linspace(0, shape[0] - 1, 12, dtype=int)
    previews = []
    with tifffile.TiffWriter(OUT_TIF, bigtiff=True) as writer:
        for z in range(shape[0]):
            gray = np.clip((ct[z].astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            m = missing[z]
            e = excess[z]
            # Blend overlays so the underlying CT remains visible.
            rgb[m] = (0.35 * rgb[m] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
            rgb[e] = (0.35 * rgb[e] + 0.65 * np.array([255, 220, 30])).astype(np.uint8)
            both = m & e
            rgb[both] = np.array([255, 0, 255], dtype=np.uint8)
            writer.write(rgb, photometric="rgb", compression="deflate")
            if z in preview_indices:
                previews.append((z, rgb.copy()))

    # Contact sheet with a simple legend.
    thumbs = []
    for z, arr in previews:
        im = Image.fromarray(arr).convert("RGB")
        im.thumbnail((260, 260))
        canvas = Image.new("RGB", (280, 300), "white")
        canvas.paste(im, ((280 - im.width) // 2, 25))
        ImageDraw.Draw(canvas).text((8, 5), f"slice {z}", fill="black")
        thumbs.append(canvas)
    sheet = Image.new("RGB", (4 * 280, 3 * 300), "white")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 4) * 280, (i // 4) * 300))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((8, 870, 22, 884), fill=(255, 40, 40))
    draw.text((28, 868), "missing material", fill="black")
    draw.rectangle((170, 870, 184, 884), fill=(255, 220, 30))
    draw.text((190, 868), "excess material", fill="black")
    sheet.save(OUT_PREVIEW, quality=92)

    print(f"CT shape: {shape}")
    print(f"Missing voxels highlighted: {int(missing.sum())}")
    print(f"Excess voxels highlighted: {int(excess.sum())}")
    print(f"Wrote: {OUT_TIF}")
    print(f"Wrote: {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
