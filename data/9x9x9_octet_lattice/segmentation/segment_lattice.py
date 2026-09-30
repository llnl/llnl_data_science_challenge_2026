from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.filters import threshold_otsu

INPUT = Path(__file__).resolve().parents[1] / "9x9x9_octet_lattice.tif"
OUT = Path(__file__).resolve().parent
SLICE_INDEX = 380
STRIDE = 8

def main():
    with tifffile.TiffFile(INPUT) as source:
        pages = source.pages
        if len(pages) == 0:
            raise ValueError("Input TIFF contains no pages")
        shape = (len(pages),) + pages[0].shape
        if len(shape) != 3 or not (0 <= SLICE_INDEX < shape[0]):
            raise ValueError(f"Slice index {SLICE_INDEX} is invalid for shape {shape}")
        # Compressed TIFFs cannot be memory-mapped; sample page-wise instead.
        sample = np.concatenate([page.asarray()[::STRIDE, ::STRIDE].ravel() for page in pages[::STRIDE]])
        threshold = float(threshold_otsu(sample))
        sampled_foreground = float((sample >= threshold).mean())
        plt.figure(figsize=(8, 4)); plt.hist(sample, bins=256, log=True, color="0.35")
        plt.axvline(threshold, color="crimson", label=f"Otsu = {threshold:.1f}")
        plt.xlabel("Voxel intensity"); plt.ylabel("Sampled voxel count (log)"); plt.legend(); plt.tight_layout()
        plt.savefig(OUT / "diagnostic_histogram.png", dpi=150); plt.close()
        foreground = 0
        with tifffile.TiffWriter(OUT / "segmented_mask.tif", bigtiff=True) as writer:
            for page in pages:
                mask_page = (page.asarray() >= threshold).astype(np.uint8)
                foreground += int(mask_page.sum())
                writer.write(mask_page, photometric="minisblack", metadata=None)
        slice_mask = (pages[SLICE_INDEX].asarray() >= threshold).astype(np.uint8) * 255
        tifffile.imwrite(OUT / "slice_380.png", slice_mask, photometric="minisblack")
        plt.figure(figsize=(8, 3)); plt.plot(slice_mask.mean(axis=1))
        plt.xlabel("Row"); plt.ylabel("Foreground fraction"); plt.tight_layout()
        plt.savefig(OUT / "diagnostic_slice_profile.png", dpi=150); plt.close()
        input_dtype = pages[0].dtype
        input_compression = pages[0].compression
    total = int(np.prod(shape)); background = total - foreground; foreground_pct = 100 * foreground / total
    (OUT / "segmentation_report.md").write_text(f"""# Segmentation report: 9x9x9 octet lattice

## Summary
- Input: `{INPUT.name}`
- Volume shape: `{shape}`
- Input dtype: `{input_dtype}`
- Selected method: global Otsu threshold on a reproducible page-wise 8-voxel-stride sample
- Selected threshold: `{threshold:.6g}`
- Iterations: 1 diagnostic threshold attempt; no further iteration justified
- Foreground voxels: `{foreground:,}`
- Background voxels: `{background:,}`
- Foreground: `{foreground_pct:.4f}%`
- Background: `{100 - foreground_pct:.4f}%`

## Iteration history
1. Otsu threshold `{threshold:.6g}` yielded sampled foreground `{100 * sampled_foreground:.4f}%`; mask and slice 380 were written.

## Diagnostics
- [Histogram](diagnostic_histogram.png)
- [Slice-380 mask](slice_380.png)
- [Slice-380 profile](diagnostic_slice_profile.png)

## Limits and failures
- Slice index 380 is valid (`0..{shape[0] - 1}`).
- Source TIFF compression tag: `{input_compression}`; processing was performed page-by-page to bound peak memory.
""", encoding="utf-8")
    print(f"threshold={threshold:.6g} foreground={foreground} background={background} percent={foreground_pct:.4f}")

if __name__ == "__main__":
    main()

