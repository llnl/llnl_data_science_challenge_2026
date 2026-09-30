from pathlib import Path

import numpy as np
import tifffile


INPUT = Path(__file__).resolve().parents[1] / "9x9x9_octet_lattice.tif"
OUT = Path(__file__).resolve().parent / "threshold_comparison"
THRESHOLDS = (0.3, 0.5, 0.7)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(INPUT) as tif:
        pages = tif.pages
        shape = (len(pages), pages[0].shape[0], pages[0].shape[1])
        dtype = pages[0].dtype
        if not np.issubdtype(dtype, np.integer):
            raise TypeError(f"Expected integer TIFF data, got {dtype}")
        scale = float(np.iinfo(dtype).max)
        total = int(np.prod(shape))
        rows = []
        writers = {
            threshold: tifffile.TiffWriter(
                OUT / f"segmented_mask_threshold_{threshold:.1f}.tif", bigtiff=True
            )
            for threshold in THRESHOLDS
        }
        counts = {threshold: 0 for threshold in THRESHOLDS}
        try:
            for page in pages:
                image = page.asarray()
                for threshold in THRESHOLDS:
                    mask = (image.astype(np.float64) / scale >= threshold).astype(np.uint8)
                    counts[threshold] += int(mask.sum())
                    writers[threshold].write(mask, photometric="minisblack", metadata=None)
        finally:
            for writer in writers.values():
                writer.close()

    report = [
        "# Threshold comparison",
        "",
        f"- Input: `{INPUT.name}`",
        f"- Shape: `{shape}`",
        f"- Input dtype: `{dtype}`",
        "- Thresholds are normalized intensity values, matching the Task 5 examples.",
        "",
        "| Threshold | Equivalent uint16 cutoff | Foreground voxels | Foreground percent | Output |",
        "|---:|---:|---:|---:|---|",
    ]
    for threshold in THRESHOLDS:
        cutoff = int(np.ceil(threshold * scale))
        foreground = counts[threshold]
        percent = 100.0 * foreground / total
        output = f"segmented_mask_threshold_{threshold:.1f}.tif"
        report.append(f"| {threshold:.1f} | {cutoff} | {foreground:,} | {percent:.4f}% | `{output}` |")
    (OUT / "threshold_comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"shape={shape} total={total}")
    for threshold in THRESHOLDS:
        print(f"threshold={threshold:.1f} cutoff={int(np.ceil(threshold * scale))} foreground={counts[threshold]} percent={100.0 * counts[threshold] / total:.4f}")


if __name__ == "__main__":
    main()
