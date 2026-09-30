"""Run reference-based PacificVis validation with compact boolean masks."""

from pathlib import Path
import json
import re
import numpy as np
from skimage.filters import threshold_otsu
from part2.synthetic_validation import _bounded_sample, _mask, _render


def label(p):
    s = p.stem.lower()
    for name in ("missing_strut", "broken_strut", "thin_strut", "inflated_strut", "bent_strut", "no_defects"):
        if name in s:
            return name
    return "unknown"


def predict(neg, pos, base):
    d = (neg + pos) / max(base, 1)
    if d < .005: return "no_defects"
    if d > .70: return "inflated_strut"
    if d > .32: return "bent_strut"
    if d > .15: return "broken_strut"
    return "missing_strut" if neg / max(base, 1) > .014 else "thin_strut"


def run(data_dir, output_dir):
    files = sorted(data_dir.glob("*_xray_recon.npy"), key=lambda p: (label(p) != "no_defects", str(p)))
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = None
    base_count = 0
    records = []
    for p in files:
        vol = np.load(p, mmap_mode="r", allow_pickle=False)
        sample = _bounded_sample(vol)
        threshold = float(threshold_otsu(sample))
        factor = max(1, int(np.ceil(max(vol.shape) / 192)))
        compact = _mask(vol, threshold, factor)
        known = label(p)
        if known == "no_defects":
            baseline = compact.copy()
            base_count = int(baseline.sum())
        neg = int(np.logical_and(baseline > 0, compact == 0).sum()) if baseline is not None else 0
        pos = int(np.logical_and(compact > 0, baseline == 0).sum()) if baseline is not None else 0
        predicted = "no_defects" if known == "no_defects" else predict(neg, pos, base_count)
        target = output_dir / p.stem
        target.mkdir(parents=True, exist_ok=True)
        image = target / "isometric.png"
        _render(compact, image, f"{p.stem} | predicted: {predicted}")
        records.append({"dataset": str(p), "known_label": known, "predicted_label": predicted,
                        "match": predicted == known, "screenshot": str(image),
                        "features": {"shape_zyx": list(map(int, vol.shape)), "threshold": threshold,
                                     "downsample_factor": factor, "sample_values": int(sample.size),
                                     "compact_material_voxels": int(compact.sum()),
                                     "negative_difference_voxels": neg, "positive_difference_voxels": pos,
                                     "difference_fraction": round((neg + pos) / max(base_count, 1), 6)}})
        del compact, sample, vol
    result = {"mode": "synthetic_validation_reference", "dataset_root": str(data_dir),
              "dataset_count": len(records), "records": records,
              "memory_strategy": "NPY mmap, bounded sampling, compact masks, no full-volume copy"}
    (output_dir / "synthetic_validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "candidate_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# PacificVis Octet Unit-Cell Validation", "", f"Datasets: {len(records)}", "",
             "| Known label | Predicted label | Match |", "|---|---|---|"]
    lines.extend(f"| {r['known_label']} | {r['predicted_label']} | {'yes' if r['match'] else 'no'} |" for r in records)
    lines += ["", "Predictions use compact-mask differences against the no-defect reference.",
              "Source NPY files were memory-mapped read-only and not modified."]
    (output_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.data_dir, a.output_dir), indent=2))
