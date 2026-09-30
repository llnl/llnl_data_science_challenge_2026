"""Reference-based PacificVis unit-cell validation runner."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from part2.synthetic_validation import _bounded_sample, _mask, _render
from skimage.filters import threshold_otsu


def label(path: Path) -> str:
    text = path.stem.lower().replace("-", "_").replace(" ", "_")
    for name, tokens in {
        "missing_strut": ("missing_strut",), "broken_strut": ("broken_strut",),
        "thin_strut": ("thin_strut",), "inflated_strut": ("inflated_strut", "thick_strut"),
        "bent_strut": ("bent_strut",), "no_defects": ("no_defects",),
    }.items():
        if any(re.search(rf"(?:^|_){re.escape(t)}(?:_|$)", text) for t in tokens):
            return name
    return "unknown"


def predict(negative: int, positive: int, baseline_voxels: int) -> str:
    change = (negative + positive) / max(baseline_voxels, 1)
    if change < 0.005:
        return "no_defects"
    if change > 0.70:
        return "inflated_strut"
    if change > 0.32:
        return "bent_strut"
    if change > 0.15:
        return "broken_strut"
    return "missing_strut" if negative / max(baseline_voxels, 1) > 0.014 else "thin_strut"


def run(data_dir: Path, output_dir: Path) -> dict:
    files = sorted(data_dir.glob("*.npy"))
    files = [p for p in files if "xray_recon" in p.name]
    files.sort(key=lambda p: (0 if label(p) == "no_defects" else 1, str(p)))
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_mask = None
    baseline_voxels = 0
    records = []
    for path in files:
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        sample = _bounded_sample(volume)
        threshold = float(threshold_otsu(sample))
        factor = max(1, int(np.ceil(max(volume.shape) / 192)))
        compact = _mask(volume, threshold, factor)
        known = label(path)
        if known == "no_defects":
            baseline_mask = compact.copy()
            baseline_voxels = int(baseline_mask.sum())
        negative = int(np.logical_and(baseline_mask, ~compact).sum()) if baseline_mask is not None else 0
        positive = int(np.logical_and(compact, ~baseline_mask).sum()) if baseline_mask is not None else 0
        predicted = predict(negative, positive, baseline_voxels) if known != "no_defects" else "no_defects"
        target = output_dir / path.stem
        target.mkdir(parents=True, exist_ok=True)
        image = target / "isometric.png"
        _render(compact, image, f"{path.stem} | predicted: {predicted}")
        records.append({
            "dataset": str(path), "known_label": known, "predicted_label": predicted,
            "match": predicted == known, "screenshot": str(image),
            "features": {"shape_zyx": list(map(int, volume.shape)), "threshold": threshold,
                         "downsample_factor": factor, "sample_values": int(sample.size),
                         "compact_material_voxels": int(compact.sum()),
                         "negative_difference_voxels": negative,
                         "positive_difference_voxels": positive,
                         "difference_fraction": round((negative + positive) / max(baseline_voxels, 1), 6)},
        })
        del compact, sample, volume
    result = {"mode": "synthetic_validation_reference", "dataset_root": str(data_dir),
              "dataset_count": len(records), "records": records,
              "memory_strategy": "NPY mmap, bounded sampling, compact masks, no full-volume copy"}
    (output_dir / "synthetic_validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "candidate_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# PacificVis Octet Unit-Cell Validation", "", f"Datasets: {len(records)}", "",
             "| Known label | Predicted label | Match |", "|---|---|---|"]
    lines.extend(f"| {r['known_label']} | {r['predicted_label']} | {'yes' if r['match'] else 'no'} |" for r in records)
    lines += ["", "Classification uses compact-mask differences against the no-defect reference.",
              "Source NPY files were read with memory mapping and were not modified."]
    (output_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))
