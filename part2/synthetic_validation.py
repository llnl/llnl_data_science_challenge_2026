"""Memory-bounded validation for PacificVis-style octet-unit-cell NPY files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv
from skimage.filters import threshold_otsu

LABELS = (
    ("no_defects", ("no_defect", "no_defects", "nodefect", "healthy")),
    ("missing_strut", ("missing_strut", "missing")),
    ("broken_strut", ("broken_strut", "broken")),
    ("thin_strut", ("thin_strut", "thin")),
    ("bent_strut", ("bent_strut", "bent")),
    ("dross", ("dross",)),
)


def _known_label(path: Path) -> str:
    text = path.stem.lower().replace("-", "_").replace(" ", "_")
    for label, tokens in LABELS:
        if any(re.search(rf"(?:^|_)" + re.escape(token) + r"(?:_|$)", text) for token in tokens):
            return label
    if "unitcell" in text or "unit_cell" in text:
        return "no_defects"
    return "unknown"


def discover(data_dir: Path) -> list[Path]:
    return [p for p in sorted(data_dir.rglob("*.npy"))
            if any(t in p.name.lower() for t in ("octet", "unitcell", "unit_cell", "defect"))]


def _bounded_sample(volume: np.ndarray, max_values: int = 250_000) -> np.ndarray:
    stride = max(1, int(np.ceil((np.prod(volume.shape) / max_values) ** (1 / 3))))
    sample = np.asarray(volume[::stride, ::stride, ::stride]).ravel()
    if sample.size > max_values:
        sample = sample[np.linspace(0, sample.size - 1, max_values, dtype=np.int64)]
    return sample


def _mask(volume: np.ndarray, threshold: float, factor: int) -> np.ndarray:
    return (np.asarray(volume[::factor, ::factor, ::factor]) >= threshold).astype(np.uint8)


def _bbox(mask: np.ndarray):
    points = np.argwhere(mask)
    if points.size == 0:
        return None, None
    return points.min(axis=0).tolist(), points.max(axis=0).tolist()


def _prediction(features: dict, baseline: dict | None) -> str:
    if baseline is None:
        return "defect_present" if features["material_fraction"] < 0.15 else "no_defects"
    missing = max(0.0, baseline["material_fraction"] - features["material_fraction"])
    added = max(0.0, features["material_fraction"] - baseline["material_fraction"])
    if max(missing, added) < 0.005:
        return "no_defects"
    return "dross" if added > missing * 1.5 else "missing_or_broken_strut"


def _normalized(label: str) -> str:
    return "missing_or_broken_strut" if label in {"missing_strut", "broken_strut", "missing_or_broken_strut"} else label


def _render(mask: np.ndarray, output_path: Path, title: str):
    grid = pv.ImageData(dimensions=(mask.shape[2], mask.shape[1], mask.shape[0]))
    grid.point_data["material"] = mask.ravel(order="F")
    surface = grid.contour([0.5], scalars="material") if mask.any() and not mask.all() else None
    plotter = pv.Plotter(off_screen=True, window_size=(1100, 850))
    plotter.set_background("#111827")
    if surface is not None:
        plotter.add_mesh(surface, color="#8fb7d9", opacity=0.8, smooth_shading=True)
    plotter.add_text(title, font_size=12, color="white")
    plotter.add_axes(line_width=2)
    plotter.view_isometric()
    plotter.show(screenshot=str(output_path), auto_close=True)


def validate(data_dir: Path, output_dir: Path, max_datasets: int = 0) -> dict:
    files = discover(data_dir)
    if max_datasets > 0:
        files = files[:max_datasets]
    output_dir.mkdir(parents=True, exist_ok=True)
    records, baseline = [], None
    for path in files:
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        if volume.ndim != 3:
            continue
        sample = _bounded_sample(volume)
        threshold = float(threshold_otsu(sample)) if sample.min() != sample.max() else float(sample.min())
        factor = max(1, int(np.ceil(max(volume.shape) / 192)))
        compact = _mask(volume, threshold, factor)
        features = {
            "shape_zyx": list(map(int, volume.shape)), "dtype": str(volume.dtype),
            "threshold": threshold, "threshold_method": "Otsu on bounded strided NPY sample",
            "downsample_factor": factor, "sample_values": int(sample.size),
            "material_voxels_in_compact_mask": int(compact.sum()),
            "material_fraction": round(float(compact.mean()), 8),
        }
        features["bbox_zyx_min"], features["bbox_zyx_max"] = _bbox(compact)
        known = _known_label(path)
        if known == "no_defects" and baseline is None:
            baseline = features
        predicted = _prediction(features, baseline if known != "no_defects" else features)
        target = output_dir / path.stem
        target.mkdir(parents=True, exist_ok=True)
        image = target / "isometric.png"
        _render(compact, image, f"{path.stem} | predicted: {predicted}")
        records.append({
            "dataset": str(path), "known_defect_type": known,
            "predicted_defect_type": predicted,
            "prediction_matches_known_label": _normalized(predicted) == _normalized(known),
            "features": features, "visualization": str(image),
        })
        del compact, sample, volume
    result = {"mode": "synthetic_validation", "dataset_root": str(data_dir),
              "dataset_count": len(records), "records": records,
              "memory_strategy": "NPY memory mapping, bounded strided sampling, compact masks"}
    (output_dir / "synthetic_validation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("part2/developer/output/synthetic_validation"))
    parser.add_argument("--max-datasets", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(validate(args.data_dir, args.output_dir, args.max_datasets), indent=2))
