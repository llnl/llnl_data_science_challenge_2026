
"""Memory-bounded first-pass anomaly detection for lattice CT stacks.

This is deliberately a candidate generator, not a defect classifier.  TIFF
pages are read one at a time and the compact downsampled mask is persisted in a
memmap.  If a registered graph is supplied, expected struts are sampled in the
mask and only sustained low-support gaps are handed to Stage 2b.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.filters import threshold_otsu


DEFAULT_SCAN = Path("data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif")
DEFAULT_GRAPH = Path("data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json")


def _page_shape(path: Path) -> tuple[tuple[int, int, int], np.dtype]:
    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF has no pages: {path}")
        page = tif.pages[0]
        if len(page.shape) != 2:
            raise ValueError("Expected a stack of 2-D TIFF pages")
        return (len(tif.pages), page.shape[0], page.shape[1]), page.dtype


def _intensity_sample(path: Path, stride: int, max_values: int) -> np.ndarray:
    """Collect a deterministic, bounded sample without materializing the stack."""
    values: list[np.ndarray] = []
    with tifffile.TiffFile(path) as tif:
        for z, page in enumerate(tif.pages):
            if z % stride:
                continue
            values.append(page.asarray()[::stride, ::stride].ravel())
    if not values:
        raise ValueError("Sampling produced no TIFF values")
    sample = np.concatenate(values)
    if sample.size > max_values:
        # A deterministic evenly spaced sub-sample bounds memory and avoids a
        # random seed becoming part of the processing provenance.
        sample = sample[np.linspace(0, sample.size - 1, max_values, dtype=np.int64)]
    return sample


def _write_downsampled_mask(scan: Path, mask_path: Path, threshold: float, factor: int) -> tuple[tuple[int, int, int], int]:
    shape, _ = _page_shape(scan)
    small_shape = tuple(math.ceil(n / factor) for n in shape)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.memmap(mask_path, dtype=np.uint8, mode="w+", shape=small_shape)
    foreground = 0
    with tifffile.TiffFile(scan) as tif:
        for z, page in enumerate(tif.pages):
            # The only sizable working array is one source slice.
            page_mask = (page.asarray()[::factor, ::factor] >= threshold).astype(np.uint8)
            mask[z // factor] |= page_mask if z % factor == 0 else 0
            # Max-pooling within a z block is conservative for thin struts.
            if z % factor == 0:
                foreground += int(page_mask.sum())
            elif page_mask.any():
                mask[z // factor] |= page_mask
    mask.flush()
    del mask
    return small_shape, foreground


def _line_points(a: np.ndarray, b: np.ndarray, count: int) -> np.ndarray:
    t = np.linspace(0.15, 0.85, count, dtype=np.float32)
    return a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]


def _local_support(mask: np.memmap, xyz: np.ndarray) -> np.ndarray:
    """Return 3-D max-pooled support for points expressed as x,y,z indices."""
    x = np.rint(xyz[:, 0]).astype(np.int64)
    y = np.rint(xyz[:, 1]).astype(np.int64)
    z = np.rint(xyz[:, 2]).astype(np.int64)
    valid = (x >= 0) & (y >= 0) & (z >= 0)
    valid &= (x < mask.shape[2]) & (y < mask.shape[1]) & (z < mask.shape[0])
    result = np.zeros(len(x), dtype=bool)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                zz, yy, xx = z + dz, y + dy, x + dx
                inside = valid & (zz >= 0) & (yy >= 0) & (xx >= 0)
                inside &= (zz < mask.shape[0]) & (yy < mask.shape[1]) & (xx < mask.shape[2])
                if inside.any():
                    result[inside] |= mask[zz[inside], yy[inside], xx[inside]].astype(bool)
    return result


def _candidates(graph_path: Path | None, mask_path: Path, mask_shape: tuple[int, int, int], factor: int,
                gap_threshold: float, min_samples: int) -> list[dict]:
    if graph_path is None or not graph_path.exists():
        return []
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    junctions = {int(j["id"]): np.asarray(j["position"], dtype=np.float32) / factor
                 for j in graph.get("junctions", [])}
    mask = np.memmap(mask_path, dtype=np.uint8, mode="r", shape=mask_shape)
    candidates: list[dict] = []
    for strut in graph.get("struts", []):
        try:
            a, b = junctions[int(strut["junction0"])], junctions[int(strut["junction1"])]
        except (KeyError, TypeError):
            continue
        length = float(np.linalg.norm((b - a) * factor))
        count = max(min_samples, int(length / max(factor, 1)))
        support = _local_support(mask, _line_points(a, b, count))
        support_fraction = float(support.mean()) if support.size else 0.0
        # Endpoints are excluded above; a low fraction therefore represents a
        # persistent unsupported member rather than a single noisy voxel.
        if support_fraction >= gap_threshold:
            continue
        midpoint = ((a + b) * 0.5 * factor).tolist()
        confidence = float(np.clip(0.5 + (gap_threshold - support_fraction), 0.0, 0.99))
        candidates.append({
            "ID": f"strut-{int(strut.get('id', len(candidates))):05d}",
            "type": "possible_missing_or_broken_strut",
            "location": {"voxel_xyz": [round(float(v), 2) for v in midpoint], "coordinate_system": "registered_CT"},
            "confidence": round(confidence, 3),
            "crop_size": {"voxels_xyz": [max(16, int(round(length * 0.35))), 32, 32], "downsample_factor": factor},
            "reason": (f"Registered expected strut has {support_fraction:.1%} local material support "
                       f"along {count} interior samples (threshold {gap_threshold:.1%})."),
            "status": "candidate_for_stage_2b_visual_validation",
            "evidence": {"strut_id": int(strut.get("id", -1)), "support_fraction": round(support_fraction, 4),
                         "sample_count": count, "physical_length_in_input_units": round(length, 2)},
        })
    del mask
    return candidates


def run(scan: Path, output_dir: Path, graph: Path | None = None, factor: int = 4,
        sample_stride: int = 8, max_sample: int = 250_000, gap_threshold: float = 0.35,
        min_samples: int = 12) -> dict:
    shape, dtype = _page_shape(scan)
    sample = _intensity_sample(scan, sample_stride, max_sample)
    threshold = float(threshold_otsu(sample))
    mask_path = output_dir / "downsampled_material_mask.uint8.memmap"
    small_shape, pooled_foreground = _write_downsampled_mask(scan, mask_path, threshold, factor)
    candidates = _candidates(graph, mask_path, small_shape, factor, gap_threshold, min_samples)
    # Keep the inter-agent contract anchored at part2/, independent of the chosen diagnostics directory.
    feedback = Path(__file__).resolve().parents[1] / "interagent_feedback" / "anomaly_candidates.json"
    feedback.parent.mkdir(parents=True, exist_ok=True)
    feedback.write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    summary = {
        "input": str(scan), "shape_zyx": list(shape), "dtype": str(dtype),
        "threshold": threshold, "threshold_method": "Otsu on bounded page-wise strided sample",
        "sample_values": int(sample.size), "downsample_factor": factor,
        "downsampled_mask_shape_zyx": list(small_shape), "pooled_foreground_voxels": pooled_foreground,
        "candidate_count": len(candidates), "candidate_output": str(feedback),
        "limitations": ["Candidates are not confirmed defects; Stage 2b must visually validate them.",
                        "Downsampling and local max pooling favor recall and can create false gaps/bridges.",
                        "No CAD registration or machine learning is performed."],
    }
    (output_dir / "anomaly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # A small diagnostic that does not retain volume data.
    plt.figure(figsize=(7, 3)); plt.hist(sample, bins=128, log=True, color="0.35"); plt.axvline(threshold, color="crimson")
    plt.xlabel("CT intensity"); plt.ylabel("bounded sample count"); plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_histogram.png", dpi=140); plt.close()
    del sample
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=Path("part2/developer/output"))
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args()
    if args.downsample < 1:
        parser.error("--downsample must be positive")
    print(json.dumps(run(args.scan, args.output_dir, args.graph, factor=args.downsample), indent=2))


if __name__ == "__main__":
    main()


