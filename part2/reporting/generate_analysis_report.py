"""Generate Stage 3 analysis from existing Stage 2 JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "part2" / "developer" / "output" / "anomaly_summary.json"
CANDIDATES = ROOT / "part2" / "interagent_feedback" / "anomaly_candidates.json"
VALIDATED = ROOT / "part2" / "interagent_feedback" / "validated_candidates_v2.json"
REPORT = ROOT / "part2" / "reporting" / "analysis_report.md"

# This is the v2 schema field. The older validated_candidates.json used a
# different field name and must not be mixed into this report.
CROP_SUPPORT_FIELD = "material_fraction_in_crop"


def _link(path_text: str) -> str:
    path = Path(path_text)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path_text.replace("\\", "/")


def generate() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    validated = json.loads(VALIDATED.read_text(encoding="utf-8"))

    lines = [
        "# Stage 3 Analysis Report",
        "",
        "## Scope",
        "",
        f"Stage 2a analyzed `{summary['input']}` using a bounded page-wise sample and produced **{len(candidates)} candidate defects**.",
        f"Stage 2b reviewed the 20 highest-priority candidates with local TIFF crops and PyVista off-screen renders.",
        "",
        "These are candidate defects, not confirmed manufacturing defects. Visual agreement with the CT data does not establish manufacturing cause, metrological accuracy, or probability of detection.",
        "",
        "## Stage 2a summary",
        "",
        f"- Candidate records: **{len(candidates)}**",
        f"- Input shape: `{summary['shape_zyx']}` (z, y, x)",
        f"- Segmentation threshold: `{summary['threshold']}` ({summary['threshold_method']})",
        f"- Downsample factor: `{summary['downsample_factor']}`",
        "- Candidate type: `possible_missing_or_broken_strut`",
        "- Candidate locations are in the registered CT voxel coordinate system.",
        "",
        "## Stage 2b reviewed candidates",
        "",
        "The `support` column is the v2 field `visual_validation.material_fraction_in_crop`; it is not `raw_local_support_fraction`.",
        "",
        "| ID | Type | Status / assessment | Confidence | Location (x, y, z) | Boundary | Support | Representative views |",
        "|---|---|---|---:|---|---|---:|---|",
    ]
    for item in validated:
        vv = item["visual_validation"]
        location = ", ".join(f"{float(v):.2f}" for v in item["location"]["voxel_xyz"])
        boundary = "inconclusive_boundary" if vv["crop_was_clipped"] or vv["near_scan_boundary"] else "clear"
        status = vv["assessment"]
        if status == "confirmed_missing_or_broken_strut":
            status = "likely missing/broken candidate; not manufacturing-confirmed"
        views = "<br>".join(f"[{name}]({_link(path)})" for name, path in zip(("iso", "front", "side"), vv["views"]))
        support = vv[CROP_SUPPORT_FIELD]
        lines.append(f"| {item['ID']} | {item['type']} | {status} | {item['confidence']:.3f} | {location} | {boundary} | {support:.6f} | {views} |")

    boundary_count = sum(v["visual_validation"]["crop_was_clipped"] or v["visual_validation"]["near_scan_boundary"] for v in validated)
    lines += [
        "",
        f"Boundary review: **{boundary_count} of {len(validated)}** reviewed candidates were boundary-affected. Boundary-affected candidates must be reported as inconclusive rather than confirmed defects.",
        "",
        "## Limitations and recommended next steps",
        "",
        "- Stage 2a uses downsampling and max pooling, which favor recall but can create false gaps or bridges.",
        "- Stage 2a did not perform CAD-to-CT registration or machine-learning classification; the registered topology is an expected-lattice reference, not proof of a manufacturing defect.",
        "- Stage 2b screenshots are local visual evidence only. Occlusion, segmentation threshold choice, partial-volume effects, and scan artifacts can mislead interpretation.",
        "- The current review supports the label `likely missing/broken candidate` for the clear-crop cases, not a confirmed manufacturing defect diagnosis.",
        "- Add strut-axis sampling, connected-component checks, local diameter estimates, and calibrated uncertainty before production use.",
        "- Boundary and clipped-crop cases, if encountered on future scans, should be routed to manual review and labeled inconclusive.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate()
