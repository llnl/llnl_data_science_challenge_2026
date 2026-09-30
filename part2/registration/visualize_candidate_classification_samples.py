#!/usr/bin/env python3
"""Render five spatially distributed samples per regenerated Stage 2a class."""
from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


PART2 = Path(__file__).resolve().parent
PROJECT = PART2.parent
VISUALIZER_PATH = PROJECT / "stage3_visualization/strut_visualizer.py"
SCAN = PART2 / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
REGISTRATION = PART2 / "registratiion_alignment_check/candidate_affine_registered.json"
NEW_INVENTORY = PART2 / "registratiion_alignment_check/stage2a_candidate_registration_output/all_struts_inventory.csv"
OLD_INVENTORY = PART2 / "stage_2a_developer_output/all_struts_inventory.csv"
OUT = PART2 / "registratiion_alignment_check/stage2a_candidate_registration_output/classification_samples_5_each"
THRESHOLD = 40081.0


def load_visualizer():
    spec = importlib.util.spec_from_file_location("candidate_strut_visualizer", VISUALIZER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {VISUALIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inventory(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_five(rows):
    """Select five samples spanning each category's registered Z extent."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["classification"]].append(row)
    selected = {}
    for classification, members in grouped.items():
        members.sort(key=lambda row: float(row["centroid_z_um"]))
        if len(members) < 5:
            raise ValueError(f"{classification} has only {len(members)} rows")
        indices = [round(i * (len(members) - 1) / 4) for i in range(5)]
        selected[classification] = [members[index] for index in indices]
    return selected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    visualizer = load_visualizer()
    current = read_inventory(NEW_INVENTORY)
    prior = {int(row["strut_id"]): row for row in read_inventory(OLD_INVENTORY)}
    selected = select_five(current)
    manifest = []

    for classification in sorted(selected):
        class_dir = OUT / classification
        for row in selected[classification]:
            strut_id = int(row["strut_id"])
            metadata = visualizer.visualize_strut(
                scan_path=SCAN,
                registration_path=REGISTRATION,
                strut_id=strut_id,
                output_dir=class_dir,
                csv_path=NEW_INVENTORY,
                margin_voxels=10.0,
                threshold=THRESHOLD,
                downsample=1,
                voxel_alpha=0.28,
            )
            old = prior[strut_id]
            manifest.append(
                {
                    "classification": classification,
                    "strut_id": strut_id,
                    "source_z_voxels": float(row["centroid_z_um"]) / float(row["voxel_size_um"]),
                    "prior_classification": old["classification"],
                    "prior_occupancy": float(old["ct_material_occupancy"]),
                    "candidate_occupancy": float(row["ct_material_occupancy"]),
                    "occupancy_change": float(row["ct_material_occupancy"])
                    - float(old["ct_material_occupancy"]),
                    "render": str(
                        Path(metadata["output_files"]["render_3d"]).relative_to(OUT)
                    ),
                    "metadata": str(
                        Path(metadata["output_files"]["metadata"]).relative_to(OUT)
                    ),
                }
            )

    with (OUT / "sample_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    provenance = {
        "visualizer": str(VISUALIZER_PATH),
        "visualizer_modified_by_this_run": False,
        "scan": str(SCAN),
        "registration": str(REGISTRATION),
        "inventory": str(NEW_INVENTORY),
        "selection": "Five quantile-spaced samples by registered source-Z centroid per category.",
        "threshold": THRESHOLD,
        "downsample": 1,
        "margin_voxels": 10.0,
        "voxel_alpha": 0.28,
        "sample_count": len(manifest),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    lines = [
        "# Candidate-registration classification samples",
        "",
        "These renders use the affine candidate registration, regenerated Stage 2a inventory,",
        "full-resolution crops, and intensity threshold 40081. Five samples per class were",
        "selected at quantiles of registered source-Z position to cover the volume.",
        "",
        "The blue segment is the candidate-registered JSON centerline. Red voxels are CT",
        "material at or above the threshold. Occupancy values come from the Stage 2a tube",
        "sampler rather than from the rendered crop.",
        "",
    ]
    for classification in sorted(selected):
        lines.extend([f"## {classification}", ""])
        for item in [row for row in manifest if row["classification"] == classification]:
            lines.extend(
                [
                    f"### Strut {item['strut_id']}",
                    "",
                    f"- Prior class: `{item['prior_classification']}`",
                    f"- Prior occupancy: `{item['prior_occupancy']:.4f}`",
                    f"- Candidate occupancy: `{item['candidate_occupancy']:.4f}`",
                    f"- Occupancy change: `{item['occupancy_change']:+.4f}`",
                    "",
                    f"![Strut {item['strut_id']}]({item['render']})",
                    "",
                    f"[Metadata]({item['metadata']})",
                    "",
                ]
            )
    lines.extend(
        [
            "## Files",
            "",
            "- [Sample manifest](sample_manifest.csv)",
            "- [Provenance](provenance.json)",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
