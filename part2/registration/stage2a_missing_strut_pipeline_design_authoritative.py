#!/usr/bin/env python3
"""Stage 2a candidate-registration run with one explicit policy modification.

The computational pipeline is imported unchanged from
``stage2a_missing_strut_pipeline.py``.  After its normal extraction, expected
0.5%-CAD omissions are made design-authoritative: every row with
``expected_by_0point5_cad=True`` is labeled ``Missing_Intentional`` regardless
of measured CT occupancy.  An audit CSV preserves every overridden result.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import stage2a_missing_strut_pipeline as core


PART2 = Path(__file__).resolve().parent
CANDIDATE_REGISTRATION = (
    PART2 / "registratiion_alignment_check/candidate_affine_registered.json"
)
OUTPUT = PART2 / "registration_alignment_check_modified_pipeline"
POLICY_NAME = "design_authoritative_expected_omissions"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_design_authoritative_policy() -> dict:
    inventory_path = OUTPUT / "all_struts_inventory.csv"
    defects_path = OUTPUT / "defect_summary.csv"
    inventory = read_rows(inventory_path)
    defects = read_rows(defects_path)

    overrides = []
    for row in inventory:
        expected = row["expected_by_0point5_cad"].lower() == "true"
        if expected and row["classification"] != "Missing_Intentional":
            overrides.append(
                {
                    "strut_id": row["strut_id"],
                    "unit_cell_indices": row["unit_cell_indices"],
                    "prior_pipeline_classification": row["classification"],
                    "modified_classification": "Missing_Intentional",
                    "ct_material_occupancy": row["ct_material_occupancy"],
                    "mean_intensity": row["mean_intensity"],
                    "reason": (
                        "Expected omission in 0.5% CAD; design-authoritative "
                        "classification policy overrides CT material evidence."
                    ),
                }
            )
            row["classification"] = "Missing_Intentional"

    override_ids = {row["strut_id"] for row in overrides}
    for row in defects:
        if row["strut_id"] in override_ids:
            row["classification"] = "Missing_Intentional"

    write_rows(inventory_path, inventory)
    write_rows(defects_path, defects)
    write_rows(OUTPUT / "expected_design_authoritative_overrides.csv", overrides)

    counts = Counter(row["classification"] for row in inventory)
    counts["Expected_Missing_But_Material_Present"] += 0
    ordered_counts = {
        "Expected_Missing_But_Material_Present": counts[
            "Expected_Missing_But_Material_Present"
        ],
        "Missing_Intentional": counts["Missing_Intentional"],
        "Missing_Unintentional": counts["Missing_Unintentional"],
        "Nominal": counts["Nominal"],
    }

    execution_path = OUTPUT / "execution_log.json"
    execution = json.loads(execution_path.read_text())
    execution["classification_counts"] = ordered_counts
    execution["classification_policy"] = {
        "name": POLICY_NAME,
        "expected_cad_is_authoritative": True,
        "overridden_rows": len(overrides),
        "ct_measurements_preserved": True,
        "audit_csv": "expected_design_authoritative_overrides.csv",
    }
    execution_path.write_text(json.dumps(execution, indent=2) + "\n")

    payload_path = OUTPUT / "developer_output.json"
    payload = json.loads(payload_path.read_text())
    payload["executed_script_path"] = str(Path(__file__).resolve())
    payload["output_directory"] = str(OUTPUT.relative_to(PART2)) + "/"
    payload["registration_input"] = str(CANDIDATE_REGISTRATION.relative_to(PART2))
    payload["modified_classification_policy"] = POLICY_NAME
    payload["original_pipeline_modified"] = False
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    report_path = OUTPUT / "report.md"
    report = report_path.read_text()
    replacement = f"CT classifications: {json.dumps(ordered_counts, sort_keys=True)}."
    report = re.sub(r"CT classifications: \{.*?\}\.", replacement, report, count=1)
    report += f"""

## Modified expected-omission classification policy

This run applies one modification after the normal Stage 2a measurements:
all `{ordered_counts['Missing_Intentional']}` struts identified as expected
omissions by the 0.5% CAD are treated as design-authoritative
`Missing_Intentional` labels. Therefore,
`Expected_Missing_But_Material_Present` is zero.

The modification changes labels only. CT occupancy, intensity, geometry, masks,
and all other measurements remain unchanged. The {len(overrides)} rows that
contained enough CT evidence to receive a different label from the unmodified
pipeline are preserved in
[`expected_design_authoritative_overrides.csv`](expected_design_authoritative_overrides.csv).
These overrides should be interpreted as design labels, not proof that CT
material is absent.
"""
    report_path.write_text(report)

    provenance = {
        "modified_pipeline": str(Path(__file__).resolve()),
        "imported_unmodified_core_pipeline": str(Path(core.__file__).resolve()),
        "candidate_registration": str(CANDIDATE_REGISTRATION),
        "output_directory": str(OUTPUT),
        "policy": POLICY_NAME,
        "policy_scope": "classification labels for expected 0.5%-CAD omissions only",
        "overridden_rows": len(overrides),
        "classification_counts": ordered_counts,
        "original_core_pipeline_modified": False,
    }
    (OUTPUT / "modified_pipeline_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def main() -> None:
    if not CANDIDATE_REGISTRATION.exists():
        raise FileNotFoundError(CANDIDATE_REGISTRATION)

    core.REGISTRATION = CANDIDATE_REGISTRATION
    core.OUT = OUTPUT
    sys.argv = [
        Path(__file__).name,
        "--analysis-downsample",
        "2",
        "--mask-downsample",
        "2",
        "--tube-radius-voxels",
        "6",
        "--trim-fraction",
        "0.20",
        "--stations",
        "21",
        "--empty-station-fraction",
        "0.05",
        "--min-mask-component-voxels",
        "4",
        "--cad-clearance-voxels",
        "2",
        "--iteration",
        "2",
        "--feedback",
        (
            "Approved minimal policy modification: make expected 0.5%-CAD "
            "omissions design-authoritative while preserving CT measurements."
        ),
    ]
    core.main()
    print(json.dumps(apply_design_authoritative_policy(), indent=2))


if __name__ == "__main__":
    main()
