#!/usr/bin/env python3
"""Render five samples per nonempty modified-pipeline classification."""
from __future__ import annotations

import json
from pathlib import Path

import part2.registration.visualize_candidate_classification_samples as samples


PART2 = Path(__file__).resolve().parent
OUTPUT = PART2 / "registration_alignment_check_modified_pipeline"
GALLERY = OUTPUT / "classification_samples_5_each"


def main() -> None:
    samples.NEW_INVENTORY = OUTPUT / "all_struts_inventory.csv"
    samples.OUT = GALLERY
    samples.main()

    provenance_path = GALLERY / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(
        {
            "modified_pipeline": str(
                PART2 / "stage2a_missing_strut_pipeline_design_authoritative.py"
            ),
            "classification_policy": "design_authoritative_expected_omissions",
            "empty_category": "Expected_Missing_But_Material_Present",
            "empty_category_count": 0,
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    report_path = GALLERY / "report.md"
    report = report_path.read_text()
    report = report.replace(
        "# Candidate-registration classification samples",
        "# Modified-pipeline classification samples",
        1,
    )
    note = """

> The modified design-authoritative policy contains zero
> `Expected_Missing_But_Material_Present` rows, so no samples can be rendered
> for that category. Five samples are provided for each of the three nonempty
> categories.
"""
    marker = "The blue segment is the candidate-registered JSON centerline."
    report = report.replace(marker, note + "\n\n" + marker, 1)
    report_path.write_text(report)


if __name__ == "__main__":
    main()
