#!/usr/bin/env python3
"""Run Stage 2a unchanged with the approved candidate registration.

This wrapper redirects the pipeline module's registration input and output
directory before calling its normal ``main`` function.  The original pipeline,
source registration, and prior Stage 2a outputs are not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import stage2a_missing_strut_pipeline as pipeline


PART2 = Path(__file__).resolve().parent
CANDIDATE_REGISTRATION = (
    PART2
    / "registratiion_alignment_check"
    / "candidate_affine_registered.json"
)
OUTPUT = (
    PART2
    / "registratiion_alignment_check"
    / "stage2a_candidate_registration_output"
)


def main() -> None:
    if not CANDIDATE_REGISTRATION.exists():
        raise FileNotFoundError(CANDIDATE_REGISTRATION)

    pipeline.REGISTRATION = CANDIDATE_REGISTRATION
    pipeline.OUT = OUTPUT
    sys.argv = [
        "stage2a_missing_strut_pipeline.py",
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
        "1",
        "--feedback",
        (
            "Approved affine candidate-registration rerun; original Stage 2a "
            "pipeline and prior outputs preserved."
        ),
    ]
    pipeline.main()

    payload_path = OUTPUT / "developer_output.json"
    payload = json.loads(payload_path.read_text())
    payload["output_directory"] = str(OUTPUT.relative_to(PART2)) + "/"
    payload["registration_input"] = str(CANDIDATE_REGISTRATION.relative_to(PART2))
    payload["source_pipeline_modified"] = False
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    provenance = {
        "wrapper": str(Path(__file__).resolve()),
        "unchanged_pipeline": str(Path(pipeline.__file__).resolve()),
        "registration_input": str(CANDIDATE_REGISTRATION),
        "output_directory": str(OUTPUT),
        "original_registration_preserved": True,
        "prior_stage2a_output_preserved": True,
        "parameters": {
            "analysis_downsample": 2,
            "mask_downsample": 2,
            "tube_radius_voxels": 6.0,
            "trim_fraction": 0.20,
            "stations": 21,
            "empty_station_fraction": 0.05,
            "min_mask_component_voxels": 4,
            "cad_clearance_voxels": 2.0,
            "iteration": 1,
        },
    }
    (OUTPUT / "candidate_registration_rerun_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
