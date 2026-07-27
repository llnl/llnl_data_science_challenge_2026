"""Tests for the eval-input generator in ``evals/perturb_lattice_json``.

These perturbations are the ground truth the scan-junctions rubrics grade
against, so a silently wrong one would make an evaluation meaningless rather
than merely failing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from overlay_registered_nodes_histogram import (  # noqa: E402
    merge_colocated_junctions,
)
from perturb_lattice_json import (  # noqa: E402
    add_phantom_slab,
    load_lattice_json,
    rotate_junctions,
)


def _cell_lattice(cells_per_axis: int = 2, pitch: float = 40.0) -> dict:
    """A small lattice with duplicated entries and a unit_cells index grid."""
    corners = [
        (i, j, k)
        for i in range(cells_per_axis + 1)
        for j in range(cells_per_axis + 1)
        for k in range(cells_per_axis + 1)
    ]
    entry_of = {corner: (2 * n, 2 * n + 1) for n, corner in enumerate(corners)}

    junctions = []
    for corner, entries in entry_of.items():
        position = [pitch * c for c in corner]
        for entry in entries:
            junctions.append({"id": entry, "position": list(position), "indices": [0.0] * 3})
    junctions.sort(key=lambda j: j["id"])

    struts = []
    for corner in corners:
        for axis in range(3):
            shifted = list(corner)
            shifted[axis] += 1
            neighbor = (shifted[0], shifted[1], shifted[2])
            if neighbor in entry_of:
                half = len(struts) % 2
                struts.append(
                    {
                        "id": len(struts),
                        "unit_cell_edge_idx": axis,
                        "junction0": entry_of[corner][half],
                        "junction1": entry_of[neighbor][half],
                        "thickness": 0.1,
                    }
                )

    unit_cells = []
    for i in range(cells_per_axis):
        for j in range(cells_per_axis):
            for k in range(cells_per_axis):
                owned = [
                    strut["id"]
                    for strut in struts
                    if all(
                        low <= entry_position <= low + 1
                        for low, entry_position in zip(
                            (i, j, k),
                            _corner_of(strut["junction0"], entry_of),
                        )
                    )
                    and all(
                        low <= entry_position <= low + 1
                        for low, entry_position in zip(
                            (i, j, k),
                            _corner_of(strut["junction1"], entry_of),
                        )
                    )
                ]
                unit_cells.append(
                    {"id": len(unit_cells), "struts": owned, "indices": [i, j, k]}
                )

    return {"junctions": junctions, "struts": struts, "unit_cells": unit_cells}


def _corner_of(entry: int, entry_of: dict) -> tuple[int, int, int]:
    for corner, entries in entry_of.items():
        if entry in entries:
            return corner
    raise AssertionError(f"entry {entry} belongs to no corner")


@pytest.fixture
def cell_lattice_path(tmp_path: Path) -> Path:
    path = tmp_path / "lattice.json"
    path.write_text(json.dumps(_cell_lattice()))
    return path


def test_rotation_displaces_the_edges_and_pins_the_centre(cell_lattice_path):
    original = np.array(
        [j["position"] for j in load_lattice_json(cell_lattice_path)["junctions"]]
    )
    rotated = np.array(
        [
            j["position"]
            for j in rotate_junctions(
                load_lattice_json(cell_lattice_path), 10.0, "z"
            )["junctions"]
        ]
    )

    displacement = np.linalg.norm(rotated - original, axis=1)
    centroid = original.mean(axis=0)
    # Only the distance from the rotation axis matters, so measure it in-plane.
    radius = np.linalg.norm((original - centroid)[:, :2], axis=1)

    # A rotation about the centroid pins the middle and swings the rim through
    # a chord of 2*r*sin(theta/2) -- checking the chord pins the magnitude, not
    # merely that something moved.
    assert displacement[np.argmin(radius)] < 1e-9
    np.testing.assert_allclose(
        displacement, 2 * radius * np.sin(np.deg2rad(10.0) / 2), atol=1e-9
    )
    # Rotating about z cannot change z.
    np.testing.assert_allclose(rotated[:, 2], original[:, 2])


def test_rotation_keeps_duplicate_entries_exactly_coincident(cell_lattice_path):
    """The merge rejects groups that are merely close, so drift here would break it."""
    rotated = rotate_junctions(load_lattice_json(cell_lattice_path), 2.0, "z")
    positions = np.array([j["position"] for j in rotated["junctions"]])

    merged, entry_to_junction = merge_colocated_junctions(positions)
    assert len(merged) * 2 == len(positions)
    assert len(np.unique(entry_to_junction)) == len(merged)


def test_phantom_slab_appends_a_connected_group_beyond_the_lattice(
    cell_lattice_path,
):
    data = add_phantom_slab(load_lattice_json(cell_lattice_path), "x")
    positions = np.array([j["position"] for j in data["junctions"]])

    original = load_lattice_json(cell_lattice_path)
    n_original = len(original["junctions"])
    phantom = positions[n_original:]
    assert len(phantom) > 0

    # The copy sits one cell pitch past the original lattice's x extent.
    original_max_x = positions[:n_original, 0].max()
    assert phantom[:, 0].max() > original_max_x

    # Phantom struts join phantom entries only, so they form their own group.
    phantom_struts = [
        strut
        for strut in data["struts"]
        if strut["junction0"] >= n_original or strut["junction1"] >= n_original
    ]
    assert phantom_struts
    assert all(
        strut["junction0"] >= n_original and strut["junction1"] >= n_original
        for strut in phantom_struts
    )


def test_phantom_slab_output_stays_schema_valid(cell_lattice_path):
    data = add_phantom_slab(load_lattice_json(cell_lattice_path), "x")

    ids = [junction["id"] for junction in data["junctions"]]
    assert ids == list(range(len(ids)))
    assert [strut["id"] for strut in data["struts"]] == list(
        range(len(data["struts"]))
    )
    for strut in data["struts"]:
        assert 0 <= strut["junction0"] < len(ids)
        assert 0 <= strut["junction1"] < len(ids)

    # Duplicated entries survive the copy, so the merge still applies.
    positions = np.array([junction["position"] for junction in data["junctions"]])
    merged, _ = merge_colocated_junctions(positions)
    assert len(merged) < len(positions)


def test_phantom_slab_without_unit_cells_explains_itself(tmp_path: Path):
    path = tmp_path / "no_cells.json"
    data = _cell_lattice()
    del data["unit_cells"]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="unit_cells"):
        add_phantom_slab(load_lattice_json(path), "x")


def test_rejects_non_sequential_junction_ids(tmp_path: Path):
    path = tmp_path / "bad_ids.json"
    data = _cell_lattice()
    data["junctions"][0]["id"] = 999
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="not 0..N-1"):
        load_lattice_json(path)


@pytest.mark.parametrize("axis", ["w", "X", ""])
def test_rejects_unknown_axes(cell_lattice_path, axis: str):
    with pytest.raises(ValueError, match="axis must be one of"):
        rotate_junctions(load_lattice_json(cell_lattice_path), 1.0, axis)
