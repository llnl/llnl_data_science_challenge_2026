"""Tests for the junction scanning and triage statistics in ``junction_scan``."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conftest import (  # noqa: E402
    BACKGROUND_LEVEL,
    MATERIAL_LEVEL,
    make_synthetic_lattice,
)
from junction_scan import (  # noqa: E402
    CSV_HEADER,
    dark_component_sizes,
    dark_neighbor_counts,
    parse_junction_ids,
    save_marked_tiff,
    save_mip_overlay_with_status,
    save_slice_overlay_with_radius,
    scan_junctions,
    summarize_scan,
    write_junction_csv,
)


def _junction_at(result, position_xyz) -> int:
    """Merged ids come out of a positional sort, so look candidates up by voxel."""
    matches = np.flatnonzero((result.voxel_xyz == np.array(position_xyz)).all(axis=1))
    assert len(matches) == 1, f"expected one junction at {position_xyz}, got {matches}"
    return int(matches[0])


def test_scan_merges_entries_and_flags_only_the_dark_junction(synthetic_lattice):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )

    assert result.n_junctions == synthetic_lattice.n_junctions == 27
    assert len(result.entry_to_junction) == synthetic_lattice.n_entries == 54
    # A 3x3x3 grid with 6-connectivity runs from a 3-strut corner to a 6-strut
    # center. Unmerged, every entry would carry half of that.
    assert (result.degree.min(), result.degree.max()) == (3, 6)

    candidates = np.flatnonzero(result.candidate)
    assert len(candidates) == 1
    assert candidates[0] == _junction_at(result, synthetic_lattice.dark_position_xyz)


def test_threshold_is_the_volume_otsu_not_the_nodes(synthetic_lattice):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    # The cut has to sit in the trough between the two modes; a threshold fitted
    # to the sampled nodes alone would drift into one of them instead.
    assert BACKGROUND_LEVEL < result.threshold < MATERIAL_LEVEL


def test_radius_absorbs_a_registration_offset(tmp_path):
    """A misregistered lattice reads as wholly absent until the radius covers the drift.

    This is the signature the radius sweep exists to expose: a defect-free
    specimen flagged in bulk at a tight radius and cleanly at a loose one is
    misaligned, not defective.
    """
    lattice = make_synthetic_lattice(
        tmp_path, dark_cell=None, json_offset=(4.0, 4.0, 4.0)
    )

    tight = scan_junctions(lattice.volume, lattice.registered_json_path, radius=1)
    assert tight.dark.all()

    forgiving = scan_junctions(lattice.volume, lattice.registered_json_path, radius=6)
    assert not forgiving.dark.any()


def test_exclude_bottom_face_removes_the_max_nominal_y_layer(synthetic_lattice):
    result = scan_junctions(
        synthetic_lattice.volume,
        synthetic_lattice.registered_json_path,
        nominal_json_path=synthetic_lattice.nominal_json_path,
        exclude_bottom_face=True,
    )

    assert result.excluded.sum() == 9  # one 3x3 face of the 3x3x3 grid
    # The dark junction is interior, so excluding the face must not hide it.
    assert result.candidate.sum() == 1


def test_exclude_bottom_face_without_nominal_json_raises(synthetic_lattice):
    with pytest.raises(ValueError, match="requires nominal_json_path"):
        scan_junctions(
            synthetic_lattice.volume,
            synthetic_lattice.registered_json_path,
            exclude_bottom_face=True,
        )


def test_exclude_junction_ids_suppresses_a_candidate(synthetic_lattice):
    baseline = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    dark_id = _junction_at(baseline, synthetic_lattice.dark_position_xyz)

    result = scan_junctions(
        synthetic_lattice.volume,
        synthetic_lattice.registered_json_path,
        exclude_junction_ids=[dark_id],
    )
    assert result.dark.sum() == 1
    assert result.candidate.sum() == 0


def test_out_of_range_exclusion_raises(synthetic_lattice):
    with pytest.raises(ValueError, match="exclude_junction_ids must lie in"):
        scan_junctions(
            synthetic_lattice.volume,
            synthetic_lattice.registered_json_path,
            exclude_junction_ids=[999],
        )


@pytest.mark.parametrize("radius", [0, -3])
def test_non_positive_radius_raises(synthetic_lattice, radius):
    with pytest.raises(ValueError, match="radius must be a positive"):
        scan_junctions(
            synthetic_lattice.volume,
            synthetic_lattice.registered_json_path,
            radius=radius,
        )


def test_non_volumetric_input_raises(synthetic_lattice):
    with pytest.raises(ValueError, match="expected a 3D volume"):
        scan_junctions(
            synthetic_lattice.volume[0], synthetic_lattice.registered_json_path
        )


# A path graph 0-1-2-3 plus an isolated 4, so one dark pair, one dark singleton,
# and a bright junction that must not be counted as anyone's neighbor.
PATH_STRUTS = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])


def test_dark_neighbor_counts_counts_incident_dark_ends():
    dark = np.array([True, True, False, True, False])
    counts = dark_neighbor_counts(PATH_STRUTS, dark, 5)
    assert counts.tolist() == [1, 1, 2, 0, 1]


def test_dark_component_sizes_never_mix_dark_and_bright():
    dark = np.array([True, True, False, True, False])
    sizes = dark_component_sizes(PATH_STRUTS, dark, 5)
    # 0-1 are a dark pair; 3 is dark but its only neighbors are bright.
    assert sizes.tolist() == [2, 2, 0, 1, 0]


def test_dark_component_sizes_with_nothing_dark():
    sizes = dark_component_sizes(PATH_STRUTS, np.zeros(5, dtype=bool), 5)
    assert sizes.tolist() == [0, 0, 0, 0, 0]


def test_summary_is_json_serializable_and_reports_an_isolated_candidate(
    synthetic_lattice,
):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    summary = summarize_scan(result)

    assert json.loads(json.dumps(summary)) == summary
    assert summary["n_junctions"] == 27
    assert summary["n_entries"] == 54
    assert summary["n_candidates"] == 1
    assert summary["dark_components"] == {
        "n_components": 1,
        "largest": 1,
        "size_histogram": [{"size": 1, "count": 1}],
    }

    (candidate,) = summary["candidates"]
    assert candidate["dark_neighbor_count"] == 0
    assert candidate["component_size"] == 1
    assert (candidate["x"], candidate["y"], candidate["z"]) == tuple(
        synthetic_lattice.dark_position_xyz
    )


def test_summary_bands_partition_the_scored_junctions(synthetic_lattice):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    summary = summarize_scan(result, n_bands=3)

    for axis in ("x", "y", "z"):
        bands = summary["dark_fraction_by_band"][axis]
        assert len(bands) == 3
        assert sum(band["n_scored"] for band in bands) == summary["n_scored"]
        assert sum(band["n_dark"] for band in bands) == summary["n_candidates"]

    octants = summary["dark_fraction_by_octant"]
    assert len(octants) == 8
    assert sum(cell["n_scored"] for cell in octants) == summary["n_scored"]


def test_summary_bands_localize_a_missing_face(synthetic_lattice):
    """Excluding nothing, a whole dark face should land in one y band."""
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    # Force the maximum-y layer dark, standing in for a machined-off face.
    max_y = result.positions_xyz[:, 1].max()
    forced = result.dark | (result.positions_xyz[:, 1] >= max_y - 0.5)
    faced = replace(result, dark=forced)

    summary = summarize_scan(faced, n_bands=3)
    y_bands = summary["dark_fraction_by_band"]["y"]
    assert y_bands[-1]["dark_fraction"] == 1.0
    # The nine face junctions plus the pre-existing dark one, which is a strut
    # away from the face and so joins its component -- a real property of the
    # statistic worth pinning: an isolated defect touching a systematic region
    # is absorbed into it and stops looking isolated.
    assert summary["dark_components"]["largest"] == 10
    assert summary["dark_components"]["n_components"] == 1


def test_csv_has_a_row_for_every_junction(synthetic_lattice, tmp_path):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    csv_path = write_junction_csv(tmp_path / "nested" / "junctions.csv", result)

    lines = csv_path.read_text().splitlines()
    assert lines[0] == CSV_HEADER
    assert len(lines) == result.n_junctions + 1

    flagged = [line for line in lines[1:] if line.split(",")[6] == "1"]
    assert len(flagged) == 1
    # Every junction carries the two source entries it was merged from.
    assert all(len(line.split(",")[-1].split()) == 2 for line in lines[1:])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slice_overlay_renders_on_every_axis(synthetic_lattice, tmp_path, axis):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    output_path = tmp_path / f"slice_axis{axis}.png"
    index = save_slice_overlay_with_radius(
        synthetic_lattice.volume, result, output_path, axis=axis
    )

    assert 0 <= index < synthetic_lattice.volume.shape[axis]
    assert output_path.stat().st_size > 0


def test_slice_overlay_rejects_an_out_of_range_index(synthetic_lattice, tmp_path):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    with pytest.raises(ValueError, match="index must be between"):
        save_slice_overlay_with_radius(
            synthetic_lattice.volume, result, tmp_path / "x.png", axis=0, index=9999
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_mip_overlay_renders_on_every_axis(synthetic_lattice, tmp_path, axis):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path
    )
    output_path = save_mip_overlay_with_status(
        synthetic_lattice.volume, result, tmp_path / f"mip_axis{axis}.png", axis=axis
    )
    assert output_path.stat().st_size > 0


def test_marked_tiff_paints_the_candidate_red(synthetic_lattice, tmp_path):
    result = scan_junctions(
        synthetic_lattice.volume, synthetic_lattice.registered_json_path, radius=3
    )
    tiff_path = save_marked_tiff(
        synthetic_lattice.volume, result, tmp_path / "marked.tif"
    )

    marked = tifffile.imread(tiff_path)
    assert marked.shape == synthetic_lattice.volume.shape + (3,)
    x, y, z = synthetic_lattice.dark_position_xyz
    assert marked[z, y, x].tolist() == [255, 0, 0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", []), ("  ", []), ("513,2682", [513, 2682]), ("513 2682", [513, 2682])],
)
def test_parse_junction_ids(text, expected):
    assert list(parse_junction_ids(text)) == expected
