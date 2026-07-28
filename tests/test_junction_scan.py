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
    save_marked_tiff,
    save_mip_overlay_with_status,
    save_slice_overlay_with_radius,
    scan_junctions,
    summarize_scan,
    write_junction_csv,
)


def _scan(lattice, **kwargs):
    """Scan a fixture lattice; only the extras vary between tests."""
    return scan_junctions(lattice.volume, lattice.registered_json_path, **kwargs)


def _junction_at(result, position_xyz) -> int:
    """Merged ids come out of a positional sort, so look junctions up by voxel."""
    matches = np.flatnonzero((result.voxel_xyz == np.array(position_xyz)).all(axis=1))
    assert len(matches) == 1, f"expected one junction at {position_xyz}, got {matches}"
    return int(matches[0])


def _column(lines: list[str], name: str) -> list[str]:
    """Read one CSV column by header name, so adding columns cannot break a test."""
    index = lines[0].split(",").index(name)
    return [line.split(",")[index] for line in lines[1:]]


def test_scan_merges_entries_and_flags_only_the_dark_junction(synthetic_lattice):
    result = _scan(synthetic_lattice)

    assert result.n_junctions == synthetic_lattice.n_junctions == 27
    assert len(result.entry_to_junction) == synthetic_lattice.n_entries == 54
    # A 3x3x3 grid with 6-connectivity runs from a 3-strut corner to a 6-strut
    # center. Unmerged, every entry would carry half of that.
    assert (result.degree.min(), result.degree.max()) == (3, 6)

    candidates = np.flatnonzero(result.dark)
    assert len(candidates) == 1
    assert candidates[0] == _junction_at(result, synthetic_lattice.dark_position_xyz)


def test_threshold_is_the_volume_otsu_not_the_nodes(synthetic_lattice):
    result = _scan(synthetic_lattice)
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

    tight = _scan(lattice, radius=1)
    assert tight.dark.all()

    forgiving = _scan(lattice, radius=6)
    assert not forgiving.dark.any()


@pytest.mark.parametrize("radius", [0, -3])
def test_non_positive_radius_raises(synthetic_lattice, radius):
    with pytest.raises(ValueError, match="radius must be a positive"):
        _scan(synthetic_lattice, radius=radius)


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
    result = _scan(synthetic_lattice)
    summary = summarize_scan(result)

    assert json.loads(json.dumps(summary)) == summary
    assert summary["n_junctions"] == 27
    assert summary["n_entries"] == 54
    assert summary["n_dark"] == 1
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


def test_summary_bands_partition_the_junctions(synthetic_lattice):
    result = _scan(synthetic_lattice)
    summary = summarize_scan(result, n_bands=3)

    for axis in ("x", "y", "z"):
        bands = summary["dark_fraction_by_band"][axis]
        assert len(bands) == 3
        assert sum(band["n_total"] for band in bands) == summary["n_junctions"]
        assert sum(band["n_flagged"] for band in bands) == summary["n_dark"]

    octants = summary["dark_fraction_by_octant"]
    assert len(octants) == 8
    assert sum(cell["n_junctions"] for cell in octants) == summary["n_junctions"]


def test_summary_bands_localize_a_missing_face(synthetic_lattice):
    """A whole dark face should land in one y band as a single component."""
    result = _scan(synthetic_lattice)
    # Force the maximum-y layer dark, standing in for a machined-off face.
    max_y = result.positions_xyz[:, 1].max()
    forced = result.dark | (result.positions_xyz[:, 1] >= max_y - 0.5)
    faced = replace(result, dark=forced)

    summary = summarize_scan(faced, n_bands=3)
    y_bands = summary["dark_fraction_by_band"]["y"]
    assert y_bands[-1]["flagged_fraction"] == 1.0
    # The nine face junctions plus the pre-existing dark one, which is a strut
    # away from the face and so joins its component -- a real property of the
    # statistic worth pinning: an isolated defect touching a systematic region
    # is absorbed into it and stops looking isolated.
    assert summary["dark_components"]["largest"] == 10
    assert summary["dark_components"]["n_components"] == 1


def test_csv_has_a_row_for_every_junction(synthetic_lattice, tmp_path):
    result = _scan(synthetic_lattice)
    csv_path = write_junction_csv(tmp_path / "nested" / "junctions.csv", result)

    lines = csv_path.read_text().splitlines()
    assert lines[0] == CSV_HEADER
    assert len(lines) == result.n_junctions + 1

    assert _column(lines, "dark").count("1") == 1
    # Every junction carries the two source entries it was merged from.
    assert all(len(entries.split()) == 2 for entries in _column(lines, "entry_ids"))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slice_overlay_renders_on_every_axis(synthetic_lattice, tmp_path, axis):
    result = _scan(synthetic_lattice)
    output_path = tmp_path / f"slice_axis{axis}.png"
    index = save_slice_overlay_with_radius(
        synthetic_lattice.volume, result, output_path, axis=axis
    )

    assert 0 <= index < synthetic_lattice.volume.shape[axis]
    assert output_path.stat().st_size > 0


def test_slice_overlay_rejects_an_out_of_range_index(synthetic_lattice, tmp_path):
    result = _scan(synthetic_lattice)
    with pytest.raises(ValueError, match="index must be between"):
        save_slice_overlay_with_radius(
            synthetic_lattice.volume, result, tmp_path / "x.png", axis=0, index=9999
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_mip_overlay_renders_on_every_axis(synthetic_lattice, tmp_path, axis):
    result = _scan(synthetic_lattice)
    output_path = save_mip_overlay_with_status(
        synthetic_lattice.volume, result, tmp_path / f"mip_axis{axis}.png", axis=axis
    )
    assert output_path.stat().st_size > 0


def test_marked_tiff_outlines_the_candidate_without_covering_it(
    synthetic_lattice, tmp_path
):
    """The marker is a shell, not a ball.

    This stack exists to answer whether there is material where a candidate
    sits, and a solid marker answers it by painting over it. So the wall is
    drawn at the sampling radius -- bounding exactly the voxels the flag came
    from -- and everything inside keeps its original grayscale.
    """
    radius, thickness = 3, 2
    result = _scan(synthetic_lattice, radius=radius)
    tiff_path = save_marked_tiff(
        synthetic_lattice.volume, result, tmp_path / "marked.tif", thickness=thickness
    )

    marked = tifffile.imread(tiff_path)
    assert marked.shape == synthetic_lattice.volume.shape + (3,)
    x, y, z = synthetic_lattice.dark_position_xyz

    # On the wall: painted. At radius 3 with a 2-voxel wall, offsets whose
    # squared distance falls in (1, 9] are the shell.
    assert marked[z, y, x + radius].tolist() == [255, 0, 0]
    assert marked[z + radius, y, x].tolist() == [255, 0, 0]

    # Inside it: untouched, so the voxels the reader came to look at survive.
    for offset in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        interior = marked[z + offset[2], y + offset[1], x + offset[0]]
        assert interior[0] == interior[1] == interior[2], (
            f"voxel at offset {offset} was painted over; it is inside the shell"
        )
