"""Tests for the capped-cylinder strut detector in ``strut_scan``."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conftest import (  # noqa: E402
    BACKGROUND_LEVEL,
    MATERIAL_LEVEL,
    STRUT_LATTICE_RADIUS,
    make_synthetic_lattice,
)
from junction_scan import scan_junctions  # noqa: E402
from strut_scan import (  # noqa: E402
    CSV_HEADER,
    STATUS_MISSING,
    STATUS_PARTIAL,
    STATUS_PRESENT,
    scan_struts,
    score_strut,
    segment_bounds,
    summarize_strut_scan,
    write_strut_csv,
)

# Large enough to clear the fixture's 2.5-voxel junction blobs, small enough to
# leave most of the 40-voxel span scored. The bleed threshold is measured
# directly in ``test_the_cap_is_what_clears_junction_bleed``.
CAP = 8


def _scan(lattice, radius=STRUT_LATTICE_RADIUS, cap_voxels=CAP, n_bisections=2):
    return scan_struts(
        lattice.volume,
        lattice.registered_json_path,
        radius=radius,
        cap_voxels=cap_voxels,
        n_bisections=n_bisections,
    )


def _column(lines: list[str], name: str) -> list[str]:
    """Read one CSV column by header name, so adding columns cannot break a test."""
    index = lines[0].split(",").index(name)
    return [line.split(",")[index] for line in lines[1:]]


@pytest.mark.parametrize("n_bisections, expected", [(1, 1), (2, 3), (3, 7)])
def test_segments_tile_the_capped_span(n_bisections, expected):
    """N bisections leave 2^N - 1 abutting segments between the two caps."""
    lower, upper = segment_bounds(np.array([100.0]), cap_voxels=10, n_bisections=n_bisections)

    assert lower.shape == upper.shape == (1, expected)
    # The caps are excluded and nothing between them is: the segments start at
    # the cap, end at the far cap, and each one's end is the next one's start.
    assert lower[0, 0] == pytest.approx(10.0)
    assert upper[0, -1] == pytest.approx(90.0)
    np.testing.assert_allclose(lower[0, 1:], upper[0, :-1])
    np.testing.assert_allclose(np.diff(lower[0]), 80.0 / expected)


def test_segment_bounds_reject_a_cap_that_leaves_no_span():
    with pytest.raises(ValueError, match="leaves no span to score on strut"):
        segment_bounds(np.array([40.0, 30.0]), cap_voxels=15, n_bisections=2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_bisections": 0}, "n_bisections must be at least 1"),
        ({"cap_voxels": -1}, "cap_voxels must not be negative"),
    ],
)
def test_segment_bounds_reject_invalid_parameters(kwargs, message):
    arguments = {"cap_voxels": 10, "n_bisections": 2, **kwargs}
    with pytest.raises(ValueError, match=message):
        segment_bounds(np.array([100.0]), **arguments)


def test_an_absent_strut_reads_missing_while_its_junctions_stay_bright(strut_lattice):
    """The defect only a strut-level probe can see.

    Both endpoint junctions keep their other five struts, so the junction scan
    finds nothing here. That asymmetry is the whole reason this module exists.
    """
    result = _scan(strut_lattice)

    assert np.flatnonzero(result.missing).tolist() == [strut_lattice.missing_strut_id]
    assert result.n_dark_segments[strut_lattice.missing_strut_id] == result.n_segments

    junctions = scan_junctions(
        strut_lattice.volume, strut_lattice.registered_json_path, radius=4
    )
    assert not junctions.dark.any()


def test_a_part_painted_strut_reads_partial(strut_lattice):
    """Material over part of the span and none over the rest is its own class.

    This is the sensitivity the capped cylinder exists to restore: the sphere
    probe it replaced could only sample the midpoint, where this strut is dark,
    so it reported the same class for a break and a total absence.
    """
    result = _scan(strut_lattice)
    strut = strut_lattice.partial_strut_id

    assert np.flatnonzero(result.partial).tolist() == [strut]
    assert 0 < result.n_dark_segments[strut] < result.n_segments
    # The painted end is the one the segment order starts from, so the dark
    # segments are the far ones.
    assert result.segment_dark[strut].tolist() == [False, True, True]
    assert result.status[strut] == STATUS_PARTIAL
    assert result.status[strut_lattice.missing_strut_id] == STATUS_MISSING


def test_the_cap_is_what_clears_junction_bleed(strut_lattice):
    """The claim the whole shape rests on, measured directly.

    An absent strut still has its two junction blobs, and a segment reaching
    into one is lit by it -- so the strut reads *partial*, exactly as if it were
    broken. The sphere probe had no parameter for this: its radius had to be
    wide enough for the registration drift, which pushed its outer sample points
    into the blobs. Here the cap trims along the axis while the radius keeps
    working perpendicular to it, so the bleed can be cleared without narrowing
    the probe.
    """
    strut = strut_lattice.missing_strut_id
    bleeding = _scan(strut_lattice, cap_voxels=1)
    clear = _scan(strut_lattice, cap_voxels=CAP)

    assert bleeding.status[strut] == STATUS_PARTIAL
    assert bleeding.segment_dark[strut].tolist() == [False, True, False]
    assert clear.status[strut] == STATUS_MISSING
    # And the radius is not what fixed it -- the same bleed is there at every
    # radius the drift would need.
    for radius in (2, 3, 4):
        assert _scan(strut_lattice, radius=radius, cap_voxels=1).status[strut] == (
            STATUS_PARTIAL
        )


def test_too_large_a_cap_hides_the_material_it_should_find(strut_lattice):
    """The failure mode at the other end, which is why the cap is not free.

    The part-painted strut carries material over its first 20% only. A cap wide
    enough to swallow that stretch leaves nothing but empty span to score, and
    the strut is promoted from partial to missing -- a break reported as a total
    absence.
    """
    strut = strut_lattice.partial_strut_id

    assert _scan(strut_lattice, cap_voxels=CAP).status[strut] == STATUS_PARTIAL
    assert _scan(strut_lattice, cap_voxels=12).status[strut] == STATUS_MISSING


def test_a_clean_lattice_flags_nothing(tmp_path):
    lattice = make_synthetic_lattice(
        tmp_path, pitch=40, origin=12, size=104, dark_cell=None
    )
    result = _scan(lattice)

    assert not result.missing.any()
    assert not result.partial.any()
    assert set(result.status) == {STATUS_PRESENT}


def test_threshold_is_the_volume_otsu(strut_lattice):
    result = _scan(strut_lattice)
    assert BACKGROUND_LEVEL < result.threshold < MATERIAL_LEVEL


def test_one_bisection_scores_the_whole_span_as_one_segment(strut_lattice):
    """At one bisection the probe is the bare capped cylinder, and partial cannot
    arise -- the same blind spot the sphere probe had, reached deliberately."""
    result = _scan(strut_lattice, n_bisections=1)

    assert result.n_segments == 1
    np.testing.assert_allclose(
        result.segment_centers_xyz[:, 0, :], result.centers_xyz, atol=1e-9
    )
    assert not result.partial.any()
    assert set(result.status) == {STATUS_PRESENT, STATUS_MISSING}


def test_subdividing_the_span_subtracts_from_missing_and_finds_breaks(strut_lattice):
    """What the segment count changes, in both directions.

    ``missing`` means *every* segment dark, so subdividing can only take struts
    out of it -- which is what makes one segment count comparable to another.
    Subdividing can, however, *add* flagged struts, and that is the whole reason
    to do it: a maximum over one long segment saturates on whatever material the
    strut still has. The part-painted strut keeps material over its first fifth
    and reads **present** at one segment; three segments find the empty rest of
    the span and call it **partial**.
    """
    coarse = _scan(strut_lattice, n_bisections=1)
    fine = _scan(strut_lattice, n_bisections=2)
    partial_strut = strut_lattice.partial_strut_id

    assert set(np.flatnonzero(fine.missing)) <= set(np.flatnonzero(coarse.missing))
    assert coarse.status[partial_strut] == STATUS_PRESENT
    assert fine.status[partial_strut] == STATUS_PARTIAL
    # The wholly absent strut is found either way; only the break needs the
    # extra segments.
    assert coarse.status[strut_lattice.missing_strut_id] == STATUS_MISSING
    assert fine.status[strut_lattice.missing_strut_id] == STATUS_MISSING


def test_score_strut_matches_the_full_scan(strut_lattice):
    """The single-element path an inspection tool uses must not measure something
    else than the scan whose candidate it is inspecting."""
    result = _scan(strut_lattice)
    strut = strut_lattice.missing_strut_id
    first, second = result.strut_junction_ids[strut]

    intensities, lower, upper = score_strut(
        strut_lattice.volume,
        result.positions_xyz[first],
        result.positions_xyz[second],
        radius=STRUT_LATTICE_RADIUS,
        cap_voxels=CAP,
        n_bisections=2,
    )

    np.testing.assert_array_equal(intensities, result.intensities[strut])
    assert lower[0] == pytest.approx(CAP)
    np.testing.assert_allclose(
        np.asarray(result.positions_xyz[first])
        + ((lower + upper) / 2.0)[:, None]
        * (result.positions_xyz[second] - result.positions_xyz[first])
        / np.linalg.norm(result.positions_xyz[second] - result.positions_xyz[first]),
        result.segment_centers_xyz[strut],
    )


def test_summary_is_json_serializable_and_reports_both_classes(strut_lattice):
    result = _scan(strut_lattice)
    summary = summarize_strut_scan(result)

    assert json.loads(json.dumps(summary)) == summary
    assert summary["n_struts"] == strut_lattice.n_struts
    assert summary["n_segments_per_strut"] == 3
    assert summary["cap_voxels"] == CAP
    assert summary["n_missing"] == 1
    assert summary["n_partial"] == 1

    by_id = {candidate["strut_id"]: candidate for candidate in summary["candidates"]}
    assert set(by_id) == {
        strut_lattice.missing_strut_id,
        strut_lattice.partial_strut_id,
    }
    missing = by_id[strut_lattice.missing_strut_id]
    assert missing["status"] == STATUS_MISSING
    assert len(missing["segment_intensities"]) == 3
    # Which segments were dark is what separates a break near one end from one
    # in the middle, so it has to survive into the summary.
    assert by_id[strut_lattice.partial_strut_id]["segment_dark"] == [False, True, True]
    # Endpoint junction ids are what lets a caller attribute a flagged strut to
    # a systematic region the junction scan found.
    assert missing["junction0"] != missing["junction1"]


def test_summary_bands_partition_the_struts(strut_lattice):
    result = _scan(strut_lattice)
    summary = summarize_strut_scan(result, n_bands=3)

    for axis in ("x", "y", "z"):
        bands = summary["missing_fraction_by_band"][axis]
        assert len(bands) == 3
        assert sum(band["n_total"] for band in bands) == summary["n_struts"]
        assert sum(band["n_flagged"] for band in bands) == summary["n_missing"]
        assert (
            sum(
                band["n_flagged"]
                for band in summary["partial_fraction_by_band"][axis]
            )
            == summary["n_partial"]
        )


def test_csv_has_a_row_for_every_strut(strut_lattice, tmp_path):
    result = _scan(strut_lattice)
    csv_path = write_strut_csv(tmp_path / "nested" / "struts.csv", result)

    lines = csv_path.read_text().splitlines()
    assert lines[0] == CSV_HEADER
    assert len(lines) == result.n_struts + 1

    status = _column(lines, "status")
    assert status.count(STATUS_MISSING) == 1
    assert status.count(STATUS_PARTIAL) == 1
    assert status[strut_lattice.missing_strut_id] == STATUS_MISSING
    assert _column(lines, "n_segments") == ["3"] * result.n_struts


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"radius": 0}, "radius must be a positive"),
        ({"n_bisections": 0}, "n_bisections must be at least 1"),
        ({"cap_voxels": 25}, "leaves no span to score"),
    ],
)
def test_scan_rejects_invalid_parameters(strut_lattice, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _scan(strut_lattice, **kwargs)


def test_scan_rejects_a_non_volumetric_array(strut_lattice):
    with pytest.raises(ValueError, match="expected a 3D volume"):
        scan_struts(
            strut_lattice.volume[0], strut_lattice.registered_json_path, radius=2
        )
