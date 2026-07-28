"""Tests for the sphere-based strut detector in ``strut_scan``."""

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
    strut_sample_points,
    summarize_strut_scan,
    write_strut_csv,
)


def _scan(lattice, radius=STRUT_LATTICE_RADIUS, n_bisections=2):
    return scan_struts(
        lattice.volume,
        lattice.registered_json_path,
        radius=radius,
        n_bisections=n_bisections,
    )


def _column(lines: list[str], name: str) -> list[str]:
    """Read one CSV column by header name, so adding columns cannot break a test."""
    index = lines[0].split(",").index(name)
    return [line.split(",")[index] for line in lines[1:]]


@pytest.mark.parametrize(
    "n_bisections, expected",
    [
        (1, [0.5]),
        (2, [0.25, 0.5, 0.75]),
        (3, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]),
    ],
)
def test_sample_points_sit_at_recursive_halves(n_bisections, expected):
    """N halvings leave 2^N - 1 interior points at k/2^N of the span."""
    start = np.array([[0.0, 0.0, 0.0]])
    end = np.array([[8.0, 0.0, 0.0]])
    points = strut_sample_points(start, end, n_bisections)

    assert points.shape == (1, 2**n_bisections - 1, 3)
    # The endpoints themselves are never sampled: a junction blob stays bright
    # when one of its struts is absent, so a probe there answers the wrong
    # question.
    assert points[0, :, 0].tolist() == [8.0 * f for f in expected]
    assert not np.isclose(points, 0.0).all(axis=-1).any()


def test_sample_points_reject_zero_bisections():
    with pytest.raises(ValueError, match="n_bisections must be at least 1"):
        strut_sample_points(np.zeros((1, 3)), np.ones((1, 3)), 0)


def test_an_absent_strut_reads_missing_while_its_junctions_stay_bright(strut_lattice):
    """The defect only a strut-level probe can see.

    Both endpoint junctions keep their other five struts, so the junction scan
    finds nothing here. That asymmetry is the whole reason this module exists.
    """
    result = _scan(strut_lattice)

    assert np.flatnonzero(result.missing).tolist() == [strut_lattice.missing_strut_id]
    assert result.n_dark_points[strut_lattice.missing_strut_id] == result.n_points

    junctions = scan_junctions(
        strut_lattice.volume, strut_lattice.registered_json_path, radius=4
    )
    assert not junctions.dark.any()


def test_a_part_painted_strut_reads_partial(strut_lattice):
    """Material over part of the span and none over the rest is its own class."""
    result = _scan(strut_lattice)
    strut = strut_lattice.partial_strut_id

    assert np.flatnonzero(result.partial).tolist() == [strut]
    assert 0 < result.n_dark_points[strut] < result.n_points
    # The painted end is the one the sampling order starts from, so the dark
    # points are the far ones.
    assert result.point_dark[strut].tolist() == [False, True, True]
    assert result.status[strut] == STATUS_PARTIAL
    assert result.status[strut_lattice.missing_strut_id] == STATUS_MISSING


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


def test_the_default_samples_the_midpoint_alone(strut_lattice):
    """One bisection is the default because it is the only placement that keeps
    the probe clear of the junctions at a radius wide enough for the drift.

    With a single point the partial class cannot arise at all, which is the
    sensitivity this default gives up.
    """
    result = scan_struts(
        strut_lattice.volume, strut_lattice.registered_json_path, radius=3
    )

    assert result.n_bisections == 1
    assert result.n_points == 1
    np.testing.assert_allclose(result.sample_xyz[:, 0, :], result.centers_xyz)
    assert not result.partial.any()
    assert set(result.status) == {STATUS_PRESENT, STATUS_MISSING}


def test_more_bisections_cannot_add_missing_struts(strut_lattice):
    """Missing means *every* point dark, so refining the sampling only subtracts.

    This is what makes a bisection count comparable to a coarser one: the
    missing set shrinks monotonically, and anything it drops moves into the
    partial class rather than disappearing.
    """
    coarse = _scan(strut_lattice, n_bisections=1)
    fine = _scan(strut_lattice, n_bisections=2)

    coarse_missing = set(np.flatnonzero(coarse.missing).tolist())
    fine_missing = set(np.flatnonzero(fine.missing).tolist())
    fine_flagged = fine_missing | set(np.flatnonzero(fine.partial).tolist())

    assert fine_missing <= coarse_missing
    # And this is what the extra points buy. The part-painted strut is dark at
    # its midpoint, so a single bisection calls it wholly missing; two points
    # more find the material near one end and demote it to partial.
    assert strut_lattice.partial_strut_id in coarse_missing
    assert strut_lattice.partial_strut_id not in fine_missing
    assert strut_lattice.partial_strut_id in fine_flagged


def test_summary_is_json_serializable_and_reports_both_classes(strut_lattice):
    result = _scan(strut_lattice)
    summary = summarize_strut_scan(result)

    assert json.loads(json.dumps(summary)) == summary
    assert summary["n_struts"] == strut_lattice.n_struts
    assert summary["n_points_per_strut"] == 3
    assert summary["n_missing"] == 1
    assert summary["n_partial"] == 1

    by_id = {candidate["strut_id"]: candidate for candidate in summary["candidates"]}
    assert set(by_id) == {
        strut_lattice.missing_strut_id,
        strut_lattice.partial_strut_id,
    }
    missing = by_id[strut_lattice.missing_strut_id]
    assert missing["status"] == STATUS_MISSING
    assert len(missing["point_intensities"]) == 3
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
    assert _column(lines, "n_points") == ["3"] * result.n_struts


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"radius": 0}, "radius must be a positive"),
        ({"n_bisections": 0}, "n_bisections must be at least 1"),
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
