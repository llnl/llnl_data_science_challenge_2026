"""Tests for the per-element projection views in ``element_views``.

These are pictures, so what a test can check is that the right files appear and
that the cropping arithmetic survives an element sitting against the edge of the
volume -- the case where a padded bounding box runs off the array and silently
produces an empty crop or a negative slice.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conftest import STRUT_LATTICE_RADIUS  # noqa: E402
from element_views import (  # noqa: E402
    AXIS_NAMES,
    save_junction_views,
    save_strut_views,
)
from strut_scan import scan_struts, segment_bounds  # noqa: E402

CAP = 8


def test_junction_views_write_one_image_per_axis(strut_lattice, tmp_path):
    paths = save_junction_views(
        strut_lattice.volume,
        np.array([52.0, 52.0, 52.0]),
        radius=STRUT_LATTICE_RADIUS,
        output_dir=tmp_path / "views",
        stem="junction_7_r3",
        dark=False,
    )

    assert [path.name for path in paths] == [
        f"junction_7_r3_proj_{axis}.png" for axis in AXIS_NAMES
    ]
    assert all(path.stat().st_size > 0 for path in paths)


def test_strut_views_write_one_image_per_axis(strut_lattice, tmp_path):
    result = scan_struts(
        strut_lattice.volume,
        strut_lattice.registered_json_path,
        radius=STRUT_LATTICE_RADIUS,
        cap_voxels=CAP,
        n_bisections=2,
    )
    strut = strut_lattice.missing_strut_id
    first, second = result.strut_junction_ids[strut]
    start, end = result.positions_xyz[first], result.positions_xyz[second]
    lower, upper = segment_bounds(
        np.array([float(np.linalg.norm(end - start))]), CAP, 2
    )

    paths = save_strut_views(
        strut_lattice.volume,
        start,
        end,
        radius=STRUT_LATTICE_RADIUS,
        cap_voxels=CAP,
        lower=lower[0],
        upper=upper[0],
        output_dir=tmp_path / "views",
        stem=f"strut_{strut}_r3_c8",
        segment_dark=result.segment_dark[strut],
    )

    assert [path.name for path in paths] == [
        f"strut_{strut}_r3_c8_proj_{axis}.png" for axis in AXIS_NAMES
    ]
    assert all(path.stat().st_size > 0 for path in paths)


@pytest.mark.parametrize(
    "position",
    [
        (0.0, 0.0, 0.0),
        (103.0, 103.0, 103.0),
        (0.0, 52.0, 103.0),
    ],
)
def test_a_junction_against_the_volume_edge_still_renders(
    strut_lattice, tmp_path, position
):
    """The padded box runs off the array at every corner of the specimen, and a
    lattice's outermost junctions are exactly where a real absence is likeliest.
    """
    paths = save_junction_views(
        strut_lattice.volume,
        np.array(position),
        radius=STRUT_LATTICE_RADIUS,
        output_dir=tmp_path / "views",
        stem="edge",
        dark=True,
    )

    assert all(path.stat().st_size > 0 for path in paths)


def test_strut_views_reject_coincident_endpoints(strut_lattice, tmp_path):
    with pytest.raises(ValueError, match="coincident"):
        save_strut_views(
            strut_lattice.volume,
            np.array([20.0, 20.0, 20.0]),
            np.array([20.0, 20.0, 20.0]),
            radius=STRUT_LATTICE_RADIUS,
            cap_voxels=CAP,
            lower=np.array([8.0]),
            upper=np.array([32.0]),
            output_dir=tmp_path / "views",
            stem="degenerate",
        )
