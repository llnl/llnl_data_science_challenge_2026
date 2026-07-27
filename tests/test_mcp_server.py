import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import mcp_server  # noqa: E402
from scipy.sparse import load_npz  # noqa: E402
from volume_io import convert_tiff_to_numpy  # noqa: E402


def test_convert_tiff_to_numpy_preserves_3d_volume(tmp_path: Path) -> None:
    input_path = tmp_path / "volume.tif"
    output_path = tmp_path / "nested" / "volume.npy"
    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    tifffile.imwrite(input_path, volume)

    result = mcp_server.convert_tiff_volume(str(input_path), str(output_path))

    assert result == f"NumPy volume saved to {output_path}"
    converted = np.load(output_path, allow_pickle=False)
    assert converted.dtype == volume.dtype
    np.testing.assert_array_equal(converted, volume)
    assert convert_tiff_to_numpy(input_path, output_path) == output_path


def test_convert_tiff_to_numpy_rejects_non_3d_tiff(tmp_path: Path) -> None:
    input_path = tmp_path / "slice.tif"
    tifffile.imwrite(input_path, np.zeros((3, 4), dtype=np.uint16))

    result = mcp_server.convert_tiff_volume(str(input_path), str(tmp_path / "slice.npy"))

    assert result.startswith("Error converting TIFF volume:")
    assert "expected a 3D TIFF volume" in result


def test_segment_ct_dataset_thresholds_and_creates_parent(tmp_path: Path) -> None:
    input_path = tmp_path / "volume.npy"
    output_path = tmp_path / "nested" / "mask.npy"
    volume = np.array(
        [[[-1.0, 0.5], [1.0, 2.0]], [[0.49, 0.5], [3.0, -2.0]]],
        dtype=np.float32,
    )
    np.save(input_path, volume)

    result = mcp_server.segment_ct_dataset(
        str(input_path), str(output_path), threshold=0.5
    )

    assert result == f"Segmentation saved to {output_path}"
    mask = np.load(output_path, allow_pickle=False)
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, (volume >= 0.5).astype(np.uint8))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_visualize_slice_creates_image(tmp_path: Path, axis: int) -> None:
    input_path = tmp_path / "volume.npy"
    output_path = tmp_path / f"images/axis-{axis}.png"
    np.save(input_path, np.arange(24).reshape(2, 3, 4))

    result = mcp_server.visualize_slice(
        str(input_path), str(output_path), slice_index=0, axis=axis
    )

    assert result == f"Slice visualization saved to {output_path}"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_skeletonize_creates_boolean_skeleton(tmp_path: Path) -> None:
    input_path = tmp_path / "mask.npy"
    output_path = tmp_path / "nested" / "skeleton.npy"
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[1:6, 3, 3] = 1
    np.save(input_path, mask)

    result = mcp_server.skeletonize(str(input_path), str(output_path))

    assert result == f"Skeleton saved to {output_path}"
    skeleton = np.load(output_path, allow_pickle=False)
    assert skeleton.dtype == np.bool_
    assert np.count_nonzero(skeleton) > 0


def test_skeletonize_does_not_write_to_standard_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Tool helpers must keep stdout exclusively available for MCP JSON-RPC."""
    input_path = tmp_path / "mask.npy"
    output_path = tmp_path / "skeleton.npy"
    np.save(input_path, np.ones((3, 3, 3), dtype=np.uint8))

    result = mcp_server.skeletonize(str(input_path), str(output_path))

    captured = capsys.readouterr()
    assert result == f"Skeleton saved to {output_path}"
    assert captured.out == ""


def test_graph_skeleton_saves_sparse_graph_coordinates_and_histogram(
    tmp_path: Path,
) -> None:
    skeleton_path = tmp_path / "skeleton.npy"
    graph_path = tmp_path / "graphs" / "skeleton_graph.npz"
    histogram_path = tmp_path / "graphs" / "node_degree_histogram.csv"
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[1:4, 2, 2] = True
    np.save(skeleton_path, skeleton)

    result = mcp_server.graph_skeleton(
        str(skeleton_path), str(graph_path), str(histogram_path)
    )

    assert result.startswith(f"Skeleton graph saved to {graph_path} with 3 nodes and 2 edges")
    graph = load_npz(graph_path)
    assert graph.shape == (3, 3)
    assert graph.nnz == 4
    coordinates = np.load(graph_path.with_suffix(".coordinates.npy"), allow_pickle=False)
    np.testing.assert_array_equal(coordinates, [[1, 2, 2], [2, 2, 2], [3, 2, 2]])
    assert histogram_path.read_text() == "degree,node_count\n1,2\n2,1\n"


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        ("segment", ("missing.npy", "output.npy", 1.0), "input file not found"),
        ("segment_nan", (), "threshold must be a finite number"),
        ("visualize_axis", (), "axis must be 0, 1, or 2"),
        ("visualize_index", (), "slice_index must be between"),
        ("visualize_negative", (), "slice_index must be between"),
        ("skeletonize_2d", (), "expected a 3D array"),
    ],
)
def test_invalid_inputs_return_errors(
    tmp_path: Path, tool: str, arguments: tuple[object, ...], message: str
) -> None:
    volume_path = tmp_path / "volume.npy"
    array_2d_path = tmp_path / "array-2d.npy"
    np.save(volume_path, np.zeros((2, 3, 4), dtype=np.float32))
    np.save(array_2d_path, np.zeros((3, 4), dtype=np.uint8))

    if tool == "segment":
        result = mcp_server.segment_ct_dataset(*arguments)
    elif tool == "segment_nan":
        result = mcp_server.segment_ct_dataset(
            str(volume_path), str(tmp_path / "output.npy"), float("nan")
        )
    elif tool == "visualize_axis":
        result = mcp_server.visualize_slice(
            str(volume_path), str(tmp_path / "slice.png"), 0, axis=3
        )
    elif tool == "visualize_index":
        result = mcp_server.visualize_slice(
            str(volume_path), str(tmp_path / "slice.png"), 2, axis=0
        )
    elif tool == "visualize_negative":
        result = mcp_server.visualize_slice(
            str(volume_path), str(tmp_path / "slice.png"), -1, axis=0
        )
    else:
        result = mcp_server.skeletonize(
            str(array_2d_path), str(tmp_path / "skeleton.npy")
        )

    assert result.startswith("Error")
    assert message in result


def test_invalid_or_malformed_npy_paths_return_errors(tmp_path: Path) -> None:
    wrong_extension = tmp_path / "volume.txt"
    malformed = tmp_path / "malformed.npy"
    wrong_extension.write_text("not an array")
    malformed.write_text("not an array")

    extension_result = mcp_server.segment_ct_dataset(
        str(wrong_extension), str(tmp_path / "mask.npy"), 1.0
    )
    malformed_result = mcp_server.segment_ct_dataset(
        str(malformed), str(tmp_path / "mask.npy"), 1.0
    )

    assert "input file must have a .npy extension" in extension_result
    assert malformed_result.startswith("Error segmenting CT dataset:")


def test_scan_lattice_junctions_reports_the_dark_junction(
    synthetic_lattice, tmp_path: Path
) -> None:
    output_dir = tmp_path / "scan"
    result = mcp_server.scan_lattice_junctions(
        str(synthetic_lattice.volume_path),
        str(synthetic_lattice.registered_json_path),
        str(output_dir),
        radius=8,
    )

    assert "Error" not in result
    assert "27 junctions (merged from 54 JSON entries, degree 3-6)" in result
    assert "1 dark of 27 junctions" in result
    assert "Largest dark component: 1 junction(s) across 1 component(s)." in result

    csv_path = output_dir / "junction_scan_r8.csv"
    summary_path = output_dir / "junction_scan_r8_summary.json"
    assert str(csv_path) in result and str(summary_path) in result

    summary = json.loads(summary_path.read_text())
    assert summary["n_dark"] == 1
    assert len(csv_path.read_text().splitlines()) == 28


def test_scan_lattice_junctions_accepts_a_tiff_volume(
    synthetic_lattice, tmp_path: Path
) -> None:
    tiff_path = tmp_path / "volume.tif"
    tifffile.imwrite(tiff_path, synthetic_lattice.volume)

    result = mcp_server.scan_lattice_junctions(
        str(tiff_path),
        str(synthetic_lattice.registered_json_path),
        str(tmp_path / "scan"),
    )
    assert "1 dark of 27 junctions" in result


def test_scan_lattice_junctions_tags_outputs_by_radius(
    synthetic_lattice, tmp_path: Path
) -> None:
    """A radius sweep must not overwrite its own earlier passes."""
    output_dir = tmp_path / "scan"
    for radius in (4, 8):
        mcp_server.scan_lattice_junctions(
            str(synthetic_lattice.volume_path),
            str(synthetic_lattice.registered_json_path),
                str(output_dir),
            radius=radius,
        )

    assert {path.name for path in output_dir.glob("*.csv")} == {
        "junction_scan_r4.csv",
        "junction_scan_r8.csv",
    }


def test_scan_lattice_junctions_writes_a_marked_tiff(
    synthetic_lattice, tmp_path: Path
) -> None:
    output_dir = tmp_path / "scan"
    result = mcp_server.scan_lattice_junctions(
        str(synthetic_lattice.volume_path),
        str(synthetic_lattice.registered_json_path),
        str(output_dir),
        radius=3,
        write_marked_tiff=True,
    )

    marked_path = output_dir / "junction_scan_r3_marked.tif"
    assert str(marked_path) in result
    assert tifffile.imread(marked_path).shape == synthetic_lattice.volume.shape + (3,)


@pytest.mark.parametrize("mode", ["slice", "mip"])
def test_visualize_junction_overlay_renders(
    synthetic_lattice, tmp_path: Path, mode: str
) -> None:
    output_path = tmp_path / f"overlay_{mode}.png"
    result = mcp_server.visualize_junction_overlay(
        str(synthetic_lattice.volume_path),
        str(synthetic_lattice.registered_json_path),
        str(output_path),
        mode=mode,
    )

    assert result.startswith(f"Junction overlay saved to {output_path}")
    assert "1 candidates of 27 junctions" in result
    assert output_path.stat().st_size > 0


def test_visualize_junction_overlay_defaults_to_the_median_slice(
    synthetic_lattice, tmp_path: Path
) -> None:
    result = mcp_server.visualize_junction_overlay(
        str(synthetic_lattice.volume_path),
        str(synthetic_lattice.registered_json_path),
        str(tmp_path / "overlay.png"),
        slice_index=-1,
    )
    # Junctions sit at z = 10, 24, 38, so the median is the middle layer.
    assert "slice z=24" in result


@pytest.mark.parametrize(
    ("keyword", "message"),
    [
        ({"volume_path": "missing.npy"}, "volume file not found"),
        ({"volume_path": "volume.txt"}, "volume file must have a .npy"),
        ({"registered_json_path": "missing.json"}, "registered_json_path not found"),
        ({"registered_json_path": "lattice.txt"}, "must have a .json extension"),
        ({"radius": 0}, "radius must be a positive"),
    ],
)
def test_scan_lattice_junctions_errors(
    synthetic_lattice, tmp_path: Path, keyword: dict, message: str
) -> None:
    (tmp_path / "volume.txt").write_text("not an array")
    (tmp_path / "lattice.txt").write_text("not a lattice")
    path_keys = {"volume_path", "registered_json_path"}
    arguments = {
        "volume_path": str(synthetic_lattice.volume_path),
        "registered_json_path": str(synthetic_lattice.registered_json_path),
        "output_dir": str(tmp_path / "scan"),
        **{
            key: str(tmp_path / value) if key in path_keys else value
            for key, value in keyword.items()
        },
    }

    result = mcp_server.scan_lattice_junctions(**arguments)
    assert result.startswith("Error scanning lattice junctions:")
    assert message in result


@pytest.mark.parametrize(
    ("keyword", "message"),
    [
        ({"mode": "projection"}, 'mode must be "slice" or "mip"'),
        ({"axis": 3}, "axis must be 0, 1, or 2"),
        ({"slice_index": 9999}, "index must be between"),
    ],
)
def test_visualize_junction_overlay_errors(
    synthetic_lattice, tmp_path: Path, keyword: dict, message: str
) -> None:
    result = mcp_server.visualize_junction_overlay(
        str(synthetic_lattice.volume_path),
        str(synthetic_lattice.registered_json_path),
        str(tmp_path / "overlay.png"),
        **keyword,
    )
    assert result.startswith("Error visualizing junction overlay:")
    assert message in result
