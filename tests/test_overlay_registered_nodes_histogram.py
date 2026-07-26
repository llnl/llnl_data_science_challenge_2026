"""Tests for the junction-merge step shared by the lattice analyses."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from overlay_registered_nodes_histogram import merge_colocated_junctions


def duplicated_lattice() -> tuple[np.ndarray, np.ndarray]:
    """Three distinct positions listed 2, 1 and 3 times, in interleaved order.

    Returns (entry_positions, expected_merged_positions). The interleaving keeps
    the test honest about grouping by value rather than by adjacency.
    """
    a, b, c = [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]
    entries = np.array([a, c, b, c, a, c], dtype=float)
    return entries, np.array([a, c, b], dtype=float)


def test_merges_duplicates_to_distinct_positions():
    entries, expected = duplicated_lattice()
    merged, entry_to_junction = merge_colocated_junctions(entries)

    assert len(merged) == 3
    assert len(entry_to_junction) == len(entries)
    # Row order is np.unique's, so compare as sets of positions.
    assert {tuple(p) for p in merged} == {tuple(p) for p in expected}
    # Every entry maps to a junction sitting exactly where the entry was.
    assert np.allclose(merged[entry_to_junction], entries)


def test_entries_at_one_position_share_a_junction():
    entries, _ = duplicated_lattice()
    _, entry_to_junction = merge_colocated_junctions(entries)

    assert entry_to_junction[0] == entry_to_junction[4]  # the two 'a' entries
    assert entry_to_junction[1] == entry_to_junction[3] == entry_to_junction[5]
    assert entry_to_junction[2] not in entry_to_junction[[0, 1]]


def test_preserves_struts_degree_and_creates_no_self_loops():
    entries, _ = duplicated_lattice()
    _, entry_to_junction = merge_colocated_junctions(entries)
    # Each entry of a duplicated junction carries part of its connectivity.
    entry_struts = np.array([[0, 2], [4, 2], [1, 2], [3, 0], [5, 4]])

    struts = entry_to_junction[entry_struts]

    assert len(struts) == len(entry_struts)
    assert not (struts[:, 0] == struts[:, 1]).any()
    assert np.bincount(struts.ravel()).sum() == 2 * len(struts)


def test_merge_is_idempotent():
    entries, _ = duplicated_lattice()
    merged, _ = merge_colocated_junctions(entries)

    remerged, entry_to_junction = merge_colocated_junctions(merged)

    assert np.allclose(remerged[entry_to_junction], merged)
    assert len(remerged) == len(merged)


def test_already_unique_lattice_is_unchanged():
    entries = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

    merged, entry_to_junction = merge_colocated_junctions(entries)

    assert np.allclose(merged[entry_to_junction], entries)
    assert len(merged) == len(entries)


def test_rejects_near_but_distinct_junctions():
    """Fusing these would invent a midpoint where no junction exists."""
    entries = np.array([[0.0, 0.0, 0.0], [0.0004, 0.0, 0.0]])

    with pytest.raises(ValueError, match="distinct junctions, not duplicates"):
        merge_colocated_junctions(entries)


def test_tolerates_float_noise_below_the_rounding_scale():
    entries = np.array([[1.0, 2.0, 3.0], [1.0 + 1e-9, 2.0, 3.0]])

    merged, entry_to_junction = merge_colocated_junctions(entries)

    assert len(merged) == 1
    assert entry_to_junction.tolist() == [0, 0]
