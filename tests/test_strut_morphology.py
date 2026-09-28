import numpy as np

from part2.morphology.strut_morphology import longest_true_run, orthonormal_basis


def test_longest_gap_is_physical_length():
    assert longest_true_run(np.array([False, True, True, False, True]), 58.09) == 116.18


def test_basis_is_orthonormal_to_direction():
    direction = np.array([0.3, 0.4, 0.8660254])
    u, v = orthonormal_basis(direction / np.linalg.norm(direction))
    assert np.isclose(np.dot(u, v), 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)
