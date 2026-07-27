"""Shared fixtures for building a miniature lattice and matching CT volume.

The real specimen is a 734x768x768 uint16 stack paired with a 10206-entry JSON,
far too large to commit and too slow to scan in a unit test. What the tests need
from it is its *structure*, and that reproduces at a tiny scale: junctions joined
by struts, a lattice JSON that lists every junction more than once with its
struts split across the duplicates, and a companion nominal JSON whose
maximum-Y layer stands in for the machined-off bottom face.

Struts are painted, not just junction blobs, for a reason that is not cosmetic.
Otsu's between-class variance is exactly constant across any empty span of the
histogram, so ``np.argmax`` breaks the tie at the first bin and a volume of two
well-separated spikes puts the threshold *on* the brightest background value --
making ``intensity < threshold`` a coin flip. Painting the struts brings the
material fraction to roughly the real specimen's, fills the falloff between the
modes, and lets Otsu land mid-trough the way it does on real data.

A junction is made dark by removing its blob *and every incident strut*, which
is the only way a junction actually goes dark: it stays bright as long as one
strut reaches it.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BACKGROUND_LEVEL = 1000
MATERIAL_LEVEL = 48000
NOISE_SIGMA = 80
EDGE_SIGMA = 1.0


@dataclass(frozen=True)
class SyntheticLattice:
    """Paths and ground truth for one generated lattice/volume pair."""

    volume: np.ndarray
    volume_path: Path
    registered_json_path: Path
    nominal_json_path: Path
    dark_position_xyz: tuple[int, int, int] | None
    n_entries: int
    n_junctions: int
    n_struts: int
    pitch: int


def _paint_sphere(material: np.ndarray, center_xyz, radius: float) -> None:
    """Union a filled sphere at an (x, y, z) center into a material mask."""
    x, y, z = center_xyz
    z_size, y_size, x_size = material.shape
    zz, yy, xx = np.ogrid[:z_size, :y_size, :x_size]
    material |= (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= radius**2


def _paint_cylinder(
    material: np.ndarray, start_xyz, end_xyz, radius: float
) -> None:
    """Union a capsule between two (x, y, z) points into a material mask.

    Distances are evaluated only inside the segment's padded bounding box, which
    keeps this cheap enough to call once per strut.
    """
    start = np.asarray(start_xyz, dtype=float)
    end = np.asarray(end_xyz, dtype=float)
    shape_xyz = np.array(material.shape[::-1])

    lo = np.clip(np.floor(np.minimum(start, end) - radius - 1), 0, shape_xyz - 1)
    hi = np.clip(np.ceil(np.maximum(start, end) + radius + 2), 0, shape_xyz)
    (x0, y0, z0), (x1, y1, z1) = lo.astype(int), hi.astype(int)

    zz, yy, xx = np.meshgrid(
        np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij"
    )
    points = np.stack([xx, yy, zz], axis=-1).astype(float)

    direction = end - start
    length_squared = float(direction @ direction)
    t = ((points - start) @ direction) / length_squared
    t = np.clip(t, 0.0, 1.0)[..., None]
    closest = start + t * direction
    inside = ((points - closest) ** 2).sum(axis=-1) <= radius**2

    material[z0:z1, y0:y1, x0:x1] |= inside


def make_synthetic_lattice(
    tmp_path: Path,
    grid: int = 3,
    pitch: int = 14,
    origin: int = 10,
    size: int = 48,
    junction_radius: float = 2.5,
    strut_radius: float = 1.8,
    dark_cell: tuple[int, int, int] | None = (1, 1, 1),
    json_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 0,
) -> SyntheticLattice:
    """Build a grid lattice, its duplicated-entry JSONs, and a matching volume.

    Args:
        tmp_path: Directory to write ``volume.npy`` and the two JSONs into.
        grid: Junctions per axis.
        pitch: Voxel spacing between junctions.
        origin: Voxel coordinate of the first junction on each axis.
        size: Edge length of the cubic volume.
        junction_radius: Radius of the blob painted at each junction.
        strut_radius: Radius of the capsule painted along each strut.
        dark_cell: Grid cell whose blob and incident struts are omitted, so the
            junction reads as absent. None paints a defect-free lattice.
        json_offset: Shift applied to the JSON positions only, not the volume.
            This is how registration error is simulated: the lattice is
            described as sitting somewhere the material is not. Offsetting along
            a diagonal moves the probe away from every axis-aligned strut at
            once, which an axis-aligned offset would not.
        seed: Seed for the additive noise.

    Returns:
        A ``SyntheticLattice``. Junction entries are duplicated, so
        ``n_entries`` is twice ``n_junctions`` and merged ids do not match JSON
        entry ids -- look junctions up by position, not by id.
    """
    rng = np.random.default_rng(seed)
    shape = (size, size, size)

    cells = [(i, j, k) for i in range(grid) for j in range(grid) for k in range(grid)]
    position_of = {cell: tuple(origin + pitch * c for c in cell) for cell in cells}
    edges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for cell in cells:
        for axis in range(3):
            shifted = list(cell)
            shifted[axis] += 1
            neighbor = (shifted[0], shifted[1], shifted[2])
            if neighbor in position_of:
                edges.append((cell, neighbor))

    material = np.zeros(shape, dtype=bool)
    for cell, position in position_of.items():
        if cell != dark_cell:
            _paint_sphere(material, position, junction_radius)
    for first, second in edges:
        if dark_cell in (first, second):
            continue
        _paint_cylinder(
            material, position_of[first], position_of[second], strut_radius
        )

    # Blurring the binary mask puts partial-volume voxels along every surface,
    # which is both what a real reconstruction looks like and what keeps Otsu
    # honest: with a perfectly empty gap between the two levels its objective is
    # flat there, and the argmax tie-break drops the threshold onto the
    # brightest background voxel.
    smooth = gaussian_filter(material.astype(np.float32), sigma=EDGE_SIGMA)
    volume = BACKGROUND_LEVEL + smooth * (MATERIAL_LEVEL - BACKGROUND_LEVEL)
    volume = volume + rng.normal(0.0, NOISE_SIGMA, size=shape)
    volume = volume.clip(0, 65535).astype(np.uint16)

    # Two entries per physical junction, interleaved with the struts split
    # between them, so a test that forgets to merge sees halved degrees.
    entry_of = {cell: (2 * index, 2 * index + 1) for index, cell in enumerate(cells)}
    junction_entries = []
    for cell, (first, second) in entry_of.items():
        x, y, z = position_of[cell]
        registered = [
            float(x) + json_offset[0],
            float(y) + json_offset[1],
            float(z) + json_offset[2],
        ]
        junction_entries.append((first, registered, list(cell)))
        junction_entries.append((second, list(registered), list(cell)))
    junction_entries.sort()

    struts = [
        {
            "id": index,
            "junction0": entry_of[first][index % 2],
            "junction1": entry_of[second][index % 2],
            "thickness": float(2 * strut_radius),
        }
        for index, (first, second) in enumerate(edges)
    ]

    registered_json_path = tmp_path / "registered.json"
    registered_json_path.write_text(
        json.dumps(
            {
                "junctions": [
                    {"id": entry, "position": position}
                    for entry, position, _ in junction_entries
                ],
                "struts": struts,
            }
        )
    )

    # The nominal design lives on a clean integer grid; its maximum-Y layer is
    # what ``bottom_layer_junctions`` treats as the machined-off face.
    nominal_json_path = tmp_path / "nominal.json"
    nominal_json_path.write_text(
        json.dumps(
            {
                "junctions": [
                    {"id": entry, "position": [float(c) for c in cell]}
                    for entry, _, cell in junction_entries
                ],
                "struts": struts,
            }
        )
    )

    volume_path = tmp_path / "volume.npy"
    np.save(volume_path, volume)

    dark_position: tuple[int, int, int] | None = None
    if dark_cell is not None:
        x, y, z = (
            int(round(c + o)) for c, o in zip(position_of[dark_cell], json_offset)
        )
        dark_position = (x, y, z)

    return SyntheticLattice(
        volume=volume,
        volume_path=volume_path,
        registered_json_path=registered_json_path,
        nominal_json_path=nominal_json_path,
        dark_position_xyz=dark_position,
        n_entries=len(junction_entries),
        n_junctions=len(cells),
        n_struts=len(struts),
        pitch=pitch,
    )


@pytest.fixture
def synthetic_lattice(tmp_path: Path) -> SyntheticLattice:
    """A 3x3x3 lattice in a 48^3 volume with exactly one junction left dark."""
    return make_synthetic_lattice(tmp_path)
