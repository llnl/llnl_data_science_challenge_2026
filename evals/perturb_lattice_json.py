"""Manufacture deliberately broken lattice JSONs for evaluating a junction scan.

The scan-junctions workflow is supposed to catch *systematic* problems before it
reports individual missing junctions. Neither failure mode occurs in the one
scan available locally, so this script injects them into a copy of a registered
JSON. The CT volume is never touched -- what changes is the lattice's claim
about where its junctions are.

Two perturbations, matching the two systematic failures the skill must catch:

``--rotate-degrees``
    Rotate every junction about the lattice centroid. Small angles move the
    outer junctions many voxels while leaving the centre nearly fixed, which is
    exactly the graded, radius-sensitive signature of a registration error.
    A degree or two is plenty: on the 9x9x9 specimen the half-extent is about
    440 voxels, so 2 degrees displaces the edge by roughly 15 voxels against
    struts only ~4 voxels thick.

``--add-phantom-slab``
    Append a copy of one outer face's junctions and their struts, shifted one
    unit-cell pitch further out into empty air. Those junctions are real entries
    in a valid lattice that simply have no material under them, so they should
    appear as one large connected dark component off to one side -- a missing
    region, not scattered defects.

Both outputs stay schema-valid: entry ids remain 0..N-1 in list order, struts
reference existing entries, and co-located duplicates stay exactly coincident so
``merge_colocated_junctions`` still accepts them.

A scan reads the registered and nominal lattices together and requires them to
describe the same entries, so a phantom slab has to be added to *both*: pass
``--nominal-input``/``--nominal-output`` and the same slab is appended to the
design file. There it lands on a fresh design layer one pitch beyond the part,
which is precisely how a real unprinted or out-of-field region would present.
Rotation does not change the entry list, so a rotated run keeps the original
nominal file unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def load_lattice_json(path: Path) -> dict:
    """Read a lattice JSON and check the entry ids this script relies on."""
    with path.open() as f:
        data = json.load(f)

    ids = [junction["id"] for junction in data["junctions"]]
    if ids != list(range(len(ids))):
        raise ValueError(f"junction ids in {path} are not 0..N-1 in list order")
    return data


def rotate_junctions(data: dict, degrees: float, axis: str) -> dict:
    """Rotate every junction position about the lattice centroid, in place.

    Duplicated entries stay exactly coincident because the same matrix is
    applied to identical inputs, so the merge step still sees them as one
    junction rather than as distinct ones a fraction of a voxel apart.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}, got {axis!r}")

    positions = np.array(
        [junction["position"] for junction in data["junctions"]], dtype=float
    )
    centroid = positions.mean(axis=0)

    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.eye(3)
    first, second = [i for i in range(3) if i != AXES[axis]]
    rotation[first, first] = cos
    rotation[first, second] = -sin
    rotation[second, first] = sin
    rotation[second, second] = cos

    rotated = (positions - centroid) @ rotation.T + centroid
    for junction, position in zip(data["junctions"], rotated):
        junction["position"] = position.tolist()

    displacement = np.linalg.norm(rotated - positions, axis=1)
    print(
        f"Rotated {len(positions)} entries {degrees}deg about {axis}: "
        f"displacement {displacement.min():.1f}-{displacement.max():.1f} voxels "
        f"(mean {displacement.mean():.1f})"
    )
    return data


def _unique_mean(positions: np.ndarray) -> np.ndarray:
    """Mean of the distinct positions, so duplicated entries do not vote twice."""
    return np.unique(np.round(positions, 6), axis=0).mean(axis=0)


def _entries_in_cells(data: dict, cells: list[dict]) -> set[int]:
    """Collect the junction entries touched by a set of unit cells' struts."""
    strut_by_id = {strut["id"]: strut for strut in data["struts"]}
    entries: set[int] = set()
    for cell in cells:
        for strut_id in cell["struts"]:
            strut = strut_by_id[strut_id]
            entries.add(strut["junction0"])
            entries.add(strut["junction1"])
    return entries


def add_phantom_slab(data: dict, axis: str) -> dict:
    """Append a copy of the outermost unit-cell layer, shifted into empty air.

    The layer is selected from ``unit_cells``, whose ``indices`` are the design's
    integer cell grid. The registered *positions* cannot be used for this: they
    are rotated relative to the design, so no registered coordinate threshold
    recovers a design layer -- the same trap that makes the machined-off bottom
    face unfindable from registered y.

    The shift is one cell pitch, measured as the offset between the outermost
    layer's junctions and the layer behind it, so it points along the lattice in
    registered space rather than along a volume axis. The copy therefore lands
    where the next layer of cells would be if the specimen continued.
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}, got {axis!r}")
    if "unit_cells" not in data:
        raise ValueError(
            "phantom slabs need a 'unit_cells' list to identify a design layer; "
            "registered coordinates are rotated and cannot be thresholded instead"
        )
    column = AXES[axis]

    cells = data["unit_cells"]
    cell_index = np.array([cell["indices"] for cell in cells], dtype=int)
    layers = np.unique(cell_index[:, column])
    if len(layers) < 2:
        raise ValueError(f"lattice has fewer than two cell layers along {axis}")

    outer_cells = [c for c, i in zip(cells, cell_index) if i[column] == layers[-1]]
    inner_cells = [c for c, i in zip(cells, cell_index) if i[column] == layers[-2]]
    outer_entries = sorted(_entries_in_cells(data, outer_cells))
    inner_entries = sorted(_entries_in_cells(data, inner_cells))

    positions = np.array(
        [junction["position"] for junction in data["junctions"]], dtype=float
    )
    # Deduplicate by position before averaging. The two cell layers hold the same
    # junctions one pitch apart, but not the same number of *entries* -- a
    # junction is listed once per cell touching it, so boundary junctions appear
    # fewer times and would drag an entry-weighted mean off the pitch.
    shift = _unique_mean(positions[outer_entries]) - _unique_mean(
        positions[inner_entries]
    )

    n_entries = len(data["junctions"])
    phantom_id = {entry: n_entries + i for i, entry in enumerate(outer_entries)}
    for entry in outer_entries:
        source = data["junctions"][entry]
        data["junctions"].append(
            {
                "id": phantom_id[entry],
                "position": (np.array(source["position"]) + shift).tolist(),
                "indices": source.get("indices", [0.0, 0.0, 0.0]),
            }
        )

    n_struts = len(data["struts"])
    added = 0
    for strut in list(data["struts"]):
        first, second = strut["junction0"], strut["junction1"]
        if first in phantom_id and second in phantom_id:
            data["struts"].append(
                {
                    "id": n_struts + added,
                    "unit_cell_edge_idx": strut.get("unit_cell_edge_idx", 0),
                    "junction0": phantom_id[first],
                    "junction1": phantom_id[second],
                    "thickness": strut.get("thickness", 0.1),
                }
            )
            added += 1

    if added == 0:
        raise ValueError(
            f"the outermost {axis} cell layer has no struts within it, so the "
            "phantom slab would be disconnected and could not form a component"
        )

    print(
        f"Added {len(outer_entries)} phantom entries and {added} struts, shifted "
        f"{np.linalg.norm(shift):.1f} voxels beyond the outermost {axis} cell layer"
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manufacture deliberately broken lattice JSONs for eval runs."
    )
    parser.add_argument("input", type=Path, help="registered lattice JSON to copy")
    parser.add_argument("output", type=Path, help="destination JSON")
    parser.add_argument(
        "--rotate-degrees",
        type=float,
        default=None,
        help="rotate all junctions this many degrees about the lattice centroid",
    )
    parser.add_argument(
        "--rotate-axis", default="z", choices=sorted(AXES), help="rotation axis"
    )
    parser.add_argument(
        "--add-phantom-slab",
        default=None,
        choices=sorted(AXES),
        help="append a phantom copy of the outermost face along this axis",
    )
    parser.add_argument(
        "--nominal-input",
        type=Path,
        default=None,
        help="design JSON matching the input, required for --add-phantom-slab",
    )
    parser.add_argument(
        "--nominal-output",
        type=Path,
        default=None,
        help="destination for the perturbed design JSON",
    )
    arguments = parser.parse_args()

    if arguments.rotate_degrees is None and arguments.add_phantom_slab is None:
        parser.error("choose at least one of --rotate-degrees or --add-phantom-slab")

    nominal_given = arguments.nominal_input or arguments.nominal_output
    if arguments.add_phantom_slab is not None:
        # A scan reads both lattices and rejects a pair with different entry
        # counts, so a slab added to only one of them would fail at load rather
        # than exercise the detector.
        if not (arguments.nominal_input and arguments.nominal_output):
            parser.error(
                "--add-phantom-slab needs --nominal-input and --nominal-output: the "
                "slab must be added to the design lattice too, or the two files no "
                "longer describe the same entries"
            )
    elif nominal_given:
        parser.error(
            "--nominal-input/--nominal-output apply only to --add-phantom-slab; "
            "rotation leaves the entry list alone, so the original design JSON "
            "still matches"
        )

    data = load_lattice_json(arguments.input)
    if arguments.rotate_degrees is not None:
        data = rotate_junctions(data, arguments.rotate_degrees, arguments.rotate_axis)
    if arguments.add_phantom_slab is not None:
        data = add_phantom_slab(data, arguments.add_phantom_slab)
    _write_lattice(arguments.output, data)

    if arguments.nominal_input is not None:
        # The same slab, selected the same way from the same unit-cell grid, so
        # the two files gain the same entries in the same order. The shift is
        # measured in each file's own coordinates, which puts the design copy on
        # a clean new design layer.
        nominal = add_phantom_slab(
            load_lattice_json(arguments.nominal_input), arguments.add_phantom_slab
        )
        if len(nominal["junctions"]) != len(data["junctions"]):
            raise ValueError(
                f"{arguments.nominal_input} and {arguments.input} produced "
                f"{len(nominal['junctions'])} and {len(data['junctions'])} entries; "
                "they are not the same lattice"
            )
        _write_lattice(arguments.nominal_output, nominal)


def _write_lattice(path: Path, data: dict) -> None:
    """Write a lattice JSON, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)
    print(
        f"Wrote {path} with {len(data['junctions'])} junction entries "
        f"and {len(data['struts'])} struts"
    )


if __name__ == "__main__":
    main()
