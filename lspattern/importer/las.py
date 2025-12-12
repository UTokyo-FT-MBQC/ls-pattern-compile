"""Convert LaSSynth solutions (lasre) into lspattern YAML canvases.

Usage (sketch)::

    from lspattern.importer.las import convert_lasre_to_yamls
    yamls = convert_lasre_to_yamls(lasre, specification,
                                   name_prefix="cnot",
                                   description="Auto-imported from LaSSynth")
    for name, text in yamls:
        Path(f"{name}.yml").write_text(text)

Design choices (agreed with user):
    - Y cubes are not supported yet -> raise a clear error.
    - K-direction pipes in LaSSynth correspond to stacked cubes in lspattern;
      only I/J pipes become lspattern `pipe` entries.
    - When a pipe exists, the two faces it connects are set to open (O) on the
      adjacent cubes. Pipe boundaries themselves use O on the connecting faces.
    - Color 0 -> X boundary, Color 1 -> Z boundary.
    - Initialisation / measurement bases are derived per stabilizer string:
        X -> InitPlus / MeasureX
        Z or . -> InitZero / MeasureZ
    - One YAML is produced for every stabilizer supplied in the specification.

The output YAML omits code distance; choose it at load time (e.g. d=3 for
debugging).

NOTE: Correlation surfaces (lasre keys CorrIJ/CorrIK/CorrJI/CorrJK/CorrKI/CorrKJ)
      are currently untouched. If/when logical observables are derived from
      these surfaces, access them directly from `lasre` inside
      `convert_lasre_to_yamls` and thread the information into YAML (e.g. as a
      top-level section or per-cube metadata).
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class LasImportError(RuntimeError):
    """Raised when the LaS → lspattern import cannot proceed."""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type Coord3 = tuple[int, int, int]
type Pipe = tuple[str, Coord3, Coord3, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_boundary(z_dir: str) -> list[str]:
    """Return [T,B,L,R] boundary chars given z_basis_direction.

    J means Z boundary is orthogonal to J → TOP/BOTTOM are Z, LEFT/RIGHT are X
    I means Z boundary is orthogonal to I → TOP/BOTTOM are X, LEFT/RIGHT are Z
    Anything else defaults to J-convention.
    """
    z_dir = (z_dir or "J").upper()
    if z_dir == "I":
        return list("XXZZ")
    return list("ZZXX")


def _color_to_axis(color: int | bool) -> str:
    if color == 0 or color is False:
        return "X"
    if color == 1 or color is True:
        return "Z"
    msg = f"Unsupported color value {color!r}; run color_z() first?"
    raise LasImportError(msg)


def _init_block(ch: str) -> str:
    return "InitPlusBlock" if ch.upper() == "X" else "InitZeroBlock"


def _meas_block(ch: str) -> str:
    return "MeasureXBlock" if ch.upper() == "X" else "MeasureZBlock"


def _stab_char(stabilizer: str, idx: int) -> str:
    if idx < len(stabilizer):
        return stabilizer[idx]
    return "."


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def _gather_cube_and_pipe_coords(
    lasre: dict[str, Any],
) -> tuple[set[Coord3], list[Pipe]]:
    """Collect cubes and I/J pipes from lasre.

    Returns
    -------
    cubes : set[tuple[int,int,int]]
        All cube coordinates to materialise in lspattern (includes pipe endpoints
        and K-extension endpoints).
    pipes : list[tuple[str, tuple[int,int,int], tuple[int,int,int], int]]
        Entries are (axis, start, end, color) with axis in {"I","J"}.
    """
    n_i, n_j, n_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]
    cubes: set[Coord3] = set()
    pipes: list[Pipe] = []

    exist_i = lasre.get("ExistI", [])
    exist_j = lasre.get("ExistJ", [])
    exist_k = lasre.get("ExistK", [])
    color_i = lasre.get("ColorI", [])
    color_j = lasre.get("ColorJ", [])

    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                if exist_i and exist_i[i][j][k]:
                    cubes.update({(i, j, k), (i + 1, j, k)})
                    pipes.append(("I", (i, j, k), (i + 1, j, k), int(color_i[i][j][k])))
                if exist_j and exist_j[i][j][k]:
                    cubes.update({(i, j, k), (i, j + 1, k)})
                    pipes.append(("J", (i, j, k), (i, j + 1, k), int(color_j[i][j][k])))
                if exist_k and exist_k[i][j][k]:
                    cubes.update({(i, j, k), (i, j, k + 1)})

    # include port cubes (they may sit on the top/bottom layer)
    cubes.update((pc[0], pc[1], pc[2]) for pc in lasre.get("port_cubes", []))

    return cubes, pipes


def _cube_orientation_map(
    spec_ports: Sequence[dict[str, Any]],
    port_cubes: Sequence[Coord3],
) -> dict[Coord3, str]:
    """Map each port cube coordinate to its z_basis_direction."""
    mapping: dict[Coord3, str] = {}
    for port, coord in zip(spec_ports, port_cubes, strict=True):
        zdir = port.get("z_basis_direction", "J")
        mapping[coord] = zdir
    return mapping


def _build_cube_boundaries(
    cubes: Iterable[Coord3],
    pipes: Iterable[Pipe],
    port_zdir_map: dict[Coord3, str],
    default_zdir: str,
) -> dict[Coord3, list[str]]:
    """Create boundary spec per cube, applying O on faces touched by pipes."""
    boundaries: dict[Coord3, list[str]] = {}

    for coord in cubes:
        zdir = port_zdir_map.get(coord, default_zdir)
        boundaries[coord] = _base_boundary(zdir)

    # apply openings from pipes
    for axis, start, end, _ in pipes:
        if axis == "I":
            # start is -I face of end and +I face of start -> LEFT/RIGHT are O
            if start in boundaries:
                boundaries[start][3] = "O"  # RIGHT
            if end in boundaries:
                boundaries[end][2] = "O"  # LEFT
        elif axis == "J":
            if start in boundaries:
                boundaries[start][1] = "O"  # BOTTOM
            if end in boundaries:
                boundaries[end][0] = "O"  # TOP

    return boundaries


def _pipe_boundary(axis: str, color: int) -> str:
    axis_char = _color_to_axis(color)
    if axis == "I":
        # T,B = axis_char, L,R = O
        return f"{axis_char}{axis_char}OO"
    if axis == "J":
        # T,B = O, L,R = axis_char
        return f"OO{axis_char}{axis_char}"
    msg = f"Unknown pipe axis {axis}"
    raise LasImportError(msg)


def _assign_blocks(
    cubes: Iterable[Coord3],
    port_cubes: Sequence[Coord3],
    lasre_ports: Sequence[dict[str, Any]],
    stabilizer: str,
) -> dict[Coord3, str]:
    """Decide block type per cube (init/measure/memory)."""
    blocks: dict[Coord3, str] = dict.fromkeys(cubes, "MemoryBlock")

    for idx, (port, coord) in enumerate(zip(lasre_ports, port_cubes, strict=True)):
        ch = _stab_char(stabilizer, idx)
        if port.get("e") == "-":  # input / bottom
            blocks[coord] = _init_block(ch)
        else:  # output / top
            blocks[coord] = _meas_block(ch)

    return blocks


def _check_no_y_cubes(lasre: dict[str, Any]) -> None:
    node_y = lasre.get("NodeY")
    if not node_y:
        return
    for plane in node_y:
        for row in plane:
            for val in row:
                if val:
                    msg = "Y cubes are not supported yet. Aborting import."
                    raise LasImportError(msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_lasre_to_yamls(
    lasre: dict[str, Any],
    specification: dict[str, Any],
    *,
    name_prefix: str = "las",
    description: str | None = None,
) -> list[tuple[str, str]]:
    """Convert a LaSSynth lasre + specification to multiple YAML canvases.

    Parameters
    ----------
    lasre : dict
        Output of LatticeSurgerySolution.lasre (after color_z recommended).
    specification : dict
        Original problem spec; must contain `ports` (with z_basis_direction)
        and `stabilizers` (list of strings). Order is assumed consistent with
        lasre["ports"].
    name_prefix : str
        Prefix for the generated canvas names.
    description : str | None
        Optional description field for YAML; defaults to name_prefix if None.

    Returns
    -------
    list[(name, yaml_text)]
        One entry per stabilizer string.
    """
    _check_no_y_cubes(lasre)

    spec_ports: list[dict[str, Any]] = specification.get("ports", [])
    lasre_ports: list[dict[str, Any]] = lasre.get("ports", [])
    stabilizers: list[str] = specification.get("stabilizers", [])
    if not spec_ports:
        msg = "specification must contain ports with z_basis_direction"
        raise LasImportError(msg)
    if not stabilizers:
        msg = "specification must contain stabilizers list"
        raise LasImportError(msg)

    cubes, pipes = _gather_cube_and_pipe_coords(lasre)
    port_cubes: list[Coord3] = [(pc[0], pc[1], pc[2]) for pc in lasre.get("port_cubes", [])]
    if len(port_cubes) != len(spec_ports):
        msg = "Mismatch between port_cubes and ports length"
        raise LasImportError(msg)

    port_zdir_map = _cube_orientation_map(spec_ports, port_cubes)
    default_zdir = next(iter(port_zdir_map.values()), "J")

    boundaries = _build_cube_boundaries(cubes, pipes, port_zdir_map, default_zdir)

    yaml_results: list[tuple[str, str]] = []

    for stab_idx, stab in enumerate(stabilizers):
        blocks = _assign_blocks(cubes, port_cubes, lasre_ports, stab)

        cube_entries = [
            {
                "position": list(coord),
                "block": blocks[coord],
                "boundary": "".join(boundaries[coord]),
            }
            for coord in sorted(cubes, key=operator.itemgetter(2, 1, 0))
        ]

        pipe_entries = [
            {
                "start": list(start),
                "end": list(end),
                "block": "MeasureXBlock" if _color_to_axis(color) == "X" else "MeasureZBlock",
                "boundary": _pipe_boundary(axis, color),
            }
            for axis, start, end, color in pipes
        ]

        desc_base = description or name_prefix
        canvas_dict: dict[str, Any] = {
            "name": f"{name_prefix}_{stab_idx}",
            "description": f"{desc_base} | stabilizer: {stab}",
            "layout": "rotated_surface_code",
            "cube": cube_entries,
            "pipe": pipe_entries,
        }

        yaml_text = yaml.safe_dump(
            canvas_dict,
            sort_keys=False,
            width=1000,  # encourage inline short lists like [0, 0, 0]
            default_flow_style=False,
        )
        yaml_results.append((canvas_dict["name"], yaml_text))

    return yaml_results
