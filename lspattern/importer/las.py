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


def _init_pipe_block(axis: str, color: int) -> str:
    """Return init block type for pipe based on axis and color.

    ColorI: 0 -> InitZero, 1 -> InitPlus
    ColorJ: 0 -> InitPlus, 1 -> InitZero
    """
    if axis == "I":
        return "InitPlusBlock" if color == 1 else "InitZeroBlock"
    # axis == "J"
    return "InitZeroBlock" if color == 1 else "InitPlusBlock"


def _meas_pipe_block(axis: str, color: int) -> str:
    """Return measure block type for pipe based on axis and color.

    ColorI: 0 -> MeasureZ, 1 -> MeasureX
    ColorJ: 0 -> MeasureX, 1 -> MeasureZ
    """
    if axis == "I":
        return "MeasureXBlock" if color == 1 else "MeasureZBlock"
    # axis == "J"
    return "MeasureZBlock" if color == 1 else "MeasureXBlock"


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
    """Return pipe boundary string based on axis and color.

    ColorI: 0 -> X, 1 -> Z
    ColorJ: 0 -> Z, 1 -> X
    """
    if axis == "I":
        # ColorI: 0 -> X, 1 -> Z
        axis_char = "X" if color == 0 else "Z"
        # T,B = axis_char, L,R = O
        return f"{axis_char}{axis_char}OO"
    if axis == "J":
        # ColorJ: 0 -> Z, 1 -> X
        axis_char = "Z" if color == 0 else "X"
        # T,B = O, L,R = axis_char
        return f"OO{axis_char}{axis_char}"
    msg = f"Unknown pipe axis {axis}"
    raise LasImportError(msg)


def _check_pipe_stacking_conflicts(pipes: list[Pipe]) -> None:
    """Check if pipe stacking (z -> z+1) creates conflicts.

    Each pipe at z is expanded to occupy both z and z+1. If another pipe
    already exists at z+1 position, this is a conflict.

    Parameters
    ----------
    pipes : list[Pipe]
        List of (axis, start, end, color) tuples.

    Raises
    ------
    LasImportError
        If any pipe's z+1 position conflicts with another pipe.
    """
    # Build set of all pipe positions keyed by (axis, i, j, k)
    pipe_positions: set[tuple[str, int, int, int]] = set()
    for axis, start, _end, _color in pipes:
        i, j, k = start
        pipe_positions.add((axis, i, j, k))

    # Check z+1 conflicts
    for axis, start, end, _color in pipes:
        i, j, k = start
        z_plus_1_pos = (axis, i, j, k + 1)
        if z_plus_1_pos in pipe_positions:
            msg = (
                f"Pipe stacking conflict: pipe at {start}->{end} would place "
                f"measure block at z={k + 1}, but another pipe already exists there."
            )
            raise LasImportError(msg)


def _k_color_to_init_block(color: int) -> str:
    """Return init block type based on K- boundary color."""
    # color=1 -> Z boundary -> InitZeroBlock
    # color=-1 -> X boundary -> InitPlusBlock
    return "InitZeroBlock" if color == 1 else "InitPlusBlock"


def _k_color_to_meas_block(color: int) -> str:
    """Return measure block type based on K+ boundary color."""
    # color=1 -> Z boundary -> MeasureZBlock
    # color=-1 -> X boundary -> MeasureXBlock
    return "MeasureZBlock" if color == 1 else "MeasureXBlock"


def _has_k_minus_connection(
    i: int, j: int, k: int, exist_k: list[Any], n_i: int, n_j: int, n_k: int
) -> bool:
    """Check if there's a K- connection (cube below connected via ExistK)."""
    if k == 0:
        return False
    if not exist_k or i >= n_i or j >= n_j or (k - 1) >= n_k:
        return False
    return bool(exist_k[i][j][k - 1])


def _has_k_plus_connection(
    i: int, j: int, k: int, exist_k: list[Any], n_i: int, n_j: int, n_k: int
) -> bool:
    """Check if there's a K+ connection (cube above connected via ExistK)."""
    if not exist_k or i >= n_i or j >= n_j or k >= n_k:
        return False
    return bool(exist_k[i][j][k])


def _get_color_km(
    i: int, j: int, k: int, color_km: list[Any], n_i: int, n_j: int, n_k: int
) -> int:
    """Get ColorKM value at (i, j, k), return 0 if out of bounds."""
    if not color_km or i >= n_i or j >= n_j or k >= n_k or k < 0:
        return 0
    return int(color_km[i][j][k])


def _get_color_kp(
    i: int, j: int, k: int, color_kp: list[Any], n_i: int, n_j: int, n_k: int
) -> int:
    """Get ColorKP value at (i, j, k), return 0 if out of bounds."""
    if not color_kp or i >= n_i or j >= n_j or k >= n_k:
        return 0
    return int(color_kp[i][j][k])


def _assign_block_for_cube(
    coord: Coord3,
    exist_k: list[Any],
    color_km: list[Any],
    color_kp: list[Any],
    n_i: int,
    n_j: int,
    n_k: int,
) -> tuple[str, Coord3 | None, str | None]:
    """Assign block type for a single non-port cube.

    Returns
    -------
    block : str
        Block type for this cube.
    meas_coord : Coord3 | None
        Coordinate for added measure cube (if any).
    meas_block : str | None
        Block type for added measure cube (if any).
    """
    i, j, k = coord
    block = "MemoryBlock"
    meas_coord: Coord3 | None = None
    meas_block: str | None = None

    # Check K- connection for init
    if not _has_k_minus_connection(i, j, k, exist_k, n_i, n_j, n_k):
        # No K- connection -> needs Init block
        # Use ColorKM at (i, j, k-1) to determine init type
        km_val = _get_color_km(i, j, k - 1, color_km, n_i, n_j, n_k)
        if km_val != 0:
            block = _k_color_to_init_block(km_val)
        else:
            # Fallback: check ColorKM at current position
            km_curr = _get_color_km(i, j, k, color_km, n_i, n_j, n_k)
            if km_curr != 0:
                block = _k_color_to_init_block(km_curr)

    # Check K+ connection for measure
    if not _has_k_plus_connection(i, j, k, exist_k, n_i, n_j, n_k):
        kp_val = _get_color_kp(i, j, k, color_kp, n_i, n_j, n_k)
        if kp_val != 0:
            meas_coord = (i, j, k + 1)
            meas_block = _k_color_to_meas_block(kp_val)

    return block, meas_coord, meas_block


def _assign_blocks_with_k_boundary(
    cubes: set[Coord3],
    port_cubes: Sequence[Coord3],
    lasre_ports: Sequence[dict[str, Any]],
    lasre: dict[str, Any],
    stabilizer: str,
) -> tuple[dict[Coord3, str], set[Coord3]]:
    """Decide block type per cube and add measure cubes for K+ boundaries.

    Parameters
    ----------
    cubes : set[Coord3]
        All cube coordinates.
    port_cubes : Sequence[Coord3]
        Port cube coordinates.
    lasre_ports : Sequence[dict[str, Any]]
        Port information from lasre.
    lasre : dict[str, Any]
        Full lasre dictionary containing ColorKM, ColorKP, ExistK.
    stabilizer : str
        Stabilizer string for port-based init/measure assignment.

    Returns
    -------
    blocks : dict[Coord3, str]
        Block type assignment for each cube.
    added_cubes : set[Coord3]
        New cubes added at z+1 for K+ boundaries.
    """
    color_km = lasre.get("ColorKM", [])
    color_kp = lasre.get("ColorKP", [])
    exist_k = lasre.get("ExistK", [])
    n_i, n_j, n_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]

    port_cube_set = set(port_cubes)
    added_cubes: set[Coord3] = set()
    blocks: dict[Coord3, str] = {}

    # First pass: assign blocks to existing cubes
    for coord in cubes:
        if coord in port_cube_set:
            blocks[coord] = "MemoryBlock"
            continue

        block, meas_coord, meas_block = _assign_block_for_cube(
            coord, exist_k, color_km, color_kp, n_i, n_j, n_k
        )
        blocks[coord] = block
        if meas_coord is not None and meas_block is not None:
            added_cubes.add(meas_coord)
            blocks[meas_coord] = meas_block

    # Override with port-based assignments (these take precedence)
    for idx, (port, coord) in enumerate(zip(lasre_ports, port_cubes, strict=True)):
        ch = _stab_char(stabilizer, idx)
        if port.get("e") == "-":  # input / bottom
            blocks[coord] = _init_block(ch)
        else:  # output / top
            blocks[coord] = _meas_block(ch)

    return blocks, added_cubes


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
        blocks, added_cubes = _assign_blocks_with_k_boundary(
            cubes, port_cubes, lasre_ports, lasre, stab
        )

        # Combine original cubes with added measure cubes
        all_cubes = cubes | added_cubes

        # Add boundaries for newly added cubes (use default boundary)
        all_boundaries = dict(boundaries)
        for coord in added_cubes:
            if coord not in all_boundaries:
                all_boundaries[coord] = _base_boundary(default_zdir)

        cube_entries = [
            {
                "position": list(coord),
                "block": blocks[coord],
                "boundary": "".join(all_boundaries[coord]),
            }
            for coord in sorted(all_cubes, key=operator.itemgetter(2, 1, 0))
        ]

        # Check for pipe stacking conflicts
        _check_pipe_stacking_conflicts(pipes)

        # Generate two-layer pipe structure: Init at z, Measure at z+1
        pipe_entries: list[dict[str, Any]] = []
        for axis, start, end, color in pipes:
            boundary = _pipe_boundary(axis, color)
            i, j, k = start

            # Two-layer structure: Init at z, Measure at z+1
            pipe_entries.extend([
                {
                    "start": list(start),
                    "end": list(end),
                    "block": _init_pipe_block(axis, color),
                    "boundary": boundary,
                },
                {
                    "start": [i, j, k + 1],
                    "end": [end[0], end[1], k + 1],
                    "block": _meas_pipe_block(axis, color),
                    "boundary": boundary,
                },
            ])

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
