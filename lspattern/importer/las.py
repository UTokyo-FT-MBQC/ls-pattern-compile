"""Convert LaSSynth solutions (lasre) into lspattern YAML canvases.

Usage (sketch)::

    from lspattern.importer.las import convert_lasre_to_yamls

    yamls = convert_lasre_to_yamls(lasre, specification, name_prefix="cnot", description="Auto-imported from LaSSynth")
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

import yaml

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


def _color_kp_to_boundary(color_kp: int, inverted: bool = False) -> list[str] | None:
    """Return [T,B,L,R] boundary chars based on ColorKP value.

    ColorKP = 0 -> ZZXX (or XXZZ if inverted)
    ColorKP = 1 -> XXZZ (or ZZXX if inverted)
    ColorKP = -1 -> None (refer to cube below)

    Parameters
    ----------
    color_kp : int
        ColorKP value (0, 1, or -1).
    inverted : bool
        If True, invert the boundary mapping (used after Hadamard).
    """
    if color_kp == -1:
        return None  # refer to cube below

    if inverted:
        # Invert the mapping
        return list("XXZZ") if color_kp == 0 else list("ZZXX")
    # Normal mapping
    return list("ZZXX") if color_kp == 0 else list("XXZZ")


def _resolve_cube_boundary(
    coord: Coord3,
    color_kp: list[Any],
    n_i: int,
    n_j: int,
    n_k: int,
    boundaries: dict[Coord3, list[str]],
    port_zdir_map: dict[Coord3, str],
    default_zdir: str,
    inverted: bool = False,
) -> list[str]:
    """Resolve boundary for a single cube based on ColorKP.

    Parameters
    ----------
    inverted : bool
        If True, invert the boundary mapping (used after Hadamard).
    """
    i, j, k = coord
    kp_val = _get_color_kp(i, j, k, color_kp, n_i, n_j, n_k)
    boundary = _color_kp_to_boundary(kp_val, inverted=inverted)

    if boundary is not None:
        return boundary

    # ColorKP = -1: refer to cube below
    below_coord = (i, j, k - 1)
    if below_coord in boundaries:
        return list(boundaries[below_coord])

    # Fallback to default
    zdir = port_zdir_map.get(coord, default_zdir)
    return _base_boundary(zdir)


def _color_to_axis(color: int | bool) -> str:
    if color == 0 or color is False:
        return "X"
    if color == 1 or color is True:
        return "Z"
    msg = f"Unsupported color value {color!r}; run color_z() first?"
    raise LasImportError(msg)


def _short_memory_pipe_block(axis: str, color: int) -> str:
    """Return short memory block type for pipe based on axis and color.

    Mapping:
        I axis, color=0 -> ShortXMemoryBlock (X boundary)
        I axis, color=1 -> ShortZMemoryBlock (Z boundary)
        J axis, color=0 -> ShortZMemoryBlock (Z boundary)
        J axis, color=1 -> ShortXMemoryBlock (X boundary)
    """
    if axis == "I":
        # ColorI: 0 -> X boundary, 1 -> Z boundary
        return "ShortXMemoryBlock" if color == 0 else "ShortZMemoryBlock"
    # axis == "J"
    # ColorJ: 0 -> Z boundary, 1 -> X boundary
    return "ShortZMemoryBlock" if color == 0 else "ShortXMemoryBlock"


def _init_block(ch: str) -> str:
    return "InitPlusBlock" if ch.upper() == "X" else "InitZeroBlock"


def _meas_block(ch: str) -> str:
    return "MeasureXBlock" if ch.upper() == "X" else "MeasureZBlock"


def _get_output_port_token(
    stab_char: str,
    boundary: list[str],
    inverted: bool,
) -> str | None:
    """Determine logical observable token for output port MeasureBlock.

    Parameters
    ----------
    stab_char : str
        Stabilizer character ('X', 'Z', or '.').
    boundary : list[str]
        Boundary specification [T, B, L, R].
    inverted : bool
        Whether invert_ancilla_order is True.

    Returns
    -------
    str | None
        Token ('TB' or 'RL') or None if stab_char is '.'.
    """
    if stab_char.upper() not in {"X", "Z"}:
        return None

    # Determine boundary type (XXZZ or ZZXX based on T,B positions)
    is_xxzz = boundary[0] == "X"  # T position

    # Apply invert if needed
    if inverted:
        is_xxzz = not is_xxzz

    if stab_char.upper() == "X":
        return "TB" if is_xxzz else "RL"
    # Z
    return "RL" if is_xxzz else "TB"


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
    lasre: dict[str, Any],
    invert_map: dict[Coord3, bool] | None = None,
) -> dict[Coord3, list[str]]:
    """Create boundary spec per cube, using ColorKP and applying O on pipe faces.

    Parameters
    ----------
    invert_map : dict[Coord3, bool] | None
        Mapping from cube coordinates to whether the boundary should be inverted
        (used after Hadamard).
    """
    if invert_map is None:
        invert_map = {}

    boundaries: dict[Coord3, list[str]] = {}

    color_kp = lasre.get("ColorKP", [])
    n_i, n_j, n_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]

    # Sort cubes by k to ensure lower cubes are processed first (for -1 reference)
    sorted_cubes = sorted(cubes, key=operator.itemgetter(2, 1, 0))

    for coord in sorted_cubes:
        inverted = invert_map.get(coord, False)
        boundaries[coord] = _resolve_cube_boundary(
            coord, color_kp, n_i, n_j, n_k, boundaries, port_zdir_map, default_zdir, inverted
        )

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


def _k_color_to_meas_bulk_block(color: int) -> str:
    """Return measure bulk block type based on K+ boundary color."""
    # color=1 -> Z boundary -> MeasureBulkZBlock
    # color=-1 -> X boundary -> MeasureBulkXBlock
    return "MeasureBulkZBlock" if color == 1 else "MeasureBulkXBlock"


def _k_color_to_short_memory_block(color: int) -> str:
    """Return short memory block type based on K boundary color."""
    # color=1 -> Z boundary -> ShortZMemoryBlock
    # color=-1 -> X boundary -> ShortXMemoryBlock
    return "ShortZMemoryBlock" if color == 1 else "ShortXMemoryBlock"


def _has_k_minus_connection(i: int, j: int, k: int, exist_k: list[Any], n_i: int, n_j: int, n_k: int) -> bool:
    """Check if there's a K- connection (cube below connected via ExistK)."""
    if k == 0:
        return False
    if not exist_k or i >= n_i or j >= n_j or (k - 1) >= n_k:
        return False
    return bool(exist_k[i][j][k - 1])


def _has_k_plus_connection(i: int, j: int, k: int, exist_k: list[Any], n_i: int, n_j: int, n_k: int) -> bool:
    """Check if there's a K+ connection (cube above connected via ExistK)."""
    if not exist_k or i >= n_i or j >= n_j or k >= n_k:
        return False
    return bool(exist_k[i][j][k])


def _get_color_km(i: int, j: int, k: int, color_km: list[Any], n_i: int, n_j: int, n_k: int) -> int:
    """Get ColorKM value at (i, j, k), return 0 if out of bounds."""
    if not color_km or i >= n_i or j >= n_j or k >= n_k or k < 0:
        return 0
    return int(color_km[i][j][k])


def _get_color_kp(i: int, j: int, k: int, color_kp: list[Any], n_i: int, n_j: int, n_k: int) -> int:
    """Get ColorKP value at (i, j, k), return 0 if out of bounds."""
    if not color_kp or i >= n_i or j >= n_j or k >= n_k:
        return 0
    return int(color_kp[i][j][k])


def _resolve_color_kp(
    i: int, j: int, k: int,
    color_kp: list[Any],
    n_i: int, n_j: int, n_k: int,
) -> int:
    """Resolve ColorKP value, following -1 references to previous k."""
    val = _get_color_kp(i, j, k, color_kp, n_i, n_j, n_k)
    if val != -1:
        return val
    # -1 means "same as previous", look at k-1
    if k > 0:
        return _resolve_color_kp(i, j, k - 1, color_kp, n_i, n_j, n_k)
    return 0  # default


def _resolve_color_km(
    i: int, j: int, k: int,
    color_km: list[Any],
    n_i: int, n_j: int, n_k: int,
) -> int:
    """Resolve ColorKM value, following -1 references to previous k."""
    val = _get_color_km(i, j, k, color_km, n_i, n_j, n_k)
    if val != -1:
        return val
    # -1 means "same as previous", look at k-1
    if k > 0:
        return _resolve_color_km(i, j, k - 1, color_km, n_i, n_j, n_k)
    return 0  # default


def _is_hadamard_cube(
    i: int, j: int, k: int,
    color_km: list[Any],
    color_kp: list[Any],
    n_i: int, n_j: int, n_k: int,
) -> bool:
    """Check if this cube has a Hadamard (resolved ColorKP[k] != resolved ColorKM[k-1]).

    A Hadamard occurs when the K+ boundary color of the current cube differs
    from the K- boundary color of the cube below.
    """
    if k == 0:
        return False

    # Resolved ColorKP of current cube
    kp_val = _resolve_color_kp(i, j, k, color_kp, n_i, n_j, n_k)

    # Resolved ColorKM of cube below (k-1)
    km_below = _resolve_color_km(i, j, k - 1, color_km, n_i, n_j, n_k)

    # Hadamard if different (0 and 1 are valid values to compare)
    return kp_val != km_below


def _compute_invert_ancilla_order(
    cubes: set[Coord3],
    color_km: list[Any],
    color_kp: list[Any],
    n_i: int, n_j: int, n_k: int,
) -> tuple[set[Coord3], dict[Coord3, bool]]:
    """Compute Hadamard cubes and invert_ancilla_order for all cubes.

    Returns
    -------
    hadamard_cubes : set[Coord3]
        Cubes where HadamardBlock should be placed.
    invert_map : dict[Coord3, bool]
        Whether each cube needs invert_ancilla_order=True.
    """
    hadamard_cubes: set[Coord3] = set()
    invert_map: dict[Coord3, bool] = {}

    # Sort cubes by k for each (i, j)
    cubes_by_ij: dict[tuple[int, int], list[int]] = {}
    for i, j, k in cubes:
        cubes_by_ij.setdefault((i, j), []).append(k)

    for (i, j), k_list in cubes_by_ij.items():
        inverted = False
        for k in sorted(k_list):
            if _is_hadamard_cube(i, j, k, color_km, color_kp, n_i, n_j, n_k):
                hadamard_cubes.add((i, j, k))
                inverted = not inverted  # toggle
            invert_map[i, j, k] = inverted

    return hadamard_cubes, invert_map


def _get_corr_value(lasre: dict[str, Any], corr_key: str, stab_idx: int, i: int, j: int, k: int) -> bool:
    """Get correlation surface value at (stab_idx, i, j, k) for given key."""
    corr = lasre.get(corr_key, [])
    if not corr:
        return False
    n_s = lasre.get("n_s", len(corr))
    n_i, n_j, n_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]
    if stab_idx >= n_s or i >= n_i or j >= n_j or k >= n_k:
        return False
    return bool(corr[stab_idx][i][j][k])


def _get_corr_ki(lasre: dict[str, Any], stab_idx: int, i: int, j: int, k: int) -> bool:
    """Get CorrKI value at (stab_idx, i, j, k)."""
    return _get_corr_value(lasre, "CorrKI", stab_idx, i, j, k)


def _get_corr_kj(lasre: dict[str, Any], stab_idx: int, i: int, j: int, k: int) -> bool:
    """Get CorrKJ value at (stab_idx, i, j, k)."""
    return _get_corr_value(lasre, "CorrKJ", stab_idx, i, j, k)


def _get_corr_ij(lasre: dict[str, Any], stab_idx: int, i: int, j: int, k: int) -> bool:
    """Get CorrIJ value at (stab_idx, i, j, k)."""
    return _get_corr_value(lasre, "CorrIJ", stab_idx, i, j, k)


def _get_corr_ik(lasre: dict[str, Any], stab_idx: int, i: int, j: int, k: int) -> bool:
    """Get CorrIK value at (stab_idx, i, j, k)."""
    return _get_corr_value(lasre, "CorrIK", stab_idx, i, j, k)


def _get_corr_ji(lasre: dict[str, Any], stab_idx: int, i: int, j: int, k: int) -> bool:
    """Get CorrJI value at (stab_idx, i, j, k)."""
    return _get_corr_value(lasre, "CorrJI", stab_idx, i, j, k)


def _get_corr_jk(lasre: dict[str, Any], stab_idx: int, i: int, j: int, k: int) -> bool:
    """Get CorrJK value at (stab_idx, i, j, k)."""
    return _get_corr_value(lasre, "CorrJK", stab_idx, i, j, k)


def _get_pipe_logical_observables(
    lasre: dict[str, Any],
    stab_idx: int,
    axis: str,
    start: Coord3,
    block: str,
) -> list[dict[str, Any]] | None:
    """Determine logical observables for a pipe from correlation surfaces.

    Parameters
    ----------
    lasre : dict[str, Any]
        Full lasre dictionary for correlation surface lookup.
    stab_idx : int
        Stabilizer index for correlation surface lookup.
    axis : str
        Pipe axis ("I" or "J").
    start : Coord3
        Pipe start coordinate (i, j, k).
    block : str
        Block type ("ShortXMemoryBlock" or "ShortZMemoryBlock").

    Returns
    -------
    list[dict[str, Any]] | None
        List of logical observable entries, or None if empty.
    """
    i, j, k = start
    observables: list[dict[str, Any]] = []

    if axis == "I":
        # I-axis pipe: check CorrIJ and CorrIK
        if _get_corr_ij(lasre, stab_idx, i, j, k):
            if block == "ShortXMemoryBlock":
                observables.append({"token": "X", "layer": 0, "sublayer": 2})
            else:  # ShortZMemoryBlock
                observables.append({"token": "Z", "layer": 0, "sublayer": 1})
        if _get_corr_ik(lasre, stab_idx, i, j, k):
            sublayer = 1 if block == "ShortXMemoryBlock" else 2
            observables.append({"token": "RL", "layer": -1, "sublayer": sublayer})
    else:  # axis == "J"
        # J-axis pipe: check CorrJI and CorrJK
        if _get_corr_ji(lasre, stab_idx, i, j, k):
            if block == "ShortXMemoryBlock":
                observables.append({"token": "X", "layer": 0, "sublayer": 2})
            else:  # ShortZMemoryBlock
                observables.append({"token": "Z", "layer": 0, "sublayer": 1})
        if _get_corr_jk(lasre, stab_idx, i, j, k):
            sublayer = 1 if block == "ShortXMemoryBlock" else 2
            observables.append({"token": "TB", "layer": -1, "sublayer": sublayer})

    return observables or None


def _get_logical_observables(
    lasre: dict[str, Any],
    stab_idx: int,
    k_pipe: Coord3,
) -> str | list[str] | None:
    """Determine logical observables for a measure block from correlation surfaces.

    Returns
    -------
    str | list[str] | None
        "TB" if CorrKI is 1, "RL" if CorrKJ is 1, ["TB", "RL"] if both,
        or None if neither.
    """
    i, j, k = k_pipe
    observables: list[str] = []
    if _get_corr_ki(lasre, stab_idx, i, j, k):
        observables.append("TB")
    if _get_corr_kj(lasre, stab_idx, i, j, k):
        observables.append("RL")
    if len(observables) == 1:
        return observables[0]
    if len(observables) > 1:
        return observables
    return None


def _assign_block_for_cube(
    coord: Coord3,
    exist_k: list[Any],
    color_km: list[Any],
    color_kp: list[Any],
    n_i: int,
    n_j: int,
    n_k: int,
    hadamard_cubes: set[Coord3] | None = None,
) -> tuple[str, Coord3 | None]:
    """Assign block type for a single non-port cube.

    Parameters
    ----------
    hadamard_cubes : set[Coord3] | None
        Set of cubes where HadamardBlock should be placed.

    Returns
    -------
    block : str
        Block type for this cube.
    k_pipe_coord : Coord3 | None
        K-pipe coordinate for correlation surface lookup (if this has K+ boundary).
    """
    i, j, k = coord

    # Check for Hadamard first
    if hadamard_cubes and coord in hadamard_cubes:
        return "HadamardBlock", None

    block = "MemoryBlock"
    k_pipe_coord: Coord3 | None = None

    # Check K- and K+ boundaries
    has_k_minus_conn = _has_k_minus_connection(i, j, k, exist_k, n_i, n_j, n_k)
    has_k_plus_conn = _has_k_plus_connection(i, j, k, exist_k, n_i, n_j, n_k)

    # Get K- color (for init)
    km_val = 0
    if not has_k_minus_conn:
        km_val = _get_color_km(i, j, k - 1, color_km, n_i, n_j, n_k)
        if km_val == 0:
            km_val = _get_color_km(i, j, k, color_km, n_i, n_j, n_k)

    # Get K+ color (for measure)
    kp_val = 0
    if not has_k_plus_conn:
        kp_val = _get_color_kp(i, j, k, color_kp, n_i, n_j, n_k)

    # Determine block type based on boundaries
    if km_val != 0 and kp_val != 0:
        # Both K- and K+ boundaries: use ShortMemoryBlock
        block = _k_color_to_short_memory_block(kp_val)
        k_pipe_coord = (i, j, k)
    elif kp_val != 0:
        # Only K+ boundary: use MeasureBulkBlock
        block = _k_color_to_meas_bulk_block(kp_val)
        k_pipe_coord = (i, j, k)
    elif km_val != 0:
        # Only K- boundary: use InitBlock
        block = _k_color_to_init_block(km_val)

    return block, k_pipe_coord


def _assign_blocks_with_k_boundary(
    cubes: set[Coord3],
    port_cubes: Sequence[Coord3],
    lasre_ports: Sequence[dict[str, Any]],
    lasre: dict[str, Any],
    stabilizer: str,
    hadamard_cubes: set[Coord3] | None = None,
) -> tuple[dict[Coord3, str], set[Coord3], dict[Coord3, Coord3]]:
    """Decide block type per cube using MeasureBulk blocks for K+ boundaries.

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
    hadamard_cubes : set[Coord3] | None
        Set of cubes where HadamardBlock should be placed.

    Returns
    -------
    blocks : dict[Coord3, str]
        Block type assignment for each cube.
    added_cubes : set[Coord3]
        Always empty (no additional cubes needed with MeasureBulk blocks).
    meas_to_k_pipe_map : dict[Coord3, Coord3]
        Mapping from MeasureBulk/ShortMemory block coordinate to K-pipe coordinate.
    """
    color_km = lasre.get("ColorKM", [])
    color_kp = lasre.get("ColorKP", [])
    exist_k = lasre.get("ExistK", [])
    n_i, n_j, n_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]

    port_cube_set = set(port_cubes)
    blocks: dict[Coord3, str] = {}
    meas_to_k_pipe_map: dict[Coord3, Coord3] = {}

    # Assign blocks to existing cubes
    for coord in cubes:
        if coord in port_cube_set:
            blocks[coord] = "MemoryBlock"
            continue

        block, k_pipe_coord = _assign_block_for_cube(
            coord, exist_k, color_km, color_kp, n_i, n_j, n_k, hadamard_cubes
        )
        blocks[coord] = block
        if k_pipe_coord is not None:
            meas_to_k_pipe_map[coord] = k_pipe_coord

    # Override with port-based assignments (these take precedence)
    for idx, (port, coord) in enumerate(zip(lasre_ports, port_cubes, strict=True)):
        ch = _stab_char(stabilizer, idx)
        if port.get("e") == "-":  # input / bottom
            blocks[coord] = _init_block(ch)
        else:  # output / top
            blocks[coord] = _meas_block(ch)

    return blocks, set(), meas_to_k_pipe_map


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


def _build_toplevel_logical_observables(
    meas_to_k_pipe_map: dict[Coord3, Coord3],
    pipes: list[Pipe],
    lasre: dict[str, Any],
    stab_idx: int,
    output_port_cubes: list[Coord3] | None = None,
) -> list[dict[str, Any]]:
    """Build top-level logical_observables section for YAML output.

    Collects all cubes and pipes that have correlation surface values set,
    and groups them into a single logical_observables entry.

    Parameters
    ----------
    meas_to_k_pipe_map : dict[Coord3, Coord3]
        Mapping from measure block coordinate to original K-pipe coordinate.
    pipes : list[Pipe]
        List of pipes (axis, start, end, color).
    lasre : dict[str, Any]
        Full lasre dictionary for correlation surface lookup.
    stab_idx : int
        Stabilizer index for correlation surface lookup.
    output_port_cubes : list[Coord3] | None
        List of output port cube coordinates to include.

    Returns
    -------
    list[dict[str, Any]]
        Top-level logical_observables entries for YAML output.
    """
    cube_coords: list[list[int]] = []
    pipe_coords: list[list[list[int]]] = []

    # Collect cube coordinates with correlation surface values
    for meas_coord, k_pipe in meas_to_k_pipe_map.items():
        i, j, k = k_pipe
        if _get_corr_ki(lasre, stab_idx, i, j, k) or _get_corr_kj(lasre, stab_idx, i, j, k):
            cube_coords.append(list(meas_coord))

    # Add output port cubes
    if output_port_cubes:
        cube_coords.extend(list(coord) for coord in output_port_cubes)

    # Collect pipe coordinates with correlation surface values
    for axis, start, end, _color in pipes:
        i, j, k = start
        has_corr = False
        if axis == "I":
            has_corr = _get_corr_ij(lasre, stab_idx, i, j, k) or _get_corr_ik(lasre, stab_idx, i, j, k)
        else:  # axis == "J"
            has_corr = _get_corr_ji(lasre, stab_idx, i, j, k) or _get_corr_jk(lasre, stab_idx, i, j, k)
        if has_corr:
            pipe_coords.append([list(start), list(end)])

    if not cube_coords and not pipe_coords:
        return []

    entry: dict[str, Any] = {}
    if cube_coords:
        entry["cube"] = cube_coords
    if pipe_coords:
        entry["pipe"] = pipe_coords

    return [entry]


def _build_cube_entries(
    all_cubes: set[Coord3],
    blocks: dict[Coord3, str],
    all_boundaries: dict[Coord3, list[str]],
    meas_to_k_pipe_map: dict[Coord3, Coord3],
    lasre: dict[str, Any],
    stab_idx: int,
    invert_map: dict[Coord3, bool] | None = None,
    output_port_info: dict[Coord3, str] | None = None,
) -> list[dict[str, Any]]:
    """Build cube entry list for YAML output.

    Parameters
    ----------
    all_cubes : set[Coord3]
        All cube coordinates.
    blocks : dict[Coord3, str]
        Block type assignment for each cube.
    all_boundaries : dict[Coord3, list[str]]
        Boundary specification for each cube.
    meas_to_k_pipe_map : dict[Coord3, Coord3]
        Mapping from measure block coordinate to original K-pipe coordinate.
    lasre : dict[str, Any]
        Full lasre dictionary for correlation surface lookup.
    stab_idx : int
        Stabilizer index for correlation surface lookup.
    invert_map : dict[Coord3, bool] | None
        Mapping from cube coordinates to whether invert_ancilla_order=True.
    output_port_info : dict[Coord3, str] | None
        Mapping from output port cube coordinates to stabilizer character.

    Returns
    -------
    list[dict[str, Any]]
        Cube entries for YAML output.
    """
    if invert_map is None:
        invert_map = {}
    if output_port_info is None:
        output_port_info = {}

    cube_entries: list[dict[str, Any]] = []
    for coord in sorted(all_cubes, key=operator.itemgetter(2, 1, 0)):
        entry: dict[str, Any] = {
            "position": list(coord),
            "block": blocks[coord],
            "boundary": "".join(all_boundaries[coord]),
        }

        # Add invert_ancilla_order if True
        if invert_map.get(coord, False):
            entry["invert_ancilla_order"] = True

        # Add logical_observables for blocks with K+ boundary based on correlation surfaces
        # This includes MeasureBulk blocks and ShortMemory blocks
        measure_blocks = {
            "MeasureBulkXBlock",
            "MeasureBulkZBlock",
            "ShortXMemoryBlock",
            "ShortZMemoryBlock",
        }
        if blocks[coord] in measure_blocks and coord in meas_to_k_pipe_map:
            k_pipe = meas_to_k_pipe_map[coord]
            observables = _get_logical_observables(lasre, stab_idx, k_pipe)
            if observables is not None:
                entry["logical_observables"] = observables

        # Add logical_observables for output port MeasureBlocks
        if coord in output_port_info:
            stab_char = output_port_info[coord]
            inverted = invert_map.get(coord, False)
            token = _get_output_port_token(stab_char, all_boundaries[coord], inverted)
            if token is not None:
                entry["logical_observables"] = token

        cube_entries.append(entry)
    return cube_entries


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

    # Compute Hadamard cubes and invert_ancilla_order for all cubes
    color_km = lasre.get("ColorKM", [])
    color_kp = lasre.get("ColorKP", [])
    n_i, n_j, n_k = lasre["n_i"], lasre["n_j"], lasre["n_k"]
    hadamard_cubes, invert_map = _compute_invert_ancilla_order(
        cubes, color_km, color_kp, n_i, n_j, n_k
    )

    # Build boundaries with invert_map
    boundaries = _build_cube_boundaries(cubes, pipes, port_zdir_map, default_zdir, lasre, invert_map)

    yaml_results: list[tuple[str, str]] = []

    for stab_idx, stab in enumerate(stabilizers):
        blocks, added_cubes, meas_to_k_pipe_map = _assign_blocks_with_k_boundary(
            cubes, port_cubes, lasre_ports, lasre, stab, hadamard_cubes
        )

        # Combine original cubes with added measure cubes
        all_cubes = cubes | added_cubes

        # Add boundaries for newly added cubes (use default boundary)
        all_boundaries = dict(boundaries)
        for coord in added_cubes:
            if coord not in all_boundaries:
                all_boundaries[coord] = _base_boundary(default_zdir)

        # Collect output port info for this stabilizer
        output_port_info: dict[Coord3, str] = {}
        output_port_cubes: list[Coord3] = []
        for idx, (port, coord) in enumerate(zip(lasre_ports, port_cubes, strict=True)):
            ch = _stab_char(stab, idx)
            if port.get("e") != "-" and ch.upper() in {"X", "Z"}:
                output_port_info[coord] = ch
                output_port_cubes.append(coord)

        cube_entries = _build_cube_entries(
            all_cubes, blocks, all_boundaries, meas_to_k_pipe_map, lasre, stab_idx, invert_map,
            output_port_info,
        )

        # Generate single-layer pipe structure using short memory blocks
        pipe_entries: list[dict[str, Any]] = []
        for axis, start, end, color in pipes:
            boundary = _pipe_boundary(axis, color)
            block = _short_memory_pipe_block(axis, color)
            entry: dict[str, Any] = {
                "start": list(start),
                "end": list(end),
                "block": block,
                "boundary": boundary,
            }

            # Add logical_observables based on correlation surfaces
            observables = _get_pipe_logical_observables(lasre, stab_idx, axis, start, block)
            if observables is not None:
                entry["logical_observables"] = observables

            pipe_entries.append(entry)

        # Build top-level logical_observables section
        toplevel_observables = _build_toplevel_logical_observables(
            meas_to_k_pipe_map, pipes, lasre, stab_idx, output_port_cubes
        )

        desc_base = description or name_prefix
        canvas_dict: dict[str, Any] = {
            "name": f"{name_prefix}_{stab_idx}",
            "description": f"{desc_base} | stabilizer: {stab}",
            "layout": "rotated_surface_code",
            "cube": cube_entries,
            "pipe": pipe_entries,
        }
        if toplevel_observables:
            canvas_dict["logical_observables"] = toplevel_observables

        yaml_text = yaml.safe_dump(
            canvas_dict,
            sort_keys=False,
            width=1000,  # encourage inline short lists like [0, 0, 0]
            default_flow_style=False,
        )
        yaml_results.append((canvas_dict["name"], yaml_text))

    return yaml_results
