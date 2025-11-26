"""Return data and X/Z ancilla coordination for rotated surface code layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.mytype import DIRECTION2D, AxisDIRECTION2D, Coord2D, Coord3D

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet


ANCILLA_EDGE_X = (
    (1, 1),
    (1, -1),
    (-1, -1),
    (-1, 1),
)  # order is optimized for the distance
ANCILLA_EDGE_Z = (
    (1, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
)  # order is optimized for the distance


def _range_step2(start: int, end: int) -> list[int]:
    """Inclusive range helper that moves in steps of 2 toward ``end``."""

    if start == end:
        return [start]
    step = 2 if end > start else -2
    return list(range(start, end + step, step))


def _merge_segments(path: Sequence[Coord2D], segment: Sequence[Coord2D]) -> None:
    """Append ``segment`` to ``path`` without duplicating the joint point."""

    if not segment:
        return
    if path and path[-1] == segment[0]:
        path.extend(segment[1:])
    else:
        path.extend(segment)


def _data_path_between_boundaries(
    data_coords: AbstractSet[Coord2D],
    side_a: BoundarySide,
    side_b: BoundarySide,
) -> list[Coord2D]:
    """Return ordered data-qubit coordinates connecting two boundaries through the patch center.

    The path follows Manhattan geometry on the data lattice:
    - Opposite sides (TOP-BOTTOM or LEFT-RIGHT): a straight center line.
    - Adjacent sides (e.g., TOP-LEFT): an ``L`` shape via the patch center,
      combining the central row and column.

    Parameters
    ----------
    data_coords : collections.abc.Set[Coord2D]
        Data-qubit coordinates for the patch (cube or pipe).
    side_a, side_b : BoundarySide
        Two boundary sides chosen from TOP, BOTTOM, LEFT, RIGHT.

    Returns
    -------
    list[Coord2D]
        Ordered coordinates from ``side_a`` toward ``side_b``.
    """

    if not data_coords:
        return []

    xs = sorted({c.x for c in data_coords})
    ys = sorted({c.y for c in data_coords})

    min_x = xs[0]
    max_x = xs[-1]
    min_y = ys[0]
    max_y = ys[-1]

    center_x = xs[len(xs) // 2]
    center_y = ys[len(ys) // 2]

    boundary_pos = {
        BoundarySide.TOP: min_y,
        BoundarySide.BOTTOM: max_y,
        BoundarySide.LEFT: min_x,
        BoundarySide.RIGHT: max_x,
    }

    vertical = {BoundarySide.TOP, BoundarySide.BOTTOM}
    horizontal = {BoundarySide.LEFT, BoundarySide.RIGHT}

    def vertical_segment(src: BoundarySide, dst: BoundarySide) -> list[Coord2D]:
        y_start = boundary_pos[src]
        y_end = boundary_pos[dst]
        return [Coord2D(center_x, y) for y in _range_step2(y_start, y_end) if Coord2D(center_x, y) in data_coords]

    def horizontal_segment(src: BoundarySide, dst: BoundarySide) -> list[Coord2D]:
        x_start = boundary_pos[src]
        x_end = boundary_pos[dst]
        return [Coord2D(x, center_y) for x in _range_step2(x_start, x_end) if Coord2D(x, center_y) in data_coords]

    # Opposite sides -> straight line
    if side_a in vertical and side_b in vertical:
        return vertical_segment(side_a, side_b)
    if side_a in horizontal and side_b in horizontal:
        return horizontal_segment(side_a, side_b)

    # Adjacent sides -> L via center, ordered from side_a to side_b
    path: list[Coord2D] = []
    if side_a in vertical and side_b in horizontal:
        # side_a to center (vertical), then center to side_b (horizontal)
        v_seg = [
            Coord2D(center_x, y)
            for y in _range_step2(boundary_pos[side_a], center_y)
            if Coord2D(center_x, y) in data_coords
        ]
        h_seg = [
            Coord2D(x, center_y)
            for x in _range_step2(center_x, boundary_pos[side_b])
            if Coord2D(x, center_y) in data_coords
        ]
        _merge_segments(path, v_seg)
        _merge_segments(path, h_seg)
        return path

    if side_a in horizontal and side_b in vertical:
        # side_a to center (horizontal), then center to side_b (vertical)
        h_seg = [
            Coord2D(x, center_y)
            for x in _range_step2(boundary_pos[side_a], center_x)
            if Coord2D(x, center_y) in data_coords
        ]
        v_seg = [
            Coord2D(center_x, y)
            for y in _range_step2(center_y, boundary_pos[side_b])
            if Coord2D(center_x, y) in data_coords
        ]
        _merge_segments(path, h_seg)
        _merge_segments(path, v_seg)
        return path

    msg = f"Unsupported boundary pair: {side_a}, {side_b}."
    raise ValueError(msg)


def rotated_surface_code_layout(  # noqa: C901
    code_distance: int,
    global_pos: Coord2D,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
) -> tuple[set[Coord2D], set[Coord2D], set[Coord2D]]:
    """Get data and ancilla coordinates for rotated surface code cube layout.

    Parameters
    ----------
    code_distance : int
        The code distance of the rotated surface code.
    global_pos : Coord2D
        The global (x, y) position offset for the layout.
    boundary : Mapping[BoundarySide, EdgeSpecValue]
        The boundary specifications for the rotated surface code.

    Returns
    -------
    tuple[set[Coord2D], set[Coord2D], set[Coord2D]]
        A tuple containing sets of coordinates for data qubits, X ancilla qubits, and Z ancilla qubits.
    """
    data_coords: set[Coord2D] = set()
    x_ancilla_coords: set[Coord2D] = set()
    z_ancilla_coords: set[Coord2D] = set()

    # Calculate offset based on global position

    offset_x = 2 * (code_distance + 1) * global_pos.x
    offset_y = 2 * (code_distance + 1) * global_pos.y
    offset_pos = Coord2D(offset_x, offset_y)

    # process internal coordinates
    for x in range(2 * code_distance - 1):
        for y in range(2 * code_distance - 1):
            if x % 2 == 0 and y % 2 == 0:
                # Data qubit
                coord = Coord2D(offset_pos.x + x, offset_pos.y + y)
                data_coords.add(coord)
            elif (x + y) % 4 == 0:
                # X ancilla qubit
                coord = Coord2D(offset_pos.x + x, offset_pos.y + y)
                x_ancilla_coords.add(coord)
            elif (x + y) % 4 == 2:  # noqa: PLR2004
                # Z ancilla qubit
                coord = Coord2D(offset_pos.x + x, offset_pos.y + y)
                z_ancilla_coords.add(coord)

    # process corner data qubits
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove(Coord2D(offset_pos.x + 2 * (code_distance - 1), offset_pos.y))
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove(Coord2D(offset_pos.x, offset_pos.y + 2 * (code_distance - 1)))
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove(Coord2D(offset_pos.x, offset_pos.y))
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove(Coord2D(offset_pos.x + 2 * (code_distance - 1), offset_pos.y + 2 * (code_distance - 1)))

    # process boundary coordinates
    for x in range(1, 2 * (code_distance - 1)):
        if boundary[BoundarySide.TOP] in {EdgeSpecValue.X, EdgeSpecValue.O} and x % 4 == 1:
            coord = Coord2D(offset_pos.x + x, offset_pos.y - 1)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.TOP] in {EdgeSpecValue.Z, EdgeSpecValue.O} and x % 4 == 3:  # noqa: PLR2004
            coord = Coord2D(offset_pos.x + x, offset_pos.y - 1)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] in {EdgeSpecValue.X, EdgeSpecValue.O} and x % 4 == 3:  # noqa: PLR2004
            coord = Coord2D(offset_pos.x + x, offset_pos.y + 2 * code_distance - 1)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] in {EdgeSpecValue.Z, EdgeSpecValue.O} and x % 4 == 1:
            coord = Coord2D(offset_pos.x + x, offset_pos.y + 2 * code_distance - 1)
            z_ancilla_coords.add(coord)

    for y in range(1, 2 * (code_distance - 1)):
        if boundary[BoundarySide.LEFT] in {EdgeSpecValue.X, EdgeSpecValue.O} and y % 4 == 1:
            coord = Coord2D(offset_pos.x - 1, offset_pos.y + y)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.LEFT] in {EdgeSpecValue.Z, EdgeSpecValue.O} and y % 4 == 3:  # noqa: PLR2004
            coord = Coord2D(offset_pos.x - 1, offset_pos.y + y)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] in {EdgeSpecValue.X, EdgeSpecValue.O} and y % 4 == 3:  # noqa: PLR2004
            coord = Coord2D(offset_pos.x + 2 * code_distance - 1, offset_pos.y + y)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] in {EdgeSpecValue.Z, EdgeSpecValue.O} and y % 4 == 1:
            coord = Coord2D(offset_pos.x + 2 * code_distance - 1, offset_pos.y + y)
            z_ancilla_coords.add(coord)

    # process corner ancilla qubits
    if boundary[BoundarySide.RIGHT] == EdgeSpecValue.O:
        if boundary[BoundarySide.TOP] == EdgeSpecValue.X or boundary[BoundarySide.TOP] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x + 2 * code_distance - 1, offset_pos.y - 1)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.Z or boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x + 2 * code_distance - 1, offset_pos.y + 2 * code_distance - 1)
            z_ancilla_coords.add(coord)

    if boundary[BoundarySide.LEFT] == EdgeSpecValue.O:
        if boundary[BoundarySide.TOP] == EdgeSpecValue.Z or boundary[BoundarySide.TOP] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x - 1, offset_pos.y - 1)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.X or boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x - 1, offset_pos.y + 2 * code_distance - 1)
            x_ancilla_coords.add(coord)

    if boundary[BoundarySide.TOP] == EdgeSpecValue.O:
        if boundary[BoundarySide.LEFT] == EdgeSpecValue.Z or boundary[BoundarySide.LEFT] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x - 1, offset_pos.y - 1)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] == EdgeSpecValue.X or boundary[BoundarySide.RIGHT] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x + 2 * code_distance - 1, offset_pos.y - 1)
            x_ancilla_coords.add(coord)

    if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O:
        if boundary[BoundarySide.LEFT] == EdgeSpecValue.X or boundary[BoundarySide.LEFT] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x - 1, offset_pos.y + 2 * code_distance - 1)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] == EdgeSpecValue.Z or boundary[BoundarySide.RIGHT] == EdgeSpecValue.O:
            coord = Coord2D(offset_pos.x + 2 * code_distance - 1, offset_pos.y + 2 * code_distance - 1)
            z_ancilla_coords.add(coord)

    return data_coords, x_ancilla_coords, z_ancilla_coords


def boundary_data_path_cube(
    code_distance: int,
    global_pos: Coord2D,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
    side_a: BoundarySide,
    side_b: BoundarySide,
) -> list[Coord2D]:
    """Get data-qubit coordinates connecting two boundaries inside a cube patch.

    Parameters
    ----------
    code_distance : int
        Code distance of the patch.
    global_pos : Coord2D
        Patch XY anchor (same as ``rotated_surface_code_layout``).
    boundary : Mapping[BoundarySide, EdgeSpecValue]
        Boundary specification for the patch (passed to layout for completeness).
    side_a, side_b : BoundarySide
        Two boundaries among TOP, BOTTOM, LEFT, RIGHT.

    Returns
    -------
    list[Coord2D]
        Ordered data-qubit coordinates forming a path from ``side_a`` to ``side_b`` via the patch center.
    """

    data_coords, _, _ = rotated_surface_code_layout(code_distance, global_pos, boundary)
    return _data_path_between_boundaries(data_coords, side_a, side_b)


def rotated_surface_code_pipe_layout(  # noqa: C901
    code_distance: int,
    global_pos_source: Coord3D,
    global_pos_target: Coord3D,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
) -> tuple[set[Coord2D], set[Coord2D], set[Coord2D]]:
    """Get data and ancilla coordinates for rotated surface code pipe layout.

    Parameters
    ----------
    code_distance : int
        The code distance of the rotated surface code.
    global_pos_source : Coord3D
        The global (x, y, z) position of the pipe source.
    global_pos_target : Coord3D
        The global (x, y, z) position of the pipe target.
    boundary : Mapping[BoundarySide, EdgeSpecValue]
        The boundary specifications for the rotated surface code.

    Returns
    -------
    tuple[set[Coord2D], set[Coord2D], set[Coord2D]]
        A tuple containing sets of coordinates for data qubits, X ancilla qubits, and Z ancilla qubits.
    """
    pipe_dir = pipe_direction(boundary)
    pipe_offset_dir = pipe_offset(code_distance, global_pos_source, global_pos_target)

    offset_x = 2 * (code_distance + 1) * global_pos_source.x
    offset_y = 2 * (code_distance + 1) * global_pos_source.y
    if pipe_offset_dir == DIRECTION2D.RIGHT:
        offset_x += 2 * code_distance
    if pipe_offset_dir == DIRECTION2D.LEFT:
        offset_x -= 2
    if pipe_offset_dir == DIRECTION2D.TOP:
        offset_y -= 2
    if pipe_offset_dir == DIRECTION2D.BOTTOM:
        offset_y += 2 * code_distance
    offset_pos = Coord2D(offset_x, offset_y)

    data_coords: set[Coord2D] = set()
    x_ancilla_coords: set[Coord2D] = set()
    z_ancilla_coords: set[Coord2D] = set()
    for l in range(2 * (code_distance - 1)):  # long line  # noqa: E741
        for s in range(-1, 2):  # short line
            if pipe_dir == AxisDIRECTION2D.H:
                x = offset_pos.x + s
                y = offset_pos.y + l
            else:  # pipe_dir == AxisDIRECTION2D.V:
                x = offset_pos.x + l
                y = offset_pos.y + s
            if x % 2 == 0 and y % 2 == 0:
                # Data qubit
                coord = Coord2D(x, y)
                data_coords.add(coord)
            elif (x + y) % 4 == 0:
                # X ancilla qubit
                coord = Coord2D(x, y)
                x_ancilla_coords.add(coord)
            elif (x + y) % 4 == 2:  # noqa: PLR2004
                # Z ancilla qubit
                coord = Coord2D(x, y)
                z_ancilla_coords.add(coord)

    # process boundary coordinates
    for i in (-1, 1):
        if pipe_dir == AxisDIRECTION2D.H:
            x = offset_pos.x + i
            y_top = offset_pos.y - 1
            y_bottom = offset_pos.y + 2 * code_distance - 1
            if boundary[BoundarySide.TOP] == EdgeSpecValue.X and (x + y_top) % 4 == 0:
                x_ancilla_coords.add(Coord2D(x, y_top))
            if boundary[BoundarySide.TOP] == EdgeSpecValue.Z and (x + y_top) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add(Coord2D(x, y_top))
            if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.X and (x + y_bottom) % 4 == 0:
                x_ancilla_coords.add(Coord2D(x, y_bottom))
            if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.Z and (x + y_bottom) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add(Coord2D(x, y_bottom))
        else:  # pipe_dir == AxisDIRECTION2D.V:
            x_left = offset_pos.x - 1
            x_right = offset_pos.x + 2 * code_distance - 1
            y = offset_pos.y + i
            if boundary[BoundarySide.LEFT] == EdgeSpecValue.X and (x_left + y) % 4 == 0:
                x_ancilla_coords.add(Coord2D(x_left, y))
            if boundary[BoundarySide.LEFT] == EdgeSpecValue.Z and (x_left + y) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add(Coord2D(x_left, y))
            if boundary[BoundarySide.RIGHT] == EdgeSpecValue.X and (x_right + y) % 4 == 0:
                x_ancilla_coords.add(Coord2D(x_right, y))
            if boundary[BoundarySide.RIGHT] == EdgeSpecValue.Z and (x_right + y) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add(Coord2D(x_right, y))

    return data_coords, x_ancilla_coords, z_ancilla_coords


def boundary_data_path_pipe(
    code_distance: int,
    global_pos_source: Coord3D,
    global_pos_target: Coord3D,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
    side_a: BoundarySide,
    side_b: BoundarySide,
) -> list[Coord2D]:
    """Get data-qubit coordinates connecting two boundaries inside a pipe patch.

    The pipe orientation and offset are derived from ``boundary`` and the
    source/target positions in the same way as ``rotated_surface_code_pipe_layout``.

    Parameters
    ----------
    code_distance : int
        Code distance of the pipe.
    global_pos_source : Coord3D
        Source patch coordinate (same semantics as pipe layout helper).
    global_pos_target : Coord3D
        Target patch coordinate.
    boundary : Mapping[BoundarySide, EdgeSpecValue]
        Boundary specification for the pipe.
    side_a, side_b : BoundarySide
        Two boundaries among TOP, BOTTOM, LEFT, RIGHT.

    Returns
    -------
    list[Coord2D]
        Ordered data-qubit coordinates forming a path from ``side_a`` to ``side_b`` via the pipe center.
    """

    data_coords, _, _ = rotated_surface_code_pipe_layout(code_distance, global_pos_source, global_pos_target, boundary)
    return _data_path_between_boundaries(data_coords, side_a, side_b)


# NOTE: This is redundant to the current 3D coord based implementation, but kept for
# long-range connectivity which might be added later.
def pipe_direction(boundary: Mapping[BoundarySide, EdgeSpecValue]) -> AxisDIRECTION2D:
    """Determine the pipe direction based on boundary specifications.

    Parameters
    ----------
    boundary : Mapping[BoundarySide, EdgeSpecValue]
        The boundary specifications for the rotated surface code.

    Returns
    -------
    AxisDIRECTION2D
        The pipe direction of the pipe based on the boundary specifications.

    Raises
    ------
    ValueError
        If the boundary specifications do not match a valid pipe direction.
    """
    horizontal: bool = False
    vertical: bool = False
    if boundary[BoundarySide.TOP] == EdgeSpecValue.O and boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O:
        horizontal = True
    if boundary[BoundarySide.LEFT] == EdgeSpecValue.O and boundary[BoundarySide.RIGHT] == EdgeSpecValue.O:
        vertical = True
    if horizontal and not vertical:
        return AxisDIRECTION2D.H
    if vertical and not horizontal:
        return AxisDIRECTION2D.V
    if horizontal and vertical:
        msg = "Both horizontal and vertical boundaries cannot be open for pipe layout."
        raise ValueError(msg)
    msg = "Either top-bottom or left-right boundaries must be open for pipe layout."
    raise ValueError(msg)


def pipe_offset(
    code_distance: int,
    global_pos_source: Coord3D,
    global_pos_target: Coord3D,
) -> DIRECTION2D:
    """Calculate the pipe offset from source and target positions.

    Parameters
    ----------
    code_distance : int
        The code distance of the rotated surface code.
    global_pos_source : Coord3D
        The global (x, y, z) position of the pipe source.
    global_pos_target : Coord3D
        The global (x, y, z) position of the pipe target.
    Returns
    -------
    DIRECTION2D
        The direction of the pipe offset from source to target.

    Raises
    ------
    ValueError
        If the pipe offset from source to target is invalid.
    """
    dx = global_pos_target.x - global_pos_source.x
    dy = global_pos_target.y - global_pos_source.y
    if dx == 1 and dy == 0:
        return DIRECTION2D.RIGHT
    if dx == -1 and dy == 0:
        return DIRECTION2D.LEFT
    if dx == 0 and dy == 1:
        return DIRECTION2D.TOP
    if dx == 0 and dy == -1:
        return DIRECTION2D.BOTTOM
    msg = f"Invalid pipe offset: source {global_pos_source}, target {global_pos_target}, code distance {code_distance}."
    raise ValueError(msg)
