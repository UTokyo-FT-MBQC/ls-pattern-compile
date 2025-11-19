"""Return data and X/Z ancilla coordination for rotated surface code layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.mytype import AxisDIRECTION2D, Coord3D, Coord2D, DIRECTION2D

if TYPE_CHECKING:
    from collections.abc import Mapping


def rotated_surface_code_layout(  # noqa: C901
    code_distance: int,
    global_pos: Coord3D,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
) -> tuple[set[Coord2D], set[Coord2D], set[Coord2D]]:
    """Get data and ancilla coordinates for rotated surface code cube layout.

    Parameters
    ----------
    code_distance : int
        The code distance of the rotated surface code.
    global_pos : Coord3D
        The global (x, y, z) position offset for the layout.
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
                coord = (offset_pos.x + x, offset_pos.y + y)
                data_coords.add(coord)
            elif (x + y) % 4 == 0:
                # X ancilla qubit
                coord = (offset_pos.x + x, offset_pos.y + y)
                x_ancilla_coords.add(coord)
            elif (x + y) % 4 == 2:  # noqa: PLR2004
                # Z ancilla qubit
                coord = (offset_pos.x + x, offset_pos.y + y)
                z_ancilla_coords.add(coord)

    # process corner data qubits
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove((offset_pos.x + 2 * (code_distance - 1), offset_pos.y - 2 * (code_distance - 1)))
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove((offset_pos.x, offset_pos.y))
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove((offset_pos.x, offset_pos.y + 2 * (code_distance - 1)))
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove((offset_pos.x + 2 * (code_distance - 1), offset_pos.y))

    # process boundary coordinates
    for x in range(1, 2 * (code_distance - 1)):
        if boundary[BoundarySide.TOP] in {EdgeSpecValue.X, EdgeSpecValue.O} and x % 4 == 1:
            coord = (offset_pos.x + x, offset_pos.y - 1)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.TOP] in {EdgeSpecValue.Z, EdgeSpecValue.O} and x % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x + x, offset_pos.y - 1)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] in {EdgeSpecValue.X, EdgeSpecValue.O} and x % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x + x, offset_pos.y + 2 * (code_distance - 1))
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] in {EdgeSpecValue.Z, EdgeSpecValue.O} and x % 4 == 1:
            coord = (offset_pos.x + x, offset_pos.y + 2 * (code_distance - 1))
            z_ancilla_coords.add(coord)

    for y in range(1, 2 * (code_distance - 1)):
        if boundary[BoundarySide.LEFT] in {EdgeSpecValue.X, EdgeSpecValue.O} and y % 4 == 1:
            coord = (offset_pos.x - 1, offset_pos.y + y)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.LEFT] in {EdgeSpecValue.Z, EdgeSpecValue.O} and y % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x - 1, offset_pos.y + y)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] in {EdgeSpecValue.X, EdgeSpecValue.O} and y % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x + 2 * (code_distance - 1), offset_pos.y + y)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] in {EdgeSpecValue.Z, EdgeSpecValue.O} and y % 4 == 1:
            coord = (offset_pos.x + 2 * (code_distance - 1), offset_pos.y + y)
            z_ancilla_coords.add(coord)

    # process corner ancilla qubits
    if boundary[BoundarySide.RIGHT] == EdgeSpecValue.O:
        if boundary[BoundarySide.TOP] == EdgeSpecValue.X or boundary[BoundarySide.TOP] == EdgeSpecValue.O:
            coord = (offset_pos.x + 2 * code_distance - 1, offset_pos.y - 1)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.Z or boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O:
            coord = (offset_pos.x + 2 * code_distance - 1, offset_pos.y + 2 * code_distance - 1)
            z_ancilla_coords.add(coord)

    if boundary[BoundarySide.LEFT] == EdgeSpecValue.O:
        if boundary[BoundarySide.TOP] == EdgeSpecValue.Z or boundary[BoundarySide.TOP] == EdgeSpecValue.O:
            coord = (offset_pos.x - 1, offset_pos.y - 1)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.X or boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O:
            coord = (offset_pos.x - 1, offset_pos.y + 2 * code_distance - 1)
            x_ancilla_coords.add(coord)

    if boundary[BoundarySide.TOP] == EdgeSpecValue.O:
        if boundary[BoundarySide.LEFT] == EdgeSpecValue.Z or boundary[BoundarySide.LEFT] == EdgeSpecValue.O:
            coord = (offset_pos.x - 1, offset_pos.y - 1)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] == EdgeSpecValue.X or boundary[BoundarySide.RIGHT] == EdgeSpecValue.O:
            coord = (offset_pos.x + 2 * code_distance - 1, offset_pos.y - 1)
            x_ancilla_coords.add(coord)

    if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O:
        if boundary[BoundarySide.LEFT] == EdgeSpecValue.X or boundary[BoundarySide.LEFT] == EdgeSpecValue.O:
            coord = (offset_pos.x - 1, offset_pos.y + 2 * code_distance - 1)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] == EdgeSpecValue.Z or boundary[BoundarySide.RIGHT] == EdgeSpecValue.O:
            coord = (offset_pos.x + 2 * code_distance - 1, offset_pos.y + 2 * code_distance - 1)
            z_ancilla_coords.add(coord)

    return data_coords, x_ancilla_coords, z_ancilla_coords


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
    offset_pos = Coord3D(offset_x, offset_y)

    data_coords: set[Coord3D] = set()
    x_ancilla_coords: set[Coord3D] = set()
    z_ancilla_coords: set[Coord3D] = set()

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
                coord = (x, y)
                data_coords.add(coord)
            elif (x + y) % 4 == 0:
                # X ancilla qubit
                coord = (x, y)
                x_ancilla_coords.add(coord)
            elif (x + y) % 4 == 2:  # noqa: PLR2004
                # Z ancilla qubit
                coord = (x, y)
                z_ancilla_coords.add(coord)

    # process boundary coordinates
    for i in (-1, 1):
        if pipe_dir == AxisDIRECTION2D.H:
            x = offset_pos.x + i
            y_top = offset_pos.y - 1
            y_bottom = offset_pos.y + 2 * code_distance - 1
            if boundary[BoundarySide.TOP] == EdgeSpecValue.X and (x + y_top) % 4 == 0:
                x_ancilla_coords.add((x, y_top))
            if boundary[BoundarySide.TOP] == EdgeSpecValue.Z and (x + y_top) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add((x, y_top))
            if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.X and (x + y_bottom) % 4 == 0:
                x_ancilla_coords.add((x, y_bottom))
            if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.Z and (x + y_bottom) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add((x, y_bottom))
        else:  # pipe_dir == AxisDIRECTION2D.V:
            x_left = offset_pos.x - 1
            x_right = offset_pos.x + 2 * code_distance - 1
            y = offset_pos.y + i
            if boundary[BoundarySide.LEFT] == EdgeSpecValue.X and (x_left + y) % 4 == 0:
                x_ancilla_coords.add((x_left, y))
            if boundary[BoundarySide.LEFT] == EdgeSpecValue.Z and (x_left + y) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add((x_left, y))
            if boundary[BoundarySide.RIGHT] == EdgeSpecValue.X and (x_right + y) % 4 == 0:
                x_ancilla_coords.add((x_right, y))
            if boundary[BoundarySide.RIGHT] == EdgeSpecValue.Z and (x_right + y) % 4 == 2:  # noqa: PLR2004
                z_ancilla_coords.add((x_right, y))

    return data_coords, x_ancilla_coords, z_ancilla_coords


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
