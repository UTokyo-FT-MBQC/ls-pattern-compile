"""Return data and X/Z ancilla coordination for rotated surface code layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.mytype import Coord3D

if TYPE_CHECKING:
    from collections.abc import Mapping


def rotated_surface_code_layout(
    code_distance: int,
    global_pos: Coord3D | tuple[Coord3D, Coord3D],
    boundary: Mapping[BoundarySide, EdgeSpecValue],
) -> tuple[set[Coord3D], set[Coord3D], set[Coord3D]]:
    """Get data and ancilla coordinates for rotated surface code cube layout.

    Parameters
    ----------
    code_distance : int
        The code distance of the rotated surface code.
    global_pos : Coord3D | tuple[Coord3D, Coord3D]
        The global (x, y, z) position offset for the layout.
        If a tuple is provided, the pipe coordinates are generated.
    boundary : Mapping[BoundarySide, EdgeSpecValue]
        The boundary specifications for the rotated surface code.

    Returns
    -------
    tuple[set[Coord3D], set[Coord3D], set[Coord3D]]
        A tuple containing sets of coordinates for data qubits, X ancilla qubits, and Z ancilla qubits.
    """
    data_coords: set[Coord3D] = set()
    x_ancilla_coords: set[Coord3D] = set()
    z_ancilla_coords: set[Coord3D] = set()

    offset_x = 2 * (code_distance + 1) * global_pos[0]
    offset_y = 2 * (code_distance + 1) * global_pos[1]
    offset_z = 2 * code_distance * global_pos[2]
    global_pos = (offset_x, offset_y, offset_z)

    # process internal coordinates
    for x in range(2 * code_distance):
        for y in range(2 * code_distance):
            if x % 2 == 0 and y % 2 == 0:
                # Data qubit
                coord = (global_pos[0] + x, global_pos[1] + y, global_pos[2])
                data_coords.add(coord)
            elif (x + y) % 4 == 0:
                # X ancilla qubit
                coord = (global_pos[0] + x, global_pos[1] + y, global_pos[2])
                x_ancilla_coords.add(coord)
            elif (x + y) % 4 == 2:
                # Z ancilla qubit
                coord = (global_pos[0] + x, global_pos[1] + y, global_pos[2])
                z_ancilla_coords.add(coord)

    # process corner data qubits
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove(
            (global_pos[0] + 2 * (code_distance - 1), global_pos[1] - 2 * (code_distance - 1), global_pos[2])
        )
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove((global_pos[0], global_pos[1], global_pos[2]))
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove((global_pos[0], global_pos[1] + 2 * (code_distance - 1), global_pos[2]))
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove((global_pos[0] + 2 * (code_distance - 1), global_pos[1], global_pos[2]))

    # process boundary coordinates
    for x in range(2 * code_distance):
        if boundary[BoundarySide.TOP] == EdgeSpecValue.X or EdgeSpecValue.O:
            if x % 4 == 1:
                coord = (global_pos[0] + x, global_pos[1] - 1, global_pos[2])
                x_ancilla_coords.add(coord)
        if boundary[BoundarySide.TOP] == EdgeSpecValue.Z or EdgeSpecValue.O:
            if x % 4 == 3:
                coord = (global_pos[0] + x, global_pos[1] - 1, global_pos[2])
                z_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.X or EdgeSpecValue.O:
            if x % 4 == 3:
                coord = (global_pos[0] + x, global_pos[1] + 2 * (code_distance - 1), global_pos[2])
                x_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.Z or EdgeSpecValue.O:
            if x % 4 == 1:
                coord = (global_pos[0] + x, global_pos[1] + 2 * (code_distance - 1), global_pos[2])
                z_ancilla_coords.add(coord)

    for y in range(2 * code_distance):
        if boundary[BoundarySide.LEFT] == EdgeSpecValue.X or EdgeSpecValue.O:
            if y % 4 == 1:
                coord = (global_pos[0] - 1, global_pos[1] + y, global_pos[2])
                x_ancilla_coords.add(coord)
        if boundary[BoundarySide.LEFT] == EdgeSpecValue.Z or EdgeSpecValue.O:
            if y % 4 == 3:
                coord = (global_pos[0] - 1, global_pos[1] + y, global_pos[2])
                z_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] == EdgeSpecValue.X or EdgeSpecValue.O:
            if y % 4 == 3:
                coord = (global_pos[0] + 2 * (code_distance - 1), global_pos[1] + y, global_pos[2])
                x_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] == EdgeSpecValue.Z or EdgeSpecValue.O:
            if y % 4 == 1:
                coord = (global_pos[0] + 2 * (code_distance - 1), global_pos[1] + y, global_pos[2])
                z_ancilla_coords.add(coord)

    return data_coords, x_ancilla_coords, z_ancilla_coords
