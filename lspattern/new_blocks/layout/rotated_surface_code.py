"""Return data and X/Z ancilla coordination for rotated surface code layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.mytype import Coord3D

if TYPE_CHECKING:
    from collections.abc import Mapping


def rotated_surface_code_layout(  # noqa: C901
    code_distance: int,
    global_pos: Coord3D,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
) -> tuple[set[Coord3D], set[Coord3D], set[Coord3D]]:
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
    tuple[set[Coord3D], set[Coord3D], set[Coord3D]]
        A tuple containing sets of coordinates for data qubits, X ancilla qubits, and Z ancilla qubits.
    """
    data_coords: set[Coord3D] = set()
    x_ancilla_coords: set[Coord3D] = set()
    z_ancilla_coords: set[Coord3D] = set()

    # Calculate offset based on global position

    offset_x = 2 * (code_distance + 1) * global_pos.x
    offset_y = 2 * (code_distance + 1) * global_pos.y
    offset_z = 2 * code_distance * global_pos.z
    offset_pos = Coord3D(offset_x, offset_y, offset_z)

    # process internal coordinates
    for x in range(2 * code_distance):
        for y in range(2 * code_distance):
            if x % 2 == 0 and y % 2 == 0:
                # Data qubit
                coord = (offset_pos.x + x, offset_pos.y + y, offset_pos.z)
                data_coords.add(coord)
            elif (x + y) % 4 == 0:
                # X ancilla qubit
                coord = (offset_pos.x + x, offset_pos.y + y, offset_pos.z)
                x_ancilla_coords.add(coord)
            elif (x + y) % 4 == 2:  # noqa: PLR2004
                # Z ancilla qubit
                coord = (offset_pos.x + x, offset_pos.y + y, offset_pos.z)
                z_ancilla_coords.add(coord)

    # process corner data qubits
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove(
            (offset_pos.x + 2 * (code_distance - 1), offset_pos.y - 2 * (code_distance - 1), offset_pos.z)
        )
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.Z, EdgeSpecValue.Z):
        data_coords.remove((offset_pos.x, offset_pos.y, offset_pos.z))
    if (boundary[BoundarySide.TOP], boundary[BoundarySide.LEFT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove((offset_pos.x, offset_pos.y + 2 * (code_distance - 1), offset_pos.z))
    if (boundary[BoundarySide.BOTTOM], boundary[BoundarySide.RIGHT]) == (EdgeSpecValue.X, EdgeSpecValue.X):
        data_coords.remove((offset_pos.x + 2 * (code_distance - 1), offset_pos.y, offset_pos.z))
    # process boundary coordinates
    for x in range(2 * code_distance):
        if boundary[BoundarySide.TOP] in {EdgeSpecValue.X, EdgeSpecValue.O} and x % 4 == 1:
            coord = (offset_pos.x + x, offset_pos.y - 1, offset_pos.z)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.TOP] in {EdgeSpecValue.Z, EdgeSpecValue.O} and x % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x + x, offset_pos.y - 1, offset_pos.z)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] in {EdgeSpecValue.X, EdgeSpecValue.O} and x % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x + x, offset_pos.y + 2 * (code_distance - 1), offset_pos.z)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.BOTTOM] in {EdgeSpecValue.Z, EdgeSpecValue.O} and x % 4 == 1:
            coord = (offset_pos.x + x, offset_pos.y + 2 * (code_distance - 1), offset_pos.z)
            z_ancilla_coords.add(coord)

    for y in range(2 * code_distance):
        if boundary[BoundarySide.LEFT] in {EdgeSpecValue.X, EdgeSpecValue.O} and y % 4 == 1:
            coord = (offset_pos.x - 1, offset_pos.y + y, offset_pos.z)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.LEFT] in {EdgeSpecValue.Z, EdgeSpecValue.O} and y % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x - 1, offset_pos.y + y, offset_pos.z)
            z_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] in {EdgeSpecValue.X, EdgeSpecValue.O} and y % 4 == 3:  # noqa: PLR2004
            coord = (offset_pos.x + 2 * (code_distance - 1), offset_pos.y + y, offset_pos.z)
            x_ancilla_coords.add(coord)
        if boundary[BoundarySide.RIGHT] in {EdgeSpecValue.Z, EdgeSpecValue.O} and y % 4 == 1:
            coord = (offset_pos.x + 2 * (code_distance - 1), offset_pos.y + y, offset_pos.z)
            z_ancilla_coords.add(coord)

    return data_coords, x_ancilla_coords, z_ancilla_coords
