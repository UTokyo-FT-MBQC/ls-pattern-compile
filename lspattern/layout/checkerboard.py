"""Checkerboard pattern generation for surface code layouts.

This module provides utilities for generating qubit coordinates
following the checkerboard pattern used in rotated surface codes.
"""

from __future__ import annotations

from lspattern.layout.coordinates import PatchBounds, PatchCoordinates
from lspattern.mytype import Coord2D


def generate_checkerboard_coords(bounds: PatchBounds) -> PatchCoordinates:
    """Generate bulk coordinates using the checkerboard pattern.

    The checkerboard pattern places qubits as follows:
    - Data qubits: absolute even x, absolute even y
    - X ancillas: absolute odd x, absolute odd y, (x + y) % 4 == 0
    - Z ancillas: absolute odd x, absolute odd y, (x + y) % 4 == 2

    Parameters
    ----------
    bounds : PatchBounds
        Bounding box for the coordinate generation.

    Returns
    -------
    PatchCoordinates
        Generated coordinates for data qubits and ancillas.

    Examples
    --------
    >>> from lspattern.layout.coordinates import PatchBounds
    >>> bounds = PatchBounds(x_min=0, x_max=4, y_min=0, y_max=4)
    >>> coords = generate_checkerboard_coords(bounds)
    >>> len(coords.data) > 0
    True
    """
    data: set[Coord2D] = set()
    ancilla_x: set[Coord2D] = set()
    ancilla_z: set[Coord2D] = set()

    for x in range(bounds.x_min, bounds.x_max + 1):
        for y in range(bounds.y_min, bounds.y_max + 1):
            # Data qubits use absolute even coordinates
            if x % 2 == 0 and y % 2 == 0:
                data.add(Coord2D(x, y))
            # Ancillas use absolute odd coordinates but pattern for X/Z distinction
            elif x % 2 == 1 and y % 2 == 1:
                if (x + y) % 4 == 0:
                    ancilla_x.add(Coord2D(x, y))
                elif (x + y) % 4 == 2:  # noqa: PLR2004
                    ancilla_z.add(Coord2D(x, y))

    return PatchCoordinates(
        data=frozenset(data),
        ancilla_x=frozenset(ancilla_x),
        ancilla_z=frozenset(ancilla_z),
    )
