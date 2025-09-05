"""Utility functions for coordinate transformations and pipe directions."""

import operator

from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import PatchCoordGlobal3D


def get_direction(source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> PIPEDIRECTION:
    """Get the pipe direction from source to sink coordinates.

    Parameters
    ----------
    source : PatchCoordGlobal3D
        Source coordinates.
    sink : PatchCoordGlobal3D
        Sink coordinates.

    Returns
    -------
    PIPEDIRECTION
        Direction enum value.

    Raises
    ------
    ValueError
        If direction is invalid.
    """
    dx = sink[0] - source[0]
    dy = sink[1] - source[1]
    dz = sink[2] - source[2]

    match (dx, dy, dz):
        case (1, 0, 0):
            return PIPEDIRECTION.RIGHT
        case (-1, 0, 0):
            return PIPEDIRECTION.LEFT
        case (0, 1, 0):
            return PIPEDIRECTION.TOP
        case (0, -1, 0):
            return PIPEDIRECTION.BOTTOM
        case (0, 0, 1):
            return PIPEDIRECTION.UP
        case (0, 0, -1):
            return PIPEDIRECTION.DOWN
        case _:
            msg = "Invalid direction"
            raise ValueError(msg)


def __tuple_sum(l_: tuple, r_: tuple) -> tuple:
    assert len(l_) == len(r_)
    return tuple(a + b for a, b in zip(l_, r_, strict=False))


# Prepare outputs as sorted lists for determinism
def sort_xy(points: set[tuple[int, int]]):
    """Sort 2D points by (y, x) coordinates for determinism."""
    return sorted(points, key=operator.itemgetter(1, 0))
