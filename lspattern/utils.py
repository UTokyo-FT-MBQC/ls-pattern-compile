from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import PatchCoordGlobal3D


def get_direction(
    source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D
) -> PIPEDIRECTION:
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
            raise ValueError("Invalid direction")


def __tuple_sum(l_: tuple, r_: tuple) -> tuple:
    assert len(l_) == len(r_)
    return tuple(a + b for a, b in zip(l_, r_))


# Prepare outputs as sorted lists for determinism
def sort_xy(points: set[tuple[int, int]]):
    return sorted(points, key=lambda p: (p[1], p[0]))
