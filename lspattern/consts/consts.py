"""Constants for lspattern library including spatial orientations and pipe directions."""

# 2D spatial orientations
import enum

DIRECTIONS2D: list[tuple[int, int]] = [
    (-1, -1),  # down left
    (-1, +1),  # down right
    (+1, -1),  # up left
    (+1, +1),  # up right
]


# 3D spatial orientations
DIRECTIONS3D: list[tuple[int, int, int]] = [
    # left, right, up, down, diagonal upright, upleft, downright, downleft
    # (-1, 0, 0),  # left
    # (1, 0, 0),  # right
    # (0, 1, 0),  # up
    # (0, -1, 0),  # down
    (1, 1, 0),  # diagonal upright
    (-1, 1, 0),  # diagonal upleft
    (1, -1, 0),  # diagonal downright
    (-1, -1, 0),  # diagonal downleft
]


class PIPEDIRECTION(enum.Enum):
    """Enumeration for pipe directions in 3D space."""

    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    UP = 4
    DOWN = 5
