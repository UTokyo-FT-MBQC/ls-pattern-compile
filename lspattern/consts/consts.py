# This class stores some constants

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
    LEFT: int = 0
    TOP: int = 1
    RIGHT: int = 2
    BOTTOM: int = 3
    UP: int = 4
    DOWN: int = 5
