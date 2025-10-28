"""Type utils"""

from __future__ import annotations

from typing import NamedTuple


class Coord2D(NamedTuple):
    x: int
    y: int


class Coord3D(NamedTuple):
    x: int
    y: int
    z: int
