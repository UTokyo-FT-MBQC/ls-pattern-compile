"""Coordinate and identifier helpers for the new block API."""

from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple


class Coord2D(NamedTuple):
    x: int
    y: int


class Coord3D(NamedTuple):
    x: int
    y: int
    z: int


class NodeRole(Enum):
    """Role of a lattice node within the coordinate-based layer."""

    DATA = auto()
    ANCILLA_X = auto()
    ANCILLA_Z = auto()


class DIRECTION2D(Enum):
    """Direction in 2D lattice."""

    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()


class AxisDIRECTION2D(Enum):
    """Axis direction in 2D lattice."""

    H = auto()
    V = auto()


QubitGroupId = int
NodeId = int
