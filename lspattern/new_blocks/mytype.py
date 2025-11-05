"""Coordinate and identifier helpers for the new block API."""

from __future__ import annotations

from typing import NamedTuple

try:
    from enum import StrEnum
except ImportError:
    try:
        from typing_extensions import StrEnum
    except ImportError as exc:  # pragma: no cover
        message = "StrEnum requires Python 3.11 or typing_extensions."
        raise ImportError(message) from exc


class Coord2D(NamedTuple):
    x: int
    y: int


class Coord3D(NamedTuple):
    x: int
    y: int
    z: int


class NodeRole(StrEnum):
    """Role of a lattice node within the coordinate-based layer."""

    DATA = "DATA"
    ANCILLA_X = "ANCILLA_X"
    ANCILLA_Z = "ANCILLA_Z"


QubitGroupId = int
NodeId = int


__all__ = ["Coord2D", "Coord3D", "NodeId", "NodeRole", "QubitGroupId"]
