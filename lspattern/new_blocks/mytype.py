"""Coordinate and identifier helpers for the new block API."""

from __future__ import annotations

import sys
from typing import NamedTuple

# TODO: Remove fallback once Python 3.11+ is required
if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    try:
        from typing_extensions import StrEnum
    except (ImportError, AttributeError):
        # Fallback for Python 3.10 without typing_extensions >= 4.2.0
        from enum import Enum

        class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
            """StrEnum fallback for Python 3.10."""


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
