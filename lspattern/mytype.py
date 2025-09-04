# coordinate system
"""Typed aliases for coordinates and ids (local/global frames).

This module defines type names for the six coordinate/id kinds discussed in
the codebase:

- Patch coordinates (local/global): 2D anchors where a logical patch sits.
- Physical qubit coordinates (local/global): 3D (x, y, z) coordinates used in RHG.
- Node indices (local/global): integer ids for graph nodes.

These are aliases and NewTypes to aid static checking/documentation without
changing runtime representations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Mapping, NewType, Set, Tuple

# ---------------------------------------------------------------------
# Core scalar ids (NewType for static distinction)
# ---------------------------------------------------------------------
NodeIdLocal = NewType("NodeIdLocal", int)
NodeIdGlobal = NewType("NodeIdGlobal", int)

LogicalIndex = NewType("LogicalIndex", int)
QubitIndex = NewType("QubitIndex", int)

# ---------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------
# Tiling coordinate

TilingCoord2D = NewType("TilingCoord2D", Tuple[int, int])
TilingConsistentQubitId = NewType("TilingConsistentQubitId", int)

# Patch coordinates are 2D integer anchors (x0, y0).
PatchCoordLocal2D = NewType("PatchCoordLocal2D", Tuple[int, int])
PatchCoordGlobal3D = NewType("PatchCoordGlobal3D", Tuple[int, int, int])
PipeCoordGlobal3D = NewType(
    "PipeCoordGlobal3D", Tuple[Tuple[int, int, int], Tuple[int, int, int]]
)

# Physical qubit coordinates are 3D integer positions (x, y, z).
PhysCoordLocal2D = NewType("PhysCoordLocal2D", Tuple[int, int])  # (x, y)
PhysCoordLocal3D = NewType("PhysCoordLocal3D", Tuple[int, int, int])  # (x, y, z)
PhysCoordGlobal3D = NewType("PhysCoordGlobal3D", Tuple[int, int, int])

# Convenience aliases for collections
NodeSetLocal = Set[NodeIdLocal]
NodeSetGlobal = Set[NodeIdGlobal]

# Ports and q-index mappings (LOCAL frame on blocks)
InPortsLocal = Dict[LogicalIndex, NodeSetLocal]
OutPortsLocal = Dict[LogicalIndex, NodeSetLocal]
OutQMapLocal = Dict[LogicalIndex, Dict[NodeIdLocal, QubitIndex]]

# Schedule and flow (LOCAL)
LocalTime = NewType("LocalTime", int)
ScheduleTuplesLocal = List[Tuple[LocalTime, NodeSetLocal]]
FlowLocal = Dict[NodeIdLocal, NodeSetLocal]

# Parity caps linking PREV global center to CURR local nodes
ParityCapsLocal = List[Tuple[NodeIdGlobal, List[NodeIdLocal]]]

# Block kind
BlockKindstr = tuple[str, str, str]


__all__ = [
    # ids
    "NodeIdLocal",
    "NodeIdGlobal",
    "LogicalIndex",
    "QubitIndex",
    # coords
    "PatchCoordLocal2D",
    "PatchCoordGlobal3D",
    "PipeCoordGlobal3D",
    "PhysCoordLocal3D",
    "PhysCoordGlobal3D",
    # sets/maps
    "NodeSetLocal",
    "NodeSetGlobal",
    # block IO
    "InPortsLocal",
    "OutPortsLocal",
    "OutQMapLocal",
    # schedule/flow
    "LocalTime",
    "ScheduleTuplesLocal",
    "FlowLocal",
    # parity
    "ParityCapsLocal",
    # kinds
    "BlockKindstr",
    # directions
]

# ---------------------------------------------------------------------
# Boundary enums and types (for per-side face specification)
# ---------------------------------------------------------------------


class BoundarySide(str, Enum):
    TOP = "TOP"  # +Y
    BOTTOM = "BOTTOM"  # -Y
    LEFT = "LEFT"  # -X
    RIGHT = "RIGHT"  # +X
    UP = "UP"  # +Z
    DOWN = "DOWN"  # -Z


# Allowed edge boundary value type and helper set
EdgeSpecValue = Literal["X", "Z", "O"]


class _EdgeSpecMeta(type):
    _allowed_keys = {"TOP", "BOTTOM", "LEFT", "RIGHT", "UP", "DOWN"}
    _allowed_vals = {"X", "Z", "O"}
    _values: Dict[str, str] = {k: "O" for k in _allowed_keys}

    def __getattr__(cls, name: str) -> str:  # type: ignore[override]
        if name in cls._allowed_keys:
            return cls._values[name]
        raise AttributeError(name)

    def __setattr__(cls, name: str, value) -> None:  # type: ignore[override]
        # Intercept assignment to the four sides and validate
        if name in ("_allowed_keys", "_allowed_vals", "_values"):
            return super().__setattr__(name, value)
        if name in cls._allowed_keys:
            if isinstance(value, str):
                v = value.upper()
            else:
                raise TypeError("EdgeSpec values must be 'X', 'Z', or 'O' (str)")
            if v not in cls._allowed_vals:
                raise ValueError("EdgeSpec value must be one of 'X', 'Z', 'O'")
            cls._values[name] = v
            return
        # Fallback to regular class attribute set
        return super().__setattr__(name, value)


class EdgeSpec(metaclass=_EdgeSpecMeta):
    """Class-level container for per-side edge specifications.

    Usage
    -----
    - Set per-side spec:  EdgeSpec.TOP = "X"
    - Read current spec:  EdgeSpec.TOP  -> "X"
    - Allowed values: "X", "Z", "O" only.
    - Sides handled: TOP, BOTTOM, LEFT, RIGHT, UP, DOWN.

    Helper methods
    --------------
    - EdgeSpec.as_dict() -> dict[str, EdgeSpecValue]
    - EdgeSpec.update({...}) to set multiple at once.
    """

    # Provide annotations for better IDE/type hints
    TOP: str
    BOTTOM: str
    LEFT: str
    RIGHT: str

    @classmethod
    def as_dict(cls) -> Dict[str, EdgeSpecValue]:
        return cls._values.copy()  # type: ignore[return-value]

    @classmethod
    def update(cls, mapping: Mapping[str, str]) -> None:
        for k, v in mapping.items():
            setattr(cls, k, v)


# Mapping from side to boundary spec
BoundarySpec = Dict[BoundarySide, EdgeSpecValue]

__all__ += [
    "BoundarySide",
    "EdgeSpec",
    "EdgeSpecValue",
    "BoundarySpec",
]
