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

# ruff: noqa: RUF022

from typing import Literal, NewType

# ---------------------------------------------------------------------
# Core scalar ids (NewType for static distinction)
# ---------------------------------------------------------------------
NodeIdLocal = NewType("NodeIdLocal", int)
NodeIdGlobal = NewType("NodeIdGlobal", int)

LogicalIndex = NewType("LogicalIndex", int)
QubitIndex = NewType("QubitIndex", int)
QubitIndexLocal = NewType("QubitIndexLocal", int)

TilingId = NewType("TilingId", int)
QubitGroupIdLocal = NewType("QubitGroupIdLocal", int)
QubitGroupIdGlobal = NewType("QubitGroupIdGlobal", int)
# ---------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------
# Tiling coordinate

TilingCoord2D = NewType("TilingCoord2D", tuple[int, int])

# Patch coordinates are 2D integer anchors (x0, y0).
PatchCoordLocal2D = NewType("PatchCoordLocal2D", tuple[int, int])
PatchCoordGlobal3D = NewType("PatchCoordGlobal3D", tuple[int, int, int])
PipeCoordGlobal3D = NewType("PipeCoordGlobal3D", tuple[PatchCoordGlobal3D, PatchCoordGlobal3D])

# Physical qubit coordinates are 3D integer positions (x, y, z).
PhysCoordLocal2D = NewType("PhysCoordLocal2D", tuple[int, int])  # (x, y)
PhysCoordLocal3D = NewType("PhysCoordLocal3D", tuple[int, int, int])  # (x, y, z)
PhysCoordGlobal3D = NewType("PhysCoordGlobal3D", tuple[int, int, int])

# Convenience aliases for collections
NodeSetLocal = set[NodeIdLocal]
NodeSetGlobal = set[NodeIdGlobal]

# Ports and q-index mappings (LOCAL frame on blocks)
InPortsLocal = dict[LogicalIndex, NodeSetLocal]
OutPortsLocal = dict[LogicalIndex, NodeSetLocal]
OutQMapLocal = dict[LogicalIndex, dict[NodeIdLocal, QubitIndex]]

# Schedule and flow (LOCAL)
LocalTime = NewType("LocalTime", int)
ScheduleTuplesLocal = list[tuple[LocalTime, NodeSetLocal]]
FlowLocal = dict[NodeIdLocal, NodeSetLocal]

# Parity caps linking PREV global center to CURR local nodes
ParityCapsLocal = list[tuple[NodeIdGlobal, list[NodeIdLocal]]]

# ---------------------------------------------------------------------
# Edge/boundary specs for scalable tilings
# ---------------------------------------------------------------------
# Allowed edge spec values for faces: X / Z / O(Open or Trimmed)
EdgeSpecValue = Literal["X", "Z", "O"]

# Boundary side literals (case-insensitive handling lives in callers)
BoundarySide = Literal["TOP", "BOTTOM", "LEFT", "RIGHT", "UP", "DOWN"]

# Dict-based spatial edge spec preferred across the codebase
SpatialEdgeSpec = dict[str, EdgeSpecValue]

# Module-level convenience default used by examples/tests.
# All sides start as open ("O"). Callers may update it locally.
EdgeSpec: SpatialEdgeSpec = {
    "TOP": "O",
    "BOTTOM": "O",
    "LEFT": "O",
    "RIGHT": "O",
    "UP": "O",
    "DOWN": "O",
}

__all__ = [
    "BoundarySide",
    "EdgeSpec",
    "FlowLocal",
    "InPortsLocal",
    "LocalTime",
    "LogicalIndex",
    "NodeIdGlobal",
    "NodeIdLocal",
    "NodeSetGlobal",
    "NodeSetLocal",
    "OutPortsLocal",
    "OutQMapLocal",
    "PatchCoordGlobal3D",
    "PatchCoordLocal2D",
    "ParityCapsLocal",
    "PhysCoordGlobal3D",
    "PhysCoordLocal3D",
    "PipeCoordGlobal3D",
    "QubitIndex",
    "ScheduleTuplesLocal",
    "SpatialEdgeSpec",
]
