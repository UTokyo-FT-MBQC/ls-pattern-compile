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

from typing import Dict, List, Mapping, MutableMapping, NewType, Set, Tuple

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
# Patch coordinates are 2D integer anchors (x0, y0).
PatchCoordLocal2D = Tuple[int, int]
PatchCoordGlobal3D = Tuple[int, int, int]
PipeCoordGlobal3D = Tuple[PatchCoordGlobal3D, PatchCoordGlobal3D]

# Physical qubit coordinates are 3D integer positions (x, y, z).
PhysCoordLocal3D = Tuple[int, int, int]
PhysCoordGlobal3D = Tuple[int, int, int]

# Convenience aliases for collections
NodeSetLocal = Set[NodeIdLocal]
NodeSetGlobal = Set[NodeIdGlobal]

NodeCoordMapLocal = Dict[NodeIdLocal, PhysCoordLocal3D]
NodeCoordMapGlobal = Dict[NodeIdGlobal, PhysCoordGlobal3D]

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
    "PatchCoordLocal",
    "PatchCoordGlobal",
    "PhysCoordLocal",
    "PhysCoordGlobal",
    # sets/maps
    "NodeSetLocal",
    "NodeSetGlobal",
    "NodeCoordMapLocal",
    "NodeCoordMapGlobal",
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
]
