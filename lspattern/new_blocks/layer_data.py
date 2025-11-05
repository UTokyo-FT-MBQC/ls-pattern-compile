"""Dataclasses for coordinate-based layer descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lspattern.new_blocks.mytype import Coord3D, NodeRole, QubitGroupId


@dataclass
class CoordBasedLayerData:
    """Coordinate-based data for a single unit layer (2 physical layers)."""

    coords_by_z: dict[int, set[Coord3D]]
    coord2role: dict[Coord3D, NodeRole]
    spatial_edges: set[tuple[Coord3D, Coord3D]]
    temporal_edges: set[tuple[Coord3D, Coord3D]]
    coord_schedule: dict[int, set[Coord3D]]
    coord_flow: dict[Coord3D, set[Coord3D]]


@dataclass
class SeamEdgeCandidate:
    """Seam edge candidate (coordinate-based)."""

    coord: Coord3D
    gid1: QubitGroupId
    gid2: QubitGroupId
