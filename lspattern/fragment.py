"""Fragment data structures for block-based graph construction.

This module defines the core data structures used to represent graph fragments
that can be merged into a Canvas. Separating these into their own module
avoids circular dependencies between canvas.py and fragment_builder.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

from lspattern.accumulator import CoordFlowAccumulator, CoordParityAccumulator, CoordScheduleAccumulator
from lspattern.consts import EdgeSpecValue
from lspattern.mytype import Coord3D

if TYPE_CHECKING:
    from graphqomb.common import Axis

    from lspattern.mytype import NodeRole


class Boundary(NamedTuple):
    """Boundary specification for a block."""

    top: EdgeSpecValue
    bottom: EdgeSpecValue
    left: EdgeSpecValue
    right: EdgeSpecValue


@dataclass(slots=True)
class GraphSpec:
    """Explicit graph fragment provided by users via YAML.

    Notes
    -----
    `coord_mode`
        - "local": coordinates are translated when the block is placed on a canvas.
        - "global": coordinates are used as-is.
    `time_mode`
        - "local": schedule times are shifted based on the block's z position.
        - "global": schedule times are used as-is.
    """

    coord_mode: Literal["local", "global"] = "local"
    time_mode: Literal["local", "global"] = "local"

    nodes: set[Coord3D] = field(default_factory=set)
    edges: set[tuple[Coord3D, Coord3D]] = field(default_factory=set)
    pauli_axes: dict[Coord3D, Axis] = field(default_factory=dict)
    coord2role: dict[Coord3D, NodeRole] = field(default_factory=dict)

    flow: CoordFlowAccumulator = field(default_factory=CoordFlowAccumulator)
    scheduler: CoordScheduleAccumulator = field(default_factory=CoordScheduleAccumulator)
    parity: CoordParityAccumulator = field(default_factory=CoordParityAccumulator)


@dataclass(slots=True)
class BoundaryFragment:
    """Fragment containing boundary information for a block.

    This separates boundary graph data from the physical graph (nodes/edges),
    allowing independent management of boundary transitions for detector analysis.
    """

    boundaries: dict[Coord3D, Boundary] = field(default_factory=dict)

    def add_boundary(self, coord: Coord3D, boundary: Boundary) -> None:
        """Add a boundary at the given coordinate."""
        self.boundaries[coord] = boundary


@dataclass(slots=True)
class BlockFragment:
    """Complete fragment for a block, combining graph and boundary data.

    This dataclass bundles GraphSpec with BoundaryFragment to provide
    all information needed to merge a block into a Canvas.

    Attributes
    ----------
    graph : GraphSpec
        The physical graph fragment (nodes, edges, schedule, parity, flow).
    boundary : BoundaryFragment
        The boundary information for detector analysis.
    cout : set[Coord3D] | None
        Optional logical observable coordinates (in local coordinates).
    """

    graph: GraphSpec
    boundary: BoundaryFragment
    cout: set[Coord3D] | None = None
