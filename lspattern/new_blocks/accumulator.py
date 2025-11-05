"""Coordinate-based accumulators for new block primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lspattern.new_blocks.mytype import Coord2D, Coord3D


@dataclass
class CoordScheduleAccumulator:
    """Coordinate-based measurement schedule."""

    schedule: dict[int, set[Coord3D]] = field(default_factory=dict)

    def add_at_time(self, time: int, coords: set[Coord3D]) -> None:
        """Add coordinates to the schedule at the given time."""
        if not coords:
            return
        if time not in self.schedule:
            self.schedule[time] = set()
        self.schedule[time].update(coords)

    def to_node_schedule(self, coord2node: Mapping[Coord3D, int]) -> dict[int, set[int]]:
        """Convert schedule coordinates to node identifiers using `coord2node`."""
        return {
            t: {coord2node[c] for c in coords if c in coord2node}
            for t, coords in self.schedule.items()
            if coords
        }


@dataclass
class CoordFlowAccumulator:
    """Coordinate-based flow (correction dependencies)."""

    flow: dict[Coord3D, set[Coord3D]] = field(default_factory=dict)

    def add_flow(self, from_coord: Coord3D, to_coord: Coord3D) -> None:
        """Add a flow edge from `from_coord` to `to_coord`."""
        if from_coord not in self.flow:
            self.flow[from_coord] = set()
        self.flow[from_coord].add(to_coord)

    def to_node_flow(self, coord2node: Mapping[Coord3D, int]) -> dict[int, set[int]]:
        """Convert flow coordinates to node identifiers using `coord2node`."""
        result: dict[int, set[int]] = {}
        for from_coord, to_coords in self.flow.items():
            if from_coord not in coord2node or not to_coords:
                continue
            from_node = coord2node[from_coord]
            mapped = {coord2node[to_c] for to_c in to_coords if to_c in coord2node}
            if mapped:
                result[from_node] = mapped
        return result


@dataclass
class CoordParityAccumulator:
    """Coordinate-based parity checks indexed by (x, y, t)."""

    checks: dict[Coord2D, dict[int, set[Coord3D]]] = field(default_factory=dict)

    def add_check(self, xy: Coord2D, time: int, coords: set[Coord3D]) -> None:
        """Add a parity check at coordinate `xy` occurring at `time`."""
        if not coords:
            return
        if xy not in self.checks:
            self.checks[xy] = {}
        if time not in self.checks[xy]:
            self.checks[xy][time] = set()
        self.checks[xy][time].update(coords)
