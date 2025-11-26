"""Coordinate-based accumulators for new block primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.new_blocks.mytype import Coord2D

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from lspattern.new_blocks.mytype import Coord3D


@dataclass
class CoordScheduleAccumulator:
    """Coordinate-based measurement schedule."""

    schedule: dict[int, set[Coord3D]] = field(default_factory=dict)
    edge_schedule: dict[int, set[tuple[Coord3D, Coord3D]]] = field(default_factory=dict)

    def add_at_time(self, time: int, coords: Collection[Coord3D]) -> None:
        """Add coordinates to the schedule at the given time.

        Parameters
        ----------
        time : int
            The time step to add the coordinates to.
        coords : collections.abc.Collection[Coord3D]
            The coordinates to add at the specified time.
        """
        if not coords:
            return
        if time not in self.schedule:
            self.schedule[time] = set()
        self.schedule[time].update(coords)

    def add_entangle_at_time(self, time: int, edges: Collection[tuple[Coord3D, Coord3D]]) -> None:
        """Add entangling edges to the schedule at the given time.

        Parameters
        ----------
        time : int
            The time step to add the edges to.
        edges : collections.abc.Collection[tuple[Coord3D, Coord3D]]
            The edges to add at the specified time.
        """
        if not edges:
            return
        if time not in self.edge_schedule:
            self.edge_schedule[time] = set()
        self.edge_schedule[time].update(edges)

    def to_node_schedule(self, coord2node: Mapping[Coord3D, int]) -> dict[int, set[int]]:
        """Convert schedule coordinates to node identifiers using `coord2node`.

        Parameters
        ----------
        coord2node : collections.abc.Mapping[Coord3D, int]
            A mapping from coordinates to node identifiers.

        Returns
        -------
        dict[int, set[int]]
            A mapping from time steps to sets of node identifiers.
        """
        return {t: {coord2node[c] for c in coords if c in coord2node} for t, coords in self.schedule.items() if coords}


@dataclass
class CoordFlowAccumulator:
    """Coordinate-based flow (correction dependencies)."""

    flow: dict[Coord3D, set[Coord3D]] = field(default_factory=dict)

    def add_flow(self, from_coord: Coord3D, to_coord: Coord3D) -> None:
        """Add a flow edge from `from_coord` to `to_coord`.

        Parameters
        ----------
        from_coord : Coord3D
            The starting coordinate of the flow edge.
        to_coord : Coord3D
            The ending coordinate of the flow edge.
        """
        if from_coord not in self.flow:
            self.flow[from_coord] = set()
        self.flow[from_coord].add(to_coord)

    def to_node_flow(self, coord2node: Mapping[Coord3D, int]) -> dict[int, set[int]]:
        """Convert flow coordinates to node identifiers using `coord2node`.

        Parameters
        ----------
        coord2node : collections.abc.Mapping[Coord3D, int]
            A mapping from coordinates to node identifiers.

        Returns
        -------
        dict[int, set[int]]
            A mapping from node identifiers to sets of dependent node identifiers.
        """
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
    """Coordinate-based parity checks indexed by (x, y, z)."""

    syndrome_meas: dict[Coord2D, dict[int, set[Coord3D]]] = field(default_factory=dict)
    non_deterministic_coords: set[Coord3D] = field(default_factory=set)

    def add_syndrome_measurement(self, xy: Coord2D, z: int, involved_coords: Collection[Coord3D]) -> None:
        """Add a syndrome measurement at coordinate `coord`.


        Parameters
        ----------
        xy : Coord2D
            The (x, y) coordinate of the syndrome measurement.
        z : int
            The z-coordinate (layer) of the syndrome measurement.
        involved_coords : collections.abc.Collection[Coord3D]
            The coordinates involved in the syndrome measurement.

        Notes
        -----
        This is a pre-processing step before constructing parity checks.
        """
        if not involved_coords:
            return
        if xy not in self.syndrome_meas:
            self.syndrome_meas[xy] = {}
        if z not in self.syndrome_meas[xy]:
            self.syndrome_meas[xy][z] = set()
        self.syndrome_meas[xy][z].update(involved_coords)

    def add_non_deterministic_coord(self, coord: Coord3D) -> None:
        """Mark a coordinate as non-deterministic.

        Parameters
        ----------
        coord : Coord3D
            The coordinate to mark as non-deterministic.
        """
        if Coord2D(coord.x, coord.y) not in self.syndrome_meas:
            msg = f"Cannot add non-deterministic coord {coord} without existing syndrome measurement at (x={coord.x}, y={coord.y})"
            raise KeyError(msg)
        if coord.z not in self.syndrome_meas[Coord2D(coord.x, coord.y)]:
            msg = f"Cannot add non-deterministic coord {coord} without existing syndrome measurement at z={coord.z}"
            raise KeyError(msg)
        self.non_deterministic_coords.add(coord)
