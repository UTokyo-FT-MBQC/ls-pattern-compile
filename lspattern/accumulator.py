"""Coordinate-based accumulators for new block primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.mytype import Coord2D

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from lspattern.mytype import Coord3D


@dataclass
class CoordScheduleAccumulator:
    """Coordinate-based measurement schedule."""

    prep_time: dict[int, set[Coord3D]] = field(default_factory=dict)
    meas_time: dict[int, set[Coord3D]] = field(default_factory=dict)
    entangle_time: dict[int, set[tuple[Coord3D, Coord3D]]] = field(default_factory=dict)

    def add_prep_at_time(self, time: int, coords: Collection[Coord3D]) -> None:
        """Add preparation coordinates to the schedule at the given time.

        Parameters
        ----------
        time : int
            The time step to add the coordinates to.
        coords : collections.abc.Collection[Coord3D]
            The coordinates to add at the specified time.
        """
        if not coords:
            return
        if time not in self.prep_time:
            self.prep_time[time] = set()
        self.prep_time[time].update(coords)

    def add_meas_at_time(self, time: int, coords: Collection[Coord3D]) -> None:
        """Add measurement coordinates to the schedule at the given time.

        Parameters
        ----------
        time : int
            The time step to add the coordinates to.
        coords : collections.abc.Collection[Coord3D]
            The coordinates to add at the specified time.
        """
        if not coords:
            return
        if time not in self.meas_time:
            self.meas_time[time] = set()
        self.meas_time[time].update(coords)

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
        if time not in self.entangle_time:
            self.entangle_time[time] = set()
        self.entangle_time[time].update(edges)

    def to_node_schedule(
        self, coord2node: Mapping[Coord3D, int]
    ) -> tuple[dict[int, int], dict[int, int], dict[tuple[int, int], int]]:
        """Convert coordinate-based schedule to node identifier-based schedule.

        Parameters
        ----------
        coord2node : collections.abc.Mapping[Coord3D, int]
            A mapping from coordinates to node identifiers.

        Returns
        -------
        tuple[dict[int, int], dict[int, int], dict[tuple[int, int], int]]
            A tuple containing three dictionaries for preparation, measurement,
            and entangling schedules indexed by time steps.
        """
        prep_schedule: dict[int, int] = {}
        meas_schedule: dict[int, int] = {}
        entangle_schedule: dict[tuple[int, int], int] = {}

        for time, coords in self.prep_time.items():
            mapped = {coord2node[c] for c in coords if c in coord2node}
            for node in mapped:
                prep_schedule[node] = time

        for time, coords in self.meas_time.items():
            mapped = {coord2node[c] for c in coords if c in coord2node}
            for node in mapped:
                meas_schedule[node] = time

        for time, edges in self.entangle_time.items():
            mapped_edge = {
                (coord2node[c1], coord2node[c2]) for c1, c2 in edges if c1 in coord2node and c2 in coord2node
            }
            if mapped_edge:
                for edge in mapped_edge:
                    entangle_schedule[edge] = time

        return prep_schedule, meas_schedule, entangle_schedule


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
    remaining_parity: dict[Coord2D, dict[int, set[Coord3D]]] = field(default_factory=dict)
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
            If empty, this signals a parity reset at this z-coordinate.

        Notes
        -----
        This is a pre-processing step before constructing parity checks.
        An empty `involved_coords` is used to indicate that the parity
        should be reset at this layer (e.g., when data qubits are removed).
        """
        if xy not in self.syndrome_meas:
            self.syndrome_meas[xy] = {}
        if z not in self.syndrome_meas[xy]:
            self.syndrome_meas[xy][z] = set()
        # Union with existing coords; empty set signals parity reset
        self.syndrome_meas[xy][z].update(involved_coords)

    def add_remaining_parity(self, xy: Coord2D, z: int, involved_coords: Collection[Coord3D]) -> None:
        """Add remaining parity information at coordinate `coord`.

        Parameters
        ----------
        xy : Coord2D
            The (x, y) coordinate of the remaining parity.
        z : int
            The z-coordinate (layer) of the remaining parity.
        involved_coords : collections.abc.Collection[Coord3D]
            The coordinates involved in the remaining parity.

        Notes
        -----
        This is a pre-processing step before constructing parity checks.
        """
        if not involved_coords:
            return
        if xy not in self.remaining_parity:
            self.remaining_parity[xy] = {}
        if z not in self.remaining_parity[xy]:
            self.remaining_parity[xy][z] = set()
        self.remaining_parity[xy][z].update(involved_coords)

    def clear_remaining_parity_at(self, xy: Coord2D, z: int) -> None:
        """Clear all remaining_parity at (xy, z).

        Parameters
        ----------
        xy : Coord2D
            The (x, y) coordinate of the remaining parity to clear.
        z : int
            The z-coordinate (layer) of the remaining parity to clear.
        """
        if xy in self.remaining_parity and z in self.remaining_parity[xy]:
            del self.remaining_parity[xy][z]
            if not self.remaining_parity[xy]:
                del self.remaining_parity[xy]

    def clear_syndrome_measurement_at(self, xy: Coord2D, z: int) -> None:
        """Clear all syndrome_meas at (xy, z).

        Parameters
        ----------
        xy : Coord2D
            The (x, y) coordinate of the syndrome measurement to clear.
        z : int
            The z-coordinate (layer) of the syndrome measurement to clear.
        """
        if xy in self.syndrome_meas and z in self.syndrome_meas[xy]:
            del self.syndrome_meas[xy][z]
            if not self.syndrome_meas[xy]:
                del self.syndrome_meas[xy]

    def add_non_deterministic_coord(self, coord: Coord3D) -> None:
        """Mark a coordinate as non-deterministic.

        Parameters
        ----------
        coord : Coord3D
            The coordinate to mark as non-deterministic.

        Notes
        -----
        If the coordinate is not in syndrome_meas (e.g., init layer ancillas),
        this method silently returns without adding the coordinate.
        Such coordinates are already excluded from detector construction.
        """
        xy = Coord2D(coord.x, coord.y)
        if xy not in self.syndrome_meas:
            return
        if coord.z not in self.syndrome_meas[xy]:
            return
        self.non_deterministic_coords.add(coord)
