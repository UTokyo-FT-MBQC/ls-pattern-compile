"""Coordinate-based accumulators for new block primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from lspattern.new_blocks.mytype import Coord2D, Coord3D


@dataclass
class CoordScheduleAccumulator:
    """Coordinate-based measurement schedule."""

    schedule: dict[int, set[Coord3D]] = field(default_factory=dict)

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

    checks: dict[Coord2D, dict[int, set[Coord3D]]] = field(default_factory=dict)

    def add_check(self, xy: Coord2D, z: int, coords: Collection[Coord3D]) -> None:
        """Add a parity check at coordinate `xy` occurring at `z`.

        Parameters
        ----------
        xy : Coord2D
            The (x, y) coordinate of the parity check.
        z : int
            The z-coordinate (layer) of the parity check.
        coords : collections.abc.Collection[Coord3D]
            The coordinates involved in the parity check.
        """
        if not coords:
            return
        if xy not in self.checks:
            self.checks[xy] = {}
        if z not in self.checks[xy]:
            self.checks[xy][z] = set()
        self.checks[xy][z].update(coords)

    def remove_check(self, xy: Coord2D, z: int) -> None:
        """Remove a parity check at coordinate `xy` occurring at `z`.

        Parameters
        ----------
        xy : Coord2D
            The (x, y) coordinate of the parity check.
        z : int
            The z-coordinate (layer) of the parity check.
        """
        if xy in self.checks and z in self.checks[xy]:
            del self.checks[xy][z]
        else:
            msg = f"Attempted to remove non-existent check at {xy} z={z}"
            raise KeyError(msg)

    def to_node_checks(self, coord2node: Mapping[Coord3D, int]) -> dict[Coord2D, dict[int, set[int]]]:
        """Convert parity check coordinates to node identifiers using `coord2node`.

        Parameters
        ----------
        coord2node : collections.abc.Mapping[Coord3D, int]
            A mapping from coordinates to node identifiers.

        Returns
        -------
        dict[Coord2D, dict[int, set[int]]]
            A mapping from (x, y) coordinates to mappings of z-layers to sets of node identifiers.
        """
        result: dict[Coord2D, dict[int, set[int]]] = {}
        for xy, z_dict in self.checks.items():
            for z, coords in z_dict.items():
                mapped = {coord2node[c] for c in coords if c in coord2node}
                if mapped:
                    if xy not in result:
                        result[xy] = {}
                    result[xy][z] = mapped
        return result
