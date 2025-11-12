"""The base definition for RHG unit layers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from lspattern.new_blocks.accumulator import CoordFlowAccumulator, CoordScheduleAccumulator
from lspattern.new_blocks.coord_utils import CoordTransform
from lspattern.new_blocks.layer_data import CoordBasedLayerData
from lspattern.new_blocks.mytype import Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphqomb.graphstate import GraphState


class UnitLayer(ABC):
    """Abstract base class for RHG unit layers (2 physical layers)."""

    @property
    @abstractmethod
    def global_pos(self) -> Coord3D:
        """Get the global position of the unit layer.

        Returns
        -------
        Coord3D
            The global (x, y, z) position of the unit layer.
        """
        ...

    @abstractmethod
    def build_metadata(
        self,
        z_offset: int,
        data2d: Sequence[tuple[int, int]],
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],
    ) -> CoordBasedLayerData:
        """Build coordinate-based metadata for this unit layer.

        Parameters
        ----------
        z_offset : int
            Starting z-coordinate for this layer.
        data2d : Sequence[tuple[int, int]]
            2D data qubit coordinates.
        x2d : Sequence[tuple[int, int]]
            2D X-ancilla coordinates.
        z2d : Sequence[tuple[int, int]]
            2D Z-ancilla coordinates.

        Returns
        -------
        CoordBasedLayerData
            Layer metadata including coordinates, roles, edges, schedule, flow.
        """
        ...

    @abstractmethod
    def materialize(self, graph: GraphState, node_map: Mapping[Coord3D, int]) -> tuple[GraphState, dict[Coord3D, int]]:
        """Materialize the unit layer into the given graph.

        Parameters
        ----------
        graph : GraphState
            The graph to materialize the unit layer into.
        node_map : Mapping[Coord3D, int]
            Existing coordinate-to-node mapping.

        Returns
        -------
        tuple[GraphState, dict[Coord3D, int]]
            Updated graph and coordinate-to-node mapping.
        """
        ...


class MemoryUnitLayer(UnitLayer):
    """Standard memory unit layer consisting of a Z-check and X-check physical layer."""

    def __init__(self, global_pos: Coord3D) -> None:
        """Initialize the memory unit layer with its global offset."""
        self._global_pos = global_pos

    @property
    def global_pos(self) -> Coord3D:
        """Return the global (x, y, z) position of the unit layer."""
        return self._global_pos

    def build_metadata(  # noqa: PLR6301
        self,
        z_offset: int,
        data2d: Sequence[tuple[int, int]],
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],
    ) -> CoordBasedLayerData:
        """Build coordinate-based metadata for this unit layer.

        Creates two physical layers:
        - Layer ``z_offset`` (even): data + Z ancillas
        - Layer ``z_offset + 1`` (odd): data + X ancillas
        """
        z1 = z_offset
        z2 = z_offset + 1

        coords_by_z, coord2role = MemoryUnitLayer._build_layers(z1, z2, data2d, x2d, z2d)
        coords_z1 = coords_by_z[z1]
        coords_z2 = coords_by_z[z2]

        spatial_edges = MemoryUnitLayer._compute_spatial_edges(coords_z1, coords_z2)
        temporal_edges = MemoryUnitLayer._compute_temporal_edges(coords_z1, coords_z2, z1)
        coord_schedule = MemoryUnitLayer._build_schedule(coords_z1, coords_z2, coord2role, z1, z2)
        coord_flow = MemoryUnitLayer._build_flow(temporal_edges)

        return CoordBasedLayerData(
            coords_by_z=coords_by_z,
            coord2role=coord2role,
            spatial_edges=spatial_edges,
            temporal_edges=temporal_edges,
            coord_schedule=coord_schedule,
            coord_flow=coord_flow,
        )

    @staticmethod
    def _build_layers(
        z1: int,
        z2: int,
        data2d: Sequence[tuple[int, int]],
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],
    ) -> tuple[dict[int, set[Coord3D]], dict[Coord3D, NodeRole]]:
        """Build coordinate sets and roles for both layers."""
        coords_by_z: dict[int, set[Coord3D]] = {}
        coord2role: dict[Coord3D, NodeRole] = {}

        # Layer 1: Z-check (even z)
        coords_z1: set[Coord3D] = set()
        for x, y in data2d:
            coord = Coord3D(x, y, z1)
            coords_z1.add(coord)
            coord2role[coord] = NodeRole.DATA
        for x, y in z2d:
            coord = Coord3D(x, y, z1)
            coords_z1.add(coord)
            coord2role[coord] = NodeRole.ANCILLA_Z
        coords_by_z[z1] = coords_z1

        # Layer 2: X-check (odd z)
        coords_z2: set[Coord3D] = set()
        for x, y in data2d:
            coord = Coord3D(x, y, z2)
            coords_z2.add(coord)
            coord2role[coord] = NodeRole.DATA
        for x, y in x2d:
            coord = Coord3D(x, y, z2)
            coords_z2.add(coord)
            coord2role[coord] = NodeRole.ANCILLA_X
        coords_by_z[z2] = coords_z2

        return coords_by_z, coord2role

    @staticmethod
    def _compute_spatial_edges(
        coords_z1: set[Coord3D],
        coords_z2: set[Coord3D],
    ) -> set[tuple[Coord3D, Coord3D]]:
        """Compute spatial edges within each layer."""
        spatial_edges: set[tuple[Coord3D, Coord3D]] = set()
        for coords in (coords_z1, coords_z2):
            for coord in coords:
                for neighbor in CoordTransform.get_neighbors_3d(coord, spatial_only=True):
                    if neighbor in coords and neighbor > coord:
                        spatial_edges.add((coord, neighbor))
        return spatial_edges

    @staticmethod
    def _compute_temporal_edges(
        coords_z1: set[Coord3D],
        coords_z2: set[Coord3D],
        z1: int,
    ) -> set[tuple[Coord3D, Coord3D]]:
        """Compute temporal edges between matching data nodes."""
        temporal_edges: set[tuple[Coord3D, Coord3D]] = set()
        for coord_upper in coords_z2:
            coord_lower = Coord3D(coord_upper.x, coord_upper.y, z1)
            if coord_lower in coords_z1:
                temporal_edges.add((coord_lower, coord_upper))
        return temporal_edges

    @staticmethod
    def _build_schedule(
        coords_z1: set[Coord3D],
        coords_z2: set[Coord3D],
        coord2role: dict[Coord3D, NodeRole],
        z1: int,
        z2: int,
    ) -> dict[int, set[Coord3D]]:
        """Build schedule separating ancilla (even time) and data (odd time)."""
        schedule_acc = CoordScheduleAccumulator()
        schedule_acc.add_at_time(2 * z1, {c for c in coords_z1 if coord2role[c] != NodeRole.DATA})
        schedule_acc.add_at_time(2 * z1 + 1, {c for c in coords_z1 if coord2role[c] == NodeRole.DATA})
        schedule_acc.add_at_time(2 * z2, {c for c in coords_z2 if coord2role[c] != NodeRole.DATA})
        schedule_acc.add_at_time(2 * z2 + 1, {c for c in coords_z2 if coord2role[c] == NodeRole.DATA})
        return schedule_acc.schedule

    @staticmethod
    def _build_flow(temporal_edges: set[tuple[Coord3D, Coord3D]]) -> dict[Coord3D, set[Coord3D]]:
        """Build flow edges between temporal partners."""
        flow_acc = CoordFlowAccumulator()
        for coord_lower, coord_upper in temporal_edges:
            flow_acc.add_flow(coord_lower, coord_upper)
        return flow_acc.flow

    def materialize(
        self,
        graph: GraphState,
        node_map: Mapping[Coord3D, int],
    ) -> tuple[GraphState, dict[Coord3D, int]]:
        """Materialize this layer into the given graph (not yet implemented)."""
        msg = "Use build_metadata instead"
        raise NotImplementedError(msg)
