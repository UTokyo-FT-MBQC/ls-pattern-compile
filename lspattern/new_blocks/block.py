"""The base definition for RHG blocks"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.new_blocks.accumulator import (
    CoordFlowAccumulator,
    CoordParityAccumulator,
    CoordScheduleAccumulator,
)
from lspattern.new_blocks.mytype import Coord2D, Coord3D
from lspattern.new_blocks.unit_layer import MemoryUnitLayer, UnitLayer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphqomb.graphstate import GraphState

    from lspattern.new_blocks.layer_data import CoordBasedLayerData


class RHGBlock(ABC):
    @property
    @abstractmethod
    def global_pos(self) -> Coord3D:
        """Get the global position of the block.

        Returns
        -------
        Coord3D
            The global (x, y, z) position of the block.
        """
        ...

    @property
    @abstractmethod
    def in_ports(self) -> set[Coord2D]:
        """Get the input ports of the block.

        Returns
        -------
        set[Coord2D]
            A set of input port coordinates.
        """
        ...

    @property
    @abstractmethod
    def out_ports(self) -> set[Coord2D]:
        """Get the output ports of the block.

        Returns
        -------
        set[Coord2D]
            A set of output port coordinates.
        """
        ...

    @property
    @abstractmethod
    def cout_ports(self) -> set[Coord3D]:
        """Get the classical output ports of the block.

        Returns
        -------
        set[Coord3D]
            A set of classical output port coordinates.
        """
        ...

    @property
    @abstractmethod
    def unit_layers(self) -> list[UnitLayer]:
        """Get the unit layers comprising the block.

        Returns
        -------
        list[UnitLayer]
            A list of unit layers in the block.
        """
        ...

    @abstractmethod
    def materialize(self, graph: GraphState, node_map: Mapping[Coord3D, int]) -> tuple[GraphState, dict[Coord3D, int]]:
        """Materialize the block into the given graph.

        Parameters
        ----------
        graph : GraphState
            The graph to materialize the block into.
        node_map : dict[Coord3D, int]
            A mapping from local coordinates to node IDs.

        Returns
        -------
        tuple[GraphState, dict[Coord3D, int]]
            A tuple containing the updated graph and an updated mapping from local coordinates to node IDs.
        """
        ...


@dataclass
class RHGCube(RHGBlock):
    """Concrete implementation of an RHG cube block."""

    _global_pos: Coord3D
    d: int
    _unit_layers: list[UnitLayer] = field(default_factory=list)
    coord2role: dict[Coord3D, str] = field(default_factory=dict)
    coord_schedule: CoordScheduleAccumulator = field(default_factory=CoordScheduleAccumulator)
    coord_flow: CoordFlowAccumulator = field(default_factory=CoordFlowAccumulator)
    coord_parity: CoordParityAccumulator = field(default_factory=CoordParityAccumulator)
    _in_ports: set[Coord2D] = field(default_factory=set)
    _out_ports: set[Coord2D] = field(default_factory=set)
    _cout_ports: set[Coord3D] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Instantiate `MemoryUnitLayer` instances when none are provided."""
        if not self._unit_layers:
            for index in range(self.d):
                z_offset = self._global_pos.z + index * 2
                layer_origin = Coord3D(self._global_pos.x, self._global_pos.y, z_offset)
                self._unit_layers.append(MemoryUnitLayer(layer_origin))

    @property
    def global_pos(self) -> Coord3D:
        """Return the global origin coordinate of the cube."""
        return self._global_pos

    @property
    def in_ports(self) -> set[Coord2D]:
        """Return the set of 2D coordinates used as input ports."""
        return self._in_ports

    @property
    def out_ports(self) -> set[Coord2D]:
        """Return the set of 2D coordinates used as output ports."""
        return self._out_ports

    @property
    def cout_ports(self) -> set[Coord3D]:
        """Return the set of 3D coordinates used as classical output ports."""
        return self._cout_ports

    @property
    def unit_layers(self) -> list[UnitLayer]:
        """Return the unit-layer stack for this cube."""
        return self._unit_layers

    def prepare(
        self,
        data2d: Sequence[tuple[int, int]],
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],
    ) -> RHGCube:
        """Populate coordinate metadata ahead of graph materialization.

        Parameters
        ----------
        data2d : collections.abc.Sequence[tuple[int, int]]
            Data-qubit 2D coordinates.
        x2d : collections.abc.Sequence[tuple[int, int]]
            X-ancilla 2D coordinates.
        z2d : collections.abc.Sequence[tuple[int, int]]
            Z-ancilla 2D coordinates.
        """
        for index, layer in enumerate(self._unit_layers):
            z_offset = self._global_pos.z + index * 2
            metadata: CoordBasedLayerData = layer.build_metadata(z_offset, data2d, x2d, z2d)

            self.coord2role.update(metadata.coord2role)

            for time, coords in metadata.coord_schedule.items():
                self.coord_schedule.add_at_time(time, coords)

            for from_coord, to_coords in metadata.coord_flow.items():
                for to_coord in to_coords:
                    self.coord_flow.add_flow(from_coord, to_coord)

        return self

    def materialize(
        self,
        graph: GraphState,
        node_map: Mapping[Coord3D, int],
    ) -> tuple[GraphState, dict[Coord3D, int]]:
        """Materialize this cube into ``graph`` and return the updated state."""
        new_node_map = self._ensure_nodes(graph, node_map)
        coords_by_z = self._group_coords_by_z()

        self._connect_spatial_neighbors(graph, new_node_map, coords_by_z)
        self._connect_temporal_neighbors(graph, new_node_map)

        return graph, new_node_map

    def _ensure_nodes(
        self,
        graph: GraphState,
        node_map: Mapping[Coord3D, int],
    ) -> dict[Coord3D, int]:
        """Ensure all coordinates are present in ``node_map``."""
        new_node_map = dict(node_map)
        for coord in self.coord2role:
            if coord not in new_node_map:
                new_node_map[coord] = graph.add_physical_node()
        return new_node_map

    def _group_coords_by_z(self) -> dict[int, set[Coord3D]]:
        """Group coordinates by their z-layer."""
        coords_by_z: dict[int, set[Coord3D]] = {}
        for coord in self.coord2role:
            coords_by_z.setdefault(coord.z, set()).add(coord)
        return coords_by_z

    def _connect_spatial_neighbors(
        self,
        graph: GraphState,
        node_map: Mapping[Coord3D, int],
        coords_by_z: Mapping[int, set[Coord3D]],
    ) -> None:
        """Connect nearest-neighbour qubits within each z-layer."""
        for coords in coords_by_z.values():
            coord_list = list(coords)
            for index, coord_a in enumerate(coord_list):
                for coord_b in coord_list[index + 1 :]:
                    if self._are_nearest_neighbours(coord_a, coord_b):
                        graph.add_physical_edge(node_map[coord_a], node_map[coord_b])

    def _connect_temporal_neighbors(
        self,
        graph: GraphState,
        node_map: Mapping[Coord3D, int],
    ) -> None:
        """Connect qubits across time according to ``coord_flow``."""
        for from_coord, to_coords in self.coord_flow.flow.items():
            start_node = node_map.get(from_coord)
            if start_node is None:
                continue
            for to_coord in to_coords:
                end_node = node_map.get(to_coord)
                if end_node is not None:
                    graph.add_physical_edge(start_node, end_node)

    @staticmethod
    def _are_nearest_neighbours(coord_a: Coord3D, coord_b: Coord3D) -> bool:
        """Return whether two coordinates are nearest neighbours in the x-y plane."""
        delta_x = abs(coord_a.x - coord_b.x)
        delta_y = abs(coord_a.y - coord_b.y)
        return delta_x + delta_y == 1
