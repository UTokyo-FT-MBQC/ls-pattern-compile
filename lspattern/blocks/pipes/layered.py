"""Layer-by-layer pipe implementations using UnitLayer sequences.

This module provides RHGPipe implementations that are constructed from sequences
of UnitLayer objects, enabling flexible composition of different layer types for
spatial connections between cubes.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, overload

from graphix_zx.graphstate import GraphState

from lspattern.blocks.layers.initialize import InitPlusUnitLayer
from lspattern.blocks.layers.memory import MemoryUnitLayer
from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.blocks.unit_layer import UnitLayer
from lspattern.consts import EdgeSpecValue, NodeRole
from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D, SpatialEdgeSpec
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

if TYPE_CHECKING:
    from lspattern.consts.consts import PIPEDIRECTION


@dataclass
class LayeredRHGPipe(RHGPipe):
    """RHGPipe constructed from a sequence of UnitLayers.

    This class enables flexible layer-by-layer construction of pipes by composing
    different types of UnitLayer objects.

    Attributes
    ----------
    unit_layers : list[UnitLayer]
        Sequence of unit layers to compose. The length of this list determines
        the effective code distance.
    """

    unit_layers: list[UnitLayer] = field(default_factory=list)

    def _build_3d_graph(
        self,
    ) -> tuple[
        GraphState,
        dict[int, tuple[int, int, int]],
        dict[tuple[int, int, int], int],
        dict[int, str],
    ]:
        """Build 3D RHG graph structure layer-by-layer for pipe.

        Returns
        -------
        tuple
            (graph, node2coord, coord2node, node2role) for the complete pipe.
        """
        g = GraphState()
        z0 = int(self.source[2]) * (2 * self.d)

        # Accumulate all layer data
        all_nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}
        all_node2coord: dict[int, tuple[int, int, int]] = {}
        all_coord2node: dict[tuple[int, int, int], int] = {}
        all_node2role: dict[int, str] = {}

        for i, unit_layer in enumerate(self.unit_layers):
            layer_z = z0 + 2 * i
            layer_data = unit_layer.build_layer(g, layer_z, self.template)

            # Merge layer data
            all_nodes_by_z.update(layer_data.nodes_by_z)
            all_node2coord.update(layer_data.node2coord)
            all_coord2node.update(layer_data.coord2node)
            all_node2role.update(layer_data.node2role)

            # Merge accumulators
            self.schedule = self.schedule.compose_sequential(layer_data.schedule)
            self.flow = self.flow.merge_with(layer_data.flow)
            self.parity = self.parity.merge_with(layer_data.parity)

        # Add final data layer if final_layer is 'O' (open)
        if self.final_layer == EdgeSpecValue.O:  # noqa: PLR1702
            data2d = list(self.template.data_coords or [])
            final_z = z0 + 2 * len(self.unit_layers)
            final_layer: dict[tuple[int, int], int] = {}

            for x, y in data2d:
                n = g.add_physical_node()
                all_node2coord[n] = (int(x), int(y), int(final_z))
                all_coord2node[int(x), int(y), int(final_z)] = n
                all_node2role[n] = NodeRole.DATA
                final_layer[int(x), int(y)] = n

            all_nodes_by_z[final_z] = final_layer

            # Add spatial edges for final layer
            UnitLayer.add_spatial_edges(g, final_layer)

            # Add temporal edges connecting to previous layer
            if all_nodes_by_z:
                prev_z = final_z - 1
                if prev_z in all_nodes_by_z:
                    prev_layer = all_nodes_by_z[prev_z]
                    for xy, u in final_layer.items():
                        v = prev_layer.get(xy)
                        if v is not None:
                            with suppress(Exception):
                                g.add_physical_edge(u, v)
                            self.flow.flow.setdefault(NodeIdLocal(v), set()).add(NodeIdLocal(u))

            # Add final layer to schedule
            final_data_nodes = set(final_layer.values())
            if final_data_nodes:
                self.schedule.schedule[2 * final_z + 1] = final_data_nodes

        return g, all_node2coord, all_coord2node, all_node2role


@dataclass
class LayeredMemoryPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for layer-by-layer memory pipe construction."""

    name: ClassVar[str] = "LayeredMemoryPipeSkeleton"

    @overload
    def to_block(self) -> LayeredMemoryPipe: ...

    @overload
    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> LayeredMemoryPipe: ...

    def to_block(
        self, source: PatchCoordGlobal3D | None = None, sink: PatchCoordGlobal3D | None = None
    ) -> LayeredMemoryPipe:
        """Materialize to a LayeredMemoryPipe using MemoryUnitLayer sequence.

        Parameters
        ----------
        source : PatchCoordGlobal3D | None
            Source coordinate for the pipe.
        sink : PatchCoordGlobal3D | None
            Sink coordinate for the pipe.

        Returns
        -------
        LayeredMemoryPipe
            Materialized pipe with d memory unit layers.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        # Create sequence of memory unit layers
        unit_layers = [MemoryUnitLayer() for _ in range(self.d)]

        block = LayeredMemoryPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
            unit_layers=unit_layers,
        )
        block.source = source
        block.sink = sink
        block.final_layer = EdgeSpecValue.O
        return block


class LayeredMemoryPipe(LayeredRHGPipe):
    """Memory pipe constructed from MemoryUnitLayer sequence."""

    name: ClassVar[str] = "LayeredMemoryPipe"

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
        unit_layers: list[UnitLayer],
    ) -> None:
        """Initialize LayeredMemoryPipe.

        Parameters
        ----------
        d : int
            Code distance.
        edgespec : SpatialEdgeSpec | None
            Spatial edge specification.
        direction : PIPEDIRECTION
            Direction of the pipe.
        unit_layers : list[UnitLayer]
            Sequence of unit layers.
        """
        super().__init__(d=d, edge_spec=edgespec or {})
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edgespec or {})
        self.unit_layers = unit_layers

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory pipe: assign all data qubits as input ports."""
        if patch_coord is not None and self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            idx_map = self.template.get_data_indices_cube()
        indices = set(idx_map.values())
        if len(indices) == 0:
            msg = "LayeredMemoryPipe: in_ports should not be empty."
            raise AssertionError(msg)
        self.in_ports = indices

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory pipe: assign all data qubits as output ports."""
        if patch_coord is not None and self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory pipe does not have classical output ports."""
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Detectors are already constructed by unit layers via parity accumulator."""


@dataclass
class LayeredInitPlusPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for layer-by-layer |+⟩ initialization pipe construction."""

    name: ClassVar[str] = "LayeredInitPlusPipeSkeleton"

    @overload
    def to_block(self) -> LayeredInitPlusPipe: ...

    @overload
    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> LayeredInitPlusPipe: ...

    def to_block(
        self, source: PatchCoordGlobal3D | None = None, sink: PatchCoordGlobal3D | None = None
    ) -> LayeredInitPlusPipe:
        """Materialize to a LayeredInitPlusPipe using InitPlusUnitLayer sequence.

        Parameters
        ----------
        source : PatchCoordGlobal3D | None
            Source coordinate for the pipe.
        sink : PatchCoordGlobal3D | None
            Sink coordinate for the pipe.

        Returns
        -------
        LayeredInitPlusPipe
            Materialized pipe with d initialization unit layers.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        # Create sequence of init plus unit layers
        unit_layers = [InitPlusUnitLayer() for _ in range(self.d)]

        block = LayeredInitPlusPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
            unit_layers=unit_layers,
        )
        block.source = source
        block.sink = sink
        block.final_layer = EdgeSpecValue.O
        return block


class LayeredInitPlusPipe(LayeredRHGPipe):
    """|+⟩ initialization pipe constructed from InitPlusUnitLayer sequence."""

    name: ClassVar[str] = "LayeredInitPlusPipe"

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
        unit_layers: list[UnitLayer],
    ) -> None:
        """Initialize LayeredInitPlusPipe.

        Parameters
        ----------
        d : int
            Code distance.
        edgespec : SpatialEdgeSpec | None
            Spatial edge specification.
        direction : PIPEDIRECTION
            Direction of the pipe.
        unit_layers : list[UnitLayer]
            Sequence of unit layers.
        """
        super().__init__(d=d, edge_spec=edgespec or {})
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edgespec or {})
        self.unit_layers = unit_layers

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init plus pipe sets no input ports."""
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init plus pipe: assign all data qubits as output ports."""
        if patch_coord is not None and self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init plus pipe sets no classical output ports."""
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Detectors are already constructed by unit layers via parity accumulator."""
