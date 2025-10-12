"""Layer-by-layer cube implementations using UnitLayer sequences.

This module provides RHGCube implementations that are constructed from sequences
of UnitLayer objects, enabling flexible composition of different layer types.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import ClassVar

from graphix_zx.graphstate import GraphState

from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.blocks.layers.initialize import InitPlusUnitLayer, InitZeroUnitLayer
from lspattern.blocks.layers.memory import MemoryUnitLayer
from lspattern.blocks.unit_layer import UnitLayer
from lspattern.consts import BoundarySide, EdgeSpecValue, NodeRole
from lspattern.mytype import NodeIdLocal


@dataclass
class LayeredRHGCube(RHGCube):
    """RHGCube constructed from a sequence of UnitLayers.

    This class enables flexible layer-by-layer construction of cubes by composing
    different types of UnitLayer objects (e.g., initialization layers followed by
    memory layers).

    Attributes
    ----------
    unit_layers : list[UnitLayer]
        Sequence of unit layers to compose. The length of this list determines
        the effective code distance.
    """

    unit_layers: list[UnitLayer] = field(default_factory=list)

    def _build_3d_graph(  # noqa: C901
        self,
    ) -> tuple[
        GraphState,
        dict[int, tuple[int, int, int]],
        dict[tuple[int, int, int], int],
        dict[int, str],
    ]:
        """Build 3D RHG graph structure layer-by-layer.

        This method overrides the base implementation to construct the graph by
        iteratively building each UnitLayer and merging the results.

        Returns
        -------
        tuple
            (graph, node2coord, coord2node, node2role) for the complete cube.
        """
        g = GraphState()
        z0 = int(self.source[2]) * (2 * self.d)

        # Accumulate all layer data
        all_nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}
        all_node2coord: dict[int, tuple[int, int, int]] = {}
        all_coord2node: dict[tuple[int, int, int], int] = {}
        all_node2role: dict[int, str] = {}

        # Track last non-empty layer for connecting across empty layers
        last_nonempty_layer_z: int | None = None

        for i, unit_layer in enumerate(self.unit_layers):
            layer_z = z0 + 2 * i
            layer_data = unit_layer.build_layer(g, layer_z, self.template)

            # Check if this layer is empty (no nodes)
            is_empty = not layer_data.nodes_by_z

            if not is_empty:
                # Merge layer data
                all_nodes_by_z.update(layer_data.nodes_by_z)
                all_node2coord.update(layer_data.node2coord)
                all_coord2node.update(layer_data.coord2node)
                all_node2role.update(layer_data.node2role)

                # Merge accumulators
                self.schedule = self.schedule.compose_sequential(layer_data.schedule)
                self.flow = self.flow.merge_with(layer_data.flow)
                self.parity = self.parity.merge_with(layer_data.parity)

                # Add temporal edges between this layer and the last non-empty layer
                if last_nonempty_layer_z is not None:
                    # Find the first z-coordinate in this layer
                    current_layer_zs = sorted(layer_data.nodes_by_z.keys())
                    if current_layer_zs:
                        first_z = current_layer_zs[0]
                        current_layer_nodes = layer_data.nodes_by_z[first_z]

                        # Connect to the last z-coordinate of the previous non-empty layer
                        prev_layer_nodes = all_nodes_by_z.get(last_nonempty_layer_z, {})

                        for xy, u in current_layer_nodes.items():
                            v = prev_layer_nodes.get(xy)
                            if v is not None:
                                with suppress(Exception):
                                    g.add_physical_edge(u, v)
                                self.flow.flow.setdefault(NodeIdLocal(v), set()).add(NodeIdLocal(u))

                # Update last non-empty layer to be the last z in this layer
                layer_zs = sorted(layer_data.nodes_by_z.keys())
                if layer_zs:
                    last_nonempty_layer_z = layer_zs[-1]

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


class LayeredMemoryCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for layer-by-layer memory cube construction."""

    name: ClassVar[str] = "LayeredMemoryCubeSkeleton"

    def to_block(self) -> LayeredMemoryCube:
        """Materialize to a LayeredMemoryCube using MemoryUnitLayer sequence.

        Returns
        -------
        LayeredMemoryCube
            Materialized cube with d memory unit layers.
        """
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)

        # Evaluate template coordinates
        self.template.to_tiling()

        # Create sequence of memory unit layers
        unit_layers = [MemoryUnitLayer() for _ in range(self.d)]

        block = LayeredMemoryCube(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
            unit_layers=unit_layers,
        )
        block.final_layer = EdgeSpecValue.O
        return block


class LayeredMemoryCube(LayeredRHGCube):
    """Memory cube constructed from MemoryUnitLayer sequence."""

    name: ClassVar[str] = "LayeredMemoryCube"

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory: assign all data qubits as input ports."""
        idx_map = self.template.get_data_indices_cube(patch_coord)
        indices = set(idx_map.values())
        if len(indices) == 0:
            msg = "LayeredMemoryCube: in_ports should not be empty."
            raise AssertionError(msg)
        self.in_ports = indices

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory: assign all data qubits as output ports."""
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory does not have classical output ports."""
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Detectors are already constructed by unit layers via parity accumulator."""
        # The parity accumulator is already populated during layer construction
        # No additional detector construction needed


class LayeredInitPlusCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for layer-by-layer |+⟩ initialization cube construction."""

    name: ClassVar[str] = "LayeredInitPlusCubeSkeleton"

    def to_block(self) -> LayeredInitPlusCube:
        """Materialize to a LayeredInitPlusCube using InitPlusUnitLayer sequence.

        Returns
        -------
        LayeredInitPlusCube
            Materialized cube with d initialization unit layers.
        """
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)

        # Evaluate template coordinates
        self.template.to_tiling()

        # Create sequence of init plus unit layers
        unit_layers = [InitPlusUnitLayer() for _ in range(self.d)]

        block = LayeredInitPlusCube(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
            unit_layers=unit_layers,
        )
        block.final_layer = EdgeSpecValue.O
        return block


class LayeredInitPlusCube(LayeredRHGCube):
    """|+⟩ initialization cube constructed from InitPlusUnitLayer sequence."""

    name: ClassVar[str] = "LayeredInitPlusCube"

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init plus sets no input ports."""
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init: assign all data qubits as output ports."""
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init plus sets no classical output ports."""
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Detectors are already constructed by unit layers via parity accumulator."""


class LayeredInitZeroCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for layer-by-layer |0⟩ initialization cube construction."""

    name: ClassVar[str] = "LayeredInitZeroCubeSkeleton"

    def to_block(self) -> LayeredInitZeroCube:
        """Materialize to a LayeredInitZeroCube using InitZeroUnitLayer sequence.

        Returns
        -------
        LayeredInitZeroCube
            Materialized cube with d initialization unit layers.
        """
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)

        # Evaluate template coordinates
        self.template.to_tiling()

        # Create sequence of init zero unit layers
        unit_layers = [InitZeroUnitLayer() for _ in range(self.d)]

        block = LayeredInitZeroCube(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
            unit_layers=unit_layers,
        )
        block.final_layer = EdgeSpecValue.O
        return block


class LayeredInitZeroCube(LayeredRHGCube):
    """|0⟩ initialization cube constructed from InitZeroUnitLayer sequence."""

    name: ClassVar[str] = "LayeredInitZeroCube"

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init zero sets no input ports."""
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init: assign all data qubits as output ports."""
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Init zero sets no classical output ports."""
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Detectors are already constructed by unit layers via parity accumulator."""
