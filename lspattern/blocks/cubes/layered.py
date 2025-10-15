"""Layer-by-layer cube implementations using UnitLayer sequences.

This module provides RHGCube implementations that are constructed from sequences
of UnitLayer objects, enabling flexible composition of different layer types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from graphix_zx.graphstate import GraphState

from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.blocks.layered_builder import build_layered_graph
from lspattern.blocks.layers.initialize import InitPlusUnitLayer, InitZeroUnitLayer
from lspattern.blocks.layers.memory import MemoryUnitLayer
from lspattern.blocks.unit_layer import UnitLayer
from lspattern.consts import BoundarySide, EdgeSpecValue, NodeRole


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

    unit_layers: Sequence[UnitLayer] = field(default_factory=list)

    def _construct_detectors(self) -> None:
        """Detectors are already constructed by unit layers via parity accumulator.

        This method does nothing as the parity accumulator is populated during
        layer construction in build_layered_graph.
        """
        # The parity accumulator is already populated during layer construction
        # No additional detector construction needed

    def _build_3d_graph(  # type: ignore[override]
        self,
    ) -> tuple[
        GraphState,
        dict[int, tuple[int, int, int]],
        dict[tuple[int, int, int], int],
        dict[int, NodeRole],
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
        return build_layered_graph(
            unit_layers=self.unit_layers,
            d=self.d,
            source=self.source,
            template=self.template,
            final_layer=self.final_layer,
            schedule_accumulator=self.schedule,
            flow_accumulator=self.flow,
            parity_accumulator=self.parity,
            graph=g,
        )


class LayeredMemoryCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for layer-by-layer memory cube construction."""

    name: ClassVar[str] = "LayeredMemoryCubeSkeleton"

    def to_block(self) -> LayeredMemoryCube:
        """Materialize to a LayeredMemoryCube using MemoryUnitLayer sequence.

        Returns
        -------
        LayeredMemoryCube
            Materialized cube with d memory unit layers.

        Raises
        ------
        ValueError
            If the number of unit layers exceeds code distance d.
        """
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)

        # Evaluate template coordinates
        self.template.to_tiling()

        # Create sequence of memory unit layers
        unit_layers = [MemoryUnitLayer() for _ in range(self.d)]

        # Validate unit_layers length
        if len(unit_layers) > self.d:
            msg = f"Unit layers length ({len(unit_layers)}) cannot exceed code distance d ({self.d})"
            raise ValueError(msg)

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

        Raises
        ------
        ValueError
            If the number of unit layers exceeds code distance d.
        """
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)

        # Evaluate template coordinates
        self.template.to_tiling()

        # Create sequence of init plus unit layers
        unit_layers: list[UnitLayer] = [InitPlusUnitLayer()]
        unit_layers += [MemoryUnitLayer() for _ in range(self.d - 1)]

        # Validate unit_layers length
        if len(unit_layers) > self.d:
            msg = f"Unit layers length ({len(unit_layers)}) cannot exceed code distance d ({self.d})"
            raise ValueError(msg)

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

        Raises
        ------
        ValueError
            If the number of unit layers exceeds code distance d.
        """
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)

        # Evaluate template coordinates
        self.template.to_tiling()

        # Create sequence of init zero unit layers
        unit_layers: list[UnitLayer] = [InitZeroUnitLayer()]
        unit_layers += [MemoryUnitLayer() for _ in range(self.d - 1)]

        # Validate unit_layers length
        if len(unit_layers) > self.d:
            msg = f"Unit layers length ({len(unit_layers)}) cannot exceed code distance d ({self.d})"
            raise ValueError(msg)

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
