"""Layer-by-layer pipe implementations using UnitLayer sequences.

This module provides RHGPipe implementations that are constructed from sequences
of UnitLayer objects, enabling flexible composition of different layer types for
spatial connections between cubes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, overload

from graphix_zx.graphstate import GraphState

from lspattern.blocks.layered_builder import build_layered_graph
from lspattern.blocks.layers.initialize import InitPlusUnitLayer
from lspattern.blocks.layers.memory import MemoryUnitLayer
from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.consts import NodeRole, TimeBoundarySpecValue
from lspattern.mytype import PatchCoordGlobal3D, SpatialEdgeSpec
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lspattern.blocks.unit_layer import UnitLayer
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
        """Build 3D RHG graph structure layer-by-layer for pipe.

        Returns
        -------
        tuple
            (graph, node2coord, coord2node, node2role) for the complete pipe.
        """
        g = GraphState()
        graph, node2coord, coord2node, node2role, schedule, flow, parity = build_layered_graph(
            unit_layers=self.unit_layers,
            d=self.d,
            source=self.source,
            template=self.template,
            final_layer=self.final_layer,
            graph=g,
        )
        # Update accumulators with returned values
        self.schedule = schedule
        self.flow = flow
        self.parity = parity
        return graph, node2coord, coord2node, node2role


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

        Raises
        ------
        ValueError
            If the number of unit layers exceeds code distance d.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        # Create sequence of memory unit layers
        unit_layers: list[UnitLayer] = [MemoryUnitLayer() for _ in range(self.d)]

        # Validate unit_layers length
        if len(unit_layers) > self.d:
            msg = f"Unit layers length ({len(unit_layers)}) cannot exceed code distance d ({self.d})"
            raise ValueError(msg)

        block = LayeredMemoryPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
            unit_layers=unit_layers,
        )
        block.source = source
        block.sink = sink
        block.final_layer = TimeBoundarySpecValue.O
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

        Raises
        ------
        ValueError
            If the number of unit layers exceeds code distance d.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        # Create sequence of init plus unit layers
        unit_layers: list[UnitLayer] = [InitPlusUnitLayer()]
        unit_layers += [MemoryUnitLayer() for _ in range(self.d - 1)]

        # Validate unit_layers length
        if len(unit_layers) > self.d:
            msg = f"Unit layers length ({len(unit_layers)}) cannot exceed code distance d ({self.d})"
            raise ValueError(msg)

        block = LayeredInitPlusPipe(d=self.d, edgespec=self.edgespec, direction=direction, unit_layers=unit_layers)
        block.source = source
        block.sink = sink
        block.final_layer = TimeBoundarySpecValue.O
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
