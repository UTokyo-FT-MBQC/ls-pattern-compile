from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from graphix_zx.graphstate import GraphState

from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.consts import EdgeSpecValue
from lspattern.mytype import (
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PhysCoordLocal2D,
    SpatialEdgeSpec,
)
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

# Type alias for the return type of _build_3d_graph method
Build3DGraphReturn = tuple[
    GraphState,
    dict[int, tuple[int, int, int]],
    dict[tuple[int, int, int], int],
    dict[int, str],
]

if TYPE_CHECKING:
    from lspattern.consts.consts import PIPEDIRECTION


@dataclass
class InitPlusPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for an InitPlus-style pipe.

    Behavior
    - If ``edgespec`` is ``None``, downstream components use direction-specific defaults:
      - Horizontal (RIGHT/LEFT): {TOP: 'O', BOTTOM: 'O', LEFT: 'X', RIGHT: 'Z'}
      - Vertical   (TOP/BOTTOM): {LEFT: 'O', RIGHT: 'O', TOP: 'X', BOTTOM: 'Z'}
    - Direction is inferred from ``source`` and ``sink`` in ``to_block`` via
      ``get_direction``.
    """

    @overload
    def to_block(self) -> InitPlusPipe: ...

    @overload
    def to_block(
        self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D
    ) -> InitPlusPipe: ...

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitPlusPipe:
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitPlusPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = EdgeSpecValue.O
        return block


class InitPlusPipe(RHGPipe):
    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        # Convert None to empty dict for compatibility
        edge_spec = edgespec or {}
        super().__init__(d=d, edge_spec=edge_spec)
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edge_spec)

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init pipe: 入力ポートは持たない
        return super().set_in_ports(patch_coord)

    def set_out_ports(
        self, patch_coord: tuple[int, int] | None = None
    ) -> None:  # noqa: ARG002
        # Init pipe: 出力はテンプレートの data 全インデックス
        if self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            # Fallback for backward compatibility (no source/sink info)
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # initialize does not have cout ports
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = int(self.source[2]) * (2 * self.d)
        height = max({coord[2] for coord in self.coord2node}, default=0) - z_offset + 1
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        # ancillas of first layer is not deterministic
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z_offset)))
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}
            self.parity.ignore_dangling[PhysCoordLocal2D((x, y))] = True
        for z in range(1, height):
            for x, y in x2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z + z_offset)))
                if node_id is None:
                    continue
                coord = PhysCoordLocal2D((x, y))
                node_group = {node_id} | dangling_detectors.get(coord, set())
                self.parity.checks.setdefault(coord, {})[z + z_offset] = node_group
                dangling_detectors[coord] = {node_id}

            for x, y in z2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z + z_offset)))
                if node_id is None:
                    continue
                coord = PhysCoordLocal2D((x, y))
                node_group = {node_id} | dangling_detectors.get(coord, set())
                self.parity.checks.setdefault(coord, {})[z + z_offset] = node_group
                dangling_detectors[coord] = {node_id}

        # add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes


@dataclass
class InitPlusPipeThinLayerSkeleton(RHGPipeSkeleton):
    """Skeleton for thin-layer Plus State initialization pipes in pipe-shaped RHG structures."""

    @overload
    def to_block(self) -> InitPlusThinLayerPipe: ...

    @overload
    def to_block(
        self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D
    ) -> InitPlusThinLayerPipe: ...

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitPlusThinLayerPipe:
        """
        Return a template-holding block for single-layer initialization.

        Returns
        -------
        InitPlusThinLayerPipe
            A block containing the template with no local graph state.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitPlusThinLayerPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = EdgeSpecValue.O
        return block


class InitPlusThinLayerPipe(RHGPipe):
    """Thin-layer Plus State initialization pipe (height=3) for compose-based initialization."""

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        # Convert None to empty dict for compatibility
        edge_spec = edgespec or {}
        super().__init__(d=d, edge_spec=edge_spec)
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edge_spec)

    def _build_3d_graph(self) -> Build3DGraphReturn:
        """Override to create single-layer graph with only 13 nodes (9 data + 4 ancilla) at z=2*d."""
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Calculate z-coordinate based on source position and 2*d
        d_val = int(self.d)
        z0 = int(self.source[2]) * (2 * d_val)  # Base z-offset per block
        start_layer_z = z0 + (2 * d_val) - 2
        max_t = 2

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for each time slice
        nodes_by_z = self._assign_nodes_by_timeslice(
            g, data2d, x2d, z2d, max_t, start_layer_z, node2coord, coord2node, node2role
        )

        self._construct_schedule(nodes_by_z, node2role)

        self._add_spatial_edges(g, nodes_by_z)
        self._add_temporal_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init pipe: 入力ポートは持たない
        return super().set_in_ports(patch_coord)

    def set_out_ports(
        self, patch_coord: tuple[int, int] | None = None
    ) -> None:  # noqa: ARG002
        # Init pipe: 出力はテンプレートの data 全インデックス
        if self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            # Fallback for backward compatibility (no source/sink info)
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # 古典出力はなし
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Construct detectors for the thin-layer initialization pipe."""
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = int(self.source[2]) * (2 * self.d)
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # add dangling detectors for connectivity to next block
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(
                PhysCoordGlobal3D((x, y, z_offset + 2 * self.d - 2))
            )
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}

        for z in range(2 * self.d - 1, 2 * self.d + 1):  # height is fixed to 2
            for x, y in x2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z + z_offset)))
                if node_id is None:
                    continue
                coord = PhysCoordLocal2D((x, y))
                node_group = {node_id} | dangling_detectors.get(coord, set())
                self.parity.checks.setdefault(coord, {})[z + z_offset] = node_group
                dangling_detectors[coord] = {node_id}

            for x, y in z2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z + z_offset)))
                if node_id is None:
                    continue
                coord = PhysCoordLocal2D((x, y))
                node_group = {node_id} | dangling_detectors.get(coord, set())
                self.parity.checks.setdefault(coord, {})[z + z_offset] = node_group
                dangling_detectors[coord] = {node_id}

        # Add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes


@dataclass
class InitZeroPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for an InitZero-style pipe.

    Behavior
    - If ``edgespec`` is ``None``, downstream components use direction-specific defaults:
      - Horizontal (RIGHT/LEFT): {TOP: 'O', BOTTOM: 'O', LEFT: 'X', RIGHT: 'Z'}
      - Vertical   (TOP/BOTTOM): {LEFT: 'O', RIGHT: 'O', TOP: 'X', BOTTOM: 'Z'}
    - Direction is inferred from ``source`` and ``sink`` in ``to_block`` via
      ``get_direction``.
    """

    @overload
    def to_block(self) -> InitZeroPipe: ...

    @overload
    def to_block(
        self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D
    ) -> InitZeroPipe: ...

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitZeroPipe:
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitZeroPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = EdgeSpecValue.O
        return block


class InitZeroPipe(RHGPipe):
    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        # Convert None to empty dict for compatibility
        edge_spec = edgespec or {}
        super().__init__(d=d, edge_spec=edge_spec)
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edge_spec)

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init pipe: 入力ポートは持たない
        return super().set_in_ports(patch_coord)

    def set_out_ports(
        self, patch_coord: tuple[int, int] | None = None
    ) -> None:  # noqa: ARG002
        # Init pipe: 出力はテンプレートの data 全インデックス
        if self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            # Fallback for backward compatibility (no source/sink info)
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # initialize does not have cout ports
        return super().set_cout_ports(patch_coord)
    
    def _build_3d_graph(self) -> Build3DGraphReturn:
        """Override to create single-layer graph with only 13 nodes (9 data + 4 ancilla) at z=2*d."""
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Calculate z-coordinate based on source position and 2*d
        d_val = int(self.d)
        z0 = int(self.source[2]) * (2 * d_val)  # Base z-offset per block
        start_layer_z = z0 + 1
        max_t = 2 * self.d - 1

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for each time slice
        nodes_by_z = self._assign_nodes_by_timeslice(
            g, data2d, x2d, z2d, max_t, start_layer_z, node2coord, coord2node, node2role
        )

        self._construct_schedule(nodes_by_z, node2role)

        self._add_spatial_edges(g, nodes_by_z)
        self._add_temporal_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = int(self.source[2]) * (2 * self.d)
        height = max({coord[2] for coord in self.coord2node}, default=0) - z_offset + 1
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        # ancillas of first layer is not deterministic
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z_offset)))
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}
            self.parity.ignore_dangling[PhysCoordLocal2D((x, y))] = True
        for z in range(1, height):
            for x, y in x2d + z2d:
                if node_id := self.coord2node.get(
                    PhysCoordGlobal3D((x, y, z + z_offset))
                ):
                    coord = PhysCoordLocal2D((x, y))
                    node_group = {node_id} | dangling_detectors.get(coord, set())

                    self.parity.checks.setdefault(coord, {})[z + z_offset] = node_group
                    dangling_detectors[coord] = {node_id}

        # add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes


@dataclass
class InitZeroPipeThinLayerSkeleton(RHGPipeSkeleton):
    """Skeleton for thin-layer Zero State initialization pipes in pipe-shaped RHG structures."""

    @overload
    def to_block(self) -> InitZeroThinLayerPipe: ...

    @overload
    def to_block(
        self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D
    ) -> InitZeroThinLayerPipe: ...

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitZeroThinLayerPipe:
        """
        Return a template-holding block for single-layer initialization.

        Returns
        -------
        InitZeroThinLayerPipe
            A block containing the template with no local graph state.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitZeroThinLayerPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = EdgeSpecValue.O
        return block


class InitZeroThinLayerPipe(RHGPipe):
    """Thin-layer Zero State initialization pipe (height=2) for compose-based initialization."""

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        # Convert None to empty dict for compatibility
        edge_spec = edgespec or {}
        super().__init__(d=d, edge_spec=edge_spec)
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edge_spec)

    def _build_3d_graph(self) -> Build3DGraphReturn:
        """Override to create single-layer graph with only 13 nodes (9 data + 4 ancilla) at z=2*d."""
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Calculate z-coordinate based on source position and 2*d
        d_val = int(self.d)
        z0 = int(self.source[2]) * (2 * d_val)  # Base z-offset per block
        start_layer_z = z0 + (2 * d_val) - 1
        max_t = 1

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for each time slice
        nodes_by_z = self._assign_nodes_by_timeslice(
            g, data2d, x2d, z2d, max_t, start_layer_z, node2coord, coord2node, node2role
        )

        self._construct_schedule(nodes_by_z, node2role)

        self._add_spatial_edges(g, nodes_by_z)
        self._add_temporal_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init pipe: 入力ポートは持たない
        return super().set_in_ports(patch_coord)

    def set_out_ports(
        self, patch_coord: tuple[int, int] | None = None
    ) -> None:  # noqa: ARG002
        # set output ports to all data indices in the template
        if self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            # Fallback for backward compatibility (no source/sink info)
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # sets no classical output ports
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Construct detectors for the thin-layer initialization pipe."""
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = int(self.source[2]) * (2 * self.d)
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # add dangling detectors for connectivity to next block
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(
                PhysCoordGlobal3D((x, y, z_offset + 2 * self.d - 1))
            )
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}

        # TODO: this code can be simplified with plus block
        for z in range(2 * self.d, 2 * self.d + 1):  # height is fixed to 1
            for x, y in x2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z + z_offset)))
                if node_id is None:
                    continue
                coord = PhysCoordLocal2D((x, y))
                node_group = {node_id} | dangling_detectors.get(coord, set())
                self.parity.checks.setdefault(coord, {})[z + z_offset] = node_group
                dangling_detectors[coord] = {node_id}

            for x, y in z2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z + z_offset)))
                if node_id is None:
                    continue
                coord = PhysCoordLocal2D((x, y))
                node_group = {node_id} | dangling_detectors.get(coord, set())
                self.parity.checks.setdefault(coord, {})[z + z_offset] = node_group
                dangling_detectors[coord] = {node_id}

        # Add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes
