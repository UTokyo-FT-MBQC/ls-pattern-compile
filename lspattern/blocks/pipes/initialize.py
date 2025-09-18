from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from graphix_zx.graphstate import GraphState

from lspattern.blocks.base import RHGBlock
from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D, PhysCoordGlobal3D, PhysCoordLocal2D, SpatialEdgeSpec
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

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
    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> InitPlusPipe: ...

    def to_block(
        self, source: PatchCoordGlobal3D | None = None, sink: PatchCoordGlobal3D | None = None
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
        block.final_layer = "O"
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

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:  # noqa: ARG002
        # Init pipe: 出力はテンプレートの data 全インデックス
        if self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices(source_2d, patch_type="pipe", sink_patch=sink_2d)
        else:
            # Fallback for backward compatibility (no source/sink info)
            idx_map = self.template.get_data_indices()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # 古典出力はなし
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        t_offset = min(self.schedule.schedule.keys(), default=0)
        height = max(self.schedule.schedule.keys(), default=0) - t_offset + 1
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        # ancillas of first layer is not deterministic
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, t_offset)))
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}
        for t in range(1, height):
            for x, y in x2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, t + t_offset)))
                if node_id is None:
                    continue
                self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), []).append(
                    {node_id} | dangling_detectors.get(PhysCoordLocal2D((x, y)), set())
                )
                dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}

            for x, y in z2d:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, t + t_offset)))
                if node_id is None:
                    continue
                self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), []).append(
                    {node_id} | dangling_detectors.get(PhysCoordLocal2D((x, y)), set())
                )
                dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}

        # add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes


@dataclass
class InitPlusPipeSingleLayerSkeleton(RHGPipeSkeleton):
    """Skeleton for a single-layer InitPlus-style pipe."""

    @overload
    def to_block(self) -> InitPlusPipeSingleLayer: ...

    @overload
    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> InitPlusPipeSingleLayer: ...

    def to_block(
        self, source: PatchCoordGlobal3D | None = None, sink: PatchCoordGlobal3D | None = None
    ) -> InitPlusPipeSingleLayer:
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitPlusPipeSingleLayer(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = "O"
        return block


class InitPlusPipeSingleLayer(RHGPipe):
    """Single-layer initialization pipe (height=1) for compose-based initialization."""

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

    def _construct_graph(self) -> tuple:
        """Override to create single-layer graph at z_max position."""

        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Generate nodes at the final layer position (z_max equivalent)
        d_val = int(self.d)
        max_t = 2 * d_val - 1  # Final layer index
        z0 = int(self.source[2]) * (2 * d_val)  # Base z-offset per block
        final_z = z0 + max_t  # Actual z coordinate for the final layer

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Create nodes at the final layer
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}
        cur: dict[tuple[int, int], int] = {}

        # Add data nodes
        for x, y in data2d:
            n = g.add_physical_node()
            node2coord[n] = (int(x), int(y), int(final_z))
            coord2node[int(x), int(y), int(final_z)] = n
            node2role[n] = "data"
            cur[int(x), int(y)] = n

        # Add ancilla nodes (based on layer parity - final layer should have appropriate ancillas)
        if (max_t % 2) == 0:
            # Even layer: X ancillas
            for x, y in x2d:
                n = g.add_physical_node()
                node2coord[n] = (int(x), int(y), int(final_z))
                coord2node[int(x), int(y), int(final_z)] = n
                node2role[n] = "ancilla_x"
                cur[int(x), int(y)] = n
        else:
            # Odd layer: Z ancillas
            for x, y in z2d:
                n = g.add_physical_node()
                node2coord[n] = (int(x), int(y), int(final_z))
                coord2node[int(x), int(y), int(final_z)] = n
                node2role[n] = "ancilla_z"
                cur[int(x), int(y)] = n

        nodes_by_z[final_z] = cur

        self._construct_schedule(nodes_by_z, node2role)

        # Add spatial edges only (no temporal edges for single layer)
        self._add_spatial_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init pipe: 入力ポートは持たない
        return super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:  # noqa: ARG002
        # Init pipe: 出力はテンプレートの data 全インデックス
        if self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices(source_2d, patch_type="pipe", sink_patch=sink_2d)
        else:
            # Fallback for backward compatibility (no source/sink info)
            idx_map = self.template.get_data_indices()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # 古典出力はなし
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Single layer only has dangling detectors, no parity checks."""
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        t_offset = min(self.schedule.schedule.keys(), default=0)
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # For single layer, all ancillas become dangling detectors
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, t_offset)))
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}

        # Add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes
