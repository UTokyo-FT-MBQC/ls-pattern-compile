"""Initialization block(s) for cube-shaped RHG structures."""

from __future__ import annotations

from typing import ClassVar

from graphix_zx.graphstate import GraphState

from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue, TemporalBoundarySpecValue
from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D, PhysCoordLocal2D

# Type alias for the return type of _build_3d_graph method
Build3DGraphReturn = tuple[GraphState, dict[int, tuple[int, int, int]], dict[tuple[int, int, int], int], dict[int, str]]


class InitPlusCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for initialization blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "InitPlusCubeSkeleton"

    def to_block(self) -> RHGCube:
        """
        Return a template-holding block (no local graph state).

        Returns
        -------
            RHGBlock: A block containing the template with no local graph state.
        """
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec[direction] == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()

        block = InitPlus(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )

        # Init 系は最終層は測定せず開放(O)
        block.final_layer = TemporalBoundarySpecValue.O

        return block


class InitPlus(RHGCube):
    name: ClassVar[str] = "InitPlus"

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init plus sets no input ports
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init: 最終スライス(z+)の data を出力ポート(テンプレートの data 全インデックス)とみなす
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # sets no classical output ports
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = self.source[2] * (2 * self.d)
        height = max({coord[2] for coord in self.coord2node}, default=0) - z_offset + 1
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        # ancillas of first layer is not deterministic
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z_offset)))
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {node_id}
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


class InitPlusCubeThinLayerSkeleton(RHGCubeSkeleton):
    """Skeleton for thin-layer Plus State initialization blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "InitPlusCubeThinLayerSkeleton"

    def to_block(self) -> RHGCube:
        """
        Return a template-holding block for single-layer initialization.

        Returns
        -------
        RHGBlock
            A block containing the template with no local graph state.
        """
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec[direction] == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()

        block = InitPlusThinLayer(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )

        block.final_layer = TemporalBoundarySpecValue.O

        return block


class InitPlusThinLayer(RHGCube):
    """Thin-layer Plus State initialization cube (height=3) for compose-based initialization."""

    name: ClassVar[str] = "InitPlusThinLayer"

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
        # Init plus sets no input ports
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init: 最終スライス(z+)の data を出力ポート(テンプレートの data 全インデックス)とみなす
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # sets no classical output ports
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Construct detectors for the thin-layer initialization block."""
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = self.source[2] * (2 * self.d)
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # add dangling detectors for connectivity to next block
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z_offset + 2 * self.d - 2)))
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


class InitZeroCubeThinLayerSkeleton(RHGCubeSkeleton):
    """Skeleton for thin-layer Zero State initialization blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "InitZeroCubeThinLayerSkeleton"

    def to_block(self) -> RHGCube:
        """
        Return a template-holding block for single-layer initialization.

        Returns
        -------
        RHGBlock
            A block containing the template with no local graph state.
        """
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec[direction] == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()

        block = InitZeroThinLayer(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )

        block.final_layer = TemporalBoundarySpecValue.O

        return block


class InitZeroThinLayer(RHGCube):
    """Thin-layer Zero State initialization cube (height=2) for compose-based initialization."""

    name: ClassVar[str] = "InitZeroThinLayer"

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
        # Init plus sets no input ports
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # set output ports to all data indices in the template
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # sets no classical output ports
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Construct detectors for the thin-layer initialization block."""
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = self.source[2] * (2 * self.d)
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # add dangling detectors for connectivity to next block
        for x, y in x2d + z2d:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z_offset + 2 * self.d - 1)))
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
