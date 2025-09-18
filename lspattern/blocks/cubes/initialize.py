"""Initialization block(s) for cube-shaped RHG structures."""

from __future__ import annotations

from typing import ClassVar, Literal

from graphix_zx.graphstate import GraphState

from lspattern.blocks.base import RHGBlock
from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D, PhysCoordLocal2D
from lspattern.tiling.template import RotatedPlanarCubeTemplate


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
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if self.edgespec[direction] == "O":
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()

        block = InitPlus(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )

        # Init 系は最終層は測定せず開放(O)
        block.final_layer = "O"

        return block


class InitPlus(RHGCube):
    name: ClassVar[str] = "InitPlus"

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init plus sets no input ports
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init: 最終スライス(z+)の data を出力ポート(テンプレートの data 全インデックス)とみなす
        idx_map = self.template.get_data_indices(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # sets no classical output ports
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


class InitPlusCubeSingleLayerSkeleton(RHGCubeSkeleton):
    """Skeleton for single-layer initialization blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "InitPlusCubeSingleLayerSkeleton"

    def to_block(self) -> RHGCube:
        """
        Return a template-holding block for single-layer initialization.

        Returns
        -------
            RHGBlock: A block containing the template with no local graph state.
        """
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if self.edgespec[direction] == "O":
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()

        block = InitPlusSingleLayer(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )

        # Init 系は最終層は測定せず開放(O)
        block.final_layer = "O"

        return block


class InitPlusSingleLayer(RHGCube):
    """Single-layer initialization cube (height=1) for compose-based initialization."""

    name: ClassVar[str] = "InitPlusSingleLayer"

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
        # Init plus sets no input ports
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Init: 最終スライス(z+)の data を出力ポート(テンプレートの data 全インデックス)とみなす
        idx_map = self.template.get_data_indices(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # sets no classical output ports
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


if __name__ == "__main__":
    # NOTE: Interactive 3D preview code omitted for brevity

    # Hardcoded options (edit here as needed)
    d = 3
    edgespec: dict[str, Literal["X", "Z", "O"]] = {
        "TOP": "X",
        "BOTTOM": "Z",
        "LEFT": "X",
        "RIGHT": "Z",
    }  # e.g., {"TOP":"X","BOTTOM":"Z",...}
    ANCILLA_MODE = "both"  # "both" | "x" | "z"
    EDGE_WIDTH = 0.5  # thicker black edges
    INTERACTIVE = True  # interactive plot

    # Build template and block
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    _ = template.to_tiling()  # populate internal coords for indices

    block = InitPlus(d=d, template=template)

    # Prepare colored point clouds (match template colors)
    # data: white, X ancilla: green, Z ancilla: blue
    color_map = {
        "data": {
            "face": "white",
            "edge": "black",
            "size": 40,
        },
        "ancilla_x": {
            "face": "#2ecc71",
            "edge": "#1e8449",
            "size": 36,
        },
        "ancilla_z": {
            "face": "#3498db",
            "edge": "#1f618d",
            "size": 36,
        },
    }
