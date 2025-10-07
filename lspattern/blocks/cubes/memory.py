from __future__ import annotations

from typing import ClassVar

from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D, PhysCoordLocal2D


class MemoryCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for memory (time-extension) blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "MemoryCubeSkeleton"

    def to_block(self) -> MemoryCube:
        """Materialize to a MemoryCube (template evaluated, no local graph yet)."""
        # Apply spatial open-boundary trimming if specified
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if str(self.edgespec.get(direction, "O")).upper() == "O":
                self.trim_spatial_boundary(direction)
        # Evaluate template coordinates
        self.template.to_tiling()

        block = MemoryCube(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )
        # Memory 系も最終層は開放(O): 次段へ受け渡し
        block.final_layer = "O"
        return block


class MemoryCube(RHGCube):
    name: ClassVar[str] = "MemoryCube"

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory: 全 data(z- 側相当)を入力ポートに割当てる。"""
        # テンプレートの data インデックスを取得
        idx_map = self.template.get_data_indices_cube(patch_coord)
        indices = set(idx_map.values())
        if len(indices) == 0:
            msg = "Memory: in_ports should not be empty."
            raise AssertionError(msg)
        self.in_ports = indices

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory: 全 data(z 側相当)を出力ポートに割当てる。

        位置は in_ports と同一集合(時間延長で同一 (x,y) を受け渡す想定)。
        """
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory does not have cout ports."""
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        z_offset = int(self.source[2]) * (2 * self.d)
        height = max(coord[2] for coord in self.coord2node) - z_offset + 1
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        for z in range(height):
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
