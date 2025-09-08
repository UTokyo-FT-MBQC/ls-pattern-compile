from __future__ import annotations

# ruff: noqa: I001

from lspattern.blocks.cubes.base import RHGCubeSkeleton, RHGCube


class MemoryCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for memory (time-extension) blocks in cube-shaped RHG structures."""

    name: str = __qualname__

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
        # Memory 系も最終層は開放（O）: 次段へ受け渡し
        block.final_layer = "O"
        return block


class MemoryCube(RHGCube):
    name: str = __qualname__

    def set_in_ports(self) -> None:
        """Memory: 全 data（z- 側相当）を入力ポートに割当てる。"""
        # テンプレートの data インデックスを取得
        idx_map = self.template.get_data_indices()
        indices = set(idx_map.values())
        if len(indices) == 0:
            msg = "Memory: in_ports は空であってはならない"
            raise AssertionError(msg)
        self.in_ports = indices

    def set_out_ports(self) -> None:
        """Memory: 全 data（z 側相当）を出力ポートに割当てる。

        位置は in_ports と同一集合（時間延長で同一 (x,y) を受け渡す想定）。
        """
        idx_map = self.template.get_data_indices()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self) -> None:
        """Memory: 古典出力は持たない。"""
        return super().set_cout_ports()
