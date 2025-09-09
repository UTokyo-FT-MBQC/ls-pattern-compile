from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.mytype import PatchCoordGlobal3D, SpatialEdgeSpec
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

if TYPE_CHECKING:
    from lspattern.consts.consts import PIPEDIRECTION


@dataclass
class MemoryPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for a Memory-style pipe (time-preserving pass-through).

    Note: edgespec は省略可能(None)。テンプレートは方向に依存して決まる。
    """

    @overload
    def to_block(self) -> MemoryPipe: ...

    @overload
    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> MemoryPipe: ...

    def to_block(
        self, source: PatchCoordGlobal3D | None = None, sink: PatchCoordGlobal3D | None = None
    ) -> MemoryPipe:
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)
        spec = self.edgespec
        block = MemoryPipe(
            d=self.d,
            edgespec=spec,
            direction=direction,
        )
        # ソース/シンク座標は後段で shift_coords により調整可能
        block.source = source
        block.sink = sink
        # Memory 系は最終層は開放(O)
        block.final_layer = "O"
        return block


class MemoryPipe(RHGPipe):
    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        # RHGPipe(dataclass) の自動 __init__ は使用せず、明示的に初期化
        super().__init__(d=d, edge_spec=edgespec or {})
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edgespec or {})

    def set_in_ports(self) -> None:
        # Pipe: data の全インデックスを in とする(z- 側相当)
        idx_map = self.template.get_data_indices()
        indices = set(idx_map.values())
        if len(indices) == 0:
            msg = "MemoryPipe: in_ports は空であってはならない"
            raise AssertionError(msg)
        self.in_ports = indices

    def set_out_ports(self) -> None:
        # Pipe: data の全インデックスを out とする(z 側相当)
        idx_map = self.template.get_data_indices()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self) -> None:
        # Memory: 古典出力は持たない
        return super().set_cout_ports()
