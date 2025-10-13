from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from lspattern.consts.consts import PIPEDIRECTION


@dataclass
class MemoryPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for a Memory-style pipe (time-preserving pass-through).

    Note: edgespec は省略可能(None)。テンプレートは方向に依存して決まる。
    """

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
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
        block.final_layer = EdgeSpecValue.O
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

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Pipe: data の全インデックスを in とする(z- 側相当)
        if patch_coord is not None and self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            # Fallback for backward compatibility (no patch coordinate or source/sink info)
            idx_map = self.template.get_data_indices_cube()
        indices = set(idx_map.values())
        if len(indices) == 0:
            msg = "MemoryPipe: in_ports should not be empty."
            raise AssertionError(msg)
        self.in_ports = indices

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # Pipe: data の全インデックスを out とする(z 側相当)
        if patch_coord is not None and self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            # Fallback for backward compatibility (no patch coordinate or source/sink info)
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        zmin = min({coord[2] for coord in self.coord2node}, default=0)
        zmax = max({coord[2] for coord in self.coord2node}, default=0)
        height = zmax - zmin + 1

        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        for dz in range(height):
            for x, y in x2d + z2d:
                coord = PhysCoordLocal2D((x, y))
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, zmin + dz)))
                if node_id is None:
                    continue
                node_group = {node_id} | dangling_detectors.pop(coord, set())
                self.parity.checks.setdefault(coord, {})[zmin + dz] = node_group
                dangling_detectors[coord] = {node_id}

        # add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes
