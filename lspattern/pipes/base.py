from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    Optional,
    Tuple,
)

from graphix_zx.graphstate import BaseGraphState
from lspattern.mytype import BlockKindstr, PatchCoordGlobal3D


@dataclass
class RHGPipeSkeleton:
    logical: int
    d: int
    origin: Optional[Tuple[int, int]] = None


@dataclass
class RHGPipe:
    source: PatchCoordGlobal3D
    sink: PatchCoordGlobal3D
    d: int
    kind: BlockKindstr
    direction: Literal["up", "down", "left", "right"]
    local_template: Any
    local_graph: Optional[BaseGraphState] = None

    def materialize(self, skeleton: RHGPipeSkeleton) -> None:
        pass

    def shift_ids(by: int = 0):
        pass

    def shift_coords(by: tuple[int, int, int] = 0):
        pass


class Memory(RHGPipe):
    pass
