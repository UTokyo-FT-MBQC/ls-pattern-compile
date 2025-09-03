from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from graphix_zx.graphstate import BaseGraphState
from lspattern.mytype import PatchCoordGlobal3D, BlockKindstr
from lspattern.pipes.base import RHGPipe

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
