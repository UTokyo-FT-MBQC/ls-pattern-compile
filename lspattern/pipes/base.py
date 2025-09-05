from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from graphix_zx.graphstate import BaseGraphState
from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import PatchCoordGlobal3D, SpatialEdgeSpec
from lspattern.tiling.template import ScalableTemplate


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

    # Direction of the pipe (spatial or temporal)
    direction: PIPEDIRECTION
    # Template or tiling backing this pipe (implementation-specific)
    template: ScalableTemplate
    # Optional spatial edge spec for this pipe
    edgespec: Optional[SpatialEdgeSpec] = None
    # Local graph fragment contributed by the pipe
    graph_local: Optional[BaseGraphState] = None
    # Optional port/coord registries for compatibility with canvas2
    in_ports: list[int] = field(default_factory=list)
    out_ports: list[int] = field(default_factory=list)
    node_coords: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    
    def shift_ids(self, by: int = 0) -> None:
        # Intentionally left minimal; concrete pipes should implement.
        return

    def shift_coords(
        self,
        patch_coord: PatchCoordGlobal3D,
        direction: PIPEDIRECTION,
    ) -> None:
        # Intentionally left minimal; concrete pipes should implement.
        return


class Memory(RHGPipe):
    pass
