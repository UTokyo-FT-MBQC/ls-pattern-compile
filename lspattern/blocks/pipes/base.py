from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Optional,
    Tuple,
)

from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton
from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import (
    PatchCoordGlobal3D,
    SpatialEdgeSpec,
)
from lspattern.tiling.template import (
    ScalableTemplate,
)


@dataclass
class RHGPipeSkeleton(RHGBlockSkeleton):
    logical: int
    d: int
    origin: Optional[Tuple[int, int]] = None


@dataclass
class RHGPipe(RHGBlock):
    source: Optional[PatchCoordGlobal3D] = None
    sink: Optional[PatchCoordGlobal3D] = None
    # Direction of the pipe (spatial or temporal)
    direction: Optional[PIPEDIRECTION] = None

    # Template or tiling backing this pipe (implementation-specific)
    template: Optional[ScalableTemplate] = None  # override type to allow None
    # Optional spatial edge spec for this pipe (alias handled in RHGBlock as edge_spec/edgespec)
    edgespec: Optional[SpatialEdgeSpec] = None  # type: ignore[assignment]
