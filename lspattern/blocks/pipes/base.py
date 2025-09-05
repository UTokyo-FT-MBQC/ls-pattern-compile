from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton

if TYPE_CHECKING:
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
    origin: tuple[int, int] | None = None


@dataclass
class RHGPipe(RHGBlock):
    source: PatchCoordGlobal3D | None = None
    sink: PatchCoordGlobal3D | None = None
    # Direction of the pipe (spatial or temporal)
    direction: PIPEDIRECTION | None = None

    # Template or tiling backing this pipe (implementation-specific)
    template: ScalableTemplate | None = None  # override type to allow None
    # Optional spatial edge spec for this pipe (alias handled in RHGBlock as edge_spec/edgespec)
    edgespec: SpatialEdgeSpec | None = None  # type: ignore[assignment]
