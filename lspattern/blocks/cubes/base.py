"""Base classes for cube-shaped RHG blocks (thin wrappers)."""

from __future__ import annotations

from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton
from lspattern.tiling.template import RotatedPlanarCubeTemplate


# Almost the same as original RHGBlock, but with cube-shaped semantics
class RHGCube(RHGBlock):
    """Alias of the core RHGCube for cube-shaped semantics."""


class RHGCubeSkeleton(RHGBlockSkeleton):
    """Skeleton for a cube-shaped RHG block using RotatedPlanarBlockTemplate."""

    def __post_init__(self) -> None:
        self.template = RotatedPlanarCubeTemplate(d=self.d, edgespec=self.edgespec)
