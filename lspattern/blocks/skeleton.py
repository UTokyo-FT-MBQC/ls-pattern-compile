from __future__ import annotations

from dataclasses import dataclass, field

from lspattern.mytype import (
    SpatialEdgeSpec,
)
from lspattern.tiling.template import RotatedPlanarTemplate, ScalableTemplate


@dataclass
class RHGBlockSkeleton:
    """A lightweight representation of a block before materialization."""

    d: int
    edgespec: SpatialEdgeSpec
    tiling: ScalableTemplate = field(init=False)

    def __post_init__(self):
        self.tiling = RotatedPlanarTemplate(d=self.d, edgespec=self.edgespec)

    def to_canvas(self) -> "RHGBlock":
        """Materialize the block and return a RHGBlock."""
        raise NotImplementedError

    def trim_spatial_boundaries(self, direction: str) -> None:
        """Trim the spatial boundaries of the tiling."""
        self.tiling.trim_spatial_boundary(direction)
