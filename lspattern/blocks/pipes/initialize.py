from __future__ import annotations

from dataclasses import dataclass

from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import PatchCoordGlobal3D, SpatialEdgeSpec
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction


@dataclass
class InitPlusPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for an InitPlus-style pipe.

    Behavior
    - If ``edgespec`` is ``None``, downstream components use direction-specific defaults:
      - Horizontal (RIGHT/LEFT): {TOP: 'O', BOTTOM: 'O', LEFT: 'X', RIGHT: 'Z'}
      - Vertical   (TOP/BOTTOM): {LEFT: 'O', RIGHT: 'O', TOP: 'X', BOTTOM: 'Z'}
    - Direction is inferred from ``source`` and ``sink`` in ``to_block`` via
      ``get_direction``.
    """

    edgespec: SpatialEdgeSpec | None = None

    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> "InitPlusPipe":
        direction = get_direction(source, sink)
        spec = self.edgespec

        block = InitPlusPipe(
            d=self.d,
            edgespec=spec,
            direction=direction,
        )
        return block


class InitPlusPipe(RHGPipe):
    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec,
        direction: PIPEDIRECTION,
    ):
        # RHGPipe(dataclass) の __init__ は direction を引数に受け取らない
        super().__init__(d=d, edge_spec=edgespec)
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edgespec)
