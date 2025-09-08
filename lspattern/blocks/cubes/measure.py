from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.common import Plane, PlannerMeasBasis

from lspattern.blocks.base import RHGBlock

if TYPE_CHECKING:
    from lspattern.canvas import RHGCanvas


class _MeasureBase(RHGBlock):
    """MBQC measurement block on the latest DATA layer (RHG parity-aware).

    Behavior
    --------
    - Determine the latest DATA layer footprint from the canvas logical boundary.
    - Create readout nodes only on DATA sites at that z-layer.
    - Register each readout node as both MBQC input and output (same q_index).
    - `out_ports` is empty: this block consumes the logical boundary.
    - Provide X-cap parity directives that close the top with the previous X layer.
    """

    def __init__(self, logical: int, basis: str) -> None:
        self.logical = logical
        self.basis = PlannerMeasBasis(Plane.XY, 0.0) if basis == "X" else PlannerMeasBasis(Plane.ZX, 0.0)

    def emit(self, canvas: RHGCanvas):
        # This detailed implementation is out of scope for this milestone.
        # Kept as a placeholder to satisfy imports without runtime use.
        msg = "Measure blocks are not implemented in this build"
        raise NotImplementedError(msg)


class MeasureX(_MeasureBase):
    """Measure a logical block in the X basis."""

    def __init__(self, logical: int) -> None:
        super().__init__(logical, "X")
