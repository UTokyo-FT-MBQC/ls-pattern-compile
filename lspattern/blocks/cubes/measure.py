from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from graphix_zx.common import Axis, AxisMeasBasis, Sign

from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton
from lspattern.mytype import PhysCoordGlobal3D, PhysCoordLocal2D

if TYPE_CHECKING:
    from lspattern.canvas import RHGCanvas

ANCILLA_TARGET_DIRECTION2D = {(1, 1), (1, -1), (-1, 1), (-1, -1)}


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

    def __init__(self, logical: int, basis: Axis) -> None:
        self.logical = logical
        self.meas_basis = AxisMeasBasis(basis, Sign.PLUS)  # is it actually override the base class's meas_basis?

    def emit(self, canvas: RHGCanvas) -> None:
        # This detailed implementation is out of scope for this milestone.
        # Kept as a placeholder to satisfy imports without runtime use.
        msg = "Measure blocks are not implemented in this build"
        raise NotImplementedError(msg)

    def set_in_ports(self) -> None:
        idx_map = self.template.get_data_indices()
        self.in_ports = set(idx_map.values())

    def set_out_ports(self) -> None:
        return super().set_out_ports()


class MeasureX(_MeasureBase):
    """Measure a logical block in the X basis."""

    def __init__(self, logical: int) -> None:
        super().__init__(logical, Axis.X)

    def set_cout_ports(self) -> None:
        pass

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords

        t = min(self.schedule.schedule.keys(), default=0)

        for x, y in x2d:
            node_group = {}
            for dx, dy in ANCILLA_TARGET_DIRECTION2D:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, t)))
                if node_id is not None:
                    node_group.add(node_id)
            self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), []).append(node_group)


class MeasureZ(_MeasureBase):
    """Measure a logical block in the Z basis."""

    def __init__(self, logical: int) -> None:
        super().__init__(logical, Axis.Z)

    def set_cout_ports(self) -> None:
        pass

    def _construct_detectors(self) -> None:
        z2d = self.template.z_coords

        t = min(self.schedule.schedule.keys(), default=0)

        for x, y in z2d:
            node_group = {}
            for dx, dy in ANCILLA_TARGET_DIRECTION2D:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, t)))
                if node_id is not None:
                    node_group.add(node_id)
            self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), []).append(node_group)


class MeasureXSkelton(RHGBlockSkeleton):
    """Skeleton for X-basis measurement blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "MeasureXSkelton"

    def to_block(self) -> MeasureX:
        """Materialize to a MeasureX (template evaluated, no local graph yet)."""
        # Apply spatial open-boundary trimming if specified
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if str(self.edgespec.get(direction, "O")).upper() == "O":
                self.trim_spatial_boundary(direction)
        # Evaluate template coordinates
        self.template.to_tiling()

        block = MeasureX(
            logical=self.d,
        )
        block.final_layer = "MX"
        return block


class MeasureZSkelton(RHGBlockSkeleton):
    """Skeleton for Z-basis measurement blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "MeasureZSkelton"

    def to_block(self) -> MeasureZ:
        """Materialize to a MeasureZ (template evaluated, no local graph yet)."""
        # Apply spatial open-boundary trimming if specified
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if str(self.edgespec.get(direction, "O")).upper() == "O":
                self.trim_spatial_boundary(direction)
        # Evaluate template coordinates
        self.template.to_tiling()

        block = MeasureZ(
            logical=self.d,
        )
        block.final_layer = "MZ"
        return block
