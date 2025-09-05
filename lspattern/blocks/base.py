from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from graphix_zx.graphstate import GraphState

from lspattern.tiling.template import (
    RotatedPlanarTemplate,
    ScalableTemplate,
)

if TYPE_CHECKING:
    from lspattern.mytype import (
        FlowLocal,
        NodeSetLocal,
        PatchCoordGlobal3D,
        QubitIndexLocal,
        ScheduleTuplesLocal,
        SpatialEdgeSpec,
    )


@dataclass
class RHGBlock:
    """RHG block with input/output ports and evaluated template."""

    # The only difference from RHGBlockSleketon is that this class is
    # has: input/output ports
    # edges trimmed
    # evaluated template
    index: int = 0
    d: int = 3
    edge_spec: SpatialEdgeSpec | None = None
    patch_coord: PatchCoordGlobal3D | None = (0, 0, 0)

    template: ScalableTemplate = field(default_factory=lambda: RotatedPlanarTemplate(d=3, edgespec={}))  # evaluated

    # Ports for this block's current logical patch boundary
    in_ports: QubitIndexLocal = field(default_factory=set)
    out_ports: QubitIndexLocal = field(default_factory=set)
    # classical output ports. One group represents one logical result (to be XORed)
    cout_ports: list[QubitIndexLocal] = field(default_factory=list)

    # Child class will handle them without any input arguments
    def set_in_ports(self) -> None:
        """Set input ports for the block."""

    def set_out_ports(self) -> None:
        """Set output ports for the block."""

    def set_cout_ports(self) -> None:
        """Set c-output ports for the block."""

    def shift_ids(self, by: int) -> None:
        """Shift all node IDs by the given offset."""
        self.template.shift_qindex(by)

    def shift_coords(self, by: PatchCoordGlobal3D) -> None:
        """Shift all coordinates by the given offset."""
        if self.patch_coord is None:
            self.patch_coord = by
        else:
            ox, oy, oz = self.patch_coord
            dx, dy, dz = by
            self.patch_coord = (ox + dx, oy + dy, oz + dz)
        self.template.shift_coords((by[0], by[1]))

    def pre_materialize(self) -> None:
        """
        Materialize its contents earlier than the standard materialization step.

        Called within the TemporalLayer only for visualization purpose.

        WARNING: Do not call this method outside of visualization/debugging context.
        """
        self.graph_local: GraphState = GraphState()
        self.schedule_local: ScheduleTuplesLocal = []
        self.flow_local: FlowLocal = {}
        self.x_checks: list[NodeSetLocal] = []
        self.z_checks: list[NodeSetLocal] = []

    # --- Compatibility aliases -------------------------------------------------
    # Some parts of the codebase use `edgespec` while this class had `edge_spec`.
    # Provide a property alias for smoother unification with pipes/templates.
    @property
    def edgespec(self) -> SpatialEdgeSpec | None:  # type: ignore[override]
        """Get edge specification."""
        return self.edge_spec

    @edgespec.setter
    def edgespec(self, v: SpatialEdgeSpec | None) -> None:
        self.edge_spec = v


@dataclass
class RHGBlockSkeleton:
    """A lightweight representation of a block before materialization."""

    d: int
    edgespec: SpatialEdgeSpec
    template: ScalableTemplate = field(init=False)

    def __post_init__(self) -> None:
        self.template = RotatedPlanarTemplate(d=self.d, edgespec=self.edgespec)

    def to_block(self) -> RHGBlock:
        """Convert skeleton to full block."""
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if self.edgespec[direction] == "O":
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()  # populate coords
        # The rest of operations are handled in Child Classes
        return RHGBlock(d=self.d, edge_spec=self.edgespec, template=self.template)

    def trim_spatial_boundary(self, direction: str) -> None:
        """Trim the spatial boundaries of the tiling."""
        self.template.trim_spatial_boundary(direction)
