from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Optional,
)

from graphix_zx.graphstate import GraphState
from lspattern.mytype import (
    FlowLocal,
    NodeSetLocal,
    PatchCoordGlobal3D,
    QubitIndexLocal,
    ScheduleTuplesLocal,
    SpatialEdgeSpec,
)
from lspattern.tiling.template import (
    RotatedPlanarTemplate,
    ScalableTemplate,
)


@dataclass
class RHGBlock:
    # The only difference from RHGBlockSleketon is that this class is
    # has: input/output ports
    # edges trimmed
    # evaluated template
    index: int = 0
    d: int = 3
    edge_spec: Optional[SpatialEdgeSpec] = None
    patch_coord: Optional[PatchCoordGlobal3D] = (0, 0, 0)

    template: ScalableTemplate = field(
        default_factory=lambda: RotatedPlanarTemplate(d=3, edgespec={})
    )  # evaluated

    # Ports for this block's current logical patch boundary
    in_ports: QubitIndexLocal = field(default_factory=set)
    out_ports: QubitIndexLocal = field(default_factory=set)
    # classical output ports. One group represents one logical result (to be XORed)
    cout_ports: list[QubitIndexLocal] = field(default_factory=list)

    # Child class will handle them without any input arguments
    def set_in_ports(self) -> None: ...
    def set_out_ports(self) -> None: ...
    def set_cout_ports(self) -> None: ...

    def shift_ids(self, by: int) -> None:
        self.template.shift_qindex(by)

    def shift_coords(self, by: PatchCoordGlobal3D) -> None:
        if self.patch_coord is None:
            self.patch_coord = by
        else:
            ox, oy, oz = self.patch_coord
            dx, dy, dz = by
            self.patch_coord = (ox + dx, oy + dy, oz + dz)
        self.template.shift_coords((by[0], by[1]))

    def pre_materialize(self) -> None:
        """
        Materialize its contents earlier than the standard mateirlization step
        (called within the TemporalLayer) only for visualization purpose

        #WARNING: Do not call this method outside of visualization/debugging context
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
    def edgespec(self) -> Optional[SpatialEdgeSpec]:  # type: ignore[override]
        return self.edge_spec

    @edgespec.setter
    def edgespec(self, v: Optional[SpatialEdgeSpec]) -> None:
        self.edge_spec = v


@dataclass
class RHGBlockSkeleton:
    """A lightweight representation of a block before materialization."""

    d: int
    edgespec: SpatialEdgeSpec
    template: ScalableTemplate = field(init=False)

    def __post_init__(self):
        self.template = RotatedPlanarTemplate(d=self.d, edgespec=self.edgespec)

    def to_block(self) -> RHGBlock:
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if self.edgespec[direction] == "O":
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()  # populate coords
        # The rest of operations are handled in Child Classes
        return RHGBlock(d=self.d, edge_spec=self.edgespec, template=self.template)

    def trim_spatial_boundary(self, direction: str) -> None:
        """Trim the spatial boundaries of the tiling."""
        self.template.trim_spatial_boundary(direction)
