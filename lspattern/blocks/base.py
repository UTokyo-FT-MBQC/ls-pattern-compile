"""RHG blocks and skeletons for lattice-surgery templates."""

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
        PatchCoordGlobal3D,
        SpatialEdgeSpec,
        QubitIndexLocal,
    )


@dataclass
class RHGBlock:
    """
    Represents a block in the RHG lattice with input/output ports, trimmed edges, and an evaluated template.

    # The only difference from RHGBlockSleketon is that this class is
    # has: input/output ports
    # edges trimmed
    # evaluated template

    Attributes
    ----------
    index : int
        The index of the block.
    d : int
        The code distance or size parameter.
    edge_spec : SpatialEdgeSpec | None
        The spatial edge specification for the block.
    template : ScalableTemplate
        The evaluated template for the block.
    in_ports : QubitIndexLocal
        The input ports for the block's logical patch boundary.
    out_ports : QubitIndexLocal
        The output ports for the block's logical patch boundary.
    cout_ports : list[QubitIndexLocal]
        The classical output ports, grouped for logical results.

    Methods
    -------
    set_in_ports()
        Set the input ports for the block.
    set_out_ports()
        Set the output ports for the block.
    set_cout_ports()
        Set the classical output ports for the block.
    shift_ids(by: int)
        Shift the qubit indices in the template by a specified integer offset.
    shift_coords(by: PatchCoordGlobal3D)
        Shift the patch coordinates and update the template coordinates accordingly.
    pre_materialize()
        Materialize contents earlier than the standard materialization step (for visualization/debugging).
    """

    index: int = 0
    d: int = 3
    edge_spec: SpatialEdgeSpec | None = None

    template: ScalableTemplate = field(
        default_factory=lambda: RotatedPlanarTemplate(d=3, edgespec={})
    )  # evaluated

    # Ports for this block's current logical patch boundary (LOCAL node sets)
    in_ports: QubitIndexLocal = field(default_factory=set)
    out_ports: QubitIndexLocal = field(default_factory=set)
    # classical output ports. One group represents one logical result (to be XORed)
    cout_ports: list[QubitIndexLocal] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Sync template parameters (d, edgespec)
        edgespec = self.edge_spec
        if getattr(self, "template", None) is None:
            self.template = RotatedPlanarTemplate(
                d=int(self.d), edgespec=edgespec or {}
            )
        else:
            # Ensure d matches
            self.template.d = int(self.d)
            # Prefer explicit edge_spec if provided; otherwise adopt from template
            if edgespec is None:
                edgespec = getattr(self.template, "edgespec", {})
                self.edge_spec = edgespec  # keep alias in sync

        # Trim spatial boundaries for explicitly open sides and precompute tiling
        es = edgespec or {}
        for side in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
            if str(es.get(side, "")).upper() == "O":
                self.template.trim_spatial_boundary(side)

        self.template.to_tiling()

    # Child class will handle them without any input arguments
    def set_in_ports(self) -> None:
        """Set the input ports for the block."""

    def set_out_ports(self) -> None:
        """Set the output ports for the block."""

    def set_cout_ports(self) -> None:
        """Set the classical output ports for the block."""

    def shift_ids(self, by: int) -> None:
        """
        Shift the qubit indices in the template by a specified integer offset.

        Parameters
        ----------
        by : int
            The amount by which to shift all qubit indices.
        """
        self.template.shift_qindex(by)

    def shift_coords(self, by: PatchCoordGlobal3D) -> None:
        """
        Shift the patch coordinates and update the template coordinates accordingly.

        Parameters
        ----------
        by : PatchCoordGlobal3D
            The (dx, dy, dz) tuple by which to shift the patch coordinates.
        """
        if self.patch_coord is None:
            self.patch_coord = by
        else:
            ox, oy, oz = self.patch_coord
            dx, dy, dz = by
            self.patch_coord = (ox + dx, oy + dy, oz + dz)
        self.template.shift_coords((by[0], by[1]))

    def pre_materialize(self) -> None:
        """
        Materialize its contents earlier than the standard mateirlization step.

        (called within the TemporalLayer) only for visualization purpose

        #WARNING: Do not call this method outside of visualization/debugging context
        """
        from lspattern.mytype import NodeSetLocal, ScheduleTuplesLocal, FlowLocal

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
        """Get or set the spatial edge specification (alias for edge_spec).

        Returns
        -------
        SpatialEdgeSpec | None
            The spatial edge specification for this block.
        """
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
        """Convert this skeleton into a fully materialized RHGBlock instance.

        Returns
        -------
            RHGBlock: A fully materialized RHGBlock instance based on this skeleton.
        """
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if self.edgespec[direction] == "O":
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()  # populate coords
        # The rest of operations are handled in Child Classes
        return RHGBlock(d=self.d, edge_spec=self.edgespec, template=self.template)

    def trim_spatial_boundary(self, direction: str) -> None:
        """Trim the spatial boundaries of the tiling."""
        self.template.trim_spatial_boundary(direction)
