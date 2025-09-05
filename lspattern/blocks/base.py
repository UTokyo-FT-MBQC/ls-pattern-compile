"""RHG blocks and skeletons for lattice-surgery templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from lspattern.tiling.template import (
    RotatedPlanarBlockTemplate,
    ScalableTemplate,
)

if TYPE_CHECKING:
    from lspattern.mytype import (
        PatchCoordLocal2D,
        PatchCoordGlobal3D,
        QubitIndexLocal,
        SpatialEdgeSpec,
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
    """

    name: ClassVar[str] = __qualname__
    d: int = 3
    edge_spec: SpatialEdgeSpec = field(default_factory=dict)
    # source
    source: PatchCoordGlobal3D = field(default_factory=lambda: (0, 0, 0))
    # When it is Pipe, we have sink and direction (Not implemented here)
    template: ScalableTemplate = field(default_factory=lambda: ScalableTemplate(d=3, edgespec={}))  # evaluated

    # Ports for this block's current logical patch boundary (qubit index sets)
    # classical output ports. One group represents one logical result (to be XORed)
    in_ports: set[QubitIndexLocal] = field(default_factory=set)
    out_ports: set[QubitIndexLocal] = field(default_factory=set)
    cout_ports: list[set[QubitIndexLocal]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Sync template parameters (d, edgespec)
        edgespec = self.edge_spec
        if getattr(self, "template", None) is None:
            self.template = RotatedPlanarBlockTemplate(d=int(self.d), edgespec=edgespec or {})
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
        """Shift the patch anchor and the template by a 2D offset.

        The block's anchor (`source`) is 3D; the underlying scalable template
        only needs the XY offset. This mirrors the behavior used by pipes.
        """
        osx, osy, osz = self.source
        dx, dy, dz = by
        self.source = (osx + dx, osy + dy, osz + dz)

        by_template: PatchCoordLocal2D = (dx, dy)
        self.template.shift_coords(by_template)

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

    name: ClassVar[str] = __qualname__
    d: int
    edgespec: SpatialEdgeSpec
    template: ScalableTemplate = field(init=False)

    def __post_init__(self) -> None:
        self.template = ScalableTemplate(d=self.d, edgespec=self.edgespec)

    def to_block(self) -> RHGBlock:
        """Convert this skeleton into a fully materialized RHGBlock instance."""
        msg = "to_block() must be implemented in subclasses."
        raise NotImplementedError(msg)

    def trim_spatial_boundary(self, direction: str) -> None:
        """Trim the spatial boundaries of the tiling."""
        self.template.trim_spatial_boundary(direction)
