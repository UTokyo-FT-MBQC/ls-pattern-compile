from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton
from lspattern.consts import CoordinateSystem
from lspattern.mytype import (
    NodeIdLocal,
    PatchCoordGlobal3D,
    PatchCoordLocal2D,
    PhysCoordGlobal3D,
    PhysCoordLocal2D,
    QubitIndexLocal,
    SpatialEdgeSpec,
)
from lspattern.tiling.template import (
    ScalableTemplate,
)
from lspattern.utils import get_direction

if TYPE_CHECKING:
    from lspattern.consts.consts import PIPEDIRECTION


@dataclass
class RHGPipeSkeleton(RHGBlockSkeleton):
    edge_spec: SpatialEdgeSpec = field(default_factory=dict)


@dataclass
class RHGPipe(RHGBlock):
    """
    Represents a pipe in the RHG block structure.

    Attributes
    ----------
    source : PatchCoordGlobal3D | None
        The source coordinate of the pipe.
    sink : PatchCoordGlobal3D | None
        The sink coordinate of the pipe.
    direction : PIPEDIRECTION | None
        Direction of the pipe (spatial or temporal).
    template : ScalableTemplate | None
        Template or tiling backing this pipe (implementation-specific).
    edgespec : SpatialEdgeSpec | None
        Optional spatial edge spec for this pipe.
    """

    edge_spec: SpatialEdgeSpec | None = field(default_factory=dict)

    source: PatchCoordGlobal3D = field(default_factory=lambda: PatchCoordGlobal3D((0, 0, 0)))
    sink: PatchCoordGlobal3D | None = field(default_factory=lambda: PatchCoordGlobal3D((0, 0, 1)))
    direction: PIPEDIRECTION = field(init=False)

    template: ScalableTemplate = field(default_factory=lambda: ScalableTemplate(d=3, edgespec={}))

    in_ports: set[QubitIndexLocal] = field(default_factory=set)
    out_ports: set[QubitIndexLocal] = field(default_factory=set)
    cout_ports: list[set[NodeIdLocal]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # get direction
        if self.sink is None:
            self.sink = PatchCoordGlobal3D((0, 0, 1))
        self.direction = get_direction(self.source, self.sink)

    def shift_coords(self, by: PatchCoordGlobal3D) -> None:
        """Shift the patch coordinates and update the template coordinates accordingly.

        For pipes, prefer a patch3d-aware shift that uses the current
        `direction` to determine how XY offsets are derived for the tiling.
        Falls back to a plain tiling2d shift if the template doesn't support
        the pipe-specific signature.
        """
        osx, osy, osz = self.source
        if self.sink is None:
            self.sink = PatchCoordGlobal3D((0, 0, 1))
        osx2, osy2, osz2 = self.sink
        dx, dy, dz = by
        self.source = PatchCoordGlobal3D((osx + dx, osy + dy, osz + dz))
        self.sink = PatchCoordGlobal3D((osx2 + dx, osy2 + dy, osz2 + dz))

        # Try pipe-specific patch3d rule (RotatedPlanarPipetemplate supports this)
        try:
            self.template.shift_coords(by, coordinate=CoordinateSystem.PATCH_3D, direction=self.direction)  # type: ignore[call-arg]
        except TypeError:
            # Fallback: treat as a raw 2D tiling shift
            by_template: PatchCoordLocal2D = PatchCoordLocal2D((by[0], by[1]))
            self.template.shift_coords(by_template)

    def _construct_detectors(self) -> None:
        """Build X/Z parity detectors, deferring the first X seam ancilla pairing."""
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        zmin = min({coord[2] for coord in self.coord2node}, default=0)
        zmax = max({coord[2] for coord in self.coord2node}, default=0)
        height = zmax - zmin + 1

        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        for dz in range(height):
            for x, y in x2d + z2d:
                coord = PhysCoordLocal2D((x, y))
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, zmin + dz)))
                if not node_id:
                    continue
                if dz == 0:
                    # ancillas of first layer is not deterministic
                    dangling_detectors[coord] = {node_id}
                    self.parity.ignore_dangling[coord] = True
                else:
                    node_group = {node_id} | dangling_detectors.pop(coord, set())
                    self.parity.checks.setdefault(coord, {})[zmin + dz] = node_group
                    dangling_detectors[coord] = {node_id}

        # add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes
