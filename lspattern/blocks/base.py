from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Optional,
)

from graphix_zx.graphstate import GraphState
from lspattern.mytype import (
    FlowLocal,
    NodeIdLocal,
    NodeSetLocal,
    PatchCoordGlobal3D,
    PhysCoordLocal3D,
    ScheduleTuplesLocal,
    SpatialEdgeSpec,
)
from lspattern.tiling.template import RotatedPlanarTemplate, ScalableTemplate


@dataclass
class RHGBlock:
    index: int = 0
    d: int = 3
    # The graph fragment contributed by the block (LOCAL ids).
    graph_local: GraphState = field(default_factory=GraphState)
    origin: Optional[tuple[int, int, int]] = (0, 0, 0)

    template: Optional[ScalableTemplate] = field(
        default_factory=lambda: RotatedPlanarTemplate(d=3, edgespec={})
    )

    edge_spec: Optional[SpatialEdgeSpec] = None

    # measurement schedule (int)--> set of measured local nodes
    schedule_local: ScheduleTuplesLocal = field(default_factory=list)
    # Flow (LOCAL ids): minimal X-flow mapping (node -> correction target nodes)
    flow_local: FlowLocal = field(default_factory=dict)

    # MBQC interface (local ids)
    # Ports for this block's current logical patch boundary
    in_ports: NodeSetLocal = field(default_factory=set)
    out_ports: NodeSetLocal = field(default_factory=set)
    # classical output ports. One group represents one logical result (to be XORed)
    cout_ports: list[NodeSetLocal] = field(default_factory=list)

    # Geometry annotations (LOCAL node -> (x, y, z))
    node2coord: dict[NodeIdLocal, PhysCoordLocal3D] = field(default_factory=dict)
    coord2node: dict[PhysCoordLocal3D, NodeIdLocal] = field(default_factory=dict)
    node2role: dict[NodeIdLocal, str] = field(default_factory=dict)

    # Parity checks contributed entirely within the block (LOCAL ids)
    x_checks: list[NodeSetLocal] = field(default_factory=list)
    z_checks: list[NodeSetLocal] = field(default_factory=list)

    def shift_ids(self, by: int) -> None:
        # increase/decrease every nodes denoted by NodeIdLocal
        if by == 0:
            return

        # schedule_local: list[(t_local, {nodes..})]
        if self.schedule_local:
            self.schedule_local = [
                (t, {int(n) + by for n in nodes}) for (t, nodes) in self.schedule_local
            ]

        # flow_local: {node -> {targets..}}
        if self.flow_local:
            self.flow_local = {
                int(src) + by: {int(v) + by for v in tgts}
                for src, tgts in self.flow_local.items()
            }

        # ports and cout ports
        if self.in_ports:
            self.in_ports = {int(n) + by for n in self.in_ports}
        if self.out_ports:
            self.out_ports = {int(n) + by for n in self.out_ports}
        if self.cout_ports:
            self.cout_ports = [
                {int(n) + by for n in group} for group in self.cout_ports
            ]

        # coords maps
        if self.node2coord:
            self.node2coord = {
                int(n) + by: coord for n, coord in self.node2coord.items()
            }
        if self.coord2node:
            self.coord2node = {
                coord: int(n) + by for coord, n in self.coord2node.items()
            }

        # parity checks
        if self.x_checks:
            self.x_checks = [{int(n) + by for n in group} for group in self.x_checks]
        if self.z_checks:
            self.z_checks = [{int(n) + by for n in group} for group in self.z_checks]

    def shift_coords(self, by: PatchCoordGlobal3D) -> None:
        # move all the coordinates PhysCoordLocal3D by `by`
        if by is None:
            return
        ox, oy, oz = by

        if self.node2coord:
            patchsize = self.d + 1  # include output layer if any
            self.node2coord = {
                n: (
                    coord[0] + ox * patchsize,
                    coord[1] + oy * patchsize,
                    coord[2] + oz * patchsize,
                )
                for n, coord in self.node2coord.items()
            }

        if self.coord2node:
            # rebuild inverse map from updated node2coord for consistency
            self.coord2node = {coord: n for n, coord in self.node2coord.items()}

        self.origin = by

    # New API compatibility: blocks produced by skeletons are already
    # materialized. Provide a no-op materialize() so Canvas can call it safely.
    def materialize(self) -> None:  # noqa: D401
        """No-op for pre-materialized blocks (skeleton.to_canvas())."""
        return


@dataclass
class RHGBlockSkeleton:
    """A lightweight representation of a block before materialization."""

    d: int
    edgespec: SpatialEdgeSpec
    tiling: ScalableTemplate = field(init=False)

    def __post_init__(self):
        self.tiling = RotatedPlanarTemplate(d=self.d, edgespec=self.edgespec)

    def to_block(self) -> RHGBlock:
        raise NotImplementedError

    def trim_spatial_boundaries(self, direction: str) -> None:
        """Trim the spatial boundaries of the tiling."""
        self.tiling.trim_spatial_boundary(direction)
