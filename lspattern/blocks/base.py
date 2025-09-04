from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
)

from graphix_zx.graphstate import GraphState
from lspattern.mytype import *
from lspattern.template.base import RotatedPlanarTemplate, ScalableTemplate
from .skeleton import RHGBlockSkeleton





@dataclass
class RHGBlock:
    index: int = 0
    d: int = 3
    # The graph fragment contributed by the block (LOCAL ids).
    graph_local: GraphState = field(default_factory=GraphState)
    origin: Optional[tuple[int, int, int]] = (0, 0, 0)
    
    boundary_spec: dict[str, str] = field(default_factory=dict)
    template: Optional[ScalableTemplate] = field(
        default_factory=lambda: RotatedPlanarTemplate(d=3, edgespec={})
    )

    # 各境界の仕様（X/Z/O=Open/Trimmed）。未設定(None/欠損)は Open(O) とみなす。
    boundary_spec: Optional[BoundarySpec] = None

    # measurement schedule (int)--> set of measured local nodes
    schedule_local: ScheduleTuplesLocal = field(default_factory=list)
    # Flow (LOCAL ids): minimal X-flow mapping (node -> correction target nodes)
    flow_local: FlowLocal = field(default_factory=dict)

    # MBQC interface (local ids)
    # Ports for this block's current logical patch boundary
    in_ports: NodeSetLocal = field(default_factory=set)
    out_ports: NodeSetLocal = field(default_factory=set)
    # classical output ports. One group represents one logical result (to be XORed)
    cout_ports: list[set[NodeIdLocal]] = field(default_factory=list)

    # Geometry annotations (LOCAL node -> (x, y, z))
    node2coord: dict[NodeIdLocal, PhysCoordLocal3D] = field(default_factory=dict)
    coord2node: dict[PhysCoordLocal3D, NodeIdLocal] = field(default_factory=dict)
    node2role: dict[NodeIdLocal, str] = field(default_factory=dict)

    # Parity checks contributed entirely within the block (LOCAL ids)
    x_checks: list[set[NodeIdLocal]] = field(default_factory=list)
    z_checks: list[set[NodeIdLocal]] = field(default_factory=list)

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

    


# ---------------------------------------------------------------------
# Block delta (the unit of mutation produced by each block)
# ---------------------------------------------------------------------
@dataclass
class BlockDelta:
    """Delta produced by a block.

    Notes
    -----
    * All node ids in this object are LOCAL to `local_graph`.
      The canvas remaps them to GLOBAL ids when merging.
    * `in_ports`/`out_ports` use LOCAL ids. `out_qmap` provides LOCAL node -> q_index.
    * `schedule_tuples` is a list of (t_local, LOCAL-node-set). Each block starts at t_local=0.
    * `parity_*_prev_global_curr_local` are unified parity directives:
         (prev_global_center, [curr_local_nodes..]).
    """

    # The graph fragment contributed by the block (LOCAL ids).
    local_graph: GraphState

    # MBQC interface (LOCAL ids)
    in_ports: Dict[int, Set[int]] = field(
        default_factory=dict
    )  # logical -> set of input-side boundary nodes
    out_ports: Dict[int, Set[int]] = field(
        default_factory=dict
    )  # logical -> set of output-side boundary nodes
    out_qmap: Dict[int, Dict[int, int]] = field(
        default_factory=dict
    )  # logical -> {LOCAL node -> q_index}

    # Geometry annotations (LOCAL node -> (x, y, z))
    node_coords: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    # Parity checks contributed entirely within the block (LOCAL ids)
    x_checks: List[Set[int]] = field(default_factory=list)
    z_checks: List[Set[int]] = field(default_factory=list)

    # Local measurement schedule: list of (t_local, LOCAL node set)
    schedule_tuples: List[Tuple[int, Set[int]]] = field(default_factory=list)

    # Flow (LOCAL ids): minimal X-flow mapping (node -> correction target nodes)
    flow_local: Dict[int, Set[int]] = field(default_factory=dict)

    # Unified parity directives that pair previous GLOBAL centers with current LOCAL nodes
    parity_x_prev_global_curr_local: List[Tuple[int, List[int]]] = field(
        default_factory=list
    )
    parity_z_prev_global_curr_local: List[Tuple[int, List[int]]] = field(
        default_factory=list
    )

    # Last ancilla layers (LOCAL) keyed by (x, y) -> LOCAL node id, for seam stitching
    seam_last_x: Dict[Tuple[int, int], int] = field(default_factory=dict)
    seam_last_z: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def shift_ids(self, by: int) -> None:
        # change index of every element carrying LOCAL node ids by an offset
        if by == 0:
            return

        # in/out ports
        if self.in_ports:
            self.in_ports = {
                li: {int(n) + by for n in nodes} for li, nodes in self.in_ports.items()
            }
        if self.out_ports:
            self.out_ports = {
                li: {int(n) + by for n in nodes} for li, nodes in self.out_ports.items()
            }

        # out_qmap: logical -> {LOCAL node -> q_index}
        if self.out_qmap:
            self.out_qmap = {
                li: {int(n) + by: q for n, q in qmap.items()}
                for li, qmap in self.out_qmap.items()
            }

        # node_coords: LOCAL node -> coord
        if self.node_coords:
            self.node_coords = {
                int(n) + by: coord for n, coord in self.node_coords.items()
            }

        # parity checks
        if self.x_checks:
            self.x_checks = [{int(n) + by for n in group} for group in self.x_checks]
        if self.z_checks:
            self.z_checks = [{int(n) + by for n in group} for group in self.z_checks]

        # schedule
        if self.schedule_tuples:
            self.schedule_tuples = [
                (t, {int(n) + by for n in nodes}) for (t, nodes) in self.schedule_tuples
            ]

        # flow
        if self.flow_local:
            self.flow_local = {
                int(src) + by: {int(v) + by for v in tgts}
                for src, tgts in self.flow_local.items()
            }

        # parity caps (prev global center unchanged; current local nodes shifted)
        if self.parity_x_prev_global_curr_local:
            self.parity_x_prev_global_curr_local = [
                (center, [int(n) + by for n in locals_list])
                for center, locals_list in self.parity_x_prev_global_curr_local
            ]
        if self.parity_z_prev_global_curr_local:
            self.parity_z_prev_global_curr_local = [
                (center, [int(n) + by for n in locals_list])
                for center, locals_list in self.parity_z_prev_global_curr_local
            ]

        # seam last layers
        if self.seam_last_x:
            self.seam_last_x = {xy: int(n) + by for xy, n in self.seam_last_x.items()}
        if self.seam_last_z:
            self.seam_last_z = {xy: int(n) + by for xy, n in self.seam_last_z.items()}

    def shift_coords(self, patch_coord: Tuple[int, int]) -> None:
        # change the coordinates of every element by translating by patch_coord
        if not self.node_coords or patch_coord is None:
            return

        # Support both 2D (x, y) and 3D (x, y, z) inputs gracefully
        if len(patch_coord) == 2:
            ox, oy = patch_coord  # type: ignore[misc]
            oz = 0
        else:
            ox, oy, oz = patch_coord  # type: ignore[assignment]

        self.node_coords = {
            n: (coord[0] + ox, coord[1] + oy, coord[2] + oz)
            for n, coord in self.node_coords.items()
        }


# ---------------------------------------------------------------------
# Block protocol
# ---------------------------------------------------------------------
class RHGBlock2(Protocol):
    """Protocol for an RHG block (structural typing)."""

    logical: int
    d: int

    def emit(self, canvas: "RHGCanvas") -> BlockDelta: ...


# ---------------------------------------------------------------------
# Small utility
# ---------------------------------------------------------------------
def choose_port_node(nodes: Set[int]) -> int:
    """Pick a representative port node from a non-empty set (deterministic)."""
    if not nodes:
        raise ValueError("Port node set is empty.")
    return min(nodes)
