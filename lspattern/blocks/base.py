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


@dataclass
class RHGBlockSkeleton:
    # サイズとkind(色)情報だけ持っている
    d: int
    kind: BlockKindstr


@dataclass
class RHGBlock:
    index: int = 0
    d: int = 3
    # The graph fragment contributed by the block (LOCAL ids).
    graph_local: GraphState = field(default_factory=GraphState)
    origin: Optional[tuple[int, int, int]] = (0, 0, 0)
    kind: BlockKindstr = field(default_factory=lambda: ("X", "X", "Z"))
    boundary_spec: dict[str, str] = field(default_factory=dict)
    template: Optional[ScalableTemplate] = field(
        default_factory=lambda: RotatedPlanarTemplate(d=3, kind=("X", "X", "Z"))
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

    def materialize(self, skeleton: RHGBlockSkeleton) -> None:
        pass

    def get_boundary_spec(self, side: BoundarySide) -> EdgeSpec:
        """指定 side の境界仕様を返す（未設定は Open/Trimmed とみなす）。"""
        return self.boundary_spec[side]

    def get_boundary_nodes(
        self, face: str, depth: list[int] = [0]
    ) -> dict[str, list[NodeIdLocal]]:
        """Get all the nodes on a given face and at a specific depth.

        The function selects nodes from a LocalGraph based on their coordinates,
        targeting a specific face of the lattice boundary and a depth relative
        to it. The lattice is assumed to be a 3D RHG lattice built from a 2D
        tiling with r-rounds.

        The depth is measured outwards from the boundary surface. For example:
        - A depth of 0 indicates nodes are on the boundary.
        - A depth of -1 indicates nodes are one step inside the bulk from the surface.
        - A depth of +1 indicates nodes are one step outside the surface (valid only then Z+, to get the output ports).

        An error will be raised if the depth is larger than the size 'd'.

        Args:
            face (str): The face to select nodes from. Must be one of
                        "X+", "X-", "Y+", "Y-", "Z+", or "Z-".
            depth (int): The inward depth from the specified face.

        Returns:
            dict: A dictionary with three keys:
                  - "data": A list of data qubit node indices.
                  - "x_check": A list of X-check qubit node indices.
                  - "z_check": A list of Z-check qubit node indices.
        """
        assert face in ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
        for dv in depth:
            if dv >= 0:
                raise ValueError(
                    "depth values must be negative (e.g., -1 for boundary)"
                )
            # steps inside from boundary = (-dv - 1)
            if (-dv - 1) >= int(self.d):
                raise ValueError("depth larger than block size d")

        # Precompute extrema
        if not self.node2coord:
            raise ValueError("Block not materialized: node2coord is empty")

        xs, ys, zs = zip(*self.node2coord.values())
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)  # note output nodes have extra 1z layer

        if face[0] == "X":
            axis = 0
            if face == "X+":
                targets = {xmax - s for s in depth}
            elif face == "X-":
                targets = {xmin + s for s in depth}
        elif face[0] == "Y":
            if face == "Y+":
                targets = {ymax - s for s in depth}
            elif face == "Y-":
                targets = {ymin + s for s in depth}
            axis = 1
        elif face[0] == "Z":
            if face == "Z+":
                # Need modification if graph contains output layer
                has_output = bool(len(self.graph_local.output_nodes) > 0)
                if has_output:
                    targets = {zmax - 1 - s for s in depth}
                else:
                    targets = {zmax - s for s in depth}
            elif face == "Z-":
                targets = {zmin + s for s in depth}
            axis = 2

        # Collect nodes by role on the selected face layers
        # data_nodes should be a mapping from local coordinates to node id
        data_nodes: dict[PhysCoordLocal3D, NodeIdLocal] = {}
        zcheck_nodes: dict[PhysCoordLocal3D, NodeIdLocal] = {}
        xcheck_nodes: dict[PhysCoordLocal3D, NodeIdLocal] = {}

        # Single pass over coords for speed
        for n, coord in self.node2coord.items():
            if coord[axis] not in targets:
                continue

            role = self.node2coord[n]
            if role == "data":
                data_nodes[coord] = n
            if role == "ancilla_x":
                xcheck_nodes[coord] = n
            elif role == "ancilla_z":
                zcheck_nodes[coord] = n
            else:
                raise ValueError(f"Unknown node role: {role}")

        return {"data": data_nodes, "x_check": xcheck_nodes, "z_check": zcheck_nodes}


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
