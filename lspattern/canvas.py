"""Canvas module for RHG (Random Hamiltonian Graph) compilation.

This module provides the main compilation framework for converting RHG blocks
into executable quantum patterns with proper temporal layering and flow management.
"""

from __future__ import annotations

from contextlib import suppress

# import layout is intentionally non-standard due to optional deps fallback
from dataclasses import dataclass, field
from operator import itemgetter
from typing import TYPE_CHECKING

from graphix_zx.graphstate import (
    GraphState,
    compose_in_parallel,
    compose_sequentially,
)

from lspattern.accumulator import FlowAccumulator, ParityAccumulator, ScheduleAccumulator
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.mytype import (
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PipeCoordGlobal3D,
    QubitGroupIdGlobal,
)
from lspattern.tiling.template import cube_offset_xy, pipe_offset_xy
from lspattern.utils import UnionFind, get_direction, is_allowed_pair

if TYPE_CHECKING:
    from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
    from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton


class MixedCodeDistanceError(Exception):
    """Raised when mixed code distances are detected in TemporalLayer.materialize."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TemporalLayer:
    """
    Represents a single temporal layer in the RHG canvas.

    Contains cubes, pipes, and associated graph state
    for a given time step (z). Handles the construction and mapping of physical nodes, ports, and accumulators.
    """

    z: int
    qubit_count: int
    patches: list[PatchCoordGlobal3D]
    lines: list[PipeCoordGlobal3D]

    in_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]
    out_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]
    cout_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]

    in_ports: list[NodeIdLocal]
    out_ports: list[NodeIdLocal]
    cout_ports: list[NodeIdLocal]

    schedule: ScheduleAccumulator
    flow: FlowAccumulator
    parity: ParityAccumulator

    local_graph: GraphState
    node2coord: dict[NodeIdLocal, PhysCoordGlobal3D]
    coord2node: dict[PhysCoordGlobal3D, NodeIdLocal]
    node2role: dict[NodeIdLocal, str]

    cubes_: dict[PatchCoordGlobal3D, RHGCube]
    pipes_: dict[PipeCoordGlobal3D, RHGPipe]

    coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal]
    allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]]

    def __init__(self, z: int) -> None:
        self.z = z
        self.qubit_count = 0
        self.d: set[int] = set()
        self.patches = []
        self.lines = []
        self.in_portset = {}
        self.out_portset = {}
        self.cout_portset = {}
        self.in_ports = []
        self.out_ports = []
        self.cout_ports = []
        self.local_graph = None
        self.node2coord = {}
        self.coord2node = {}
        self.node2role = {}
        # accumulators
        self.schedule = ScheduleAccumulator()
        self.flow = FlowAccumulator()
        self.parity = ParityAccumulator()

        self.cubes_ = {}
        self.pipes_ = {}
        self.tiling_node_maps: dict[str, dict[int, int]] = {}

        self.coord2tid: dict[PatchCoordGlobal3D, str] = {}
        self.coord2gid = {}
        self.allowed_gid_pairs = set()

    def __post_init__(self) -> None:
        """Post-initialization hook."""

    def add_cubes(self, cubes: dict[PatchCoordGlobal3D, RHGCube]) -> None:
        """Add multiple cubes to this temporal layer."""
        for pos, cube in cubes.items():
            self.add_cube(pos, cube)

    def add_pipes(self, pipes: dict[PipeCoordGlobal3D, RHGPipe]) -> None:
        """Add multiple pipes to this temporal layer."""
        for pipe_coord, pipe in pipes.items():
            source, sink = pipe_coord
            self.add_pipe(source, sink, pipe)

    def add_cube(self, pos: PatchCoordGlobal3D, cube: RHGCube) -> None:
        """Add a materialized cube to this temporal layer and place it at `pos`."""
        self.cubes_[pos] = cube
        self.patches.append(pos)

    def add_pipe(
        self,
        source: PatchCoordGlobal3D,
        sink: PatchCoordGlobal3D,
        spatial_pipe: RHGPipe,
    ) -> None:
        """Register a spatial pipe within this layer between `source` and `sink`."""
        pipe_coord: PipeCoordGlobal3D = (source, sink)  # type: ignore[assignment]
        self.pipes_[pipe_coord] = spatial_pipe
        self.lines.append(pipe_coord)

    def _update_qubit_group_index(self) -> None:
        """Update qubit group index mapping.

        Internal function called inside compile() to update qubit group index mapping.
        """
        # scan through the pipes and do union-find and update coord2gid
        # return is None

    def compile(self) -> None:
        """Compile the temporal layer into a quantum pattern.

        Aggregates coordinates and patch groups, processes cubes and pipes,
        and builds the quantum graph state with proper port mappings.
        """
        # Aggregate absolute 2D coords and patch groups (reserved for future use)
        # XY(2D) -> gid mapping used for same-z gating
        coord_gid_2d: dict[tuple[int, int], QubitGroupIdGlobal] = {}
        allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]] = set()

        # Union-Find (DSU) over tiling ids to compute connected groups
        uf = UnionFind()

        # 1) Initialize
        for c in self.cubes_.values():
            uf.add(QubitGroupIdGlobal(c.get_tiling_id()))
        for pipe in self.pipes_.values():
            uf.add(QubitGroupIdGlobal(pipe.get_tiling_id()))
        # 2) Union-Find: Unify cube<->pipe<->cube per spatial pipe, and record allowed cube pairs
        for pipe_coord in self.pipes_:
            source, sink = pipe_coord
            uf.union(
                QubitGroupIdGlobal(self.cubes_[source].get_tiling_id()),
                QubitGroupIdGlobal(self.pipes_[pipe_coord].get_tiling_id()),
            )
            uf.union(
                QubitGroupIdGlobal(self.pipes_[pipe_coord].get_tiling_id()),
                QubitGroupIdGlobal(self.cubes_[sink].get_tiling_id()),
            )
        # 3) set_tiling_id
        coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal] = {}

        for pos, cube in self.cubes_.items():
            cube.set_tiling_id(uf.find(cube.get_tiling_id()))
            self.cubes_[pos] = cube
            coord2gid.update(cube.coord2gid)
        for pos, pipe in self.pipes_.items():
            pipe.set_tiling_id(uf.find(pipe.get_tiling_id()))
            self.pipes_[pos] = pipe
            coord2gid.update(pipe.coord2gid)

        allowed_gid_pairs.update(
            (
                QubitGroupIdGlobal(self.cubes_[source].get_tiling_id()),
                QubitGroupIdGlobal(self.cubes_[sink].get_tiling_id()),
            )
            for source, sink in self.pipes_
        )

        self.coord2gid = coord2gid
        self.allowed_gid_pairs = allowed_gid_pairs

        # Build GraphState by composing pre-materialized block graphs in parallel (T33)
        # Acceptance: do not create new nodes here; use compose_in_parallel per block/pipe.
        # Now the new cz spanning algorithm has disappeared.
        # Helper to remap existing registries by a node map
        def _remap_current_regs(node_map: dict[int, int]) -> None:
            if not node_map:
                return
            self.node2coord = {NodeIdLocal(node_map.get(n, n)): c for n, c in self.node2coord.items()}
            self.coord2node = {c: NodeIdLocal(node_map.get(n, n)) for c, n in self.coord2node.items()}
            self.node2role = {NodeIdLocal(node_map.get(n, n)): r for n, r in self.node2role.items()}
            # Portsets and flat lists
            for p, nodes in list(self.in_portset.items()):
                self.in_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
            for p, nodes in list(self.out_portset.items()):
                self.out_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
            for p, nodes in list(self.cout_portset.items()):
                self.cout_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
            self.in_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.in_ports]
            self.out_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.out_ports]

        g: GraphState | None = None
        self.node2coord = {}
        self.coord2node = {}
        self.node2role = {}

        # Compose cube graphs
        for pos, blk in self.cubes_.items():
            d_val = int(blk.d)
            z_base = int(pos[2]) * (2 * d_val)

            g2 = blk.local_graph

            if g is None:
                g = g2
                node_map1: dict[int, int] = {}
                node_map2: dict[int, int] = {n: n for n in getattr(g2, "physical_nodes", [])}
            else:
                g_new, node_map1, node_map2 = compose_in_parallel(g, g2)
                _remap_current_regs(node_map1)
                g = g_new

            # Compute z-shift from block-local coords to target z_base
            try:
                bmin_z = min(c[2] for c in blk.node2coord.values())
            except ValueError:
                bmin_z = z_base
            z_shift = int(z_base - bmin_z)

            # Ingest coords/roles via node_map2; XY already absolute (shifted before materialize)
            for old_n, coord in blk.node2coord.items():
                new_n = node_map2.get(old_n)
                if new_n is None:
                    continue
                x, y, z = (
                    int(coord[0]),
                    int(coord[1]),
                    int(coord[2]) + z_shift,
                )
                c_new = PhysCoordGlobal3D((x, y, z))
                self.node2coord[NodeIdLocal(new_n)] = c_new
                self.coord2node[c_new] = NodeIdLocal(new_n)
            for old_n, role in blk.node2role.items():
                new_n = node_map2.get(old_n)
                if new_n is not None:
                    self.node2role[NodeIdLocal(new_n)] = role

            # Ports (if any) per position
            if getattr(blk, "in_ports", None):
                self.in_portset[pos] = [NodeIdLocal(node_map2[n]) for n in blk.in_ports if n in node_map2]
                self.in_ports.extend(self.in_portset[pos])
            if getattr(blk, "out_ports", None):
                self.out_portset[pos] = [NodeIdLocal(node_map2[n]) for n in blk.out_ports if n in node_map2]
                self.out_ports.extend(self.out_portset[pos])
            if getattr(blk, "cout_ports", None):
                self.cout_portset[pos] = [
                    NodeIdLocal(node_map2[n]) for s in blk.cout_ports for n in s if n in node_map2
                ]

        # Compose pipe graphs (spatial pipes in this layer)
        for pipe_coord, pipe in self.pipes_.items():
            source, _sink = pipe_coord
            d_val = int(pipe.d)
            z_base = int(source[2]) * (2 * d_val)

            g2 = pipe.local_graph
            if g2 is None:
                pipe_mat = pipe.materialize()
                g2 = pipe_mat.local_graph

            if g is None:
                g = g2
                node_map1 = {}
                node_map2 = {n: n for n in getattr(g2, "physical_nodes", [])}
            else:
                g_new, node_map1, node_map2 = compose_in_parallel(g, g2)
                _remap_current_regs(node_map1)
                g = g_new

            try:
                bmin_z = min(c[2] for c in pipe.node2coord.values())
            except ValueError:
                bmin_z = z_base
            z_shift = int(z_base - bmin_z)

            for old_n, coord in pipe.node2coord.items():
                new_n = node_map2.get(old_n)
                if new_n is None:
                    continue
                # XY はテンプレートで既に絶対座標化済み(to_temporal_layer で shift 済み)
                x, y, z = (
                    int(coord[0]),
                    int(coord[1]),
                    int(coord[2]) + z_shift,
                )
                c_new = PhysCoordGlobal3D((x, y, z))
                self.node2coord[NodeIdLocal(new_n)] = c_new
                self.coord2node[c_new] = NodeIdLocal(new_n)
            for old_n, role in pipe.node2role.items():
                new_n = node_map2.get(old_n)
                if new_n is not None:
                    self.node2role[NodeIdLocal(new_n)] = role

        # T37: Add CZ edges across cube↔pipe seams within the same temporal layer (same z)
        # Only connect (x,y,z)↔(x+dx,y+dy,z) for diagonal DIRECTIONS3D if
        # - one XY belongs to a cube region and the other to a pipe region, and
        # - their XY-group ids (gid) form an allowed pair in allowed_gid_pairs.
        if g is None:
            g = GraphState()
        # Build absolute XY region sets for cubes and pipes
        cube_xy_all: set[tuple[int, int]] = set()
        pipe_xy_all: set[tuple[int, int]] = set()
        # Use materialized templates (already XY-shifted by to_temporal_layer)
        for blk in self.cubes_.values():
            t = blk.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                for x, y in coord_list or []:
                    xy = (int(x), int(y))
                    cube_xy_all.add(xy)
                    coord_gid_2d[xy] = QubitGroupIdGlobal(blk.get_tiling_id())
        for pipe in self.pipes_.values():
            t = pipe.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                for x, y in coord_list or []:
                    xy = (int(x), int(y))
                    pipe_xy_all.add(xy)
                    coord_gid_2d[xy] = QubitGroupIdGlobal(pipe.get_tiling_id())

        # Fast lookup for existing edges to avoid duplicates where possible
        try:
            existing = {tuple(sorted((int(u), int(v)))) for (u, v) in (getattr(g, "physical_edges", []) or [])}
        except (AttributeError, TypeError, ValueError):
            existing = set()

        for u, coord_u in list(self.node2coord.items()):
            xu, yu, zu = int(coord_u[0]), int(coord_u[1]), int(coord_u[2])
            xy_u = (xu, yu)
            gid_u = coord_gid_2d.get(xy_u)
            if gid_u is None:
                continue
            for dx, dy, dz in DIRECTIONS3D:
                if dz != 0:
                    continue  # we only connect within the same z plane
                xv, yv, zv = xu + int(dx), yu + int(dy), zu
                coord_v = PhysCoordGlobal3D((xv, yv, zv))
                v = self.coord2node.get(coord_v)
                if v is None or v == u:
                    continue
                xy_v = (xv, yv)
                gid_v = coord_gid_2d.get(xy_v)
                if gid_v is None:
                    continue
                # Must be across cube vs pipe regions
                cross_region = (xy_u in cube_xy_all and xy_v in pipe_xy_all) or (
                    xy_v in cube_xy_all and xy_u in pipe_xy_all
                )
                if not cross_region:
                    continue
                # Allowed only if (gid_u, gid_v) is permitted (order-insensitive via helper)
                if not is_allowed_pair(gid_u, gid_v, self.allowed_gid_pairs):
                    continue
                a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
                if (a, b) in existing:
                    continue
                with suppress(Exception):
                    g.add_physical_edge(a, b)
                    existing.add((a, b))

        # Finalize
        if g is None:
            g = GraphState()
        self.local_graph = g
        self.qubit_count = len(getattr(g, "physical_nodes", []) or [])

        # Preserve simple XY maps for inspection (optional)
        data2d = sorted(cube_xy_all.union(pipe_xy_all))
        self.tiling_node_maps = {
            "xy": {xy: i for i, xy in enumerate(data2d)},
        }

    # ---- T25: boundary queries ---------------------------------------------
    def get_boundary_nodes(
        self,
        *,
        face: str,
        depth: list[int] | None = None,
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Return nodes on a given face at the requested depths, grouped by role.

        Parameters
        ----------
        face : {'x+','x-','y+','y-','z+','z-'}
            Boundary face. Case-insensitive; '+' means max side, '-' means min side.
        depth : list[int] | None
            Offsets inward from the boundary. For example, for 'z-':
            - depth=[0] selects z == z_min
            - depth=[1] selects z == z_min+1
            Negative values are clamped to 0 (i.e., treated as the boundary).

        Returns
        -------
        dict[str, list[PhysCoordGlobal3D]]
            Mapping with keys 'data', 'xcheck', 'zcheck' of coordinate triples.
        """
        if not self.node2coord:
            # Nothing compiled yet
            return {"data": [], "xcheck": [], "zcheck": []}

        roles = self.node2role or {}
        coords = list(self.node2coord.values())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)

        f = face.strip().lower()
        if f not in {"x+", "x-", "y+", "y-", "z+", "z-"}:
            msg = "face must be one of: x+/x-/y+/y-/z+/z-"
            raise ValueError(msg)
        depths = [max(int(d), 0) for d in (depth or [0])]

        def on_face(c: tuple[int, int, int]) -> bool:
            x, y, z = c
            if f == "x+":
                return x in {xmax - d for d in depths}
            if f == "x-":
                return x in {xmin + d for d in depths}
            if f == "y+":
                return y in {ymax - d for d in depths}
            if f == "y-":
                return y in {ymin + d for d in depths}
            if f == "z+":
                return z in {zmax - d for d in depths}
            # f == 'z-'
            return z in {zmin + d for d in depths}

        data: list[PhysCoordGlobal3D] = []
        xcheck: list[PhysCoordGlobal3D] = []
        zcheck: list[PhysCoordGlobal3D] = []

        for nid, c in self.node2coord.items():
            if not on_face(c):
                continue
            role = (roles.get(nid) or "").lower()
            if role == "ancilla_x":
                xcheck.append(c)
            elif role == "ancilla_z":
                zcheck.append(c)
            else:
                data.append(c)

        return {"data": data, "xcheck": xcheck, "zcheck": zcheck}

    def get_node_maps(self) -> dict[str, dict[tuple[int, int], int]]:
        """
        Return node_maps from ConnectedTiling (compute lazily if missing).

        Returns
        -------
            dict[str, dict[tuple[int, int], int]]: The node_maps from ConnectedTiling.
        """
        if not self.tiling_node_maps:
            self.compile()
        return self.tiling_node_maps


# (removed duplicate CompiledRHGCanvas definition)


@dataclass
class CompiledRHGCanvas:
    """
    Represents a compiled RHG canvas, containing temporal layers, the global graph,
    coordinate mappings, port sets, and accumulators for schedule, flow, and parity.

    Attributes
    ----------
    layers : list[TemporalLayer]
        The temporal layers of the canvas.
    global_graph : GraphState | None
        The global graph state after compilation.
    coord2node : dict[PhysCoordGlobal3D, int]
        Mapping from physical coordinates to node IDs.
    in_portset : dict[PatchCoordGlobal3D, list[int]]
        Input port sets for each patch.
    out_portset : dict[PatchCoordGlobal3D, list[int]]
        Output port sets for each patch.
    cout_portset : dict[PatchCoordGlobal3D, list[int]]
        Ancilla output port sets for each patch.
    schedule : ScheduleAccumulator
        Accumulator for scheduling information.
    flow : FlowAccumulator
        Accumulator for flow information.
    parity : ParityAccumulator
        Accumulator for parity checks.
    zlist : list[int]
        The current temporal layer indices.
    """

    # Non-default fields must come first
    layers: list[TemporalLayer]

    # Optional/defaulted fields follow
    global_graph: GraphState | None = None
    coord2node: dict[PhysCoordGlobal3D, int] = field(default_factory=dict)

    in_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    out_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    cout_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)

    # Give defaults to satisfy dataclass ordering; caller may override later
    schedule: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(default_factory=ParityAccumulator)
    zlist: list[int] = field(default_factory=list)

    # Optional placeholders
    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    # (deprecated) debug seam pairs: removed

    # def generate_stim_circuit(self) -> stim.Circuit:
    #     pass

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> CompiledRHGCanvas:
        # Remap GraphState by copying nodes/edges according to node_map.
        def _remap_graphstate(gsrc: GraphState | None, nmap: dict[int, int]) -> GraphState | None:
            if gsrc is None:
                return None
            gdst = GraphState()
            created: dict[int, int] = {}
            for old in gsrc.physical_nodes:
                new_id = nmap.get(old, old)
                if new_id in created:
                    continue
                created[new_id] = gdst.add_physical_node()
            for old, new_id in nmap.items():
                mb = gsrc.meas_bases.get(old)
                if mb is not None:
                    with suppress(Exception):
                        gdst.assign_meas_basis(created.get(new_id, new_id), mb)
            for u, v in gsrc.physical_edges:
                nu = nmap.get(u, u)
                nv = nmap.get(v, v)
                with suppress(Exception):
                    gdst.add_physical_edge(created.get(nu, nu), created.get(nv, nv))
            return gdst

        new_cgraph = CompiledRHGCanvas(
            layers=self.layers.copy(),
            global_graph=_remap_graphstate(self.global_graph, node_map),
            coord2node={},
            in_portset={},
            out_portset={},
            cout_portset={},
            schedule=self.schedule.remap_nodes(node_map),
            flow=self.flow.remap_nodes(node_map),
            parity=self.parity.remap_nodes(node_map),
            cubes_=self.cubes_.copy(),
            pipes_=self.pipes_.copy(),
            zlist=list(self.zlist),
        )

        # Remap coord2node
        for coord, old_nodeid in self.coord2node.items():
            new_cgraph.coord2node[coord] = node_map[old_nodeid]

        # Remap portsets
        for pos, nodes in self.in_portset.items():
            new_cgraph.in_portset[pos] = [node_map[n] for n in nodes]
        for pos, nodes in self.out_portset.items():
            new_cgraph.out_portset[pos] = [node_map[n] for n in nodes]
        for pos, nodes in self.cout_portset.items():
            new_cgraph.cout_portset[pos] = [node_map[n] for n in nodes]

        return new_cgraph

    # ---- T25: boundary queries ---------------------------------------------
    def get_boundary_nodes(
        self,
        *,
        face: str,
        depth: list[int] | None = None,
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Boundary query after temporal composition on the compiled canvas.

        Operates on the global coord2node map using the same semantics as
        TemporalLayer.get_boundary_nodes.
        """
        if not self.coord2node:
            return {"data": [], "xcheck": [], "zcheck": []}

        coords = list(self.coord2node.keys())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)

        f = face.strip().lower()
        if f not in {"x+", "x-", "y+", "y-", "z+", "z-"}:
            msg = "face must be one of: x+/x-/y+/y-/z+/z-"
            raise ValueError(msg)
        depths = [max(int(d), 0) for d in (depth or [0])]

        def on_face(c: tuple[int, int, int]) -> bool:
            x, y, z = c
            if f == "x+":
                return x in {xmax - d for d in depths}
            if f == "x-":
                return x in {xmin + d for d in depths}
            if f == "y+":
                return y in {ymax - d for d in depths}
            if f == "y-":
                return y in {ymin + d for d in depths}
            if f == "z+":
                return z in {zmax - d for d in depths}
            return z in {zmin + d for d in depths}

        # Without global role info, conservatively return all as 'data'.
        selected = [c for c in coords if on_face(c)]
        return {"data": selected, "xcheck": [], "zcheck": []}

    # ---- T25: method form of temporal composition --------------------------
    def add_temporal_layer(self, next_layer: TemporalLayer, *, pipes: list[RHGPipe] | None = None) -> CompiledRHGCanvas:
        """Compose this compiled canvas with `next_layer`.

        Convenience instance-method wrapper around the module-level
        `add_temporal_layer` with optional `pipes` gating cross-time connections.
        """
        return add_temporal_layer(self, next_layer, list(pipes or []))


@dataclass
class RHGCanvasSkeleton:  # BlockGraph in tqec
    name: str = "Blank Canvas Skeleton"
    # Optional template placeholder for future use
    template: object | None = None
    # {(0,0,0): InitPlusSkeleton(), ..}
    cubes_: dict[PatchCoordGlobal3D, RHGCubeSkeleton] = field(default_factory=dict)
    # {((0,0,0),(1,0,0)): StabilizeSkeleton(), ((0,0,0), (0,0,1)): MeasureSkeleton(basis=X)}
    pipes_: dict[PipeCoordGlobal3D, RHGPipeSkeleton] = field(default_factory=dict)

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCubeSkeleton) -> None:
        self.cubes_[position] = cube

    def add_pipe(self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipeSkeleton) -> None:
        self.pipes_[start, end] = pipe

    def trim_spatial_boundaries(self) -> None:
        """
        Function trim spatial boundary (tiling from Scalable tiling class)
            case direction
            match direction
            if Xplus then
            axis to target is max d
            if Xminus then
            axis to 0
            target = -1
            for x y ancillas
                do
                if coord.dim(acos) hits the target
                remove it
            end
        end
        """
        # Iterate spatial pipes and trim facing boundaries of adjacent cubes.
        # Pipes are keyed by ((x,y,z),(x,y,z)); detect spatial adjacency by dx/dy.
        for pipe_coord, _pipe in list(self.pipes_.items()):
            u, v = pipe_coord
            ux, uy, uz = u
            vx, vy, vz = v
            # Temporal pipes are not handled here
            if uz != vz:
                continue

            dx, dy = vx - ux, vy - uy

            left = self.cubes_.get(u)
            right = self.cubes_.get(v)

            if dx == 1 and dy == 0:
                # X+ direction: trim RIGHT of left cube and LEFT of right cube
                if left is not None:
                    left.trim_spatial_boundary("RIGHT")
                if right is not None:
                    right.trim_spatial_boundary("LEFT")
            elif dx == -1 and dy == 0:
                # X- direction
                if left is not None:
                    left.trim_spatial_boundary("LEFT")
                if right is not None:
                    right.trim_spatial_boundary("RIGHT")
            elif dy == 1 and dx == 0:
                # Y+ direction
                if left is not None:
                    left.trim_spatial_boundary("TOP")
                if right is not None:
                    right.trim_spatial_boundary("BOTTOM")
            elif dy == -1 and dx == 0:
                # Y- direction
                if left is not None:
                    left.trim_spatial_boundary("BOTTOM")
                if right is not None:
                    right.trim_spatial_boundary("TOP")
            else:
                # Not an axis-aligned spatial neighbor; ignore.
                continue

    def to_canvas(self) -> RHGCanvas:
        self.trim_spatial_boundaries()

        trimmed_cubes_skeleton = self.cubes_.copy()
        trimmed_pipes_skeleton = self.pipes_.copy()

        cubes_ = {}
        for pos, c in trimmed_cubes_skeleton.items():
            # Materialize block and attach its 3D anchor so z-offset is correct
            blk = c.to_block()
            with suppress(Exception):
                # Ensure blocks know their placement (x, y, z)
                blk.source = pos
            cubes_[pos] = blk
        pipes_ = {}
        for (start, end), p in trimmed_pipes_skeleton.items():
            pipes_[start, end] = p.to_block(start, end)

        return RHGCanvas(name=self.name, cubes_=cubes_, pipes_=pipes_)


@dataclass
class RHGCanvas:  # TopologicalComputationGraph in tqec
    name: str = "Blank Canvas"

    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    layers: list[TemporalLayer] | None = None

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCube) -> None:
        self.cubes_[position] = cube
        # Reset one-shot guard so layers can be rebuilt after topology changes
        with suppress(AttributeError):
            self._to_temporal_layers_called = False

    def add_pipe(self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipe) -> None:
        self.pipes_[start, end] = pipe
        # Reset one-shot guard so layers can be rebuilt after topology changes
        with suppress(AttributeError):
            self._to_temporal_layers_called = False

    def to_temporal_layers(self) -> dict[int, TemporalLayer]:
        # Disallow multiple calls to prevent duplicate XY shifts on templates.
        if getattr(self, "_to_temporal_layers_called", False):
            msg = (
                "RHGCanvas.to_temporal_layers() can be called at most once per canvas. "
                "Rebuild the canvas (or use RHGCanvasSkeleton.to_canvas()) before calling again."
            )
            raise RuntimeError(msg)
        temporal_layers: dict[int, TemporalLayer] = {}

        for z in range(max(self.cubes_.keys(), key=itemgetter(2))[2] + 1):
            cubes = {pos: c for pos, c in self.cubes_.items() if pos[2] == z}
            pipes = {(u, v): p for (u, v), p in self.pipes_.items() if u[2] == z and v[2] == z}

            layer = to_temporal_layer(z, cubes, pipes)
            temporal_layers[z] = layer

        with suppress(AttributeError):
            self._to_temporal_layers_called = True
        return temporal_layers

    def compile(self) -> CompiledRHGCanvas:
        temporal_layers = self.to_temporal_layers()
        # Initialize an empty compiled canvas with required accumulators
        cgraph = CompiledRHGCanvas(
            layers=[],
            global_graph=None,
            coord2node={},
            in_portset={},
            out_portset={},
            cout_portset={},
            schedule=ScheduleAccumulator(),
            flow=FlowAccumulator(),
            parity=ParityAccumulator(),
            zlist=[],
        )

        # Compose layers in increasing temporal order, wiring any cross-layer pipes
        for z in sorted(temporal_layers.keys()):
            layer = temporal_layers[z]
            # Select pipes whose start.z is the last compiled z and end.z is this layer z
            prev_z = cgraph.zlist[-1] if getattr(cgraph, "zlist", []) else None
            if prev_z is None:
                pipes: list[RHGPipe] = []
            else:
                pipes = []
                for pipe_coord, pipe in self.pipes_.items():
                    u, v = pipe_coord
                    if u[2] == prev_z and v[2] == z:
                        # 明示的に端点情報を埋めて渡す(skeleton->block で保持されないため)
                        with suppress(Exception):
                            pipe.source = u
                            pipe.sink = v
                            pipe.direction = get_direction(u, v)
                        pipes.append(pipe)
            cgraph = add_temporal_layer(cgraph, layer, pipes)
        return cgraph


def to_temporal_layer(
    z: int,
    cubes: dict[PatchCoordGlobal3D, RHGCube],
    pipes: dict[PipeCoordGlobal3D, RHGPipe],
) -> TemporalLayer:
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer(z)

    # shift position before materialization(テンプレートは ScalableTemplate を保持したままXY移動)
    for pos, c in cubes.items():
        dx, dy = cube_offset_xy(c.d, pos)
        # 直接テンプレートをXY移動(inplace=True)
        c.template.shift_coords((dx, dy), coordinate="tiling2d", inplace=True)
    for (source, sink), p in pipes.items():
        direction = get_direction(source, sink)
        dx, dy = pipe_offset_xy(p.d, source, sink, direction)
        # 直接テンプレートをXY移動(inplace=True)
        p.template.shift_coords((dx, dy), coordinate="tiling2d", inplace=True)

    # materialize blocks before adding
    cubes_mat = {pos: blk.materialize() for pos, blk in cubes.items()}
    pipes_mat = {(u, v): p.materialize() for (u, v), p in pipes.items()}

    layer.add_cubes(cubes_mat)
    layer.add_pipes(pipes_mat)

    # compile this layer
    layer.compile()
    return layer


def add_temporal_layer(cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, pipes: list[RHGPipe]) -> CompiledRHGCanvas:
    """Compose the compiled canvas with the next temporal layer.

    Additionally, if temporal pipes are provided, connect corresponding out->in
    port nodes by adding CZ edges (modeled as physical edges here) between paired nodes.

    NOTE: This assumes `next_layer.local_graph` is a canonical GraphState built
    from its cubes/pipes. If it's None, we keep cgraph unchanged.

    Returns
    -------
    CompiledRHGCanvas
        The composed canvas with the next temporal layer integrated.
    """
    # If the canvas is empty, this is the first layer.
    if cgraph.global_graph is None:
        return CompiledRHGCanvas(
            layers=[next_layer],
            global_graph=next_layer.local_graph,
            coord2node={k: int(v) for k, v in next_layer.coord2node.items()},  # type: ignore[misc]
            in_portset={k: [int(v) for v in vs] for k, vs in next_layer.in_portset.items()},  # type: ignore[misc]
            out_portset={k: [int(v) for v in vs] for k, vs in next_layer.out_portset.items()},  # type: ignore[misc]
            cout_portset={k: [int(v) for v in vs] for k, vs in next_layer.cout_portset.items()},  # type: ignore[misc]
            schedule=next_layer.schedule,
            zlist=[next_layer.z],
            cubes_=next_layer.cubes_,
            pipes_=next_layer.pipes_,
        )
    # else: non-empty canvas; continue with composition
    graph1 = cgraph.global_graph
    graph2 = next_layer.local_graph

    # Compose sequentially with node remaps
    try:
        # TODO: See T49 for compose_sequentially improvements
        new_graph, node_map1, node_map2 = compose_sequentially(graph1, graph2)
    except (ValueError, TypeError, AttributeError):
        # Fallback: relaxed composition when canonical form is not satisfied.
        # Copy nodes/edges from both graphs into a fresh GraphState.
        g = GraphState()
        node_map1 = {}
        node_map2 = {}
        # copy graph1 nodes
        for n in graph1.physical_nodes:
            nn = g.add_physical_node()
            mb = graph1.meas_bases.get(n)
            if mb is not None:
                g.assign_meas_basis(nn, mb)
            node_map1[n] = nn
        # copy graph2 nodes
        for n in graph2.physical_nodes:
            nn = g.add_physical_node()
            mb = graph2.meas_bases.get(n)
            if mb is not None:
                g.assign_meas_basis(nn, mb)
            node_map2[n] = nn
        # copy edges
        for u, v in graph1.physical_edges:
            g.add_physical_edge(node_map1[u], node_map1[v])
        for u, v in graph2.physical_edges:
            g.add_physical_edge(node_map2[u], node_map2[v])
        new_graph = g
    # Remap registries for both sides into the composed id-space
    cgraph = cgraph.remap_nodes(node_map1)
    # Remap next_layer registries into composed id-space
    next_layer.coord2node = {c: node_map2.get(n, n) for c, n in next_layer.coord2node.items()}
    next_layer.node2coord = {node_map2.get(n, n): c for n, c in next_layer.node2coord.items()}
    next_layer.node2role = {node_map2.get(n, n): r for n, r in next_layer.node2role.items()}

    # Create a new CompiledRHGCanvas to hold the merged result.
    new_layers = [*cgraph.layers, next_layer]
    new_z = next_layer.z

    # Build merged coord2node map (already remapped above)
    new_coord2node: dict[PhysCoordGlobal3D, int] = {
        **cgraph.coord2node,
        **next_layer.coord2node,
    }

    # Remap portsets
    in_portset = {pos: [node_map2[n] for n in nodes] for pos, nodes in next_layer.in_portset.items()}
    out_portset = {pos: [node_map1[n] for n in nodes] for pos, nodes in cgraph.out_portset.items()}
    cout_portset = {
        **{pos: [node_map1[n] for n in nodes] for pos, nodes in cgraph.cout_portset.items()},
        **{pos: [node_map2[n] for n in nodes] for pos, nodes in next_layer.cout_portset.items()},
    }

    new_coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal] = {}
    for _pos, cube in [*cgraph.cubes_.items(), *next_layer.cubes_.items()]:
        new_coord2gid.update(cube.coord2gid)
    for _pos, pipe in [*cgraph.pipes_.items(), *next_layer.pipes_.items()]:
        new_coord2gid.update(pipe.coord2gid)

    # Build seam node pairs by XY identity at z- (prev) and z+ (next) boundaries
    # Always ON: do not expose any toggle; connect only when temporal pipes exist.
    allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]] = set()

    for p in pipes:
        allowed_gid_pairs.add(
            (
                QubitGroupIdGlobal(cgraph.cubes_[p.source].get_tiling_id()),
                QubitGroupIdGlobal(next_layer.cubes_[p.sink].get_tiling_id()),
            )
        )
        for source in next_layer.get_boundary_nodes(face="z-", depth=[-1])["data"]:
            sink_coord = PhysCoordGlobal3D((source[0], source[1], source[2] - 1))

            source_gid = new_coord2gid.get(PhysCoordGlobal3D(source))
            sink_gid = new_coord2gid.get(sink_coord)

            if is_allowed_pair(source_gid, sink_gid, allowed_gid_pairs):
                source_node = new_coord2node.get(PhysCoordGlobal3D(source))
                sink_node = new_coord2node.get(sink_coord)
                new_graph.add_physical_edge(source_node, sink_node)

    # Update accumulators at the new layer's z- boundary (ancillas)
    # Use next_layer boundary (local ids remapped earlier) and the composed graph.
    try:
        z_minus_ancillas = []
        bn = next_layer.get_boundary_nodes(face="z-", depth=[0])
        # collect anchor node ids for ancilla X/Z
        for c in bn.get("xcheck", []):
            nid = next_layer.coord2node.get(c)
            if nid is not None:
                z_minus_ancillas.append(nid)
        for c in bn.get("zcheck", []):
            nid = next_layer.coord2node.get(c)
            if nid is not None:
                z_minus_ancillas.append(nid)

        for anchor in z_minus_ancillas:
            new_schedule = cgraph.schedule
            new_parity_acc = cgraph.parity
            new_flow_acc = cgraph.flow
            # Monotone updates (assertions inside ensure non-decreasing)
            # Gating argument removed: always operate on the composed graph.
            new_schedule.update_at(anchor, new_graph)
            new_parity_acc.update_at(anchor, new_graph)
            new_flow_acc.update_at(anchor, new_graph)
        # assign back (no-op if same instances)
        cgraph.schedule = new_schedule
        cgraph.parity = new_parity_acc
        cgraph.flow = new_flow_acc
    except (ValueError, KeyError, AttributeError):
        # Be tolerant: boundary queries are best-effort at this milestone
        cgraph.schedule = cgraph.schedule

    # Merge schedule, flow, parity
    new_schedule = cgraph.schedule.compose_sequential(next_layer.schedule)

    # Start from unions of existing x/z flows
    xflow_combined: dict[int, set[int]] = {}
    zflow_combined: dict[int, set[int]] = {}
    for src, dsts in cgraph.flow.xflow.items():
        xflow_combined[src] = set(dsts)
    for src, dsts in next_layer.flow.xflow.items():
        xflow_combined.setdefault(src, set()).update(dsts)
    for src, dsts in cgraph.flow.zflow.items():
        zflow_combined[src] = set(dsts)
    for src, dsts in next_layer.flow.zflow.items():
        zflow_combined.setdefault(src, set()).update(dsts)
    # No explicit seam corrections here; physical edges were added when pipes exist.

    # Convert int keys to NodeIdLocal for FlowAccumulator
    xflow_typed = {NodeIdLocal(k): {NodeIdLocal(v) for v in vs} for k, vs in xflow_combined.items()}
    zflow_typed = {NodeIdLocal(k): {NodeIdLocal(v) for v in vs} for k, vs in zflow_combined.items()}
    merged_flow = FlowAccumulator(xflow=xflow_typed, zflow=zflow_typed)
    new_parity = ParityAccumulator(
        x_checks=cgraph.parity.x_checks + next_layer.parity.x_checks,
        z_checks=cgraph.parity.z_checks + next_layer.parity.z_checks,
    )

    return CompiledRHGCanvas(
        layers=new_layers,
        global_graph=new_graph,
        coord2node={k: int(v) for k, v in new_coord2node.items()},  # type: ignore[misc]
        in_portset={k: [int(v) for v in vs] for k, vs in in_portset.items()},  # type: ignore[misc]
        out_portset={k: [int(v) for v in vs] for k, vs in out_portset.items()},  # type: ignore[misc]
        cout_portset={k: [int(v) for v in vs] for k, vs in cout_portset.items()},  # type: ignore[misc]
        schedule=new_schedule,
        flow=merged_flow,
        parity=new_parity,
        zlist=[*list(getattr(cgraph, "zlist", [])), new_z],
    )
