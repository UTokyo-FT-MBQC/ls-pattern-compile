from __future__ import annotations

from dataclasses import dataclass, field

from lspattern.tiling.base import Tiling

try:
    from graphix_zx.graphstate import (
        GraphState,
        compose_in_parallel,
        compose_sequentially,
    )
except Exception:  # fallback for repo-local execution without install
    import sys as _sys, pathlib as _pathlib

    _ROOT = _pathlib.Path(__file__).resolve().parents[1]
    _SRC = _ROOT / "src"
    _SRC_GRAPHIX = _SRC / "graphix_zx"
    for _p in (str(_SRC_GRAPHIX), str(_SRC)):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    from graphix_zx.graphstate import (
        GraphState,
        compose_in_parallel,
        compose_sequentially,
    )
from graphix_zx.common import Plane, PlannerMeasBasis
from lspattern.accumulator import (
    FlowAccumulator,
    ParityAccumulator,
    ScheduleAccumulator,
)
from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.mytype import (
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PipeCoordGlobal3D,
)
from lspattern.tiling.template import (
    cube_offset_xy,
    offset_tiling,
    pipe_offset_xy,
)
from lspattern.utils import get_direction


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

    def __init__(self, z: int):
        self.z = z
        self.qubit_count = 0
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
        self.tiling_node_maps = {}

    def __post_init__(self):
        pass

    def add_cubes(self, cubes: dict[PatchCoordGlobal3D, RHGCube]) -> None:
        for pos, cube in cubes.items():
            self.add_cube(pos, cube)

    def add_pipes(self, pipes: dict[PipeCoordGlobal3D, RHGPipe]) -> None:
        for (source, sink), pipe in pipes.items():
            self.add_pipe(source, sink, pipe)

    def compile(self) -> None:
        # Blocks are expected to be materialized already.

        cube_tilings_abs: dict[PatchCoordGlobal3D, Tiling] = {}
        pipe_tilings_abs: dict[PipeCoordGlobal3D, Tiling] = {}
        dset: set[int] = set()

        for pos, b in self.cubes_.items():
            d_val = int(b.d)
            dset.add(d_val)
            dx, dy = cube_offset_xy(d_val, pos)
            cube_tilings_abs[pos] = offset_tiling(b.template, dx, dy)

        for (source, sink), p in self.pipes_.items():
            d_val = int(p.d)
            dset.add(d_val)
            direction = get_direction(source, sink)
            dx, dy = pipe_offset_xy(d_val, source, sink, direction)
            pipe_tilings_abs[(source, sink)] = offset_tiling(p.template, dx, dy)

        if len(dset) > 1:
            msg = (
                "TemporalLayer.compile: mixed code distances (d) are not supported yet"
            )
            raise MixedCodeDistanceError(msg)

        # Aggregate absolute 2D coords and patch groups
        data2d_set: set[tuple[int, int]] = set()
        x2d_set: set[tuple[int, int]] = set()
        z2d_set: set[tuple[int, int]] = set()
        # Mapping XY -> tiling group id (computed via union of tiles connected by pipes)
        coord_gid: dict[tuple[int, int], int] = {}
        # Mapping XY -> tiling id (per-part unique id_)
        coord_tid: dict[tuple[int, int], int] = {}
        # Track tile ids for cubes/pipes separately
        cube_pos2tid: dict[PatchCoordGlobal3D, int] = {}
        pipe_seg2tid: dict[PipeCoordGlobal3D, int] = {}

        for pos, t in cube_tilings_abs.items():
            for x, y in t.data_coords:
                xy = (int(x), int(y))
                data2d_set.add(xy)
                coord_tid[xy] = int(t.id_)
            for x, y in t.x_coords:
                xy = (int(x), int(y))
                x2d_set.add(xy)
                coord_tid[xy] = int(t.id_)
            for x, y in t.z_coords:
                xy = (int(x), int(y))
                z2d_set.add(xy)
                coord_tid[xy] = int(t.id_)
            cube_pos2tid[pos] = int(t.id_)

        for seg, t in pipe_tilings_abs.items():
            for x, y in t.data_coords:
                xy = (int(x), int(y))
                data2d_set.add(xy)
                coord_tid[xy] = int(t.id_)
            for x, y in t.x_coords:
                xy = (int(x), int(y))
                x2d_set.add(xy)
                coord_tid[xy] = int(t.id_)
            for x, y in t.z_coords:
                xy = (int(x), int(y))
                z2d_set.add(xy)
                coord_tid[xy] = int(t.id_)
            pipe_seg2tid[seg] = int(t.id_)

        # --- Build allowed pairs and unify groups (gid) across tiles via pipes ---
        # Allowed pairs are the (source cube tid, sink cube tid) induced by pipes
        allowed_pairs: set[tuple[int, int]] = set()
        # Union-Find (DSU) over tiling ids to compute connected groups
        parent: dict[int, int] = {}

        def find(a: int) -> int:
            parent.setdefault(a, a)
            if parent[a] != a:
                parent[a] = find(parent[a])
            return parent[a]

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # unify to the smaller id for determinism
            m = min(ra, rb)
            parent[ra] = m
            parent[rb] = m

        # Initialize DSU parents for all tiles
        for t in cube_tilings_abs.values():
            parent.setdefault(int(t.id_), int(t.id_))
        for t in pipe_tilings_abs.values():
            parent.setdefault(int(t.id_), int(t.id_))

        # Unify cube<->pipe<->cube per spatial pipe, and record allowed cube pairs
        for (source, sink), p in self.pipes_.items():
            tid_src = cube_pos2tid.get(source)
            tid_snk = cube_pos2tid.get(sink)
            tid_pipe = pipe_seg2tid.get((source, sink))
            if tid_src is None or tid_snk is None or tid_pipe is None:
                continue
            union(tid_src, tid_pipe)
            union(tid_pipe, tid_snk)
            allowed_pairs.add((tid_src, tid_snk))

        # Map each XY coord to its unified group id (gid)
        for xy, tid in coord_tid.items():
            gid = find(int(tid)) if tid is not None else None
            if gid is not None:
                coord_gid[xy] = int(gid)

        # Build GraphState (T12 policy)
        d = next(iter(dset)) if dset else 0
        max_t = 2 * d
        # TODO: read the comment below and do it
        # [T33] T33.md - Remove redundant graphstate construction
        # Read the Task T31-32 where we have changed the policy to materialize the RHG
        # graph before reaching TemporalLayer.compile. So we do not need to add all the nodes here.
        # Modify the codes below to add **only difference** of the nodes, edges, schedules, flows, parities.
        # AGENTS.md、.local/*のコメントたちを読んで現状の課題を認識して、その後にレポジトリ全体lspattern/*を確認して。残っているタスク間の依存関係を踏まえつつタスクを進めて．
        # 受け入れ要件: compile()関数内のRHGBlock.materialize()でつくったlocal_graph, node2coord, coord2node, node2role等を最大限活用することで本関数内のコードを短くする．また，compose_parallelをpipe/blockの数だけcallすることが受け入れ要件．さらに，新しく"g.add_physical_node()”をcompile()ないで呼ばないことも受け入れ要件．これらが満たされるまでiterateして頑張って．physical nodesはmaterializeしたところから引っ張ってきてnode_mapをする

        g = GraphState()
        node2coord = {}
        coord2node = {}
        node2role = {}

        data2d = sorted(data2d_set)
        x2d = sorted(x2d_set)
        z2d = sorted(z2d_set)

        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}
        for t in range(max_t + 1):
            cur = {}
            if t != max_t:
                # normal node assignment
                for x, y in data2d:
                    n = g.add_physical_node()
                    node2coord[n] = (x, y, t)
                    coord2node[x, y, t] = n
                    node2role[n] = "data"
                    cur[x, y] = n
                if t % 2 == 0:
                    for x, y in x2d:
                        n = g.add_physical_node()
                        node2coord[n] = (x, y, t)
                        coord2node[x, y, t] = n
                        node2role[n] = "ancilla_x"
                        cur[x, y] = n
                else:
                    for x, y in z2d:
                        n = g.add_physical_node()
                        node2coord[n] = (x, y, t)
                        coord2node[x, y, t] = n
                        node2role[n] = "ancilla_z"
                        cur[x, y] = n
            # No special handling for final t: already added data nodes above.

            nodes_by_z[t] = cur

        # Helper: allowed to connect if two coords belong to the same unified tiling-group
        def _same_gid(xy1: tuple[int, int], xy2: tuple[int, int]) -> bool:
            g1 = coord_gid.get(xy1)
            g2 = coord_gid.get(xy2)
            return (g1 is not None) and (g1 == g2)

        for t, cur in nodes_by_z.items():
            for (x, y), u in cur.items():
                for dx, dy, dz in DIRECTIONS3D:
                    if dz != 0:
                        continue
                    xy2 = (x + dx, y + dy)
                    v = cur.get(xy2)
                    if v is not None and v > u and _same_gid((x, y), xy2):
                        # Connect CZ only within unified groups (includes cube<->pipe<->cube)
                        g.add_physical_edge(u, v)

        for t in range(1, max_t + 1):
            cur = nodes_by_z[t]
            prev = nodes_by_z[t - 1]
            for xy, u in cur.items():
                v = prev.get(xy)
                if v is not None:
                    g.add_physical_edge(u, v)

        self.local_graph = g
        self.node2coord = node2coord
        self.coord2node = coord2node
        self.node2role = node2role
        self.qubit_count = len(g.physical_nodes)
        # simple node_maps for inspection
        self.tiling_node_maps = {
            "data": {xy: i for i, xy in enumerate(data2d)},
            "x": {xy: i for i, xy in enumerate(x2d)},
            "z": {xy: i for i, xy in enumerate(z2d)},
        }

        return

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
        if f not in ("x+", "x-", "y+", "y-", "z+", "z-"):
            raise ValueError("face must be one of: x+/x-/y+/y-/z+/z-")
        depths = [int(d) if int(d) >= 0 else 0 for d in (depth or [0])]

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

    def add_cube(self, pos: PatchCoordGlobal3D, cube: RHGCube) -> None:
        """Add a materialized cube to this temporal layer and place it at `pos`."""
        self.cubes_[pos] = cube
        self.patches.append(pos)
        return

    def add_pipe(
        self,
        source: PatchCoordGlobal3D,
        sink: PatchCoordGlobal3D,
        spatial_pipe: RHGPipe,
    ) -> None:
        """Register a spatial pipe within this layer between `source` and `sink`."""
        self.pipes_[(source, sink)] = spatial_pipe
        self.lines.append((source, sink))
        return


@dataclass  # noqa: E302
class CompiledRHGCanvas:
    layers: list[TemporalLayer]

    global_graph: GraphState | None = None
    coord2node: dict[PhysCoordGlobal3D, int] = field(default_factory=dict)

    in_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    out_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    cout_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)

    # Give defaults to satisfy dataclass ordering; caller may override later
    schedule: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(default_factory=ParityAccumulator)
    z: int = 0

    # def generate_stim_circuit(self) -> stim.Circuit:
    #     pass

    def remap_nodes(
        self, node_map: dict[NodeIdLocal, NodeIdLocal]
    ) -> CompiledRHGCanvas:
        # Remap GraphState by copying nodes/edges according to node_map.
        def _remap_graphstate(
            gsrc: GraphState | None, nmap: dict[int, int]
        ) -> GraphState | None:
            if gsrc is None:
                return None
            gdst = GraphState()
            # Ensure we create as many nodes as needed to host remapped ids.
            # Build reverse map new_id -> created node index
            created: dict[int, int] = {}
            for old in gsrc.physical_nodes:
                new_id = nmap.get(old, old)
                if new_id in created:
                    continue
                # create placeholder nodes up to the maximum new_id index
                # We cannot directly set indices; instead, allocate sequentially
                created[new_id] = gdst.add_physical_node()
            # Assign meas bases approximately (best-effort)
            for old, new_id in nmap.items():
                mb = gsrc.meas_bases.get(old)
                if mb is not None:
                    try:
                        gdst.assign_meas_basis(created.get(new_id, new_id), mb)
                    except Exception:
                        pass
            # Copy edges
            for u, v in gsrc.physical_edges:
                nu = nmap.get(u, u)
                nv = nmap.get(v, v)
                try:
                    gdst.add_physical_edge(created.get(nu, nu), created.get(nv, nv))
                except Exception:
                    pass
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
            z=self.z,
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


@dataclass
class RHGCanvasSkeleton:  # BlockGraph in tqec
    name: str = "Blank Canvas Skeleton"
    # Optional template placeholder for future use
    template: object | None = None
    # {(0,0,0): InitPlusSkeleton(), ..}
    cubes_: dict[PatchCoordGlobal3D, RHGCubeSkeleton] = field(default_factory=dict)
    # {((0,0,0),(1,0,0)): StabilizeSkeleton(), ((0,0,0), (0,0,1)): MeasureSkeleton(basis=X)}
    pipes_: dict[PipeCoordGlobal3D, RHGPipeSkeleton] = field(default_factory=dict)


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
    z : int
        The current temporal layer index.
    """

    layers: list[TemporalLayer]

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

    # def generate_stim_circuit(self) -> stim.Circuit:
    #     pass

    def remap_nodes(
        self, node_map: dict[NodeIdLocal, NodeIdLocal]
    ) -> CompiledRHGCanvas:
        # Remap GraphState by copying nodes/edges according to node_map.
        def _remap_graphstate(
            gsrc: GraphState | None, nmap: dict[int, int]
        ) -> GraphState | None:
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
                    try:
                        gdst.assign_meas_basis(created.get(new_id, new_id), mb)
                    except Exception:
                        pass
            for u, v in gsrc.physical_edges:
                nu = nmap.get(u, u)
                nv = nmap.get(v, v)
                try:
                    gdst.add_physical_edge(created.get(nu, nu), created.get(nv, nv))
                except Exception:
                    pass
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
        if f not in ("x+", "x-", "y+", "y-", "z+", "z-"):
            raise ValueError("face must be one of: x+/x-/y+/y-/z+/z-")
        depths = [int(d) if int(d) >= 0 else 0 for d in (depth or [0])]

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
    def add_temporal_layer(
        self, next_layer: TemporalLayer, *, pipes: list[RHGPipe] | None = None
    ) -> "CompiledRHGCanvas":
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

    def materialize(self) -> RHGCube:
        "Materialize the internal template assuming that the boundaries are trimmed"

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCubeSkeleton) -> None:
        self.cubes_[position] = cube

    def add_pipe(
        self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipeSkeleton
    ) -> None:
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
        for (u, v), _pipe in list(self.pipes_.items()):
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
            cubes_[pos] = c.to_block()
        pipes_ = {}
        for (start, end), p in trimmed_pipes_skeleton.items():
            pipes_[start, end] = p.to_block(start, end)

        canvas = RHGCanvas(name=self.name, cubes_=cubes_, pipes_=pipes_)
        return canvas


@dataclass
class RHGCanvas:  # TopologicalComputationGraph in tqec
    name: str = "Blank Canvas"

    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    layers: list[TemporalLayer] | None = None

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCube) -> None:
        self.cubes_[position] = cube

    def add_pipe(
        self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipe
    ) -> None:
        self.pipes_[start, end] = pipe

    def to_temporal_layers(self) -> dict[int, TemporalLayer]:
        temporal_layers: dict[int, TemporalLayer] = {}
        for z in range(max(self.cubes_.keys(), key=lambda pos: pos[2])[2] + 1):
            cubes = {pos: c for pos, c in self.cubes_.items() if pos[2] == z}
            pipes = {
                (u, v): p
                for (u, v), p in self.pipes_.items()
                if u[2] == z and v[2] == z
            }

            layer = to_temporal_layer(z, cubes, pipes)
            temporal_layers[z] = layer

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
            # Select pipes whose start.z is the current compiled z and end.z is the next layer z
            pipes: list[RHGPipe] = [
                pipe
                for (u, v), pipe in self.pipes_.items()
                if u[2] == cgraph.z and v[2] == z
            ]
            cgraph = add_temporal_layer(cgraph, layer, pipes)
        return cgraph


def to_temporal_layer(
    z: int,
    cubes: dict[PatchCoordGlobal3D, RHGCube],
    pipes: dict[PipeCoordGlobal3D, RHGPipe],
) -> TemporalLayer:
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer(z)

    # materialize blocks before adding
    cubes_mat = {pos: blk.materialize() for pos, blk in cubes.items()}
    pipes_mat = {(u, v): p.materialize() for (u, v), p in pipes.items()}

    layer.add_cubes(cubes_mat)
    layer.add_pipes(pipes_mat)

    # compile this layer
    layer.compile()
    return layer


def add_temporal_layer(
    cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, pipes: list[RHGPipe]
) -> CompiledRHGCanvas:
    """Compose the compiled canvas with the next temporal layer.

    Follows the legacy-canvas pattern:
      - If there is no existing global graph, ingest the layer as-is.
      - Otherwise, compose sequentially and remap existing registries via node_map1,
        then ingest the layer's locals via node_map2.

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
        new_cgraph = CompiledRHGCanvas(
            layers=[next_layer],
            global_graph=next_layer.local_graph,
            coord2node=next_layer.coord2node,
            in_portset=next_layer.in_portset,
            out_portset=next_layer.out_portset,
            cout_portset=next_layer.cout_portset,
            schedule=next_layer.schedule,
            zlist=[next_layer.z],
        )
        return new_cgraph

    graph1 = cgraph.global_graph
    graph2 = next_layer.local_graph

    # Compose sequentially with node remaps
    try:
        new_graph, node_map1, node_map2 = compose_sequentially(graph1, graph2)
    except Exception:
        # Fallback: relaxed composition when canonical form is not satisfied.
        # Copy nodes/edges from both graphs into a fresh GraphState.
        g = GraphState()
        node_map1 = {}
        node_map2 = {}
        # copy graph1 nodes
        for n in getattr(graph1, "physical_nodes", []) or []:
            nn = g.add_physical_node()
            mb = getattr(graph1, "meas_bases", {}).get(n)
            if mb is not None:
                try:
                    g.assign_meas_basis(nn, mb)
                except Exception:
                    pass
            node_map1[n] = nn
        # copy graph2 nodes
        for n in getattr(graph2, "physical_nodes", []) or []:
            nn = g.add_physical_node()
            mb = getattr(graph2, "meas_bases", {}).get(n)
            if mb is not None:
                try:
                    g.assign_meas_basis(nn, mb)
                except Exception:
                    pass
            node_map2[n] = nn
        # copy edges
        for u, v in getattr(graph1, "physical_edges", []) or []:
            try:
                g.add_physical_edge(node_map1[u], node_map1[v])
            except Exception:
                pass
        for u, v in getattr(graph2, "physical_edges", []) or []:
            try:
                g.add_physical_edge(node_map2[u], node_map2[v])
            except Exception:
                pass
        new_graph = g
    # Remap registries for both sides into the composed id-space
    cgraph = cgraph.remap_nodes(node_map1)
    # Remap next_layer coord2node into composed id-space
    next_layer.coord2node = {
        c: node_map2.get(n, n) for c, n in next_layer.coord2node.items()
    }

    # Create a new CompiledRHGCanvas to hold the merged result.
    new_layers = cgraph.layers + [next_layer]
    new_z = next_layer.z

    # Build merged coord2node map (already remapped above)
    new_coord2node: dict[PhysCoordGlobal3D, int] = {}
    for coord, nid in cgraph.coord2node.items():
        new_coord2node[coord] = nid
    for coord, nid in next_layer.coord2node.items():
        new_coord2node[coord] = nid

    # Remap portsets
    in_portset = {}
    out_portset = {}
    cout_portset = {}

    for pos, nodes in next_layer.in_portset.items():
        in_portset[pos] = [node_map2[n] for n in nodes]
    for pos, nodes in cgraph.out_portset.items():
        out_portset[pos] = [node_map1[n] for n in nodes]
    for pos, nodes in cgraph.cout_portset.items():
        cout_portset[pos] = [node_map1[n] for n in nodes]
    for pos, nodes in next_layer.cout_portset.items():
        cout_portset[pos] = [node_map2[n] for n in nodes]

    # Stitch across time by matching (x, y) at the temporal seam; add CZ edges
    try:
        prev_last_z = max(c[2] for c in cgraph.coord2node.keys())
    except ValueError:
        prev_last_z = None
    try:
        next_first_z = min(c[2] for c in next_layer.coord2node.keys())
    except ValueError:
        next_first_z = None

    seam_pairs: list[tuple[int, int]] = []
    if prev_last_z is not None and next_first_z is not None and pipes:
        prev_xy_to_node = {
            (x, y): nid
            for (x, y, z), nid in cgraph.coord2node.items()
            if z == prev_last_z
        }
        next_xy_to_node = {
            (x, y): nid
            for (x, y, z), nid in next_layer.coord2node.items()
            if z == next_first_z
        }
        for xy, u in prev_xy_to_node.items():
            v = next_xy_to_node.get(xy)
            if v is not None and u != v:
                new_graph.add_physical_edge(u, v)
                seam_pairs.append((u, v))

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
            new_schedule.update_at(anchor, new_graph)
            new_parity_acc.update_at(anchor, new_graph)
            new_flow_acc.update_at(anchor, new_graph)
        # assign back (no-op if same instances)
        cgraph.schedule = new_schedule
        cgraph.parity = new_parity_acc
        cgraph.flow = new_flow_acc
    except Exception:
        # Be tolerant: boundary queries are best-effort at this milestone
        pass

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
    # Add seam corrections (prev -> next)
    for u, v in seam_pairs:
        xflow_combined.setdefault(u, set()).add(v)

    merged_flow = FlowAccumulator(xflow=xflow_combined, zflow=zflow_combined)
    new_parity = ParityAccumulator(
        x_checks=cgraph.parity.x_checks + next_layer.parity.x_checks,
        z_checks=cgraph.parity.z_checks + next_layer.parity.z_checks,
    )

    cgraph = CompiledRHGCanvas(
        layers=new_layers,
        global_graph=new_graph,
        coord2node=new_coord2node,
        in_portset=in_portset,
        out_portset=out_portset,
        cout_portset=cout_portset,
        schedule=new_schedule,
        flow=merged_flow,
        parity=new_parity,
        z=new_z,
    )

    return cgraph
