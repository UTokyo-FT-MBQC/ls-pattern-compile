from __future__ import annotations

from dataclasses import dataclass, field

from graphix_zx.graphstate import GraphState, compose_in_parallel, compose_sequentially
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
from lspattern.tiling.base import ConnectedTiling
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

    def materialize(self) -> None:
        for b in self.cubes_.values():
            b.template.to_tiling()
        for p in self.pipes_.values():
            p.template.to_tiling()

        # New path: build absolute 2D tilings from patch positions, then connect
        # TODO: ConnectedTilingが無造作にpatch同士をつなげてしまうバグが見つかった。
        # どうやら今のコードだとPatch同士がつながっているか、ただ隣同士にいるだけなのか区別できないようだ
        # ConnectedTilingにはcube_tilingsとpipe_tilingsを分けて渡すようにする
        # でConnectedTiling内で接続があればcoord2idを同一化させるように修正する
        # ancillaはこれまでと同様に全方向を探索するが、coord2idが同一のdata/ancilla間のみを接続するように修正する
        # 以上をT19.mdに仕様書として書き出し、遂行しなさい。
        tilings_abs: list = []
        dset: set[int] = set()

        for pos, b in self.cubes_.items():
            if getattr(b, "template", None) is None:
                continue
            d_val = int(getattr(b, "d", getattr(b.template, "d", 0)))
            dset.add(d_val)
            dx, dy = cube_offset_xy(d_val, pos, anchor="inner")
            tilings_abs.append(offset_tiling(b.template, dx, dy))

        for (source, sink), p in self.pipes_.items():
            if getattr(p, "template", None) is None:
                continue
            d_val = int(getattr(p, "d", getattr(p.template, "d", 0)))
            dset.add(d_val)
            direction = get_direction(source, sink)
            dx, dy = pipe_offset_xy(d_val, source, sink, direction)
            tilings_abs.append(offset_tiling(p.template, dx, dy))

        if len(dset) > 1:
            msg = "TemporalLayer.materialize: mixed code distances (d) are not supported yet"
            raise MixedCodeDistanceError(msg)

        ct = ConnectedTiling(tilings_abs, check_collisions=True)
        self.tiling_node_maps = {
            "data": dict(ct.node_maps.get("data", {})),
            "x": dict(ct.node_maps.get("x", {})),
            "z": dict(ct.node_maps.get("z", {})),
        }
        # Build GraphState for this layer from ConnectedTiling (T12 policy)
        d = next(iter(dset)) if dset else 0
        max_t = 2 * d
        g = GraphState()
        node2coord = {}
        coord2node = {}
        node2role = {}

        data2d = list(ct.data_coords)
        x2d = list(ct.x_coords)
        z2d = list(ct.z_coords)

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

        for t, cur in nodes_by_z.items():
            for (x, y), u in cur.items():
                for dx, dy, dz in DIRECTIONS3D:
                    if dz != 0:
                        continue
                    v = cur.get((x + dx, y + dy))
                    if v is not None and v > u:
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

        # 2D 連結（パッチ位置のオフセットは未考慮）
        # ct = ConnectedTiling(tilings, check_collisions=True)
        # # base/ConnectedTiling の node_maps をそのまま引き継ぐ
        # self.tiling_node_maps = {
        #     "data": dict(ct.node_maps.get("data", {})),
        #     "x": dict(ct.node_maps.get("x", {})),
        #     "z": dict(ct.node_maps.get("z", {})),
        # }
        # # TODO: ctはmaterializeするもの。なんかぁE�E��E�感じに設計したい
        # # graph, coord2node, ... = ct.materialize()

        # # TODO: input_nodeset, output_nodesetの設定をnodemapをもとに適切に実装する必要があります
        # return

        return

    def get_node_maps(self) -> dict[str, dict[tuple[int, int], int]]:
        """
        Return node_maps from ConnectedTiling (compute lazily if missing).

        Returns
        -------
            dict[str, dict[tuple[int, int], int]]: The node_maps from ConnectedTiling.
        """
        if not getattr(self, "tiling_node_maps", None):
            self.materialize()
        return self.tiling_node_maps

    def add_cube(self, pos: PatchCoordGlobal3D, cube: RHGCube) -> None:
        """Add a materialized cube to this temporal layer and place it at `pos`."""
        self.cubes_[pos] = cube
        self.patches.append(pos)
        # Shift only coordinates for placement (template keeps local ids)
        cube.shift_coords(pos)
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

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> CompiledRHGCanvas:
        new_cgraph = CompiledRHGCanvas(
            layers=self.layers.copy(),
            global_graph=self.global_graph.remap_nodes(node_map),
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
    z: int = 0

    # def generate_stim_circuit(self) -> stim.Circuit:
    #     pass

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> CompiledRHGCanvas:
        new_cgraph = CompiledRHGCanvas(
            layers=self.layers.copy(),
            global_graph=self.global_graph.remap_nodes(node_map),
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

    def materialize(self) -> RHGCube:
        "Materialize the internal template assuming that the boundaries are trimmed"

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

    def add_pipe(self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipe) -> None:
        self.pipes_[start, end] = pipe

    def to_temporal_layers(self) -> dict[int, TemporalLayer]:
        temporal_layers: dict[int, TemporalLayer] = {}
        for z in range(max(self.cubes_.keys(), key=lambda pos: pos[2])[2] + 1):
            cubes = {pos: c for pos, c in self.cubes_.items() if pos[2] == z}
            pipes = {(u, v): p for (u, v), p in self.pipes_.items() if u[2] == z and v[2] == z}

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
            z=0,
        )

        # Compose layers in increasing temporal order, wiring any cross-layer pipes
        for z in sorted(temporal_layers.keys()):
            layer = temporal_layers[z]
            # Select pipes whose start.z is the current compiled z and end.z is the next layer z
            pipes: list[RHGPipe] = [pipe for (u, v), pipe in self.pipes_.items() if u[2] == cgraph.z and v[2] == z]
            cgraph = add_temporal_layer(cgraph, layer, pipes)
        return cgraph


def to_temporal_layer(
    z: int,
    cubes: dict[PatchCoordGlobal3D, RHGCube],
    pipes: dict[PipeCoordGlobal3D, RHGPipe],
) -> TemporalLayer:
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer(z)

    layer.add_cubes(cubes)
    layer.add_pipes(pipes)

    # call materialize here
    layer.materialize()
    return layer


def add_temporal_layer(cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, pipes: list[RHGPipe]) -> CompiledRHGCanvas:
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
            z=next_layer.z,
        )
        return new_cgraph

    graph1 = cgraph.global_graph
    graph2 = next_layer.local_graph

    # Compose sequentially with node remaps
    new_graph, node_map1, node_map2 = compose_sequentially(graph1, graph2)
    # Remap registries for both sides into the composed id-space
    cgraph = cgraph.remap_nodes(node_map1)
    # Remap next_layer coord2node into composed id-space
    next_layer.coord2node = {c: node_map2.get(n, n) for c, n in next_layer.coord2node.items()}

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
    if prev_last_z is not None and next_first_z is not None:
        prev_xy_to_node = {(x, y): nid for (x, y, z), nid in cgraph.coord2node.items() if z == prev_last_z}
        next_xy_to_node = {(x, y): nid for (x, y, z), nid in next_layer.coord2node.items() if z == next_first_z}
        for xy, u in prev_xy_to_node.items():
            v = next_xy_to_node.get(xy)
            if v is not None and u != v:
                new_graph.add_physical_edge(u, v)
                seam_pairs.append((u, v))

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
