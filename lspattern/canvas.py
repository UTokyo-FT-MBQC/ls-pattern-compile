from __future__ import annotations

from dataclasses import dataclass, field

# import stim
# graphix_zx pieces
from graphix_zx.graphstate import GraphState, compose_in_parallel, compose_sequentially

from lspattern.accumulator import (
    FlowAccumulator,
    ParityAccumulator,
    ScheduleAccumulator,
)
from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton
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
    block_offset_xy,
    offset_tiling,
    pipe_offset_xy,
)
from lspattern.utils import get_direction


class TemporalLayer:
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

    blocks_: dict[PatchCoordGlobal3D, RHGBlock]
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
        # ConnectedTiling/テンプレート連結用の保持領域
        self.blocks_ = {}
        self.pipes_ = {}
        self.tiling_node_maps = {}

    def materialize(self) -> None:
        # TODO: 2次元結合 -> populateの構想
        # Allow untrimmed templates for single-block layers; trimming is handled
        # earlier at the skeleton-canvas stage when applicable.
        for b in self.blocks_.values():
            b.template.to_tiling()
        for p in self.pipes_.values():
            p.template.to_tiling()

        # New path: build absolute 2D tilings from patch positions, then connect
        tilings_abs: list = []
        dset: set[int] = set()

        for pos, b in self.blocks_.items():
            if getattr(b, "template", None) is None:
                continue
            d_val = int(getattr(b, "d", getattr(b.template, "d", 0)))
            dset.add(d_val)
            dx, dy = block_offset_xy(d_val, pos, anchor="inner")
            tilings_abs.append(offset_tiling(b.template, dx, dy))

        for (source, sink), p in self.pipes_.items():
            if getattr(p, "template", None) is None:
                continue
            d_val = int(getattr(p, "d", getattr(p.template, "d", 0)))
            dset.add(d_val)
            direction = get_direction(source, sink)
            dx, dy = pipe_offset_xy(d_val, source, sink, direction)
            tilings_abs.append(offset_tiling(p.template, dx, dy))

        if not tilings_abs:
            self.tiling_node_maps = {}
            return

        if len(dset) > 1:
            raise ValueError(
                "TemporalLayer.materialize: mixed code distances (d) are not supported yet"
            )

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

        nodes_by_z = {}
        for t in range(max_t + 1):
            cur = {}
            for (x, y) in data2d:
                n = g.add_physical_node()
                node2coord[n] = (x, y, t)
                coord2node[x, y, t] = n
                node2role[n] = "data"
                cur[x, y] = n
            if t != max_t:
                if t % 2 == 0:
                    for (x, y) in x2d:
                        n = g.add_physical_node()
                        node2coord[n] = (x, y, t)
                        coord2node[x, y, t] = n
                        node2role[n] = "ancilla_x"
                        cur[x, y] = n
                else:
                    for (x, y) in z2d:
                        n = g.add_physical_node()
                        node2coord[n] = (x, y, t)
                        coord2node[x, y, t] = n
                        node2role[n] = "ancilla_z"
                        cur[x, y] = n
            nodes_by_z[t] = cur

        for t, cur in nodes_by_z.items():
            for (x, y), u in cur.items():
                for dx, dy, dz in DIRECTIONS3D:
                    if dz != 0:
                        continue
                    v = cur.get((x + dx, y + dy))
                    if v is not None and v > u:
                        try:
                            g.add_physical_edge(u, v)
                        except Exception:
                            pass

        for t in range(1, max_t + 1):
            cur = nodes_by_z[t]
            prev = nodes_by_z[t - 1]
            for xy, u in cur.items():
                v = prev.get(xy)
                if v is not None:
                    try:
                        g.add_physical_edge(u, v)
                    except Exception:
                        pass

        self.local_graph = g
        self.node2coord = node2coord
        self.coord2node = coord2node
        self.node2role = node2role
        self.qubit_count = len(g.physical_nodes)
        return

        # 2D 連結（パッチ位置のオフセットは未考慮）
        tilings = [
            *(b.template for b in self.blocks_.values() if getattr(b, "template", None)),
            *(p.template for p in self.pipes_.values() if getattr(p, "template", None)),
        ]
        if not tilings:
            self.tiling_node_maps = {}
            return
        # TODO: fix Bug
        # Before this line we assume that template Coord2D are correctly shifted according to block/pipes positions
        # Need to check
        # tilings does not have position information, this is absolutely wrong.
        ct = ConnectedTiling(tilings, check_collisions=True)
        # base/ConnectedTiling の node_maps をそのまま公開
        self.tiling_node_maps = {
            "data": dict(ct.node_maps.get("data", {})),
            "x": dict(ct.node_maps.get("x", {})),
            "z": dict(ct.node_maps.get("z", {})),
        }
        # TODO: ctはmaterializeするもの。なんかいい感じに設計したい
        # graph, coord2node, ... = ct.materialize()

        # TODO: input_nodeset, output_nodesetの設定をnodemapをもとに適切に実装する必要がある
        return

    def get_node_maps(self) -> dict[str, dict[tuple[int, int], int]]:
        """ConnectedTiling 由来の node_maps を返す（必要なら遅延計算）。"""
        if not getattr(self, "tiling_node_maps", None):
            self.materialize()
        return self.tiling_node_maps

    def get_connected_tiling(self, anchor: str = "inner") -> ConnectedTiling:
        """ブロック/パイプを絶対2D座標に再配置して ConnectedTiling を返す。

        - materialize() と同等のオフセット計算を一時的に行う（キャッシュは任意）
        - d が混在する場合は ValueError を送出
        """
        # to_tiling を先に呼び出して内部座標を確実に持たせる
        for b in self.blocks_.values():
            if getattr(b, "template", None) is not None:
                b.template.to_tiling()
        for p in self.pipes_.values():
            if getattr(p, "template", None) is not None:
                p.template.to_tiling()

        tilings_abs: list = []
        dset: set[int] = set()

        for pos, b in self.blocks_.items():
            if getattr(b, "template", None) is None:
                continue
            d_val = int(getattr(b, "d", getattr(b.template, "d", 0)))
            dset.add(d_val)
            dx, dy = block_offset_xy(d_val, pos, anchor=anchor)
            tilings_abs.append(offset_tiling(b.template, dx, dy))

        for (source, sink), p in self.pipes_.items():
            if getattr(p, "template", None) is None:
                continue
            d_val = int(getattr(p, "d", getattr(p.template, "d", 0)))
            dset.add(d_val)
            direction = get_direction(source, sink)
            dx, dy = pipe_offset_xy(d_val, source, sink, direction)
            tilings_abs.append(offset_tiling(p.template, dx, dy))

        if not tilings_abs:
            return ConnectedTiling([])
        if len(dset) > 1:
            raise ValueError("Mixed code distances (d) are not supported in a single layer")

        return ConnectedTiling(tilings_abs, check_collisions=True)

    def add_block(self, pos: PatchCoordGlobal3D, block: RHGBlockSkeleton) -> None:
        # Accept either a pre-materialized block or a skeleton.
        if isinstance(block, RHGBlockSkeleton):
            block = block.to_block()
        # Require a template and a materialized local graph
        if getattr(block, "template", None) is None:
            raise ValueError(
                "Block has no template; set block.template before add_block()."
            )
        # TODO: Materialize 方針について再考する
        block.materialize()
        if getattr(block, "graph_local", None) is None:
            raise ValueError("Block.materialize() did not produce graph_local.")
        if not getattr(block, "node2coord", {}):
            raise ValueError("Block has empty node2coord after materialize().")

        # shift coordinates for placement (ids remain local to block graph)
        block.shift_coords(pos)

        # ConnectedTiling 用に保持
        self.blocks_[pos] = block

        # Update patch registry (ports will be set after composition)
        self.patches.append(pos)
        # T12: Do not compose block graphs here; layer graph is built in materialize()
        return

        # Compose this block's graph in parallel with the existing layer graph
        g2: GraphState = getattr(block, "graph_local", None)
        node_map1: dict[int, int] = {}
        node_map2: dict[int, int] = {}
        if getattr(self, "local_graph", None) is None:
            # First block in the layer: adopt directly; identity node_map2
            self.local_graph = g2
            node_map2 = {n: n for n in g2.physical_nodes}
        else:
            # Compose in parallel and adopt the new graph
            g1: GraphState = self.local_graph
            g_new, node_map1, node_map2 = compose_in_parallel(g1, g2)
            self.local_graph = g_new

            # Remap existing registries by node_map1
            if node_map1:
                self.node2coord = {
                    node_map1.get(n, n): c for n, c in self.node2coord.items()
                }
                self.coord2node = {
                    c: node_map1.get(n, n) for c, n in self.coord2node.items()
                }
                self.node2role = {
                    node_map1.get(n, n): r for n, r in self.node2role.items()
                }

                # Remap existing port sets and flat lists
                for p, nodes in list(self.in_portset.items()):
                    self.in_portset[p] = [node_map1.get(n, n) for n in nodes]
                for p, nodes in list(self.out_portset.items()):
                    self.out_portset[p] = [node_map1.get(n, n) for n in nodes]
                if hasattr(self, "cout_portset") and isinstance(
                    self.cout_portset, dict
                ):
                    for p, nodes in list(self.cout_portset.items()):
                        self.cout_portset[p] = [node_map1.get(n, n) for n in nodes]
                self.in_ports = [node_map1.get(n, n) for n in self.in_ports]
                self.out_ports = [node_map1.get(n, n) for n in self.out_ports]

        # Set in/out ports for this block using node_map2
        self.in_portset[pos] = [
            node_map2[n] for n in getattr(block, "in_ports", []) if n in node_map2
        ]
        self.out_portset[pos] = [
            node_map2[n] for n in getattr(block, "out_ports", []) if n in node_map2
        ]
        self.in_ports.extend(self.in_portset.get(pos, []))
        self.out_ports.extend(self.out_portset.get(pos, []))

        # Add the new block geometry via node_map2
        for old_n, coord in (getattr(block, "node2coord", {}) or {}).items():
            nn = node_map2.get(old_n)
            if nn is None:
                continue
            # Detect coordinate collisions with already-placed blocks/nodes
            if coord in self.coord2node:
                existing_nn = self.coord2node[coord]
                raise ValueError(
                    f"Coordinate collision: {coord} already occupied by node {existing_nn} "
                    f"when adding block at {pos}."
                )
            self.node2coord[nn] = coord
            self.coord2node[coord] = nn

        # Record roles if provided by the block (for visualization)
        for old_n, role in (getattr(block, "node2role", {}) or {}).items():
            nn = node_map2.get(old_n)
            if nn is not None:
                self.node2role[nn] = role

        self.qubit_count = len(self.local_graph.physical_nodes)

    def add_pipe(
        self,
        source: PatchCoordGlobal3D,
        sink: PatchCoordGlobal3D,
        spatial_pipe: RHGPipe,
    ) -> None:
        """
        This is the function to add a pipe to the temporal layer.
        - It adds a pipe between two blocks. Its shift coordinate is given from source and direction derived from source->sink directionality
        - In addition, it connects two blocks of the same z in PatchCoordGlobal3D (do assert). Accordingly we need to modify
        - 1.

        """
        # Shift pipe-local ids (defensive; concrete pipes may override)
        spatial_pipe.shift_ids(by=self.qubit_count)
        # Position the pipe according to source->sink direction (seam anchoring)
        spatial_pipe.shift_coords(
            source, direction=get_direction(source, sink), sink=sink
        )

        # ConnectedTiling 用に保持
        self.pipes_[source, sink] = spatial_pipe

        # Compose the pipe's local graph into the layer graph (if any)
        node_map1: dict[int, int] = {}
        node_map2: dict[int, int] = {}
        g2 = spatial_pipe.graph_local

        if g2 is not None:
            if getattr(self, "local_graph", None) is None:
                # First ingestion into the layer
                self.local_graph = g2
                node_map2 = {n: n for n in g2.physical_nodes}
            else:
                g1 = self.local_graph
                g_new, node_map1, node_map2 = compose_in_parallel(g1, g2)
                self.local_graph = g_new

                # Remap existing registries for prior nodes
                if node_map1:
                    self.node2coord = {
                        node_map1.get(n, n): c for n, c in self.node2coord.items()
                    }
                    self.coord2node = {
                        c: node_map1.get(n, n) for c, n in self.coord2node.items()
                    }
                    self.node2role = {
                        node_map1.get(n, n): r for n, r in self.node2role.items()
                    }
                    # Remap existing portsets and flat lists
                    for p, nodes in list(self.in_portset.items()):
                        self.in_portset[p] = [node_map1.get(n, n) for n in nodes]
                    for p, nodes in list(self.out_portset.items()):
                        self.out_portset[p] = [node_map1.get(n, n) for n in nodes]
                    self.in_ports = [node_map1.get(n, n) for n in self.in_ports]
                    self.out_ports = [node_map1.get(n, n) for n in self.out_ports]

        # Record the pipe endpoints for visualization
        self.lines.append((source, sink))

        # Register in/out ports (remapped if composed)
        if getattr(spatial_pipe, "in_ports", None):
            self.in_portset[source] = [
                node_map2.get(n, n) for n in spatial_pipe.in_ports
            ]
            self.in_ports.extend(self.in_portset[source])
        if getattr(spatial_pipe, "out_ports", None):
            self.out_portset[sink] = [
                node_map2.get(n, n) for n in spatial_pipe.out_ports
            ]
            self.out_ports.extend(self.out_portset[sink])

        # Update coord registries for the newly added pipe nodes
        for old_n, coord in (getattr(spatial_pipe, "node_coords", {}) or {}).items():
            nn = node_map2.get(old_n, old_n)
            # Collision check against existing coordinates
            if coord in self.coord2node and self.coord2node[coord] != nn:
                raise ValueError(
                    f"Coordinate collision: {coord} already occupied by node {self.coord2node[coord]} "
                    f"when adding pipe {source}->{sink}."
                )
            self.node2coord[nn] = coord
            self.coord2node[coord] = nn

        # Stitch RHG seam edges: connect new nodes to any neighbor-at-offset nodes
        # present in the layer at the same z slice.
        if getattr(self, "local_graph", None) is not None and spatial_pipe.node_coords:
            for old_n, coord in spatial_pipe.node_coords.items():
                nn = node_map2.get(old_n, old_n)
                x, y, z = coord
                for dx, dy, dz in DIRECTIONS3D:
                    if dz != 0:
                        continue
                    nbr_coord = (x + dx, y + dy, z)
                    nbr = self.coord2node.get(nbr_coord)
                    if nbr is None or nbr == nn:
                        continue
                    try:
                        self.local_graph.add_physical_edge(nn, nbr)
                    except Exception:
                        # Ignore duplicate/self edges or invalid connections
                        pass

        # Update qubit count if we now have a layer graph
        if getattr(self, "local_graph", None) is not None:
            self.qubit_count = len(self.local_graph.physical_nodes)

    def add_blocks(self, blocks: dict[PatchCoordGlobal3D, RHGBlockSkeleton]) -> None:
        for pos, block in blocks.items():
            self.add_block(pos, block)

    def add_pipes(self, pipes: dict[PipeCoordGlobal3D, RHGBlockSkeleton]) -> None:
        for (start, end), pipe in pipes.items():
            self.add_pipe(start, end, pipe)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> TemporalLayer:
        """Return a copy of this layer with all node ids remapped by `node_map`."""
        new_layer = TemporalLayer(self.z)
        new_layer.qubit_count = self.qubit_count
        new_layer.patches = self.patches.copy()
        new_layer.lines = self.lines.copy()

        new_layer.in_portset = {
            pos: [node_map.get(n, n) for n in nodes]
            for pos, nodes in self.in_portset.items()
        }
        new_layer.out_portset = {
            pos: [node_map.get(n, n) for n in nodes]
            for pos, nodes in self.out_portset.items()
        }
        new_layer.in_ports = [node_map.get(n, n) for n in self.in_ports]
        new_layer.out_ports = [node_map.get(n, n) for n in self.out_ports]

        if self.local_graph is not None:
            new_layer.local_graph = self.local_graph.remap_nodes(node_map)
        new_layer.node2coord = {
            node_map.get(n, n): c for n, c in self.node2coord.items()
        }
        new_layer.coord2node = {
            c: node_map.get(n, n) for c, n in self.coord2node.items()
        }
        new_layer.node2role = {node_map.get(n, n): r for n, r in self.node2role.items()}

        new_layer.schedule = self.schedule.remap_nodes(node_map)
        new_layer.flow = self.flow.remap_nodes(node_map)
        new_layer.parity = self.parity.remap_nodes(node_map)
        return new_layer


@dataclass
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
    blocks_: dict[PatchCoordGlobal3D, RHGBlockSkeleton] = field(default_factory=dict)
    # {((0,0,0),(1,0,0)): StabilizeSkeleton(), ((0,0,0), (0,0,1)): MeasureSkeleton(basis=X)}
    pipes_: dict[PipeCoordGlobal3D, RHGPipeSkeleton] = field(default_factory=dict)

    def materialize(self) -> RHGBlock:
        "Materialize the internal template assuming that the boundaries are trimmed"

    def add_block(self, position: PatchCoordGlobal3D, block: RHGBlockSkeleton) -> None:
        self.blocks_[position] = block

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
        # Iterate spatial pipes and trim facing boundaries of adjacent blocks.
        # Pipes are keyed by ((x,y,z),(x,y,z)); detect spatial adjacency by dx/dy.
        for (u, v), _pipe in list(self.pipes_.items()):
            ux, uy, uz = u
            vx, vy, vz = v
            # Temporal pipes are not handled here
            if uz != vz:
                continue

            dx, dy = vx - ux, vy - uy

            left = self.blocks_.get(u)
            right = self.blocks_.get(v)

            if dx == 1 and dy == 0:
                # X+ direction: trim RIGHT of left block and LEFT of right block
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

        trimmed_blocks_skeleton = self.blocks_.copy()
        trimmed_pipes_skeleton = self.pipes_.copy()

        # Block は to_block() があればそれを、無ければ materialize() を使う
        blocks_ = {}
        for pos, blk in trimmed_blocks_skeleton.items():
            to_block = getattr(blk, "to_block", None)
            if callable(to_block):
                blocks_[pos] = to_block()
            else:
                blocks_[pos] = blk.materialize()
        # Pipe は skeleton でも concrete でも受け付ける
        pipes_ = {}
        for (start, end), p in trimmed_pipes_skeleton.items():
            pipe_obj = p
            to_pipe = getattr(p, "to_pipe", None)
            if callable(to_pipe):
                try:
                    pipe_obj = to_pipe(start, end)
                except TypeError:
                    pipe_obj = to_pipe()
            pipes_[start, end] = pipe_obj

        canvas = RHGCanvas(name=self.name, blocks_=blocks_, pipes_=pipes_)
        return canvas


@dataclass
class RHGCanvas:  # TopologicalComputationGraph in tqec
    name: str = "Blank Canvas"
    # blocks pipesは最後までmateiralizeされることはない。してもいいけど。tilingはmaterializeできる
    blocks_: dict[PatchCoordGlobal3D, RHGBlock] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    layers: list[TemporalLayer] | None = None

    def add_block(self, position: PatchCoordGlobal3D, block: RHGBlock) -> None:
        self.blocks_[position] = block

    def add_pipe(
        self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipe
    ) -> None:
        self.pipes_[start, end] = pipe

    def to_temporal_layers(self) -> dict[int, TemporalLayer]:
        temporal_layers: dict[int, TemporalLayer] = {}
        for z in range(max(self.blocks_.keys(), key=lambda pos: pos[2])[2] + 1):
            blocks = {pos: blk for pos, blk in self.blocks_.items() if pos[2] == z}
            pipes = {
                (u, v): p
                for (u, v), p in self.pipes_.items()
                if u[2] == z and v[2] == z
            }

            layer = to_temporal_layer(z, blocks, pipes)
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
            pipes: list[RHGPipe] = [
                pipe
                for (u, v), pipe in self.pipes_.items()
                if u[2] == cgraph.z and v[2] == z
            ]
            cgraph = add_temporal_layer(cgraph, layer, pipes)
        return cgraph


def to_temporal_layer(
    z: int,
    blocks: dict[PatchCoordGlobal3D, RHGBlockSkeleton],
    pipes: dict[PipeCoordGlobal3D, RHGBlockSkeleton],
) -> TemporalLayer:
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer(z)

    layer.add_blocks(blocks)
    layer.add_pipes(pipes)

    # call materialize here
    layer.materialize()
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
    from its blocks/pipes. If it's None, we keep cgraph unchanged.
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
    next_layer = next_layer.remap_nodes(node_map2)

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
                try:
                    new_graph.add_physical_edge(u, v)
                except Exception:
                    pass
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
