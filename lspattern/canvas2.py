from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set

# import stim
# graphix_zx pieces
from graphix_zx.graphstate import GraphState, compose_in_parallel, compose_sequentially
from lspattern.accumulator import (
    FlowAccumulator,
    ParityAccumulator,
    ScheduleAccumulator,
)
from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.mytype import (
    NodeIdGlobal,
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PipeCoordGlobal3D,
)
from lspattern.pipes.base import RHGPipe
from lspattern.utils import __tuple_sum, get_direction


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

    def __init__(self, z: int):
        self.z = z
        self.qubit_count = 0
        self.patches = []
        self.lines = []
        self.in_portset = {}
        self.out_portset = {}
        self.in_ports = []
        self.out_ports = []
        self.local_graph = None
        self.node2coord = {}
        self.coord2node = {}

    def add_block(self, pos: PatchCoordGlobal3D, block: RHGBlock) -> None:
        # Require a template and a materialized local graph
        if getattr(block, "template", None) is None:
            raise ValueError(
                "Block has no template; set block.template before add_block()."
            )
        block.materialize()
        if getattr(block, "graph_local", None) is None:
            raise ValueError("Block.materialize() did not produce graph_local.")
        if not getattr(block, "node2coord", {}):
            raise ValueError("Block has empty node2coord after materialize().")

        # shift coordinates for placement (ids remain local to block graph)
        block.shift_coords(pos)

        # Update patch, input/output ports
        self.patches.append(pos)
        self.in_portset[pos] = list(getattr(block, "in_ports", []))
        self.out_portset[pos] = list(getattr(block, "out_ports", []))

        # (Defer geometry mapping until after graph composition)

        # Compose this block's graph by cloning it into the layer graph
        g2: GraphState = getattr(block, "graph_local", None)
        if getattr(self, "local_graph", None) is None:
            self.local_graph = GraphState()
        g_dst: GraphState = self.local_graph

        node_map2: dict[int, int] = {}
        # Clone nodes and meas bases
        for n in sorted(g2.physical_nodes):
            nn = g_dst.add_physical_node()
            node_map2[n] = nn
            mb = g2.meas_bases.get(n)
            if mb is not None:
                g_dst.assign_meas_basis(nn, mb)
        # Clone edges
        for u, v in g2.physical_edges:
            g_dst.add_physical_edge(node_map2[u], node_map2[v])
        # Carry over IO markers
        for in_n in g2.input_node_indices.keys():
            g_dst.register_input(node_map2[in_n])
        for out_n, qidx in g2.output_node_indices.items():
            g_dst.register_output(node_map2[out_n], qidx)

        # Add the new block geometry via node_map2
        for old_n, coord in (getattr(block, "node2coord", {}) or {}).items():
            nn = node_map2.get(old_n)
            if nn is None:
                continue
            self.node2coord[nn] = coord
            self.coord2node[coord] = nn

        if pos in self.in_portset:
            self.in_portset[pos] = [node_map2[n] for n in self.in_portset[pos] if n in node_map2]
        if pos in self.out_portset:
            self.out_portset[pos] = [node_map2[n] for n in self.out_portset[pos] if n in node_map2]

        self.in_ports.extend(self.in_portset.get(pos, []))
        self.out_ports.extend(self.out_portset.get(pos, []))

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
        # TODO: Implement add_pipe functionality
        # shift qubit ids and coordinate
        spatial_pipe.shift_ids(by=self.qubit_count)
        spatial_pipe.shift_coords(
            patch_coord=source, direction=get_direction(source, sink)
        )

        # Update pipe3d, input/output ports
        self.lines.append((source, sink))
        self.in_portset[source] = spatial_pipe.in_ports
        self.out_portset[sink] = spatial_pipe.out_ports

        # update node2coord and coord2node
        for n, coord in spatial_pipe.node_coords.items():
            self.node2coord[n] = coord
            self.coord2node[coord] = n

        # search for new RHG CZ-connection
        for node in spatial_pipe.local_graph.physical_nodes:
            for direction in DIRECTIONS3D:
                u: PhysCoordGlobal3D = node  # type: ignore[assignment]
                v: PhysCoordGlobal3D = __tuple_sum(node, direction)  # type: ignore[assignment]

        # search for new detector groups
        for node in spatial_pipe.local_graph.physical_nodes:
            for direction in DIRECTIONS3D:
                u: PhysCoordGlobal3D = node  # type: ignore[assignment]
                v: PhysCoordGlobal3D = __tuple_sum(node, direction)  # type: ignore[assignment]

    def add_blocks(self, blocks: dict[PatchCoordGlobal3D, BlockDelta]) -> None:
        for pos, block in blocks.items():
            self.add_block(pos, block)

    def add_pipes(self, pipes: dict[PipeCoordGlobal3D, BlockDelta]) -> None:
        for (start, end), pipe in pipes.items():
            self.add_pipe(start, end, pipe)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> TemporalLayer:
        new_layer = TemporalLayer(self.z)
        new_layer.qubit_count = self.qubit_count
        new_layer.patches = self.patches.copy()
        new_layer.lines = self.lines.copy()

        new_layer.in_portset = {
            pos: [node_map[n] for n in nodes] for pos, nodes in self.in_portset.items()
        }
        new_layer.out_portset = {
            pos: [node_map[n] for n in nodes] for pos, nodes in self.out_portset.items()
        }

        new_layer.in_ports = [node_map[n] for n in self.in_ports]
        new_layer.out_ports = [node_map[n] for n in self.out_ports]

        new_layer.local_graph = self.local_graph.remap_nodes(node_map)
        new_layer.node2coord = {
            node_map[n]: coord for n, coord in self.node2coord.items()
        }
        new_layer.coord2node = {
            coord: node_map[n] for n, coord in self.node2coord.items()
        }

        # TODO: implement remap_nodes for schedule, flow, parity
        new_layer.schedule = self.schedule.remap_nodes(node_map)
        new_layer.flow = self.flow.remap_nodes(node_map)
        new_layer.parity = self.parity.remap_nodes(node_map)

        return new_layer

    def remap_nodes(
        self, node_map: dict[NodeIdLocal, NodeIdLocal]
    ) -> "ScheduleAccumulator":
        if not self.schedule:
            return ScheduleAccumulator()
        # Times are unchanged; remap node ids per time slot.
        remapped: dict[int, Set[NodeIdGlobal]] = {}
        for t, nodes in self.schedule.items():
            # Use get for robustness if node_map is partial in some contexts
            remapped[t] = {node_map.get(n, n) for n in nodes}
        return ScheduleAccumulator(remapped)


@dataclass
class CompiledRHGCanvas:
    layers: list[TemporalLayer]

    global_graph: Optional[GraphState] = None
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
class RHGCanvas2:
    # Graphは持たない
    name: str = "Blank Canvas"
    # {(0,0,0): InitPlus(), ..}
    block3d: dict[PatchCoordGlobal3D, RHGBlock] = field(default_factory=dict)
    # {((0,0,0),(1,0,0)): Stabilize(), ((0,0,0), (0,0,1)): Measure(basis=X)}
    pipe3d: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    layers: Optional[list[TemporalLayer]] = None

    def add_block(self, position: PatchCoordGlobal3D, block: RHGBlock) -> None:
        self.block3d[position] = block

    def add_pipe(
        self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipe
    ) -> None:
        self.pipe3d[(start, end)] = pipe

    def to_temporal_layers(self) -> dict[int, TemporalLayer]:
        temporal_layers: dict[int, TemporalLayer] = {}
        for z in range(max(self.block3d.keys(), key=lambda pos: pos[2])[2] + 1):
            blocks = {pos: blk for pos, blk in self.block3d.items() if pos[2] == z}
            pipes = {
                (u, v): p
                for (u, v), p in self.pipe3d.items()
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
                for (u, v), pipe in self.pipe3d.items()
                if u[2] == cgraph.z and v[2] == z
            ]
            cgraph = add_temporal_layer(cgraph, layer, pipes)
        return cgraph


def to_temporal_layer(
    z: int,
    blocks: dict[PatchCoordGlobal3D, RHGBlock],
    pipes: dict[PipeCoordGlobal3D, RHGPipe],
) -> TemporalLayer:
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer(z)
    # 2) Add blocks
    layer.add_blocks(blocks)
    # 3) Add pipes
    layer.add_pipes(pipes)
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
            scheduler=next_layer.scheduler,
            z=next_layer.z,
        )
        return new_cgraph

    graph1 = cgraph.global_graph
    graph2 = next_layer.local_graph

    # Compose sequentially with node remaps
    new_graph, node_map1, node_map2 = compose_sequentially(graph1, graph2)
    # remap nodes for cgraph and next_layer
    cgraph = cgraph.remap_nodes(node_map1)
    next_layer = next_layer.remap_nodes(node_map2)

    # Create a new CompiledRHGCanvas to hold the merged result.
    new_layers = cgraph.layers + [next_layer]
    new_z = next_layer.z

    # Remap nodes
    new_coord2node: dict[PhysCoordGlobal3D, int] = {}
    for coord, old_nodeid in cgraph.coord2node.items():
        new_coord2node[coord] = node_map1[old_nodeid]
    for coord, old_nodeid in next_layer.coord2node.items():
        new_coord2node[coord] = node_map2[old_nodeid]

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

    # scheduler, xflow, parity
    new_schedule = cgraph.schedule.compose_sequential(next_layer.schedule)
    new_xflow = FlowAccumulator(xflow=merged_xflow, zflow=merged_zflow)
    new_parity = ParityAccumulator(
        x_checks=remapped_prev_x_checks + remapped_next_x_checks,
        z_checks=remapped_prev_z_checks + remapped_next_z_checks,
    )
    cgraph = CompiledRHGCanvas(
        layers=new_layers,
        global_graph=new_graph,
        coord2node=new_cgraph.coord2node,
        in_portset=in_portset,
        out_portset=out_portset,
        cout_portset=cout_portset,
        scheduler=new_schedule,
        xflow=new_xflow,
        parity=new_parity,
        z=new_z,
    )

    return cgraph
