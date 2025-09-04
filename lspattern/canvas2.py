from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set

import stim

# graphix_zx pieces
from graphix_zx.graphstate import GraphState, compose_sequentially

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.mytype import (
    FlowLocal,
    NodeIdGlobal,
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PipeCoordGlobal3D,
)
from lspattern.pipes.base import RHGPipe
from lspattern.utils import __tuple_sum, get_direction


# Flow helpers (node maps guaranteed to contain all keys)
def _remap_flow(flow: FlowLocal, node_map: dict[NodeIdLocal, NodeIdLocal]) -> FlowLocal:
    return {
        node_map[src]: {node_map[dst] for dst in dsts} for src, dsts in flow.items()
    }


def _merge_flow(a: FlowLocal, b: FlowLocal) -> FlowLocal:
    out: FlowLocal = {}
    for src, dsts in a.items():
        if not dsts:
            continue
        out.setdefault(src, set()).update(dsts)
    for src, dsts in b.items():
        if not dsts:
            continue
        out.setdefault(src, set()).update(dsts)
    return out


# Parity groups: remap list[set[int]] via node maps and concatenate.
def _remap_groups(
    groups: list[set[NodeIdLocal]],
    node_map: dict[NodeIdLocal, NodeIdLocal],
) -> list[set[NodeIdLocal]]:
    return [
        {node_map.get(n, n) for n in grp}
        for grp in groups
        if grp  # skip empty
    ]


@dataclass
class ParityAccumulator:
    # Parity check groups (local ids)
    x_checks: list[set[NodeIdLocal]] = field(default_factory=list)
    z_checks: list[set[NodeIdLocal]] = field(default_factory=list)


@dataclass
class FlowAccumulator:
    xflow: dict[NodeIdLocal, set[NodeIdLocal]] = field(default_factory=dict)
    zflow: dict[NodeIdLocal, set[NodeIdLocal]] = field(default_factory=dict)


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
        # shift qubit ids and coordinate
        block.shift_ids(by=self.qubit_count)
        block.shift_coords(patch_coord=pos)

        # Update patch, input/output ports
        self.patches.append(pos)
        self.in_portset[pos] = block.in_ports
        self.out_portset[pos] = block.out_ports

        # Update node2coord and coord2node
        for n, coord in block.node2coords.items():
            self.node2coord[n] = coord
            self.coord2node[coord] = n

    def add_pipe(
        self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D, pipe: RHGPipe
    ) -> None:
        """
        This is the function to add a pipe to the temporal layer.
        - It adds a pipe between two blocks. Its shift coordinate is given from source and direction derived from source->sink directionality
        - In addition, it connects two blocks of the same z in PatchCoordGlobal3D (do assert). Accordingly we need to modify
        - 1.

        """
        # TODO: Implement add_pipe functionality
        # shift qubit ids and coordinate
        pipe.shift_ids(by=self.qubit_count)
        pipe.shift_coords(patch_coord=source, direction=get_direction(source, sink))

        # Update pipe3d, input/output ports
        self.lines.append((source, sink))
        self.in_portset[source] = pipe.in_ports
        self.out_portset[sink] = pipe.out_ports

        # update node2coord and coord2node
        for n, coord in pipe.node_coords.items():
            self.node2coord[n] = coord
            self.coord2node[coord] = n

        # search for new RHG CZ-connection
        for node in pipe.local_graph.physical_nodes:
            for direction in DIRECTIONS3D:
                u: PhysCoordGlobal3D = node  # type: ignore[assignment]
                v: PhysCoordGlobal3D = __tuple_sum(node, direction)  # type: ignore[assignment]

        # search for new detector groups
        for node in pipe.local_graph.physical_nodes:
            for direction in DIRECTIONS3D:
                u: PhysCoordGlobal3D = node  # type: ignore[assignment]
                v: PhysCoordGlobal3D = __tuple_sum(node, direction)  # type: ignore[assignment]

    def add_blocks(self, blocks: dict[PatchCoordGlobal3D, BlockDelta]) -> None:
        for pos, block in blocks.items():
            self.add_block(pos, block)

    def add_pipes(self, pipes: dict[PipeCoordGlobal3D, BlockDelta]) -> None:
        for (start, end), pipe in pipes.items():
            self.add_pipe(start, end, pipe)


@dataclass
class ScheduleAccumulator:
    schedule: dict[int, Set[NodeIdGlobal]] = field(default_factory=dict)

    def compose_parallel(self, other: ScheduleAccumulator) -> ScheduleAccumulator:
        new_schedule = self.schedule.copy()
        for t, nodes in other.schedule.items():
            if t in new_schedule:
                new_schedule[t].update(nodes)
            else:
                new_schedule[t] = nodes
        return ScheduleAccumulator(new_schedule)

    def shift_z(self, z_by: int) -> None:
        new_schedule = {}
        for t, nodes in self.schedule.items():
            new_schedule[t + z_by] = nodes
        self.schedule = new_schedule

    def compose_sequential(
        self, late_schedule: ScheduleAccumulator
    ) -> ScheduleAccumulator:
        new_schedule = self.schedule.copy()
        late_schedule.shift_z(max(self.schedule.keys()) + 1)
        for t, nodes in late_schedule.schedule.items():
            new_schedule[t] = new_schedule.get(t, set()).union(nodes)
        return ScheduleAccumulator(new_schedule)


@dataclass
class CompiledRHGCanvas:
    layers: list[TemporalLayer]

    global_graph: Optional[GraphState] = None
    coord2node: dict[PhysCoordGlobal3D, int] = field(default_factory=dict)

    in_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    out_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    cout_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)

    schedule: ScheduleAccumulator
    flow: FlowAccumulator
    parity: ParityAccumulator
    z: int = 0

    def generate_stim_circuit(self) -> stim.Circuit:
        pass


@dataclass
class RHGCanvas2:
    # Graphは持たない
    name: str = "Blank Canvas"
    block3d: dict[PatchCoordGlobal3D, BlockDelta] = {}  # {(0,0,0): InitPlus(), ...}
    pipe3d: dict[
        PipeCoordGlobal3D, BlockDelta
    ] = {}  # {((0,0,0),(1,0,0)): Stabilize(), ((0,0,0), (0,0,1)): Measure(basis=X)}

    def add_block(self, position: PatchCoordGlobal3D, block: BlockDelta) -> None:
        self.block3d[position] = block

    def add_pipe(
        self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: BlockDelta
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

    # Create a new CompiledRHGCanvas to hold the merged result.
    new_layers = cgraph.layers + [next_layer]
    new_z = next_layer.z

    # Remap nodes
    for coord, old_nodeid in graph1.coord2node.items():
        new_cgraph.coord2node[coord] = node_map1[old_nodeid]
    for coord, old_nodeid in graph2.coord2node.items():
        new_cgraph.coord2node[coord] = node_map2[old_nodeid]

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

    # Remap existing flows via node_map1 and new layer flows via node_map2, then merge.

    # TODO: add new stabilizers, xflow, parity coming from merged boundaries
    remapped_prev_x = _remap_flow(cgraph.flow.xflow, node_map1)
    remapped_prev_z = _remap_flow(cgraph.flow.zflow, node_map1)
    remapped_next_x = _remap_flow(next_layer.flow.xflow, node_map2)
    remapped_next_z = _remap_flow(next_layer.flow.zflow, node_map2)

    merged_xflow = _merge_flow(remapped_prev_x, remapped_next_x)
    merged_zflow = _merge_flow(remapped_prev_z, remapped_next_z)

    new_xflow = FlowAccumulator(xflow=merged_xflow, zflow=merged_zflow)

    remapped_prev_x_checks = _remap_groups(cgraph.parity.x_checks, node_map1)
    remapped_prev_z_checks = _remap_groups(cgraph.parity.z_checks, node_map1)
    remapped_next_x_checks = _remap_groups(next_layer.parity.x_checks, node_map2)
    remapped_next_z_checks = _remap_groups(next_layer.parity.z_checks, node_map2)

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
