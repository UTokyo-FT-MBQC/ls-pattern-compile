from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union
import stim


# graphix_zx pieces
from graphix_zx.graphstate import BaseGraphState, compose_sequentially, GraphState
from graphix_zx.graphstate import compose_in_parallel

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.compile import compile_canvas
from lspattern.geom.tiler import PatchTiler
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.utils import get_direction

from lspattern.mytype import (
    PhysCoordGlobal3D,
    PatchCoordGlobal3D,
    PipeCoordGlobal3D,
    NodeIdGlobal,
    NodeIdLocal,
    BlockKindstr,
)
from lspattern.pipes.base import RHGPipe
from lspattern.utils import __tuple_sum
from mytype import *


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
        for n, coord in block.node_coords.items():
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
class Scheduler:
    schedule: dict[int, Set[NodeIdGlobal]] = field(default_factory=dict)

    def compose_parallel(self, other: Scheduler) -> Scheduler:
        new_schedule = self.schedule.copy()
        for t, nodes in other.schedule.items():
            if t in new_schedule:
                new_schedule[t].update(nodes)
            else:
                new_schedule[t] = nodes
        return Scheduler(new_schedule)

    def shift_z(self, z_by: int) -> None:
        new_schedule = {}
        for t, nodes in self.schedule.items():
            new_schedule[t + z_by] = nodes
        self.schedule = new_schedule

    def compose_sequential(self, late_schedule: Scheduler) -> Scheduler:
        new_schedule = self.schedule.copy()
        late_schedule.shift_z(max(self.schedule.keys()) + 1)
        for t, nodes in late_schedule.schedule.items():
            new_schedule[t] = new_schedule.get(t, set()).union(nodes)
        return Scheduler(new_schedule)


@dataclass
class CompiledRHGCanvas:
    layers: list[TemporalLayer]

    global_graph: Optional[GraphState] = None
    coord2node: dict[PhysCoordGlobal3D, int] = field(default_factory=dict)

    in_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    out_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)
    cout_portset: dict[PatchCoordGlobal3D, list[int]] = field(default_factory=dict)

    scheduler: Scheduler
    flower: "Flower"
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
        cgraph = CompiledRHGCanvas()

        for layer in temporal_layers:
            cgraph = add_temporal_layer(cgraph, layer, pipes)
        return cgraph


def to_temporal_layer(
    z: int,
    blocks: dict[PatchCoordGlobal3D, RHGBlock],
    pipes: dict[PipeCoordGlobal3D, RHGPipe],
) -> TemporalLayer:
    # The workflow is below
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
    # Implement temporal composition logic here
    out_portset = cgraph.out_portset
    in_portset = next_layer.in_portset

    for pipe in pipes:
        source, sink = pipe.source, pipe.sink
        out_ports = out_portset[source]
        in_ports = in_portset[sink]
        # simply connect the two
        # note that in this case pipe has NO physical qubits. Simply connect via CZ and extra stabilizer network
        graph = compose_sequentially(graph)
