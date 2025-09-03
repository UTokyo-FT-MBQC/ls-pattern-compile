from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

# graphix_zx pieces
from graphix_zx.graphstate import BaseGraphState, compose_sequentially

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.compile import compile_canvas
from lspattern.geom.tiler import PatchTiler
from lspattern.consts.consts import DIRECTIONS3D

from mytype import PhysCoordGlobal3D, NodeIdGlobal, NodeIdLocal, BlockKindstr


def __tuple_sum(l: tuple, r: tuple) -> tuple:
    assert len(l) == len(r)
    return tuple(a + b for a, b in zip(l, r))


class TemporalLayer:
    z: int
    qubit_count: int
    patches: list[BlockPosition]
    lines: list[PipePosition]

    in_portset: dict[AbstractPosition, list[int]]
    out_portset: dict[AbstractPosition, list[int]]
    cout_portset: dict[AbstractPosition, list[int]]

    in_ports: list[int]
    out_ports: list[int]
    cout_ports: list[int]

    local_graph: BaseGraphState
    node2coord: dict[int, PhysicalQubitPosition]
    coord2node: dict[PhysicalQubitPosition, int]

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

    def add_block(self, pos: BlockPosition, block: BlockDelta) -> None:
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
        self, start: PipePosition, end: PipePosition, pipe: BlockDelta
    ) -> None:
        # shift qubit ids and coordinate
        pipe.shift_ids(by=self.qubit_count)
        pipe.shift_coords(patch_coord=start)

        # Update pipe3d, input/output ports
        self.lines.append((start, end))
        self.in_portset[start] = pipe.in_ports
        self.out_portset[end] = pipe.out_ports

        # update node2coord and coord2node
        for n, coord in pipe.node_coords.items():
            self.node2coord[n] = coord
            self.coord2node[coord] = n

        # search for new RHG CZ-connection
        for node in pipe.local_graph.physical_nodes:
            for direction in DIRECTIONS3D:
                u: PhysicalQubitPosition = node
                v: PhysicalQubitPosition = __tuple_sum(node, direction)

        # search for new detector groups
        for node in pipe.local_graph.physical_nodes:
            for direction in DIRECTIONS3D:
                u: PhysicalQubitPosition = node
                v: PhysicalQubitPosition = __tuple_sum(node, direction)

    def add_blocks(self, blocks: dict[BlockPosition, BlockDelta]) -> None:
        for pos, block in blocks.items():
            self.add_block(pos, block)

    def add_pipes(self, pipes: dict[PipePosition, BlockDelta]) -> None:
        for (start, end), pipe in pipes.items():
            self.add_pipe(start, end, pipe)


@dataclass
class CompiledRHGCanvas:
    layers: list[TemporalLayer]

    global_graph: Optional[BaseGraphState] = None
    coord2node: dict[PhysicalQubitPosition, int] = field(default_factory=dict)

    scheduler: "Scheduler"
    flower: "Flower"


@dataclass
class RHGCanvas2:
    # Graphは持たない
    block3d: dict[BlockPosition, BlockDelta] = {}  # {(0,0,0): InitPlus(), ...}
    pipe3d: dict[PipePosition, BlockDelta] = (
        {}
    )  # {((0,0,0),(1,0,0)): Stabilize(), ((0,0,0), (0,0,1)): Measure(basis=X)}

    def add_block(self, position: BlockPosition, block: BlockDelta) -> None:
        self.block3d[position] = block

    def add_pipe(
        self, start: PipePosition, end: PipePosition, pipe: BlockDelta
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

            layer = to_temporal_layer(blocks, pipes)
            temporal_layers[z] = layer

        return temporal_layers

    # def compile(self) -> "CompiledRHG":
    #     temporal_layers = self.to_temporal_layers()
    #     cgraph = CompiledRHGCanvas()
    #     graph = temporal_layers[0]

    #     for i in range(1, len(temporal_layers)):
    #         graph = compose_temporal_layers(graph, temporal_layers[i])

    #     return graph


def to_temporal_layer(
    blocks: dict[BlockPosition, BlockDelta], pipes: dict[PipePosition, BlockDelta]
) -> TemporalLayer:
    # The workflow is below
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer()
    # 2) Add blocks
    layer.add_blocks(blocks)
    # 3) Add pipes
    layer.add_pipes(pipes)
    return layer


# def compose_temporal_layers(
#     graph: "ComposedGraph",
#     layer2: TemporalLayer,
#     pipes: list[PositionedPipeDelta],
# ) -> "ComposedGraph":
#     # Implement temporal composition logic here
#     out_portset = graph.out_portset
#     in_portset = layer2.in_portset

#     for pipe in pipes:
#         u, v = pipe.start, pipe.end
#         out_ports = graph.out_portset[u]
#         int_ports = layer2.in_portset[v]
#         # simply connect the two
#         graph = compose_sequentially(graph)


# def compose_sequentially(graph, graph2):
#     # compute shifts
#     qshift = max(graph.physical_qubits)
#     graph2.shift_ids(by=qshift + 1)
