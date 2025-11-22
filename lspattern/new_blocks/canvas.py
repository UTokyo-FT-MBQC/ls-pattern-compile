from __future__ import annotations

from dataclasses import dataclass
from graphqomb.common import Axis

from lspattern.new_blocks.mytype import Coord3D
from lspattern.new_blocks.accumulator import CoordParityAccumulator, CoordFlowAccumulator, CoordScheduleAccumulator
from lspattern.new_blocks.mytype import NodeRole
from lspattern.new_blocks.block import Node, Edge
from lspattern.new_blocks.layout.rotated_surface_code import (
    rotated_surface_code_layout,
    rotated_surface_code_pipe_layout,
    ANCILLA_EDGE,
)  # this will be dynamically loaded based on config
from lspattern.new_blocks.loader import BlockConfig


@dataclass
class CanvasConfig:
    """Configuration for a canvas.

    Attributes
    ----------
    name : str
        Name of the canvas.
    description : str
        Description of the canvas.
    d : int
        Code distance.
    tiling : str
        Tiling type (e.g., "rotated_surface_code").
    """

    name: str
    description: str
    d: int
    tiling: str


class Canvas:
    config: CanvasConfig
    __nodes: set[Coord3D]
    __edges: set[tuple[Coord3D, Coord3D]]
    __pauli_axes: dict[Coord3D, Axis]
    __coord2role: dict[Coord3D, NodeRole]
    __parity: CoordParityAccumulator
    __flow: CoordFlowAccumulator
    __schedule: CoordScheduleAccumulator

    def __init__(self, config: CanvasConfig) -> None:
        self.config = config
        self.__nodes = set()
        self.__edges = set()
        self.__pauli_axes = {}
        self.__coord2role = {}
        self.__parity = CoordParityAccumulator()
        self.__flow = CoordFlowAccumulator()
        self.__schedule = CoordScheduleAccumulator()

    @property
    def nodes(self) -> set[Coord3D]:
        return set(self.__nodes)

    @property
    def edges(self) -> set[tuple[Coord3D, Coord3D]]:
        return set(self.__edges)

    @property
    def coord2role(self) -> dict[Coord3D, NodeRole]:
        return dict(self.__coord2role)

    @property
    def pauli_axes(self) -> dict[Coord3D, Axis]:
        return dict(self.__pauli_axes)

    def add_cube(self, global_pos: Coord3D, block_config: BlockConfig) -> None:
        data2d, ancilla_x2d, ancilla_z2d = rotated_surface_code_layout(self.config.d, global_pos, block_config.boundary)

        offset_z = global_pos.z * 2 * self.config.d

        # Build graph layer by layer
        for layer_idx, layer_cfg in enumerate(block_config):
            z = offset_z + layer_idx * 2

            if layer_cfg.layer1.basis is not None:
                for x, y in data2d:
                    self.__nodes.add(Coord3D(x, y, z))
                    self.__coord2role[Coord3D(x, y, z)] = NodeRole.DATA
                    self.__pauli_axes[Coord3D(x, y, z)] = layer_cfg.layer1.basis
                    # temporal edge
                    if Coord3D(x, y, z - 1) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z - 1), Coord3D(x, y, z)))
                        self.__flow.add_flow(Coord3D(x, y, z - 1), Coord3D(x, y, z))

            if layer_cfg.layer2.basis is not None:
                for x, y in data2d:
                    self.__nodes.add(Coord3D(x, y, z + 1))
                    self.__coord2role[Coord3D(x, y, z + 1)] = NodeRole.DATA
                    self.__pauli_axes[Coord3D(x, y, z + 1)] = layer_cfg.layer2.basis
                    # temporal edge
                    if Coord3D(x, y, z) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z), Coord3D(x, y, z + 1)))
                        self.__flow.add_flow(Coord3D(x, y, z), Coord3D(x, y, z + 1))

            if layer_cfg.layer1.ancilla:
                for x, y in ancilla_z2d:
                    if layer_cfg.layer1.ancilla:
                        self.__nodes.add(Coord3D(x, y, z))
                        self.__coord2role[Coord3D(x, y, z)] = NodeRole.ANCILLA_Z
                        self.__pauli_axes[Coord3D(x, y, z)] = Axis.X
                        for dx, dy in ANCILLA_EDGE:
                            self.__edges.add((Coord3D(x, y, z), Coord3D(x + dx, y + dy, z)))

            if layer_cfg.layer2.ancilla:
                for x, y in ancilla_x2d:
                    self.__nodes.add(Coord3D(x, y, z + 1))
                    self.__coord2role[Coord3D(x, y, z + 1)] = NodeRole.ANCILLA_X
                    self.__pauli_axes[Coord3D(x, y, z + 1)] = Axis.X
                    for dx, dy in ANCILLA_EDGE:
                        self.__edges.add((Coord3D(x, y, z + 1), Coord3D(x + dx, y + dy, z + 1)))

    def add_pipe(self, global_edge: tuple[Coord3D, Coord3D], block_config: BlockConfig) -> None:
        pass
