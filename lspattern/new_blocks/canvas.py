from __future__ import annotations

from dataclasses import dataclass

from lspattern.new_blocks.mytype import Coord3D
from lspattern.new_blocks.accumulator import CoordParityAccumulator, CoordFlowAccumulator, CoordScheduleAccumulator
from lspattern.new_blocks.mytype import NodeRole
from lspattern.new_blocks.block import RHGCube, RHGPipe


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
    __cooord2node: dict[Coord3D, NodeRole]
    __parity: CoordParityAccumulator
    __flow: CoordFlowAccumulator
    __schedule: CoordScheduleAccumulator

    def __init__(self, config: CanvasConfig) -> None:
        self.config = config
        self.__nodes = set()
        self.__edges = set()
        self.__parity = CoordParityAccumulator()
        self.__flow = CoordFlowAccumulator()
        self.__schedule = CoordScheduleAccumulator()

    def add_cube(self, global_pos: Coord3D, cube: RHGCube) -> None:
        pass

    def add_pipe(self, source_pos: Coord3D, target_pos: Coord3D, pipe: RHGPipe) -> None:
        pass
