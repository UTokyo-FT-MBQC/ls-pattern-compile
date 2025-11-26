from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from graphqomb.common import Axis

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.accumulator import CoordFlowAccumulator, CoordParityAccumulator, CoordScheduleAccumulator
from lspattern.new_blocks.layout.rotated_surface_code import (
    ANCILLA_EDGE,
    rotated_surface_code_layout,
)  # this will be dynamically loaded based on config
from lspattern.new_blocks.loader import BlockConfig
from lspattern.new_blocks.mytype import DIRECTION2D, Coord2D, Coord3D, NodeRole


class Boundary(NamedTuple):
    top: EdgeSpecValue
    bottom: EdgeSpecValue
    left: EdgeSpecValue
    right: EdgeSpecValue


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


@dataclass
class BoundaryGraph:
    """Graph representing boundary conditions."""

    boundary_map: dict[Coord3D, Boundary]

    def add_boundary(self, coord: Coord3D, boundary: Boundary) -> None:
        self.boundary_map[coord] = boundary

    def check_bulk_init(self, coord: Coord3D) -> bool:
        """Check if the coordinate is in the bulk for initialization."""
        return Coord3D(coord.x, coord.y, coord.z - 1) not in self.boundary_map

    def check_boundary_init(self, coord: Coord3D) -> set[DIRECTION2D]:
        prev_boundary = self.boundary_map.get(Coord3D(coord.x, coord.y, coord.z - 1), None)
        if prev_boundary is None:
            msg = f"No boundary info for coordinate {Coord3D(coord.x, coord.y, coord.z - 1)}"
            raise KeyError(msg)
        current_boundary = self.boundary_map.get(coord, None)
        if current_boundary is None:
            msg = f"No boundary info for coordinate {coord}"
            raise KeyError(msg)

        # check boundary change for each direction
        changed_directions: set[DIRECTION2D] = set()
        if prev_boundary.top != current_boundary.top:
            changed_directions.add(DIRECTION2D.TOP)
        if prev_boundary.bottom != current_boundary.bottom:
            changed_directions.add(DIRECTION2D.BOTTOM)
        if prev_boundary.left != current_boundary.left:
            changed_directions.add(DIRECTION2D.LEFT)
        if prev_boundary.right != current_boundary.right:
            changed_directions.add(DIRECTION2D.RIGHT)
        return changed_directions


class Canvas:
    config: CanvasConfig
    __nodes: set[Coord3D]
    __edges: set[tuple[Coord3D, Coord3D]]
    __pauli_axes: dict[Coord3D, Axis]
    __coord2role: dict[Coord3D, NodeRole]

    couts: dict[Coord3D, set[Coord3D]]

    __parity: CoordParityAccumulator
    flow: CoordFlowAccumulator
    scheduler: CoordScheduleAccumulator

    cube_config: dict[Coord3D, BlockConfig]
    pipe_config: dict[tuple[Coord3D, Coord3D], BlockConfig]

    bgraph: BoundaryGraph  # NOTE: boundary info is duplicated in configs

    def __init__(self, config: CanvasConfig) -> None:
        self.config = config
        self.__nodes = set()
        self.__edges = set()
        self.__pauli_axes = {}
        self.__coord2role = {}
        self.couts = {}
        self.__parity = CoordParityAccumulator()
        self.flow = CoordFlowAccumulator()
        self.scheduler = CoordScheduleAccumulator()

        self.cube_config = {}
        self.pipe_config = {}
        self.bgraph = BoundaryGraph(boundary_map={})

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

    @property
    def parity_accumulator(self) -> CoordParityAccumulator:
        return self.__parity

    def add_cube(self, global_pos: Coord3D, block_config: BlockConfig) -> None:
        self.cube_config[global_pos] = block_config
        boundary = Boundary(
            top=block_config.boundary[BoundarySide.TOP],
            bottom=block_config.boundary[BoundarySide.BOTTOM],
            left=block_config.boundary[BoundarySide.LEFT],
            right=block_config.boundary[BoundarySide.RIGHT],
        )
        self.bgraph.add_boundary(global_pos, boundary)

        data2d, ancilla_x2d, ancilla_z2d = rotated_surface_code_layout(
            self.config.d, Coord2D(global_pos.x, global_pos.y), block_config.boundary
        )

        offset_z = global_pos.z * 2 * self.config.d

        # Build graph layer by layer
        for layer_idx, layer_cfg in enumerate(block_config):  # noqa: PLR1702
            z = offset_z + layer_idx * 2

            if layer_cfg.layer1.basis is not None:
                for x, y in data2d:
                    self.__nodes.add(Coord3D(x, y, z))
                    self.__coord2role[Coord3D(x, y, z)] = NodeRole.DATA
                    self.__pauli_axes[Coord3D(x, y, z)] = layer_cfg.layer1.basis
                    # temporal edge
                    if Coord3D(x, y, z - 1) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z - 1), Coord3D(x, y, z)))
                        self.flow.add_flow(Coord3D(x, y, z - 1), Coord3D(x, y, z))

                # should construct parity check with data qubits
                if not layer_cfg.layer1.ancilla:
                    parity_offset = 1 if layer_cfg.layer1.basis == Axis.X else 0  # NOTE: only X and Z are allowed
                    ancilla_2d = ancilla_z2d if layer_cfg.layer1.basis == Axis.Z else ancilla_x2d
                    for x, y in ancilla_2d:
                        data_collection: set[Coord3D] = set()
                        for dx, dy in ANCILLA_EDGE:
                            if Coord3D(x + dx, y + dy, z) in self.__nodes:
                                data_collection.add(Coord3D(x + dx, y + dy, z))
                        if data_collection:
                            self.__parity.add_syndrome_measurement(Coord2D(x, y), z + parity_offset, data_collection)

            if layer_cfg.layer2.basis is not None:
                for x, y in data2d:
                    self.__nodes.add(Coord3D(x, y, z + 1))
                    self.__coord2role[Coord3D(x, y, z + 1)] = NodeRole.DATA
                    self.__pauli_axes[Coord3D(x, y, z + 1)] = layer_cfg.layer2.basis
                    # temporal edge
                    if Coord3D(x, y, z) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z), Coord3D(x, y, z + 1)))
                        self.flow.add_flow(Coord3D(x, y, z), Coord3D(x, y, z + 1))

                # NOTE: Redundant in layer2?
                # should construct parity check with data qubits
                if not layer_cfg.layer2.ancilla:
                    parity_offset = 0 if layer_cfg.layer2.basis == Axis.X else 1  # NOTE: only X and Z are allowed
                    ancilla_2d = ancilla_z2d if layer_cfg.layer2.basis == Axis.Z else ancilla_x2d
                    for x, y in ancilla_2d:
                        data_collection = set()
                        for dx, dy in ANCILLA_EDGE:
                            if Coord3D(x + dx, y + dy, z + 1) in self.__nodes:
                                data_collection.add(Coord3D(x + dx, y + dy, z + 1))
                        if data_collection:
                            self.__parity.add_syndrome_measurement(
                                Coord2D(x, y), z + 1 + parity_offset, data_collection
                            )

            if layer_cfg.layer1.ancilla:
                for x, y in ancilla_z2d:
                    if layer_cfg.layer1.ancilla:
                        self.__nodes.add(Coord3D(x, y, z))
                        self.__coord2role[Coord3D(x, y, z)] = NodeRole.ANCILLA_Z
                        self.__pauli_axes[Coord3D(x, y, z)] = Axis.X
                        for dx, dy in ANCILLA_EDGE:
                            if Coord3D(x + dx, y + dy, z) in self.__nodes:
                                self.__edges.add((Coord3D(x, y, z), Coord3D(x + dx, y + dy, z)))
                        self.__parity.add_syndrome_measurement(Coord2D(x, y), z, {Coord3D(x, y, z)})

            if layer_cfg.layer2.ancilla:
                for x, y in ancilla_x2d:
                    self.__nodes.add(Coord3D(x, y, z + 1))
                    self.__coord2role[Coord3D(x, y, z + 1)] = NodeRole.ANCILLA_X
                    self.__pauli_axes[Coord3D(x, y, z + 1)] = Axis.X
                    for dx, dy in ANCILLA_EDGE:
                        if Coord3D(x + dx, y + dy, z + 1) in self.__nodes:
                            self.__edges.add((Coord3D(x, y, z + 1), Coord3D(x + dx, y + dy, z + 1)))
                    self.__parity.add_syndrome_measurement(Coord2D(x, y), z + 1, {Coord3D(x, y, z + 1)})

    def add_pipe(self, global_edge: tuple[Coord3D, Coord3D], block_config: BlockConfig) -> None:
        pass
