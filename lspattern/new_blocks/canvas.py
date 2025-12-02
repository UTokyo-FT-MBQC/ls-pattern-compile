from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from graphqomb.common import Axis

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.accumulator import CoordFlowAccumulator, CoordParityAccumulator, CoordScheduleAccumulator
from lspattern.new_blocks.layout import (
    ANCILLA_EDGE_X,
    ANCILLA_EDGE_Z,
    RotatedSurfaceCodeLayoutBuilder,
)
from lspattern.new_blocks.loader import BlockConfig
from lspattern.new_blocks.mytype import Coord2D, Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from lspattern.new_blocks.canvas_loader import LogicalObservableSpec


_TOKEN_TO_SIDES: dict[str, BoundarySide] = {
    "T": BoundarySide.TOP,
    "B": BoundarySide.BOTTOM,
    "L": BoundarySide.LEFT,
    "R": BoundarySide.RIGHT,
}

_PHYSICAL_CLOCK = 2
ANCILLA_LENGTH = len(ANCILLA_EDGE_X)  # assuming both have the same length


def _token_to_boundary_sides(token: str) -> tuple[BoundarySide, BoundarySide]:
    """Convert logical observable token (2-char) to boundary side pair.

    Parameters
    ----------
    token : str
        A two-character token like "TB", "LR", "TL", etc.

    Returns
    -------
    tuple[BoundarySide, BoundarySide]
        A pair of boundary sides.

    Raises
    ------
    ValueError
        If the token is not a valid 2-character boundary token.
    """
    if len(token) == 2 and token[0] in _TOKEN_TO_SIDES and token[1] in _TOKEN_TO_SIDES:
        return _TOKEN_TO_SIDES[token[0]], _TOKEN_TO_SIDES[token[1]]
    msg = f"Unknown logical observable token: {token}"
    raise ValueError(msg)


def _edge_spec_to_pauli_set(edge_spec: EdgeSpecValue) -> frozenset[Axis]:
    """Convert EdgeSpecValue to a set of Pauli axes.

    O boundary contains both X and Z.

    Parameters
    ----------
    edge_spec : EdgeSpecValue
        The edge specification value (X, Z, or O).

    Returns
    -------
    frozenset[Axis]
        The set of Pauli axes contained in this boundary type.
    """
    if edge_spec == EdgeSpecValue.X:
        return frozenset({Axis.X})
    if edge_spec == EdgeSpecValue.Z:
        return frozenset({Axis.Z})
    # O boundary contains both X and Z
    return frozenset({Axis.X, Axis.Z})


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

    def check_boundary_init(self, coord: Coord3D) -> dict[BoundarySide, frozenset[Axis]]:
        """Check boundary changes and return added Pauli axes for each direction.

        Parameters
        ----------
        coord : Coord3D
            The coordinate to check.

        Returns
        -------
        dict[BoundarySide, frozenset[Axis]]
            A dictionary mapping each changed direction to the set of Pauli axes
            that were added in the transition. Only directions with added Paulis
            are included (e.g., O->X transitions are excluded since no Pauli is added).

        Raises
        ------
        KeyError
            If boundary info is missing for the coordinate or its predecessor.
        """
        prev_boundary = self.boundary_map.get(Coord3D(coord.x, coord.y, coord.z - 1), None)
        if prev_boundary is None:
            msg = f"No boundary info for coordinate {Coord3D(coord.x, coord.y, coord.z - 1)}"
            raise KeyError(msg)
        current_boundary = self.boundary_map.get(coord, None)
        if current_boundary is None:
            msg = f"No boundary info for coordinate {coord}"
            raise KeyError(msg)

        result: dict[BoundarySide, frozenset[Axis]] = {}

        directions_and_boundaries = [
            (BoundarySide.TOP, prev_boundary.top, current_boundary.top),
            (BoundarySide.BOTTOM, prev_boundary.bottom, current_boundary.bottom),
            (BoundarySide.LEFT, prev_boundary.left, current_boundary.left),
            (BoundarySide.RIGHT, prev_boundary.right, current_boundary.right),
        ]

        for direction, prev_edge, current_edge in directions_and_boundaries:
            if prev_edge != current_edge:
                added = _edge_spec_to_pauli_set(current_edge) - _edge_spec_to_pauli_set(prev_edge)
                if added:
                    result[direction] = added

        return result


class Canvas:
    config: CanvasConfig
    __nodes: set[Coord3D]
    __edges: set[tuple[Coord3D, Coord3D]]
    __pauli_axes: dict[Coord3D, Axis]
    __coord2role: dict[Coord3D, NodeRole]

    couts: dict[Coord3D, set[Coord3D]]
    pipe_couts: dict[tuple[Coord3D, Coord3D], set[Coord3D]]

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
        self.pipe_couts = {}
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

    def add_cube(
        self,
        global_pos: Coord3D,
        block_config: BlockConfig,
        logical_observable: LogicalObservableSpec | None = None,
    ) -> None:
        self.cube_config[global_pos] = block_config
        boundary = Boundary(
            top=block_config.boundary[BoundarySide.TOP],
            bottom=block_config.boundary[BoundarySide.BOTTOM],
            left=block_config.boundary[BoundarySide.LEFT],
            right=block_config.boundary[BoundarySide.RIGHT],
        )
        self.bgraph.add_boundary(global_pos, boundary)

        data2d, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.cube(
            self.config.d, Coord2D(global_pos.x, global_pos.y), block_config.boundary
        ).to_mutable_sets()

        current_time = global_pos.z * (
            2 * self.config.d * (_PHYSICAL_CLOCK + ANCILLA_LENGTH)
        )  # time offset for scheduler

        # Compute couts if logical_observable is specified
        if logical_observable is not None:
            self._compute_cout_from_logical_observable(
                global_pos, block_config, logical_observable, ancilla_x2d, ancilla_z2d
            )

        offset_z = global_pos.z * 2 * self.config.d

        # Build graph layer by layer
        for layer_idx, layer_cfg in enumerate(block_config):  # noqa: PLR1702
            z = offset_z + layer_idx * 2
            layer_time = current_time + layer_idx * 2 * (_PHYSICAL_CLOCK + ANCILLA_LENGTH)

            if layer_cfg.layer1.basis is not None:
                layer1_coords: list[Coord3D] = []
                for x, y in data2d:
                    coord = Coord3D(x, y, z)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.DATA
                    self.__pauli_axes[coord] = layer_cfg.layer1.basis
                    layer1_coords.append(coord)
                    # temporal edge
                    if Coord3D(x, y, z - 1) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z - 1), coord))
                        self.flow.add_flow(Coord3D(x, y, z - 1), coord)
                        self.scheduler.add_entangle_at_time(layer_time, {(Coord3D(x, y, z - 1), coord)})
                # Add layer1 data qubits to scheduler
                self.scheduler.add_prep_at_time(layer_time, layer1_coords)
                self.scheduler.add_meas_at_time(
                    layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH + 1,
                    layer1_coords,
                )

                # should construct parity check with data qubits
                if not layer_cfg.layer1.ancilla:
                    parity_offset = 1 if layer_cfg.layer1.basis == Axis.X else 0  # NOTE: only X and Z are allowed
                    ancilla_2d = ancilla_z2d if layer_cfg.layer1.basis == Axis.Z else ancilla_x2d
                    for x, y in ancilla_2d:
                        data_collection: set[Coord3D] = set()
                        for dx, dy in ANCILLA_EDGE_Z:
                            if Coord2D(x + dx, y + dy) in data2d:
                                data_collection.add(Coord3D(x + dx, y + dy, z))
                        if data_collection:
                            self.__parity.add_syndrome_measurement(Coord2D(x, y), z + parity_offset, data_collection)

            if layer_cfg.layer2.basis is not None:
                layer2_coords: list[Coord3D] = []
                for x, y in data2d:
                    coord = Coord3D(x, y, z + 1)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.DATA
                    self.__pauli_axes[coord] = layer_cfg.layer2.basis
                    layer2_coords.append(coord)
                    # temporal edge
                    if Coord3D(x, y, z) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z), coord))
                        self.flow.add_flow(Coord3D(x, y, z), coord)
                        self.scheduler.add_entangle_at_time(
                            layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH, {(Coord3D(x, y, z), coord)}
                        )
                # Add layer2 data qubits to scheduler
                self.scheduler.add_prep_at_time(
                    layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH,
                    layer2_coords,
                )
                self.scheduler.add_meas_at_time(
                    layer_time + 2 * (_PHYSICAL_CLOCK + ANCILLA_LENGTH) + 1,
                    layer2_coords,
                )

                # NOTE: Redundant in layer2?
                # should construct parity check with data qubits
                if not layer_cfg.layer2.ancilla:
                    parity_offset = 0 if layer_cfg.layer2.basis == Axis.X else 1  # NOTE: only X and Z are allowed
                    ancilla_2d = ancilla_z2d if layer_cfg.layer2.basis == Axis.Z else ancilla_x2d
                    for x, y in ancilla_2d:
                        data_collection = set()
                        for dx, dy in ANCILLA_EDGE_X:
                            if Coord2D(x + dx, y + dy) in data2d:
                                data_collection.add(Coord3D(x + dx, y + dy, z + 1))
                        if data_collection:
                            self.__parity.add_syndrome_measurement(
                                Coord2D(x, y), z + 1 + parity_offset, data_collection
                            )

            if layer_cfg.layer1.ancilla:
                ancilla_z_coords: list[Coord3D] = []
                for x, y in ancilla_z2d:
                    coord = Coord3D(x, y, z)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.ANCILLA_Z
                    self.__pauli_axes[coord] = Axis.X
                    ancilla_z_coords.append(coord)
                    for i, (dx, dy) in enumerate(ANCILLA_EDGE_Z):
                        if Coord3D(x + dx, y + dy, z) in self.__nodes:
                            self.__edges.add((coord, Coord3D(x + dx, y + dy, z)))
                            self.scheduler.add_entangle_at_time(
                                layer_time + 1 + i, {(coord, Coord3D(x + dx, y + dy, z))}
                            )
                    self.__parity.add_syndrome_measurement(Coord2D(x, y), z, {coord})
                # Add ancilla_z qubits to scheduler
                self.scheduler.add_prep_at_time(layer_time, ancilla_z_coords)
                self.scheduler.add_meas_at_time(layer_time + ANCILLA_LENGTH + 1, ancilla_z_coords)

            if layer_cfg.layer2.ancilla:
                ancilla_x_coords: list[Coord3D] = []
                for x, y in ancilla_x2d:
                    coord = Coord3D(x, y, z + 1)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.ANCILLA_X
                    self.__pauli_axes[coord] = Axis.X
                    ancilla_x_coords.append(coord)
                    for i, (dx, dy) in enumerate(ANCILLA_EDGE_X):
                        if Coord3D(x + dx, y + dy, z + 1) in self.__nodes:
                            self.__edges.add((coord, Coord3D(x + dx, y + dy, z + 1)))
                            self.scheduler.add_entangle_at_time(
                                layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH + 1 + i,
                                {(coord, Coord3D(x + dx, y + dy, z + 1))},
                            )
                    self.__parity.add_syndrome_measurement(Coord2D(x, y), z + 1, {coord})
                # Add ancilla_x qubits to scheduler
                self.scheduler.add_prep_at_time(
                    layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH,
                    ancilla_x_coords,
                )
                self.scheduler.add_meas_at_time(
                    layer_time + _PHYSICAL_CLOCK + 2 * ANCILLA_LENGTH + 1,
                    ancilla_x_coords,
                )

    def _compute_cout_from_logical_observable(
        self,
        global_pos: Coord3D,
        block_config: BlockConfig,
        logical_observable: LogicalObservableSpec,
        ancilla_x2d: AbstractSet[Coord2D],
        ancilla_z2d: AbstractSet[Coord2D],
    ) -> None:
        """Compute cout coordinates from logical observable specification.

        Parameters
        ----------
        global_pos : Coord3D
            The global position of the cube.
        block_config : BlockConfig
            The block configuration.
        logical_observable : LogicalObservableSpec
            The logical observable specification.
        ancilla_x2d : collections.abc.Set[Coord2D]
            X ancilla 2D coordinates for this cube.
        ancilla_z2d : collections.abc.Set[Coord2D]
            Z ancilla 2D coordinates for this cube.

        Notes
        -----
        Token types:
        - "TB", "LR", etc. (2-char): Use boundary_data_path_cube() to get data qubit path
        - "X": Select all ANCILLA_X nodes in the cube's final layer
        - "Z": Select all ANCILLA_Z nodes in the cube's final layer
        """
        observable_token = logical_observable.token
        offset_z = global_pos.z * 2 * self.config.d

        if observable_token == "X":  # noqa: S105
            # Select ANCILLA_X nodes at final layer within cube's XY range
            cout_coords = {Coord3D(c.x, c.y, offset_z + 1) for c in ancilla_x2d}
        elif observable_token == "Z":  # noqa: S105
            # Select ANCILLA_Z nodes at final layer within cube's XY range
            cout_coords = {Coord3D(c.x, c.y, offset_z) for c in ancilla_z2d}
        else:
            # TB, LR, etc. - use cube_boundary_path for data qubit path
            side_a, side_b = _token_to_boundary_sides(observable_token)
            path_2d = RotatedSurfaceCodeLayoutBuilder.cube_boundary_path(
                self.config.d,
                Coord2D(global_pos.x, global_pos.y),
                block_config.boundary,
                side_a,
                side_b,
            )
            cout_coords = {Coord3D(c.x, c.y, offset_z) for c in path_2d}

        self.couts[global_pos] = cout_coords

    def compute_pipe_cout_from_logical_observable(
        self,
        global_edge: tuple[Coord3D, Coord3D],
        block_config: BlockConfig,
        logical_observable: LogicalObservableSpec,
        ancilla_x2d: AbstractSet[Coord2D],
        ancilla_z2d: AbstractSet[Coord2D],
    ) -> None:
        """Compute cout coordinates from logical observable specification for pipe.

        Parameters
        ----------
        global_edge : tuple[Coord3D, Coord3D]
            The global edge (start, end) of the pipe.
        block_config : BlockConfig
            The block configuration.
        logical_observable : LogicalObservableSpec
            The logical observable specification.
        ancilla_x2d : collections.abc.Set[Coord2D]
            X ancilla 2D coordinates for this pipe.
        ancilla_z2d : collections.abc.Set[Coord2D]
            Z ancilla 2D coordinates for this pipe.

        Notes
        -----
        Token types:
        - "RL", "TB", etc. (2-char): Use pipe_boundary_path() to get data qubit path
        - "X": Select all ANCILLA_X nodes in the pipe's first layer
        - "Z": Select all ANCILLA_Z nodes in the pipe's first layer
        """
        observable_token = logical_observable.token
        start, end = global_edge
        offset_z = start.z * 2 * self.config.d

        if observable_token == "X":  # noqa: S105
            # Select ANCILLA_X nodes at first layer within pipe's XY range
            cout_coords = {Coord3D(c.x, c.y, offset_z + 1) for c in ancilla_x2d}
        elif observable_token == "Z":  # noqa: S105
            # Select ANCILLA_Z nodes at first layer within pipe's XY range
            cout_coords = {Coord3D(c.x, c.y, offset_z) for c in ancilla_z2d}
        else:
            # RL, TB, etc. - use pipe_boundary_path for data qubit path
            side_a, side_b = _token_to_boundary_sides(observable_token)
            path_2d = RotatedSurfaceCodeLayoutBuilder.pipe_boundary_path(
                self.config.d,
                start,
                end,
                block_config.boundary,
                side_a,
                side_b,
            )
            cout_coords = {Coord3D(c.x, c.y, offset_z) for c in path_2d}

        self.pipe_couts[global_edge] = cout_coords

    def add_pipe(
        self,
        global_edge: tuple[Coord3D, Coord3D],
        block_config: BlockConfig,
        logical_observable: LogicalObservableSpec | None = None,
    ) -> None:
        """Add a pipe to the canvas.

        Parameters
        ----------
        global_edge : tuple[Coord3D, Coord3D]
            The global edge (start, end) of the pipe.
        block_config : BlockConfig
            The block configuration for the pipe.
        logical_observable : LogicalObservableSpec | None
            Optional logical observable specification.
        """
        start, end = global_edge

        # Store config
        self.pipe_config[global_edge] = block_config

        # Create boundary and register BOTH coordinates
        boundary = Boundary(
            top=block_config.boundary[BoundarySide.TOP],
            bottom=block_config.boundary[BoundarySide.BOTTOM],
            left=block_config.boundary[BoundarySide.LEFT],
            right=block_config.boundary[BoundarySide.RIGHT],
        )
        self.bgraph.add_boundary(start, boundary)
        self.bgraph.add_boundary(end, boundary)

        # Get 2D coordinates from layout builder
        data2d, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.pipe(
            self.config.d, start, end, block_config.boundary
        ).to_mutable_sets()
        print(f"pipe data2d: {data2d}, ancilla_x2d: {ancilla_x2d}, ancilla_z2d: {ancilla_z2d}")

        # Calculate time offsets using start.z
        current_time = start.z * (2 * self.config.d * (_PHYSICAL_CLOCK + ANCILLA_LENGTH))
        offset_z = start.z * 2 * self.config.d

        # Compute couts if logical_observable is specified
        if logical_observable is not None:
            self.compute_pipe_cout_from_logical_observable(
                global_edge, block_config, logical_observable, ancilla_x2d, ancilla_z2d
            )

        # Build graph layer by layer (same structure as add_cube)
        for layer_idx, layer_cfg in enumerate(block_config):  # noqa: PLR1702
            z = offset_z + layer_idx * 2
            layer_time = current_time + layer_idx * 2 * (_PHYSICAL_CLOCK + ANCILLA_LENGTH)

            if layer_cfg.layer1.basis is not None:
                layer1_coords: list[Coord3D] = []
                for x, y in data2d:
                    coord = Coord3D(x, y, z)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.DATA
                    self.__pauli_axes[coord] = layer_cfg.layer1.basis
                    layer1_coords.append(coord)
                    # temporal edge
                    if Coord3D(x, y, z - 1) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z - 1), coord))
                        self.flow.add_flow(Coord3D(x, y, z - 1), coord)
                        self.scheduler.add_entangle_at_time(layer_time, {(Coord3D(x, y, z - 1), coord)})
                # Add layer1 data qubits to scheduler
                self.scheduler.add_prep_at_time(layer_time, layer1_coords)
                self.scheduler.add_meas_at_time(
                    layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH + 1,
                    layer1_coords,
                )

                # should construct parity check with data qubits
                if not layer_cfg.layer1.ancilla:
                    parity_offset = 1 if layer_cfg.layer1.basis == Axis.X else 0
                    ancilla_2d = ancilla_z2d if layer_cfg.layer1.basis == Axis.Z else ancilla_x2d
                    for x, y in ancilla_2d:
                        data_collection: set[Coord3D] = set()
                        for dx, dy in ANCILLA_EDGE_Z:
                            if Coord2D(x + dx, y + dy) in data2d:
                                data_collection.add(Coord3D(x + dx, y + dy, z))
                        if data_collection:
                            self.__parity.add_syndrome_measurement(Coord2D(x, y), z + parity_offset, data_collection)

            if layer_cfg.layer2.basis is not None:
                layer2_coords: list[Coord3D] = []
                for x, y in data2d:
                    coord = Coord3D(x, y, z + 1)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.DATA
                    self.__pauli_axes[coord] = layer_cfg.layer2.basis
                    layer2_coords.append(coord)
                    # temporal edge
                    if Coord3D(x, y, z) in self.__nodes:
                        self.__edges.add((Coord3D(x, y, z), coord))
                        self.flow.add_flow(Coord3D(x, y, z), coord)
                        self.scheduler.add_entangle_at_time(
                            layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH, {(Coord3D(x, y, z), coord)}
                        )
                # Add layer2 data qubits to scheduler
                self.scheduler.add_prep_at_time(
                    layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH,
                    layer2_coords,
                )
                self.scheduler.add_meas_at_time(
                    layer_time + 2 * (_PHYSICAL_CLOCK + ANCILLA_LENGTH) + 1,
                    layer2_coords,
                )

                # should construct parity check with data qubits
                if not layer_cfg.layer2.ancilla:
                    parity_offset = 0 if layer_cfg.layer2.basis == Axis.X else 1
                    ancilla_2d = ancilla_z2d if layer_cfg.layer2.basis == Axis.Z else ancilla_x2d
                    for x, y in ancilla_2d:
                        data_collection = set()
                        for dx, dy in ANCILLA_EDGE_X:
                            if Coord2D(x + dx, y + dy) in data2d:
                                data_collection.add(Coord3D(x + dx, y + dy, z + 1))
                        if data_collection:
                            self.__parity.add_syndrome_measurement(
                                Coord2D(x, y), z + 1 + parity_offset, data_collection
                            )

            if layer_cfg.layer1.ancilla:
                ancilla_z_coords: list[Coord3D] = []
                for x, y in ancilla_z2d:
                    coord = Coord3D(x, y, z)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.ANCILLA_Z
                    self.__pauli_axes[coord] = Axis.X
                    ancilla_z_coords.append(coord)
                    for i, (dx, dy) in enumerate(ANCILLA_EDGE_Z):
                        if Coord3D(x + dx, y + dy, z) in self.__nodes:
                            self.__edges.add((coord, Coord3D(x + dx, y + dy, z)))
                            self.scheduler.add_entangle_at_time(
                                layer_time + 1 + i, {(coord, Coord3D(x + dx, y + dy, z))}
                            )
                    self.__parity.add_syndrome_measurement(Coord2D(x, y), z, {coord})
                # Add ancilla_z qubits to scheduler
                self.scheduler.add_prep_at_time(layer_time, ancilla_z_coords)
                self.scheduler.add_meas_at_time(layer_time + ANCILLA_LENGTH + 1, ancilla_z_coords)

            if layer_cfg.layer2.ancilla:
                ancilla_x_coords: list[Coord3D] = []
                for x, y in ancilla_x2d:
                    coord = Coord3D(x, y, z + 1)
                    self.__nodes.add(coord)
                    self.__coord2role[coord] = NodeRole.ANCILLA_X
                    self.__pauli_axes[coord] = Axis.X
                    ancilla_x_coords.append(coord)
                    for i, (dx, dy) in enumerate(ANCILLA_EDGE_X):
                        if Coord3D(x + dx, y + dy, z + 1) in self.__nodes:
                            self.__edges.add((coord, Coord3D(x + dx, y + dy, z + 1)))
                            self.scheduler.add_entangle_at_time(
                                layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH + 1 + i,
                                {(coord, Coord3D(x + dx, y + dy, z + 1))},
                            )
                    self.__parity.add_syndrome_measurement(Coord2D(x, y), z + 1, {coord})
                # Add ancilla_x qubits to scheduler
                self.scheduler.add_prep_at_time(
                    layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH,
                    ancilla_x_coords,
                )
                self.scheduler.add_meas_at_time(
                    layer_time + _PHYSICAL_CLOCK + 2 * ANCILLA_LENGTH + 1,
                    ancilla_x_coords,
                )
