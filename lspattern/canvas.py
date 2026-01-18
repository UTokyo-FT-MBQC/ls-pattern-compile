from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphqomb.common import Axis

from lspattern.accumulator import CoordFlowAccumulator, CoordParityAccumulator, CoordScheduleAccumulator
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.fragment_builder import build_patch_cube_fragment, build_patch_pipe_fragment
from lspattern.layout import (
    ANCILLA_EDGE_X,
    RotatedSurfaceCodeLayoutBuilder,
)
from lspattern.mytype import Coord2D, Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from lspattern.canvas_loader import CompositeLogicalObservableSpec, LogicalObservableSpec
    from lspattern.fragment import Boundary, GraphSpec
    from lspattern.loader import BlockConfig


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
    if len(token) == 2 and token[0] in _TOKEN_TO_SIDES and token[1] in _TOKEN_TO_SIDES:  # noqa: PLR2004
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

    couts: dict[Coord3D, dict[str, set[Coord3D]]]
    pipe_couts: dict[tuple[Coord3D, Coord3D], dict[str, set[Coord3D]]]

    __parity: CoordParityAccumulator
    flow: CoordFlowAccumulator
    scheduler: CoordScheduleAccumulator

    cube_config: dict[Coord3D, BlockConfig]
    pipe_config: dict[tuple[Coord3D, Coord3D], BlockConfig]

    bgraph: BoundaryGraph  # NOTE: boundary info is duplicated in configs

    logical_observables: tuple[CompositeLogicalObservableSpec, ...]

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
        self.logical_observables = ()

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

    def _merge_graph_spec(self, graph_spec: GraphSpec, *, coord_offset: Coord3D, time_offset: int) -> None:  # noqa: C901
        def translate_coord(coord: Coord3D) -> Coord3D:
            if graph_spec.coord_mode == "global":
                return coord
            return Coord3D(coord.x + coord_offset.x, coord.y + coord_offset.y, coord.z + coord_offset.z)

        def translate_xy(xy: Coord2D) -> Coord2D:
            if graph_spec.coord_mode == "global":
                return xy
            return Coord2D(xy.x + coord_offset.x, xy.y + coord_offset.y)

        def translate_z(z: int) -> int:
            if graph_spec.coord_mode == "global":
                return z
            return z + coord_offset.z

        def translate_time(time: int) -> int:
            if graph_spec.time_mode == "global":
                return time
            return time + time_offset

        # Nodes, roles, measurement bases
        translated_nodes: set[Coord3D] = set()
        for coord in graph_spec.nodes:
            translated = translate_coord(coord)
            translated_nodes.add(translated)
            self.__nodes.add(translated)
            self.__coord2role[translated] = graph_spec.coord2role.get(coord, NodeRole.DATA)
            self.__pauli_axes[translated] = graph_spec.pauli_axes[coord]

        # Inter-block temporal edges (z-direction edges connecting to previous blocks)
        inter_block_edges: list[tuple[Coord3D, Coord3D]] = []
        for translated in translated_nodes:
            prev_coord = Coord3D(translated.x, translated.y, translated.z - 1)
            # Connect if prev_coord exists in canvas but is not part of current fragment
            if prev_coord in self.__nodes and prev_coord not in translated_nodes:
                edge = (prev_coord, translated)
                inter_block_edges.append(edge)
                self.__edges.add(edge)
                self.flow.add_flow(prev_coord, translated)

        # Schedule inter-block entanglements at time_offset (start of this block)
        if inter_block_edges:
            self.scheduler.add_entangle_at_time(time_offset, inter_block_edges)

        # Physical edges
        for a, b in graph_spec.edges:
            self.__edges.add((translate_coord(a), translate_coord(b)))

        # Flow (xflow)
        for from_coord, to_coords in graph_spec.flow.flow.items():
            from_translated = translate_coord(from_coord)
            for to_coord in to_coords:
                self.flow.add_flow(from_translated, translate_coord(to_coord))

        # Scheduler
        for time, coords in graph_spec.scheduler.prep_time.items():
            self.scheduler.add_prep_at_time(translate_time(time), [translate_coord(coord) for coord in coords])
        for time, coords in graph_spec.scheduler.meas_time.items():
            self.scheduler.add_meas_at_time(translate_time(time), [translate_coord(coord) for coord in coords])
        for time, edges in graph_spec.scheduler.entangle_time.items():
            translated_edges = [(translate_coord(a), translate_coord(b)) for a, b in edges]
            self.scheduler.add_entangle_at_time(translate_time(time), translated_edges)
            self.__edges.update(translated_edges)

        # Parity / detector candidates
        for xy, z_map in graph_spec.parity.syndrome_meas.items():
            xy_translated = translate_xy(xy)
            for z, coords in z_map.items():
                self.__parity.add_syndrome_measurement(
                    xy_translated,
                    translate_z(z),
                    [translate_coord(coord) for coord in coords],
                )

        for xy, z_map in graph_spec.parity.remaining_parity.items():
            xy_translated = translate_xy(xy)
            for z, coords in z_map.items():
                self.__parity.add_remaining_parity(
                    xy_translated,
                    translate_z(z),
                    [translate_coord(coord) for coord in coords],
                )

        for det_coord in graph_spec.parity.non_deterministic_coords:
            self.__parity.add_non_deterministic_coord(translate_coord(det_coord))

    def add_cube(
        self,
        global_pos: Coord3D,
        block_config: BlockConfig,
        logical_observables: Sequence[LogicalObservableSpec] | None = None,
    ) -> None:
        """Add a cube block to the canvas.

        Parameters
        ----------
        global_pos : Coord3D
            Global (x, y, z) position of the cube in slot coordinates.
        block_config : BlockConfig
            Block configuration containing boundary and layer definitions.
        logical_observables : Sequence[LogicalObservableSpec] | None
            Optional logical observable specifications (multiple supported).
        """
        self.cube_config[global_pos] = block_config

        # Calculate coordinate and time offsets for this position
        coord_offset = Coord3D(
            2 * (self.config.d + 1) * global_pos.x,
            2 * (self.config.d + 1) * global_pos.y,
            global_pos.z * 2 * self.config.d,
        )
        time_offset = global_pos.z * (2 * self.config.d * (_PHYSICAL_CLOCK + ANCILLA_LENGTH))

        # Handle user-supplied graph spec
        if block_config.graph_spec is not None:
            self._merge_graph_spec(block_config.graph_spec, coord_offset=coord_offset, time_offset=time_offset)
            if logical_observables is not None:
                result: dict[str, set[Coord3D]] = {}
                for idx, obs in enumerate(logical_observables):
                    label = obs.label if obs.label is not None else str(idx)
                    if obs.nodes:
                        unknown = set(obs.nodes) - block_config.graph_spec.nodes
                        if unknown:
                            msg = f"logical_observables.nodes references undefined graph nodes: {sorted(unknown)}"
                            raise ValueError(msg)
                        if block_config.graph_spec.coord_mode == "global":
                            result[label] = set(obs.nodes)
                        else:
                            result[label] = {
                                Coord3D(coord.x + coord_offset.x, coord.y + coord_offset.y, coord.z + coord_offset.z)
                                for coord in obs.nodes
                            }
                    else:
                        msg = "graph-based blocks require logical_observables.nodes (list of node coordinates)."
                        raise ValueError(msg)
                self.couts[global_pos] = result
            return

        # Patch-based cube: use fragment builder
        fragment = build_patch_cube_fragment(self.config.d, block_config)

        # Merge graph spec
        self._merge_graph_spec(fragment.graph, coord_offset=coord_offset, time_offset=time_offset)

        # Merge boundary fragment into bgraph
        for local_coord, boundary in fragment.boundary.boundaries.items():
            global_coord = Coord3D(
                global_pos.x + local_coord.x,
                global_pos.y + local_coord.y,
                global_pos.z + local_coord.z,
            )
            self.bgraph.add_boundary(global_coord, boundary)

        # Compute couts if logical_observables is specified
        if logical_observables is not None:
            _, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.cube(
                self.config.d, Coord2D(global_pos.x, global_pos.y), block_config.boundary
            ).to_mutable_sets()
            # Each unit layer has 2 physical layers (layer1 and layer2)
            num_unit_layers = len(block_config) // 2
            self._compute_cout_from_logical_observables(
                global_pos, block_config, logical_observables, ancilla_x2d, ancilla_z2d, num_unit_layers
            )

    def _compute_cout_from_logical_observables(
        self,
        global_pos: Coord3D,
        block_config: BlockConfig,
        logical_observables: Sequence[LogicalObservableSpec],
        ancilla_x2d: AbstractSet[Coord2D],
        ancilla_z2d: AbstractSet[Coord2D],
        num_unit_layers: int,
    ) -> None:
        """Compute cout coordinates from logical observable specifications.

        Parameters
        ----------
        global_pos : Coord3D
            The global position of the cube.
        block_config : BlockConfig
            The block configuration.
        logical_observables : Sequence[LogicalObservableSpec]
            The logical observable specifications (multiple supported).
        ancilla_x2d : collections.abc.Set[Coord2D]
            X ancilla 2D coordinates for this cube.
        ancilla_z2d : collections.abc.Set[Coord2D]
            Z ancilla 2D coordinates for this cube.
        num_unit_layers : int
            Number of unit layers in this block.

        Notes
        -----
        Token types:
        - "TB", "LR", etc. (2-char): Use boundary_data_path_cube() to get data qubit path
        - "X": Select all ANCILLA_X nodes at the specified layer
        - "Z": Select all ANCILLA_Z nodes at the specified layer

        Physical z coordinate calculation:
        - block_base_z = global_pos.z * 2 * d
        - physical_z = block_base_z + (unit_layer_idx * 2) + sublayer_offset
        - sublayer_offset: 0 for layer1, 1 for layer2
        """
        result: dict[str, set[Coord3D]] = {}
        block_base_z = global_pos.z * 2 * self.config.d

        for idx, obs in enumerate(logical_observables):
            label = obs.label if obs.label is not None else str(idx)
            observable_token = obs.token

            if observable_token is None:
                msg = "Patch-based logical_observables must be specified as a token string (e.g. 'TB', 'X', 'Z')."
                raise ValueError(msg)

            # Determine unit layer index
            if obs.layer is not None:
                unit_layer_idx = obs.layer
                if unit_layer_idx < 0:
                    unit_layer_idx = num_unit_layers + unit_layer_idx
                if unit_layer_idx < 0 or unit_layer_idx >= num_unit_layers:
                    msg = f"layer index {obs.layer} out of range for block with {num_unit_layers} unit layers"
                    raise ValueError(msg)
            else:
                unit_layer_idx = 0  # Default: first unit layer

            # Determine sublayer offset
            if obs.sublayer is not None:
                sublayer_offset = obs.sublayer - 1  # 1→0, 2→1
            elif observable_token == "X":  # noqa: S105
                sublayer_offset = 1  # X → layer2
            else:
                sublayer_offset = 0  # Z, TB, etc. → layer1

            offset_z = block_base_z + (unit_layer_idx * 2) + sublayer_offset

            if observable_token == "X":  # noqa: S105
                # Select ANCILLA_X nodes at specified layer within cube's XY range
                cout_coords = {Coord3D(c.x, c.y, offset_z) for c in ancilla_x2d}
            elif observable_token == "Z":  # noqa: S105
                # Select ANCILLA_Z nodes at specified layer within cube's XY range
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

            result[label] = cout_coords

        self.couts[global_pos] = result

    def _compute_pipe_cout_from_logical_observables(
        self,
        global_edge: tuple[Coord3D, Coord3D],
        block_config: BlockConfig,
        logical_observables: Sequence[LogicalObservableSpec],
        ancilla_x2d: AbstractSet[Coord2D],
        ancilla_z2d: AbstractSet[Coord2D],
        num_unit_layers: int,
    ) -> None:
        """Compute cout coordinates from logical observable specifications for pipe.

        Parameters
        ----------
        global_edge : tuple[Coord3D, Coord3D]
            The global edge (start, end) of the pipe.
        block_config : BlockConfig
            The block configuration.
        logical_observables : Sequence[LogicalObservableSpec]
            The logical observable specifications (multiple supported).
        ancilla_x2d : collections.abc.Set[Coord2D]
            X ancilla 2D coordinates for this pipe.
        ancilla_z2d : collections.abc.Set[Coord2D]
            Z ancilla 2D coordinates for this pipe.
        num_unit_layers : int
            Number of unit layers in this block.

        Notes
        -----
        Token types:
        - "RL", "TB", etc. (2-char): Use pipe_boundary_path() to get data qubit path
        - "X": Select all ANCILLA_X nodes at the specified layer
        - "Z": Select all ANCILLA_Z nodes at the specified layer
        """
        result: dict[str, set[Coord3D]] = {}
        start, end = global_edge
        block_base_z = start.z * 2 * self.config.d

        for idx, obs in enumerate(logical_observables):
            label = obs.label if obs.label is not None else str(idx)
            observable_token = obs.token

            if observable_token is None:
                msg = "Patch-based logical_observables must be specified as a token string (e.g. 'TB', 'X', 'Z')."
                raise ValueError(msg)

            # Determine unit layer index
            if obs.layer is not None:
                unit_layer_idx = obs.layer
                if unit_layer_idx < 0:
                    unit_layer_idx = num_unit_layers + unit_layer_idx
                if unit_layer_idx < 0 or unit_layer_idx >= num_unit_layers:
                    msg = f"layer index {obs.layer} out of range for block with {num_unit_layers} unit layers"
                    raise ValueError(msg)
            else:
                unit_layer_idx = 0  # Default: first unit layer

            # Determine sublayer offset
            if obs.sublayer is not None:
                sublayer_offset = obs.sublayer - 1  # 1→0, 2→1
            elif observable_token == "X":  # noqa: S105
                sublayer_offset = 1  # X → layer2
            else:
                sublayer_offset = 0  # Z, TB, etc. → layer1

            offset_z = block_base_z + (unit_layer_idx * 2) + sublayer_offset

            if observable_token == "X":  # noqa: S105
                # Select ANCILLA_X nodes at specified layer within pipe's XY range
                cout_coords = {Coord3D(c.x, c.y, offset_z) for c in ancilla_x2d}
            elif observable_token == "Z":  # noqa: S105
                # Select ANCILLA_Z nodes at specified layer within pipe's XY range
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

            result[label] = cout_coords

        self.pipe_couts[global_edge] = result

    def add_pipe(  # noqa: C901
        self,
        global_edge: tuple[Coord3D, Coord3D],
        block_config: BlockConfig,
        logical_observables: Sequence[LogicalObservableSpec] | None = None,
    ) -> None:
        """Add a pipe to the canvas.

        Parameters
        ----------
        global_edge : tuple[Coord3D, Coord3D]
            The global edge (start, end) of the pipe.
        block_config : BlockConfig
            The block configuration for the pipe.
        logical_observables : Sequence[LogicalObservableSpec] | None
            Optional logical observable specifications (multiple supported).
        """
        start, end = global_edge

        # Store config
        self.pipe_config[global_edge] = block_config

        # Calculate pipe direction and coordinate offset
        pipe_dir = RotatedSurfaceCodeLayoutBuilder.pipe_offset(start, end)
        offset_x = 2 * (self.config.d + 1) * start.x
        offset_y = 2 * (self.config.d + 1) * start.y
        if pipe_dir == BoundarySide.RIGHT:
            offset_x += 2 * self.config.d
        elif pipe_dir == BoundarySide.LEFT:
            offset_x -= 2
        elif pipe_dir == BoundarySide.TOP:
            offset_y -= 2
        elif pipe_dir == BoundarySide.BOTTOM:
            offset_y += 2 * self.config.d
        coord_offset = Coord3D(offset_x, offset_y, start.z * 2 * self.config.d)
        time_offset = start.z * (2 * self.config.d * (_PHYSICAL_CLOCK + ANCILLA_LENGTH))

        # Handle user-supplied graph spec
        if block_config.graph_spec is not None:
            self._merge_graph_spec(block_config.graph_spec, coord_offset=coord_offset, time_offset=time_offset)
            if logical_observables is not None:
                result: dict[str, set[Coord3D]] = {}
                for idx, obs in enumerate(logical_observables):
                    label = obs.label if obs.label is not None else str(idx)
                    if obs.nodes:
                        unknown = set(obs.nodes) - block_config.graph_spec.nodes
                        if unknown:
                            msg = f"logical_observables.nodes references undefined graph nodes: {sorted(unknown)}"
                            raise ValueError(msg)
                        if block_config.graph_spec.coord_mode == "global":
                            result[label] = set(obs.nodes)
                        else:
                            result[label] = {
                                Coord3D(coord.x + coord_offset.x, coord.y + coord_offset.y, coord.z + coord_offset.z)
                                for coord in obs.nodes
                            }
                    else:
                        msg = "graph-based blocks require logical_observables.nodes (list of node coordinates)."
                        raise ValueError(msg)
                self.pipe_couts[global_edge] = result
            return

        # Patch-based pipe: use fragment builder
        fragment = build_patch_pipe_fragment(self.config.d, pipe_dir, block_config)

        # Merge graph spec
        self._merge_graph_spec(fragment.graph, coord_offset=coord_offset, time_offset=time_offset)

        # Merge boundary fragment into bgraph
        # For pipe, boundaries are stored at start and end positions
        for local_coord, boundary in fragment.boundary.boundaries.items():
            global_coord = Coord3D(
                start.x + local_coord.x,
                start.y + local_coord.y,
                start.z + local_coord.z,
            )
            self.bgraph.add_boundary(global_coord, boundary)

        # Compute couts if logical_observables is specified
        if logical_observables is not None:
            _, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.pipe(
                self.config.d, start, end, block_config.boundary
            ).to_mutable_sets()
            # Each unit layer has 2 physical layers (layer1 and layer2)
            num_unit_layers = len(block_config) // 2
            self._compute_pipe_cout_from_logical_observables(
                global_edge, block_config, logical_observables, ancilla_x2d, ancilla_z2d, num_unit_layers
            )
