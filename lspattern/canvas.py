from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lspattern.accumulator import CoordFlowAccumulator, CoordParityAccumulator, CoordScheduleAccumulator
from lspattern.consts import BoundarySide
from lspattern.fragment_builder import build_patch_cube_fragment, build_patch_pipe_fragment
from lspattern.layout import (
    ANCILLA_EDGE_X,
    RotatedSurfaceCodeLayoutBuilder,
)
from lspattern.mytype import Coord2D, Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.common import Axis

    from lspattern.canvas_loader import CompositeLogicalObservableSpec, LogicalObservableSpec
    from lspattern.fragment import GraphSpec
    from lspattern.init_flow_analysis import InitFlowOverrides
    from lspattern.loader import BlockConfig


_TOKEN_TO_SIDES: dict[str, BoundarySide] = {
    "T": BoundarySide.TOP,
    "B": BoundarySide.BOTTOM,
    "L": BoundarySide.LEFT,
    "R": BoundarySide.RIGHT,
}

_PHYSICAL_CLOCK = 2
ANCILLA_LENGTH = len(ANCILLA_EDGE_X)  # assuming both have the same length


def _compute_sublayer_offset(
    observable_token: str,
    explicit_sublayer: int | None,
    invert_ancilla_order: bool,
) -> int:
    """Compute sublayer offset for logical observable coordinate calculation.

    Parameters
    ----------
    observable_token : str
        The observable token (X, Z, TB, LR, etc.).
    explicit_sublayer : int | None
        Explicit sublayer specification (1 or 2), or None for default.
    invert_ancilla_order : bool
        Whether ancilla order is inverted.

    Returns
    -------
    int
        Sublayer offset (0 for layer1, 1 for layer2).
    """
    if explicit_sublayer is not None:
        return explicit_sublayer - 1  # 1→0, 2→1
    if observable_token == "X":  # noqa: S105
        # X-ancilla: layer2 by default, layer1 if inverted
        return 0 if invert_ancilla_order else 1
    if observable_token == "Z":  # noqa: S105
        # Z-ancilla: layer1 by default, layer2 if inverted
        return 1 if invert_ancilla_order else 0
    # TB, LR, etc. → layer1
    return 0


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

    couts: dict[Coord3D, dict[str, set[Coord3D]]]
    pipe_couts: dict[tuple[Coord3D, Coord3D], dict[str, set[Coord3D]]]

    __parity: CoordParityAccumulator
    flow: CoordFlowAccumulator
    scheduler: CoordScheduleAccumulator

    cube_config: dict[Coord3D, BlockConfig]
    pipe_config: dict[tuple[Coord3D, Coord3D], BlockConfig]

    logical_observables: tuple[CompositeLogicalObservableSpec, ...]
    init_flow_overrides: InitFlowOverrides | None

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
        self.logical_observables = ()
        self.init_flow_overrides = None

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

        # Compute couts if logical_observables is specified
        if logical_observables is not None:
            _, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.cube(
                self.config.d, Coord2D(global_pos.x, global_pos.y), block_config.boundary
            ).to_mutable_sets()
            self._compute_cout_from_logical_observables(
                global_pos, block_config, logical_observables, ancilla_x2d, ancilla_z2d
            )

    def _compute_cout_from_logical_observables(
        self,
        global_pos: Coord3D,
        block_config: BlockConfig,
        logical_observables: Sequence[LogicalObservableSpec],
        ancilla_x2d: AbstractSet[Coord2D],
        ancilla_z2d: AbstractSet[Coord2D],
    ) -> None:
        """Compute cout coordinates from logical observable specifications.

        Parameters
        ----------
        global_pos : Coord3D
            The global position of the cube.
        block_config : BlockConfig
            The block configuration (length determines number of unit layers).
        logical_observables : Sequence[LogicalObservableSpec]
            The logical observable specifications (multiple supported).
        ancilla_x2d : collections.abc.Set[Coord2D]
            X ancilla 2D coordinates for this cube.
        ancilla_z2d : collections.abc.Set[Coord2D]
            Z ancilla 2D coordinates for this cube.

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
        num_unit_layers = len(block_config)

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
                    msg = (
                        f"layer index {obs.layer} (resolved to {unit_layer_idx}) out of range "
                        f"for block with {num_unit_layers} unit layers. "
                        f"Valid range: 0 to {num_unit_layers - 1} or -{num_unit_layers} to -1"
                    )
                    raise ValueError(msg)
            else:
                unit_layer_idx = 0  # Default: first unit layer

            sublayer_offset = _compute_sublayer_offset(
                observable_token, obs.sublayer, block_config.invert_ancilla_order
            )
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
    ) -> None:
        """Compute cout coordinates from logical observable specifications for pipe.

        Parameters
        ----------
        global_edge : tuple[Coord3D, Coord3D]
            The global edge (start, end) of the pipe.
        block_config : BlockConfig
            The block configuration (length determines number of unit layers).
        logical_observables : Sequence[LogicalObservableSpec]
            The logical observable specifications (multiple supported).
        ancilla_x2d : collections.abc.Set[Coord2D]
            X ancilla 2D coordinates for this pipe.
        ancilla_z2d : collections.abc.Set[Coord2D]
            Z ancilla 2D coordinates for this pipe.

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
        num_unit_layers = len(block_config)

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
                    msg = (
                        f"layer index {obs.layer} (resolved to {unit_layer_idx}) out of range "
                        f"for block with {num_unit_layers} unit layers. "
                        f"Valid range: 0 to {num_unit_layers - 1} or -{num_unit_layers} to -1"
                    )
                    raise ValueError(msg)
            else:
                unit_layer_idx = 0  # Default: first unit layer

            sublayer_offset = _compute_sublayer_offset(
                observable_token, obs.sublayer, block_config.invert_ancilla_order
            )
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

    def _clear_cube_syndrome_meas_for_pipe_init(
        self,
        global_edge: tuple[Coord3D, Coord3D],
        block_config: BlockConfig,
        coord_offset: Coord3D,
    ) -> None:
        """Clear cube's syndrome_meas entries for pipe's init layer regions.

        This is called before merging pipe's graph spec to ensure that cube's
        syndrome_meas entries are cleared for positions where the pipe has
        init layers. Init layers should be excluded from detector construction.
        Note: remaining_parity is NOT cleared here, as it is needed for parity
        chain tracking.

        Parameters
        ----------
        global_edge : tuple[Coord3D, Coord3D]
            The global edge (start, end) of the pipe.
        block_config : BlockConfig
            Block configuration containing layer definitions.
        coord_offset : Coord3D
            Coordinate offset for translating local to global coordinates.
        """
        start, end = global_edge
        _, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.pipe(
            self.config.d, start, end, block_config.boundary
        ).to_mutable_sets()

        for layer_idx, layer_cfg in enumerate(block_config):
            z = coord_offset.z + layer_idx * 2

            # Layer1: clear syndrome_meas only if init=true
            # Use appropriate ancilla set based on invert_ancilla_order
            if layer_cfg.layer1.init and layer_cfg.layer1.ancilla:
                ancilla_2d = ancilla_x2d if block_config.invert_ancilla_order else ancilla_z2d
                for coord in ancilla_2d:
                    xy = Coord2D(coord.x, coord.y)
                    self.__parity.clear_syndrome_measurement_at(xy, z)

            # Layer2: clear syndrome_meas only if init=true
            if layer_cfg.layer2.init and layer_cfg.layer2.ancilla:
                ancilla_2d = ancilla_z2d if block_config.invert_ancilla_order else ancilla_x2d
                for coord in ancilla_2d:
                    xy = Coord2D(coord.x, coord.y)
                    self.__parity.clear_syndrome_measurement_at(xy, z + 1)

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

        # Clear cube's syndrome_meas for pipe's init layer regions BEFORE merging pipe's parity
        self._clear_cube_syndrome_meas_for_pipe_init(global_edge, block_config, coord_offset)

        # Merge graph spec
        self._merge_graph_spec(fragment.graph, coord_offset=coord_offset, time_offset=time_offset)

        # Compute couts if logical_observables is specified
        if logical_observables is not None:
            _, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.pipe(
                self.config.d, start, end, block_config.boundary
            ).to_mutable_sets()
            self._compute_pipe_cout_from_logical_observables(
                global_edge, block_config, logical_observables, ancilla_x2d, ancilla_z2d
            )
