"""Interfaces for init-layer ancilla flow direction analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import Coord2D, Coord3D

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lspattern.canvas_loader import CanvasPipeSpec, CanvasSpec
    from lspattern.loader import BlockConfig


class InitFlowLayerKey(NamedTuple):
    """Identify a specific unit layer and sublayer within a block."""

    unit_layer_index: int
    sublayer: int  # 1=layer1, 2=layer2


InitFlowDirectionMap = dict[InitFlowLayerKey, Coord2D]

# Per-boundary-side set of data qubit coordinates in adjacent pipe
AdjacentPipeData = dict[BoundarySide, frozenset[Coord2D]]


@dataclass(slots=True)
class InitFlowDirections:
    """Per-block init-layer flow directions in local (block) coordinates."""

    cube: dict[Coord3D, InitFlowDirectionMap] = field(default_factory=dict)
    pipe: dict[tuple[Coord3D, Coord3D], InitFlowDirectionMap] = field(default_factory=dict)
    cube_adjacent_pipe_data: dict[Coord3D, AdjacentPipeData] = field(default_factory=dict)

    def cube_directions(self, position: Coord3D) -> InitFlowDirectionMap:
        return self.cube.get(position, {})

    def pipe_directions(self, start: Coord3D, end: Coord3D) -> InitFlowDirectionMap:
        return self.pipe.get((start, end), {})

    def cube_adjacent_data(self, position: Coord3D) -> AdjacentPipeData:
        """Return adjacent pipe data for a cube, or empty dict if none."""
        return self.cube_adjacent_pipe_data.get(position, {})


_SIDE_ORDER = (
    BoundarySide.TOP,
    BoundarySide.BOTTOM,
    BoundarySide.LEFT,
    BoundarySide.RIGHT,
)

_SIDE_TO_VEC: dict[BoundarySide, Coord2D] = {
    BoundarySide.TOP: Coord2D(0, -1),  # TOP = min_y, move toward smaller y
    BoundarySide.BOTTOM: Coord2D(0, +1),  # BOTTOM = max_y, move toward larger y
    BoundarySide.LEFT: Coord2D(-1, 0),  # LEFT = min_x, move toward smaller x
    BoundarySide.RIGHT: Coord2D(+1, 0),  # RIGHT = max_x, move toward larger x
}

_SUBLAYER_1 = 1
_SUBLAYER_2 = 2


def _ancilla_type_for_init_layer(sublayer: int, invert_ancilla_order: bool) -> EdgeSpecValue:
    if sublayer == _SUBLAYER_1:
        return EdgeSpecValue.X if invert_ancilla_order else EdgeSpecValue.Z
    if sublayer == _SUBLAYER_2:
        return EdgeSpecValue.Z if invert_ancilla_order else EdgeSpecValue.X
    msg = f"Invalid init sublayer: {sublayer}."
    raise ValueError(msg)


def _candidate_sides(
    boundary: Mapping[BoundarySide, EdgeSpecValue],
    ancilla_type: EdgeSpecValue,
) -> set[BoundarySide]:
    return {side for side in _SIDE_ORDER if boundary[side] != ancilla_type}


def _adjacent_pairs(positions: set[Coord3D]) -> list[tuple[Coord3D, Coord3D, BoundarySide]]:
    pairs: list[tuple[Coord3D, Coord3D, BoundarySide]] = []
    for pos in positions:
        right = Coord3D(pos.x + 1, pos.y, pos.z)
        if right in positions:
            pairs.append((pos, right, BoundarySide.RIGHT))
        # y+1 is BOTTOM direction (TOP is y-1 per _SIDE_TO_VEC)
        bottom = Coord3D(pos.x, pos.y + 1, pos.z)
        if bottom in positions:
            pairs.append((pos, bottom, BoundarySide.BOTTOM))
    return pairs


def _violates_pair(dir_a: BoundarySide, dir_b: BoundarySide, dir_a_to_b: BoundarySide) -> bool:
    if dir_a_to_b == BoundarySide.RIGHT:
        return dir_a == BoundarySide.RIGHT and dir_b == BoundarySide.LEFT
    if dir_a_to_b == BoundarySide.BOTTOM:
        return dir_a == BoundarySide.BOTTOM and dir_b == BoundarySide.TOP
    return False


def _is_choice_consistent(
    pos: Coord3D,
    choice: BoundarySide,
    assignments: dict[Coord3D, BoundarySide],
    pairs: list[tuple[Coord3D, Coord3D, BoundarySide]],
) -> bool:
    for left, right, dir_left_to_right in pairs:
        if pos == left:
            other = right
            if other in assignments and _violates_pair(choice, assignments[other], dir_left_to_right):
                return False
        elif pos == right:
            other = left
            if other in assignments and _violates_pair(assignments[other], choice, dir_left_to_right):
                return False
    return True


def _solve_direction_assignment(
    candidates: Mapping[Coord3D, set[BoundarySide]],
    *,
    label: str,
) -> dict[Coord3D, BoundarySide]:
    positions = list(candidates.keys())
    if not positions:
        return {}

    pairs = _adjacent_pairs(set(positions))
    assignments: dict[Coord3D, BoundarySide] = {}
    ordered_positions = sorted(positions, key=lambda pos: len(candidates[pos]))

    def backtrack(idx: int) -> bool:
        if idx == len(ordered_positions):
            return True
        pos = ordered_positions[idx]
        for choice in _SIDE_ORDER:
            if choice not in candidates[pos]:
                continue
            if _is_choice_consistent(pos, choice, assignments, pairs):
                assignments[pos] = choice
                if backtrack(idx + 1):
                    return True
                assignments.pop(pos)
        return False

    if not backtrack(0):
        msg = f"No feasible init-flow direction assignment for {label}."
        raise ValueError(msg)
    return assignments


def _register_init_layer_candidates(
    group_candidates: dict[InitFlowLayerKey, dict[Coord3D, set[BoundarySide]]],
    cube_position: Coord3D,
    cube_boundary: Mapping[BoundarySide, EdgeSpecValue],
    invert_ancilla_order: bool,
    layer_idx: int,
    sublayer: int,
    cube_positions: set[Coord3D],
) -> None:
    key = InitFlowLayerKey(layer_idx, sublayer)
    ancilla_type = _ancilla_type_for_init_layer(sublayer, invert_ancilla_order)
    candidates = _candidate_sides(cube_boundary, ancilla_type)

    # Exclude directions where the adjacent cube has a cube below (z-1).
    # Init flow toward such a direction would target data qubits connected
    # via temporal edges to z-1, causing zflow cycles.
    to_exclude = set()
    for side in candidates:
        vec = _SIDE_TO_VEC[side]
        neighbor = Coord3D(cube_position.x + vec.x, cube_position.y + vec.y, cube_position.z)
        below_neighbor = Coord3D(neighbor.x, neighbor.y, neighbor.z - 1)
        if below_neighbor in cube_positions:
            to_exclude.add(side)
    candidates -= to_exclude

    if not candidates:
        return  # No valid directions; temporal flow from below handles this cube
    group_candidates.setdefault(key, {})[cube_position] = candidates


def _find_adjacent_pipe(
    cube_pos: Coord3D,
    direction: BoundarySide,
    pipes: Sequence[CanvasPipeSpec],
) -> CanvasPipeSpec | None:
    """Find pipe adjacent to cube in given direction.

    Parameters
    ----------
    cube_pos : Coord3D
        Position of the cube.
    direction : BoundarySide
        Direction to search for adjacent pipe (TOP, BOTTOM, LEFT, RIGHT).
    pipes : Sequence[CanvasPipeSpec]
        List of all pipes in the canvas.

    Returns
    -------
    CanvasPipeSpec | None
        The adjacent pipe if found, None otherwise.
    """
    # Compute the neighbor position based on direction
    if direction == BoundarySide.TOP:
        neighbor = Coord3D(cube_pos.x, cube_pos.y - 1, cube_pos.z)
    elif direction == BoundarySide.BOTTOM:
        neighbor = Coord3D(cube_pos.x, cube_pos.y + 1, cube_pos.z)
    elif direction == BoundarySide.LEFT:
        neighbor = Coord3D(cube_pos.x - 1, cube_pos.y, cube_pos.z)
    else:  # BoundarySide.RIGHT
        neighbor = Coord3D(cube_pos.x + 1, cube_pos.y, cube_pos.z)

    # Find a pipe connecting cube_pos and neighbor
    for pipe in pipes:
        if (pipe.start == cube_pos and pipe.end == neighbor) or (pipe.start == neighbor and pipe.end == cube_pos):
            return pipe
    return None


def _compute_pipe_data_for_direction(
    code_distance: int,
    direction: BoundarySide,
    pipe: CanvasPipeSpec,
) -> frozenset[Coord2D]:
    """Compute pipe data in cube-local coordinates.

    The pipe data is computed using local endpoints (as if the cube is at origin).
    This ensures the coordinates can be merged directly with the cube's data
    in construct_initial_ancilla_flow().

    Parameters
    ----------
    code_distance : int
        Code distance of the surface code.
    direction : BoundarySide
        Direction from cube to pipe.
    pipe : CanvasPipeSpec
        The pipe specification.

    Returns
    -------
    frozenset[Coord2D]
        Set of data qubit coordinates in cube-local 2D coordinates.
    """
    from lspattern.layout import RotatedSurfaceCodeLayoutBuilder  # noqa: PLC0415

    # Use local endpoints: cube at origin, neighbor at +/-1 in direction
    local_start = Coord3D(0, 0, 0)
    if direction == BoundarySide.TOP:
        local_end = Coord3D(0, -1, 0)
    elif direction == BoundarySide.BOTTOM:
        local_end = Coord3D(0, 1, 0)
    elif direction == BoundarySide.LEFT:
        local_end = Coord3D(-1, 0, 0)
    else:  # BoundarySide.RIGHT
        local_end = Coord3D(1, 0, 0)

    # Get pipe coordinates using local endpoints
    pipe_coords = RotatedSurfaceCodeLayoutBuilder.pipe(code_distance, local_start, local_end, pipe.boundary)
    return pipe_coords.data


def _collect_cube_adjacent_pipe_data(
    spec: CanvasSpec,
    code_distance: int,
) -> dict[Coord3D, AdjacentPipeData]:
    """Collect adjacent pipe data for all cubes with O (open) boundaries.

    Parameters
    ----------
    spec : CanvasSpec
        Canvas specification containing cubes and pipes.
    code_distance : int
        Code distance of the surface code.

    Returns
    -------
    dict[Coord3D, AdjacentPipeData]
        Mapping from cube position to adjacent pipe data by boundary side.
    """
    result: dict[Coord3D, AdjacentPipeData] = {}
    for cube in spec.cubes:
        adjacent_data: AdjacentPipeData = {}
        for side in _SIDE_ORDER:
            if cube.boundary.get(side) == EdgeSpecValue.O:
                pipe = _find_adjacent_pipe(cube.position, side, spec.pipes)
                if pipe is not None:
                    pipe_data = _compute_pipe_data_for_direction(code_distance, side, pipe)
                    adjacent_data[side] = pipe_data
        if adjacent_data:
            result[cube.position] = adjacent_data
    return result


def analyze_init_flow_directions(
    spec: CanvasSpec,
    *,
    code_distance: int,
    extra_paths: Sequence[Path | str] = (),
) -> InitFlowDirections:
    """Analyze boundary relationships and return init-layer flow directions."""
    if spec.layout != "rotated_surface_code":
        return InitFlowDirections()

    from lspattern.canvas_loader import load_block_config_from_name  # noqa: PLC0415

    directions = InitFlowDirections()
    block_cache: dict[str, BlockConfig] = {}
    merged_paths: tuple[Path | str, ...] = (*spec.search_paths, *extra_paths)

    cube_positions = {cube.position for cube in spec.cubes}

    group_candidates: dict[InitFlowLayerKey, dict[Coord3D, set[BoundarySide]]] = {}
    for cube in spec.cubes:
        block_config = block_cache.get(cube.block)
        if block_config is None:
            block_config = load_block_config_from_name(
                cube.block, code_distance=code_distance, extra_paths=merged_paths
            )
            block_cache[cube.block] = block_config

        if getattr(block_config, "graph_spec", None) is not None:
            continue

        for layer_idx, layer_cfg in enumerate(block_config):
            if layer_cfg.layer1.init:
                _register_init_layer_candidates(
                    group_candidates, cube.position, cube.boundary, cube.invert_ancilla_order, layer_idx, _SUBLAYER_1,
                    cube_positions,
                )
            if layer_cfg.layer2.init:
                _register_init_layer_candidates(
                    group_candidates, cube.position, cube.boundary, cube.invert_ancilla_order, layer_idx, _SUBLAYER_2,
                    cube_positions,
                )

    for key, candidates in group_candidates.items():
        assignment = _solve_direction_assignment(
            candidates, label=f"unit_layer_index={key.unit_layer_index}, sublayer={key.sublayer}"
        )
        for pos, side in assignment.items():
            directions.cube.setdefault(pos, {})[key] = _SIDE_TO_VEC[side]

    # Phase 3: Compute adjacent pipe data for cubes with O (open) boundaries
    directions.cube_adjacent_pipe_data = _collect_cube_adjacent_pipe_data(spec, code_distance)

    return directions
