"""Interfaces for init-layer ancilla flow direction analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import Coord2D, Coord3D

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from lspattern.canvas_loader import CanvasSpec
    from lspattern.loader import BlockConfig


class InitFlowLayerKey(NamedTuple):
    """Identify a specific unit layer and sublayer within a block."""

    unit_layer_index: int
    sublayer: int  # 1=layer1, 2=layer2


InitFlowDirectionMap = dict[InitFlowLayerKey, Coord2D]


@dataclass(slots=True)
class InitFlowDirections:
    """Per-block init-layer flow directions in local (block) coordinates."""

    cube: dict[Coord3D, InitFlowDirectionMap] = field(default_factory=dict)
    pipe: dict[tuple[Coord3D, Coord3D], InitFlowDirectionMap] = field(default_factory=dict)

    def cube_directions(self, position: Coord3D) -> InitFlowDirectionMap:
        return self.cube.get(position, {})

    def pipe_directions(self, start: Coord3D, end: Coord3D) -> InitFlowDirectionMap:
        return self.pipe.get((start, end), {})


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
) -> None:
    key = InitFlowLayerKey(layer_idx, sublayer)
    ancilla_type = _ancilla_type_for_init_layer(sublayer, invert_ancilla_order)
    candidates = _candidate_sides(cube_boundary, ancilla_type)
    if not candidates:
        msg = f"No candidate directions for cube {cube_position} layer{sublayer} init."
        raise ValueError(msg)
    group_candidates.setdefault(key, {})[cube_position] = candidates


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
                    group_candidates, cube.position, cube.boundary, cube.invert_ancilla_order, layer_idx, _SUBLAYER_1
                )
            if layer_cfg.layer2.init:
                _register_init_layer_candidates(
                    group_candidates, cube.position, cube.boundary, cube.invert_ancilla_order, layer_idx, _SUBLAYER_2
                )

    for key, candidates in group_candidates.items():
        assignment = _solve_direction_assignment(
            candidates, label=f"unit_layer_index={key.unit_layer_index}, sublayer={key.sublayer}"
        )
        for pos, side in assignment.items():
            directions.cube.setdefault(pos, {})[key] = _SIDE_TO_VEC[side]

    return directions
