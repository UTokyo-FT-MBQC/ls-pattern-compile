from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from lspattern.canvas_loader import CanvasCubeSpec, CanvasPipeSpec, CanvasSpec
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.init_flow_analysis import (
    InitFlowLayerKey,
    _adjacent_pairs,
    _ancilla_type_for_init_layer,
    _candidate_sides,
    _solve_direction_assignment,
    _violates_pair,
    analyze_init_flow_directions,
)
from lspattern.mytype import Coord2D, Coord3D


def _boundary(
    *,
    top: EdgeSpecValue,
    bottom: EdgeSpecValue,
    left: EdgeSpecValue,
    right: EdgeSpecValue,
) -> dict[BoundarySide, EdgeSpecValue]:
    return {
        BoundarySide.TOP: top,
        BoundarySide.BOTTOM: bottom,
        BoundarySide.LEFT: left,
        BoundarySide.RIGHT: right,
    }


def _spec_with_cubes(
    *cubes: CanvasCubeSpec,
    pipes: tuple[CanvasPipeSpec, ...] = (),
    search_paths: tuple[Path, ...] = (),
) -> CanvasSpec:
    return CanvasSpec(
        name="test-canvas",
        description="",
        layout="rotated_surface_code",
        cubes=list(cubes),
        pipes=list(pipes),
        search_paths=search_paths,
        logical_observables=(),
    )


def _pipe(
    start: Coord3D,
    end: Coord3D,
    boundary: dict[BoundarySide, EdgeSpecValue],
    *,
    block: str = "memory_block.yml",
) -> CanvasPipeSpec:
    return CanvasPipeSpec(
        start=start,
        end=end,
        block=block,
        boundary=boundary,
        logical_observables=None,
        invert_ancilla_order=False,
    )


def test_init_flow_direction_single_cube_prefers_left_candidate() -> None:
    boundary = _boundary(
        top=EdgeSpecValue.Z,
        bottom=EdgeSpecValue.Z,
        left=EdgeSpecValue.X,
        right=EdgeSpecValue.X,
    )
    pos = Coord3D(0, 0, 0)
    spec = _spec_with_cubes(
        CanvasCubeSpec(
            position=pos,
            block="init_plus_block.yml",
            boundary=boundary,
            logical_observables=None,
            invert_ancilla_order=False,
        )
    )

    directions = analyze_init_flow_directions(spec, code_distance=3)
    key = InitFlowLayerKey(0, 1)
    assert directions.cube_directions(pos)[key] == Coord2D(-1, 0)


def test_init_flow_direction_avoids_opposing_adjacent() -> None:
    left_pos = Coord3D(0, 0, 0)
    right_pos = Coord3D(1, 0, 0)

    left_boundary = _boundary(
        top=EdgeSpecValue.X,
        bottom=EdgeSpecValue.Z,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.X,
    )
    right_boundary = _boundary(
        top=EdgeSpecValue.Z,
        bottom=EdgeSpecValue.X,
        left=EdgeSpecValue.X,
        right=EdgeSpecValue.Z,
    )

    spec = _spec_with_cubes(
        CanvasCubeSpec(
            position=left_pos,
            block="init_plus_block.yml",
            boundary=left_boundary,
            logical_observables=None,
            invert_ancilla_order=False,
        ),
        CanvasCubeSpec(
            position=right_pos,
            block="init_plus_block.yml",
            boundary=right_boundary,
            logical_observables=None,
            invert_ancilla_order=False,
        ),
    )

    directions = analyze_init_flow_directions(spec, code_distance=3)
    key = InitFlowLayerKey(0, 1)

    # TOP = (0, -1), BOTTOM = (0, +1) in the coordinate system
    assert directions.cube_directions(left_pos)[key] == Coord2D(0, -1)  # TOP
    assert directions.cube_directions(right_pos)[key] == Coord2D(0, +1)  # BOTTOM


def test_init_flow_direction_excludes_side_with_same_slot_pipe_basis_non_null_at_physical_z_minus_1(
    tmp_path: Path,
) -> None:
    delayed_init_block = dedent("""
    name: DelayedInitBlock
    description: init layer appears after one memory unit
    layers:
      - type: MemoryUnit
        num_layers: 1
      - type: InitPlusUnit
        num_layers: 1
    """)
    (tmp_path / "delayed_init_block.yml").write_text(delayed_init_block, encoding="utf-8")

    pos = Coord3D(0, 0, 1)
    boundary = _boundary(
        top=EdgeSpecValue.Z,
        bottom=EdgeSpecValue.Z,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.O,
    )
    pipe_boundary = _boundary(
        top=EdgeSpecValue.X,
        bottom=EdgeSpecValue.X,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.Z,
    )
    spec = _spec_with_cubes(
        CanvasCubeSpec(
            position=pos,
            block="delayed_init_block.yml",
            boundary=boundary,
            logical_observables=None,
            invert_ancilla_order=False,
        ),
        pipes=(
            _pipe(Coord3D(0, 0, 1), Coord3D(1, 0, 1), pipe_boundary),
        ),
        search_paths=(tmp_path,),
    )

    directions = analyze_init_flow_directions(spec, code_distance=3)
    key = InitFlowLayerKey(1, 1)
    assert key not in directions.cube_directions(pos)


def test_init_flow_direction_excludes_side_with_lower_slot_pipe_basis_non_null_at_physical_z_minus_1() -> None:
    pos = Coord3D(0, 0, 1)
    boundary = _boundary(
        top=EdgeSpecValue.Z,
        bottom=EdgeSpecValue.Z,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.O,
    )
    pipe_boundary = _boundary(
        top=EdgeSpecValue.X,
        bottom=EdgeSpecValue.X,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.Z,
    )
    spec = _spec_with_cubes(
        CanvasCubeSpec(
            position=pos,
            block="init_plus_block.yml",
            boundary=boundary,
            logical_observables=None,
            invert_ancilla_order=False,
        ),
        pipes=(
            # same-slot pipe exists but its z-1 slice basis is null at d=3
            _pipe(Coord3D(0, 0, 1), Coord3D(1, 0, 1), pipe_boundary, block="measure_x_block.yml"),
            # lower-slot pipe has non-null basis at the same physical z-1
            _pipe(Coord3D(0, 0, 0), Coord3D(1, 0, 0), pipe_boundary, block="memory_block.yml"),
        ),
    )

    directions = analyze_init_flow_directions(spec, code_distance=3)
    key = InitFlowLayerKey(0, 1)
    assert key not in directions.cube_directions(pos)


def test_init_flow_direction_not_excluded_when_lower_slot_pipe_basis_null_at_physical_z_minus_1() -> None:
    pos = Coord3D(0, 0, 1)
    boundary = _boundary(
        top=EdgeSpecValue.Z,
        bottom=EdgeSpecValue.Z,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.O,
    )
    pipe_boundary = _boundary(
        top=EdgeSpecValue.X,
        bottom=EdgeSpecValue.X,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.Z,
    )
    spec = _spec_with_cubes(
        CanvasCubeSpec(
            position=pos,
            block="init_plus_block.yml",
            boundary=boundary,
            logical_observables=None,
            invert_ancilla_order=False,
        ),
        pipes=(
            # lower-slot pipe exists but its local z=5 basis is null for measure_x_block at d=3
            _pipe(Coord3D(0, 0, 0), Coord3D(1, 0, 0), pipe_boundary, block="measure_x_block.yml"),
        ),
    )

    directions = analyze_init_flow_directions(spec, code_distance=3)
    key = InitFlowLayerKey(0, 1)
    assert directions.cube_directions(pos)[key] == Coord2D(1, 0)


def test_init_flow_direction_not_excluded_by_other_side_lower_slot_pipe() -> None:
    pos = Coord3D(0, 0, 1)
    boundary = _boundary(
        top=EdgeSpecValue.Z,
        bottom=EdgeSpecValue.Z,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.O,
    )
    pipe_boundary = _boundary(
        top=EdgeSpecValue.X,
        bottom=EdgeSpecValue.X,
        left=EdgeSpecValue.Z,
        right=EdgeSpecValue.Z,
    )
    spec = _spec_with_cubes(
        CanvasCubeSpec(
            position=pos,
            block="init_plus_block.yml",
            boundary=boundary,
            logical_observables=None,
            invert_ancilla_order=False,
        ),
        pipes=(
            _pipe(Coord3D(0, 0, 1), Coord3D(1, 0, 1), pipe_boundary, block="measure_x_block.yml"),
            _pipe(Coord3D(0, 0, 0), Coord3D(-1, 0, 0), pipe_boundary, block="memory_block.yml"),
        ),
    )

    directions = analyze_init_flow_directions(spec, code_distance=3)
    key = InitFlowLayerKey(0, 1)
    assert directions.cube_directions(pos)[key] == Coord2D(1, 0)


# =============================================================================
# Tests for _ancilla_type_for_init_layer
# =============================================================================


class TestAncillaTypeForInitLayer:
    @pytest.mark.parametrize(
        ("sublayer", "invert", "expected"),
        [
            (1, False, EdgeSpecValue.Z),
            (1, True, EdgeSpecValue.X),
            (2, False, EdgeSpecValue.X),
            (2, True, EdgeSpecValue.Z),
        ],
    )
    def test_valid_sublayer(self, sublayer: int, invert: bool, expected: EdgeSpecValue) -> None:
        assert _ancilla_type_for_init_layer(sublayer, invert) == expected

    @pytest.mark.parametrize("sublayer", [0, 3, -1, 100])
    def test_invalid_sublayer_raises(self, sublayer: int) -> None:
        with pytest.raises(ValueError, match="Invalid init sublayer"):
            _ancilla_type_for_init_layer(sublayer, False)


# =============================================================================
# Tests for _candidate_sides
# =============================================================================


class TestCandidateSides:
    def test_returns_non_z_sides_when_ancilla_is_z(self) -> None:
        boundary = _boundary(
            top=EdgeSpecValue.Z,
            bottom=EdgeSpecValue.Z,
            left=EdgeSpecValue.X,
            right=EdgeSpecValue.X,
        )
        result = _candidate_sides(boundary, EdgeSpecValue.Z)
        assert result == {BoundarySide.LEFT, BoundarySide.RIGHT}

    def test_returns_non_x_sides_when_ancilla_is_x(self) -> None:
        boundary = _boundary(
            top=EdgeSpecValue.Z,
            bottom=EdgeSpecValue.Z,
            left=EdgeSpecValue.X,
            right=EdgeSpecValue.X,
        )
        result = _candidate_sides(boundary, EdgeSpecValue.X)
        assert result == {BoundarySide.TOP, BoundarySide.BOTTOM}

    def test_all_sides_same_returns_all_when_different(self) -> None:
        boundary = _boundary(
            top=EdgeSpecValue.Z,
            bottom=EdgeSpecValue.Z,
            left=EdgeSpecValue.Z,
            right=EdgeSpecValue.Z,
        )
        result = _candidate_sides(boundary, EdgeSpecValue.X)
        assert result == {
            BoundarySide.TOP,
            BoundarySide.BOTTOM,
            BoundarySide.LEFT,
            BoundarySide.RIGHT,
        }

    def test_all_sides_same_returns_empty_when_same(self) -> None:
        boundary = _boundary(
            top=EdgeSpecValue.Z,
            bottom=EdgeSpecValue.Z,
            left=EdgeSpecValue.Z,
            right=EdgeSpecValue.Z,
        )
        result = _candidate_sides(boundary, EdgeSpecValue.Z)
        assert result == set()

    def test_with_open_boundary_o_returns_for_z_ancilla(self) -> None:
        boundary = _boundary(
            top=EdgeSpecValue.O,
            bottom=EdgeSpecValue.Z,
            left=EdgeSpecValue.X,
            right=EdgeSpecValue.X,
        )
        result = _candidate_sides(boundary, EdgeSpecValue.Z)
        assert result == {BoundarySide.TOP, BoundarySide.LEFT, BoundarySide.RIGHT}

    def test_with_open_boundary_o_returns_for_x_ancilla(self) -> None:
        boundary = _boundary(
            top=EdgeSpecValue.O,
            bottom=EdgeSpecValue.O,
            left=EdgeSpecValue.X,
            right=EdgeSpecValue.Z,
        )
        result = _candidate_sides(boundary, EdgeSpecValue.X)
        assert result == {BoundarySide.TOP, BoundarySide.BOTTOM, BoundarySide.RIGHT}

    def test_all_open_boundaries_returns_all(self) -> None:
        boundary = _boundary(
            top=EdgeSpecValue.O,
            bottom=EdgeSpecValue.O,
            left=EdgeSpecValue.O,
            right=EdgeSpecValue.O,
        )
        for ancilla_type in [EdgeSpecValue.Z, EdgeSpecValue.X]:
            result = _candidate_sides(boundary, ancilla_type)
            assert result == {
                BoundarySide.TOP,
                BoundarySide.BOTTOM,
                BoundarySide.LEFT,
                BoundarySide.RIGHT,
            }


# =============================================================================
# Tests for _adjacent_pairs
# =============================================================================


class TestAdjacentPairs:
    def test_empty_set(self) -> None:
        assert _adjacent_pairs(set()) == []

    def test_single_position(self) -> None:
        assert _adjacent_pairs({Coord3D(0, 0, 0)}) == []

    def test_horizontal_adjacent_pair(self) -> None:
        positions = {Coord3D(0, 0, 0), Coord3D(1, 0, 0)}
        result = _adjacent_pairs(positions)
        assert len(result) == 1
        assert result[0] == (Coord3D(0, 0, 0), Coord3D(1, 0, 0), BoundarySide.RIGHT)

    def test_vertical_adjacent_pair(self) -> None:
        # y+1 is BOTTOM direction (TOP is y-1 per _SIDE_TO_VEC)
        positions = {Coord3D(0, 0, 0), Coord3D(0, 1, 0)}
        result = _adjacent_pairs(positions)
        assert len(result) == 1
        assert result[0] == (Coord3D(0, 0, 0), Coord3D(0, 1, 0), BoundarySide.BOTTOM)

    def test_non_adjacent_positions(self) -> None:
        positions = {Coord3D(0, 0, 0), Coord3D(2, 0, 0)}
        result = _adjacent_pairs(positions)
        assert result == []

    def test_2x2_grid(self) -> None:
        positions = {
            Coord3D(0, 0, 0),
            Coord3D(1, 0, 0),
            Coord3D(0, 1, 0),
            Coord3D(1, 1, 0),
        }
        result = _adjacent_pairs(positions)
        # Should find 4 pairs: (0,0)-(1,0), (0,1)-(1,1), (0,0)-(0,1), (1,0)-(1,1)
        assert len(result) == 4

    def test_different_z_not_adjacent(self) -> None:
        positions = {Coord3D(0, 0, 0), Coord3D(0, 0, 1)}
        result = _adjacent_pairs(positions)
        assert result == []


# =============================================================================
# Tests for _violates_pair
# =============================================================================


class TestViolatesPair:
    def test_horizontal_opposing_violates(self) -> None:
        assert _violates_pair(BoundarySide.RIGHT, BoundarySide.LEFT, BoundarySide.RIGHT)

    def test_horizontal_same_direction_not_violates(self) -> None:
        assert not _violates_pair(BoundarySide.RIGHT, BoundarySide.RIGHT, BoundarySide.RIGHT)

    def test_vertical_opposing_violates(self) -> None:
        # dir_a_to_b is BOTTOM (y+1 direction), violation if a=BOTTOM, b=TOP
        assert _violates_pair(BoundarySide.BOTTOM, BoundarySide.TOP, BoundarySide.BOTTOM)

    def test_vertical_same_direction_not_violates(self) -> None:
        # dir_a_to_b is BOTTOM; same direction doesn't violate
        assert not _violates_pair(BoundarySide.BOTTOM, BoundarySide.BOTTOM, BoundarySide.BOTTOM)

    def test_orthogonal_directions_not_violates(self) -> None:
        assert not _violates_pair(BoundarySide.LEFT, BoundarySide.TOP, BoundarySide.RIGHT)
        assert not _violates_pair(BoundarySide.RIGHT, BoundarySide.BOTTOM, BoundarySide.RIGHT)

    def test_left_direction_between_pair_not_violates(self) -> None:
        # dir_a_to_b is neither RIGHT nor TOP, so always returns False
        assert not _violates_pair(BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.LEFT)
        assert not _violates_pair(BoundarySide.TOP, BoundarySide.BOTTOM, BoundarySide.BOTTOM)


# =============================================================================
# Tests for _solve_direction_assignment
# =============================================================================


class TestSolveDirectionAssignment:
    def test_empty_candidates(self) -> None:
        result = _solve_direction_assignment({}, label="test")
        assert result == {}

    def test_single_position_returns_first_candidate_by_priority(self) -> None:
        candidates = {Coord3D(0, 0, 0): {BoundarySide.LEFT, BoundarySide.RIGHT}}
        result = _solve_direction_assignment(candidates, label="test")
        # Priority order: TOP, BOTTOM, LEFT, RIGHT
        assert result[Coord3D(0, 0, 0)] == BoundarySide.LEFT

    def test_single_position_with_top_candidate(self) -> None:
        candidates = {Coord3D(0, 0, 0): {BoundarySide.TOP, BoundarySide.BOTTOM}}
        result = _solve_direction_assignment(candidates, label="test")
        assert result[Coord3D(0, 0, 0)] == BoundarySide.TOP

    def test_multiple_non_adjacent_positions(self) -> None:
        candidates = {
            Coord3D(0, 0, 0): {BoundarySide.LEFT, BoundarySide.RIGHT},
            Coord3D(5, 5, 0): {BoundarySide.TOP, BoundarySide.BOTTOM},
        }
        result = _solve_direction_assignment(candidates, label="test")
        assert result[Coord3D(0, 0, 0)] == BoundarySide.LEFT
        assert result[Coord3D(5, 5, 0)] == BoundarySide.TOP

    def test_adjacent_positions_avoids_conflict(self) -> None:
        # Two horizontally adjacent positions that would conflict if both point inward
        candidates = {
            Coord3D(0, 0, 0): {BoundarySide.RIGHT, BoundarySide.TOP},
            Coord3D(1, 0, 0): {BoundarySide.LEFT, BoundarySide.BOTTOM},
        }
        result = _solve_direction_assignment(candidates, label="test")
        # Should not have both pointing at each other
        assert not (result[Coord3D(0, 0, 0)] == BoundarySide.RIGHT and result[Coord3D(1, 0, 0)] == BoundarySide.LEFT)

    def test_vertically_adjacent_avoids_conflict(self) -> None:
        # (0,0,0) is at y+1 direction (BOTTOM) from perspective of (0,0,0) to (0,1,0)
        # Conflict occurs when (0,0,0) points BOTTOM and (0,1,0) points TOP
        candidates = {
            Coord3D(0, 0, 0): {BoundarySide.BOTTOM, BoundarySide.LEFT},
            Coord3D(0, 1, 0): {BoundarySide.TOP, BoundarySide.RIGHT},
        }
        result = _solve_direction_assignment(candidates, label="test")
        # Should not have both pointing at each other
        assert not (result[Coord3D(0, 0, 0)] == BoundarySide.BOTTOM and result[Coord3D(0, 1, 0)] == BoundarySide.TOP)

    def test_no_solution_raises_value_error(self) -> None:
        # Two horizontally adjacent positions, but both can only point inward
        candidates = {
            Coord3D(0, 0, 0): {BoundarySide.RIGHT},
            Coord3D(1, 0, 0): {BoundarySide.LEFT},
        }
        with pytest.raises(ValueError, match="No feasible init-flow direction"):
            _solve_direction_assignment(candidates, label="test")
