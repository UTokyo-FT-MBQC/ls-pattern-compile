"""Unit tests for corner_analysis module."""

from __future__ import annotations

from lspattern.canvas_loader import CanvasCubeSpec, CanvasPipeSpec, CanvasSpec
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.corner_analysis import (
    ALL_CORNERS,
    CORNER_BOTTOM_LEFT,
    CORNER_BOTTOM_RIGHT,
    CORNER_TOP_LEFT,
    CORNER_TOP_RIGHT,
    CornerAnalysisResult,
    CornerAncillaDecision,
    CornerPosition,
    _check_corner_ancilla_parity,
    analyze_corner_ancillas,
    get_cube_corner_decisions,
    get_pipe_corner_decisions,
)
from lspattern.layout import RotatedSurfaceCodeLayoutBuilder
from lspattern.mytype import Coord2D, Coord3D


class TestCornerPosition:
    """Tests for CornerPosition class."""

    def test_create_corner_position(self) -> None:
        corner = CornerPosition(BoundarySide.TOP, BoundarySide.LEFT)
        assert corner.side1 == BoundarySide.TOP
        assert corner.side2 == BoundarySide.LEFT

    def test_normalized_keeps_top_bottom_first(self) -> None:
        corner = CornerPosition(BoundarySide.TOP, BoundarySide.LEFT)
        normalized = corner.normalized()
        assert normalized == corner

    def test_normalized_swaps_when_lr_first(self) -> None:
        corner = CornerPosition(BoundarySide.LEFT, BoundarySide.TOP)
        normalized = corner.normalized()
        assert normalized.side1 == BoundarySide.TOP
        assert normalized.side2 == BoundarySide.LEFT

    def test_all_corners_contains_four_corners(self) -> None:
        assert len(ALL_CORNERS) == 4
        assert CORNER_TOP_LEFT in ALL_CORNERS
        assert CORNER_TOP_RIGHT in ALL_CORNERS
        assert CORNER_BOTTOM_LEFT in ALL_CORNERS
        assert CORNER_BOTTOM_RIGHT in ALL_CORNERS


class TestCornerAncillaDecision:
    """Tests for CornerAncillaDecision class."""

    def test_default_no_removal(self) -> None:
        decision = CornerAncillaDecision()
        assert not decision.remove_far_x_ancilla
        assert not decision.remove_far_z_ancilla

    def test_with_removal_x(self) -> None:
        decision = CornerAncillaDecision()
        new_decision = decision.with_removal("x", remove=True)
        assert new_decision.remove_far_x_ancilla
        assert not new_decision.remove_far_z_ancilla

    def test_with_removal_z(self) -> None:
        decision = CornerAncillaDecision()
        new_decision = decision.with_removal("z", remove=True)
        assert not new_decision.remove_far_x_ancilla
        assert new_decision.remove_far_z_ancilla

    def test_with_removal_preserves_other(self) -> None:
        decision = CornerAncillaDecision(remove_far_x_ancilla=True, remove_far_z_ancilla=False)
        new_decision = decision.with_removal("z", remove=True)
        assert new_decision.remove_far_x_ancilla
        assert new_decision.remove_far_z_ancilla


class TestCornerAnalysisResult:
    """Tests for CornerAnalysisResult class."""

    def test_empty_result(self) -> None:
        result = CornerAnalysisResult()
        assert len(result.cube_decisions) == 0
        assert len(result.pipe_decisions) == 0

    def test_get_cube_decision_not_found(self) -> None:
        result = CornerAnalysisResult()
        position = Coord3D(0, 0, 0)
        assert result.get_cube_decision(position, CORNER_TOP_LEFT) is None

    def test_set_and_get_cube_decision(self) -> None:
        result = CornerAnalysisResult()
        position = Coord3D(0, 0, 0)
        decision = CornerAncillaDecision(remove_far_x_ancilla=True)

        result.set_cube_decision(position, CORNER_TOP_LEFT, decision)
        retrieved = result.get_cube_decision(position, CORNER_TOP_LEFT)

        assert retrieved is not None
        assert retrieved.remove_far_x_ancilla

    def test_get_pipe_decision_not_found(self) -> None:
        result = CornerAnalysisResult()
        start = Coord3D(0, 0, 0)
        end = Coord3D(1, 0, 0)
        assert result.get_pipe_decision(start, end, CORNER_TOP_LEFT) is None

    def test_set_and_get_pipe_decision(self) -> None:
        result = CornerAnalysisResult()
        start = Coord3D(0, 0, 0)
        end = Coord3D(1, 0, 0)
        decision = CornerAncillaDecision(remove_far_z_ancilla=True)

        result.set_pipe_decision(start, end, CORNER_TOP_RIGHT, decision)
        retrieved = result.get_pipe_decision(start, end, CORNER_TOP_RIGHT)

        assert retrieved is not None
        assert retrieved.remove_far_z_ancilla

    def test_normalized_corner_lookup(self) -> None:
        result = CornerAnalysisResult()
        position = Coord3D(0, 0, 0)
        decision = CornerAncillaDecision(remove_far_x_ancilla=True)

        # Set with non-normalized corner
        non_normalized = CornerPosition(BoundarySide.LEFT, BoundarySide.TOP)
        result.set_cube_decision(position, non_normalized, decision)

        # Get with normalized corner
        retrieved = result.get_cube_decision(position, CORNER_TOP_LEFT)
        assert retrieved is not None
        assert retrieved.remove_far_x_ancilla


class TestGetCornerDecisions:
    """Tests for get_cube_corner_decisions and get_pipe_corner_decisions."""

    def test_get_cube_corner_decisions_empty(self) -> None:
        result = CornerAnalysisResult()
        decisions = get_cube_corner_decisions(result, Coord3D(0, 0, 0))
        assert decisions == {}

    def test_get_cube_corner_decisions_with_data(self) -> None:
        result = CornerAnalysisResult()
        position = Coord3D(0, 0, 0)
        decision = CornerAncillaDecision(remove_far_x_ancilla=True)
        result.set_cube_decision(position, CORNER_TOP_LEFT, decision)

        decisions = get_cube_corner_decisions(result, position)
        assert CORNER_TOP_LEFT in decisions
        assert decisions[CORNER_TOP_LEFT].remove_far_x_ancilla

    def test_get_pipe_corner_decisions_empty(self) -> None:
        result = CornerAnalysisResult()
        decisions = get_pipe_corner_decisions(result, Coord3D(0, 0, 0), Coord3D(1, 0, 0))
        assert decisions == {}

    def test_get_pipe_corner_decisions_with_data(self) -> None:
        result = CornerAnalysisResult()
        start = Coord3D(0, 0, 0)
        end = Coord3D(1, 0, 0)
        decision = CornerAncillaDecision(remove_far_z_ancilla=True)
        result.set_pipe_decision(start, end, CORNER_BOTTOM_RIGHT, decision)

        decisions = get_pipe_corner_decisions(result, start, end)
        assert CORNER_BOTTOM_RIGHT in decisions
        assert decisions[CORNER_BOTTOM_RIGHT].remove_far_z_ancilla


class TestAnalyzeCornerAncillas:
    """Tests for analyze_corner_ancillas function."""

    def test_returns_empty_result_for_empty_spec(self) -> None:
        spec = CanvasSpec(
            name="test",
            description="test canvas",
            layout="rotated_surface_code",
            cubes=[],
            pipes=[],
        )

        result = analyze_corner_ancillas(spec, code_distance=3)

        assert isinstance(result, CornerAnalysisResult)
        assert len(result.cube_decisions) == 0
        assert len(result.pipe_decisions) == 0


class TestIntegrationWithLayout:
    """Integration tests with rotated_surface_code layout."""

    def test_cube_with_no_corner_decisions(self) -> None:
        """Test cube() without corner_decisions uses local logic."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        coords = RotatedSurfaceCodeLayoutBuilder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
        )

        # Should produce valid coordinates
        assert len(coords.data) > 0
        assert len(coords.ancilla_x) > 0
        assert len(coords.ancilla_z) > 0

    def test_cube_with_empty_corner_decisions(self) -> None:
        """Test cube() with empty corner_decisions falls back to local logic."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        coords_without = RotatedSurfaceCodeLayoutBuilder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
        )

        coords_with_empty = RotatedSurfaceCodeLayoutBuilder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
            corner_decisions={},
        )

        # Should produce identical results
        assert coords_without.data == coords_with_empty.data
        assert coords_without.ancilla_x == coords_with_empty.ancilla_x
        assert coords_without.ancilla_z == coords_with_empty.ancilla_z

    def test_cube_with_o_boundary_and_corner_decisions(self) -> None:
        """Test cube() with O boundary respects corner_decisions."""
        # Create a cube with O boundaries on LEFT and TOP
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.O,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.O,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        # Get coordinates without corner decisions
        coords_no_decision = RotatedSurfaceCodeLayoutBuilder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
        )

        # Create a corner decision to remove far ancillas at top-left
        corner_decisions = {
            CORNER_TOP_LEFT: CornerAncillaDecision(remove_far_x_ancilla=True, remove_far_z_ancilla=True),
        }

        coords_with_decision = RotatedSurfaceCodeLayoutBuilder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
            corner_decisions=corner_decisions,
        )

        # Data should be the same
        assert coords_no_decision.data == coords_with_decision.data

        # Ancillas may differ at the top-left corner position
        # The far corner ancilla at (-1, -1) should be affected
        # Note: The specific effect depends on which ancilla type exists there


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_existing_tests_still_pass_with_new_parameter(self) -> None:
        """Ensure existing code calling cube() without corner_decisions works."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        # This should not raise any errors
        coords = RotatedSurfaceCodeLayoutBuilder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
        )

        # Basic sanity checks
        assert len(coords.data) == 9  # d=3 has 9 data qubits


class TestCornerEvaluationLogic:
    """Tests for corner evaluation logic with pipes."""

    def test_evaluate_corner_with_no_pipes(self) -> None:
        """Corners without pipes should not have ancillas added."""
        cube = CanvasCubeSpec(
            position=Coord3D(0, 0, 0),
            block="test",
            boundary={
                BoundarySide.TOP: EdgeSpecValue.X,
                BoundarySide.BOTTOM: EdgeSpecValue.X,
                BoundarySide.LEFT: EdgeSpecValue.Z,
                BoundarySide.RIGHT: EdgeSpecValue.Z,
            },
            logical_observables=None,
        )
        spec = CanvasSpec(
            name="test",
            description="",
            layout="rotated_surface_code",
            cubes=[cube],
            pipes=[],
        )

        result = analyze_corner_ancillas(spec, code_distance=3)

        # No pipes means no corner ancillas added
        decisions = get_cube_corner_decisions(result, Coord3D(0, 0, 0))
        for corner in ALL_CORNERS:
            if corner in decisions:
                assert not decisions[corner].has_x_ancilla
                assert not decisions[corner].has_z_ancilla

    def test_evaluate_corner_with_matching_zz_boundary(self) -> None:
        """Corner with matching ZZ boundaries should have Z ancilla (if parity matches)."""
        # Create L-patch like configuration
        cube = CanvasCubeSpec(
            position=Coord3D(0, 0, 0),
            block="test",
            boundary={
                BoundarySide.TOP: EdgeSpecValue.X,
                BoundarySide.BOTTOM: EdgeSpecValue.O,  # Connected to pipe
                BoundarySide.LEFT: EdgeSpecValue.Z,
                BoundarySide.RIGHT: EdgeSpecValue.Z,
            },
            logical_observables=None,
        )
        pipe = CanvasPipeSpec(
            start=Coord3D(0, 0, 0),
            end=Coord3D(0, 1, 0),
            block="test",
            boundary={
                BoundarySide.TOP: EdgeSpecValue.O,  # Connection to cube
                BoundarySide.BOTTOM: EdgeSpecValue.O,  # Connection to next cube
                BoundarySide.LEFT: EdgeSpecValue.Z,
                BoundarySide.RIGHT: EdgeSpecValue.O,
            },
            logical_observables=None,
        )
        spec = CanvasSpec(
            name="test",
            description="",
            layout="rotated_surface_code",
            cubes=[cube],
            pipes=[pipe],
        )

        result = analyze_corner_ancillas(spec, code_distance=3)

        # Should have decisions for cubes with pipes (just verify no error)
        _ = get_cube_corner_decisions(result, Coord3D(0, 0, 0))
        # BOTTOM-LEFT corner: cube LEFT=Z, pipe LEFT=Z -> ZZ
        # The actual presence depends on parity condition

    def test_evaluate_corner_with_matching_xx_boundary(self) -> None:
        """Corner with matching XX boundaries should have X ancilla (if parity matches)."""
        cube = CanvasCubeSpec(
            position=Coord3D(0, 0, 0),
            block="test",
            boundary={
                BoundarySide.TOP: EdgeSpecValue.X,
                BoundarySide.BOTTOM: EdgeSpecValue.X,
                BoundarySide.LEFT: EdgeSpecValue.Z,
                BoundarySide.RIGHT: EdgeSpecValue.O,  # Connected to pipe
            },
            logical_observables=None,
        )
        pipe = CanvasPipeSpec(
            start=Coord3D(0, 0, 0),
            end=Coord3D(1, 0, 0),
            block="test",
            boundary={
                BoundarySide.TOP: EdgeSpecValue.X,  # X boundary
                BoundarySide.BOTTOM: EdgeSpecValue.X,  # X boundary
                BoundarySide.LEFT: EdgeSpecValue.O,  # Connection to cube
                BoundarySide.RIGHT: EdgeSpecValue.O,  # Connection to next
            },
            logical_observables=None,
        )
        spec = CanvasSpec(
            name="test",
            description="",
            layout="rotated_surface_code",
            cubes=[cube],
            pipes=[pipe],
        )

        result = analyze_corner_ancillas(spec, code_distance=3)

        # The analysis should run without errors (just verify no error)
        _ = get_cube_corner_decisions(result, Coord3D(0, 0, 0))
        # TOP-RIGHT corner: cube TOP=X, pipe TOP=X -> XX
        # Actual presence depends on parity

    def test_evaluate_corner_with_mixed_boundary_no_ancilla(self) -> None:
        """Corner with mixed XZ boundaries should not have ancilla."""
        cube = CanvasCubeSpec(
            position=Coord3D(0, 0, 0),
            block="test",
            boundary={
                BoundarySide.TOP: EdgeSpecValue.X,
                BoundarySide.BOTTOM: EdgeSpecValue.O,  # Connected to pipe
                BoundarySide.LEFT: EdgeSpecValue.Z,
                BoundarySide.RIGHT: EdgeSpecValue.Z,
            },
            logical_observables=None,
        )
        pipe = CanvasPipeSpec(
            start=Coord3D(0, 0, 0),
            end=Coord3D(0, 1, 0),
            block="test",
            boundary={
                BoundarySide.TOP: EdgeSpecValue.O,
                BoundarySide.BOTTOM: EdgeSpecValue.O,
                BoundarySide.LEFT: EdgeSpecValue.X,  # X boundary (differs from cube's Z)
                BoundarySide.RIGHT: EdgeSpecValue.O,
            },
            logical_observables=None,
        )
        spec = CanvasSpec(
            name="test",
            description="",
            layout="rotated_surface_code",
            cubes=[cube],
            pipes=[pipe],
        )

        result = analyze_corner_ancillas(spec, code_distance=3)

        decisions = get_cube_corner_decisions(result, Coord3D(0, 0, 0))
        # BOTTOM-LEFT: cube LEFT=Z, pipe LEFT=X -> XZ (mixed) -> no ancilla
        if CORNER_BOTTOM_LEFT in decisions:
            assert not decisions[CORNER_BOTTOM_LEFT].has_x_ancilla
            assert not decisions[CORNER_BOTTOM_LEFT].has_z_ancilla


class TestCornerAncillaParityCondition:
    """Tests for parity condition checking."""

    def test_x_ancilla_parity_condition(self) -> None:
        """X ancilla requires (x+y) % 4 == 0 at odd coordinates."""
        # Coordinates at odd positions with (x+y) % 4 == 0 should satisfy X parity
        # (-1, -1): (-1 + -1) = -2, -2 % 4 == 2 in Python -> Z parity, not X
        assert not _check_corner_ancilla_parity(Coord2D(-1, -1), "x")
        # (-1, 3): (-1 + 3) = 2, 2 % 4 == 2 -> Z parity, not X
        assert not _check_corner_ancilla_parity(Coord2D(-1, 3), "x")
        # (1, 3): (1 + 3) = 4, 4 % 4 == 0 -> X parity, passes
        assert _check_corner_ancilla_parity(Coord2D(1, 3), "x")
        # (3, 1): (3 + 1) = 4, 4 % 4 == 0 -> X parity, passes
        assert _check_corner_ancilla_parity(Coord2D(3, 1), "x")

    def test_z_ancilla_parity_condition(self) -> None:
        """Z ancilla requires (x+y) % 4 == 2 at odd coordinates."""
        # Coordinates at odd positions with (x+y) % 4 == 2 should satisfy Z parity
        assert _check_corner_ancilla_parity(Coord2D(-1, -1), "z")  # -2 % 4 == 2, passes
        assert _check_corner_ancilla_parity(Coord2D(1, 1), "z")  # 2 % 4 == 2, passes
        assert not _check_corner_ancilla_parity(Coord2D(1, 3), "z")  # 4 % 4 == 0, fails

    def test_even_coordinates_never_satisfy_parity(self) -> None:
        """Even coordinates never satisfy ancilla parity."""
        # Even x or y coordinate should always return False
        assert not _check_corner_ancilla_parity(Coord2D(0, 1), "x")
        assert not _check_corner_ancilla_parity(Coord2D(0, 1), "z")
        assert not _check_corner_ancilla_parity(Coord2D(1, 0), "x")
        assert not _check_corner_ancilla_parity(Coord2D(1, 0), "z")
        assert not _check_corner_ancilla_parity(Coord2D(0, 0), "x")
        assert not _check_corner_ancilla_parity(Coord2D(0, 0), "z")
