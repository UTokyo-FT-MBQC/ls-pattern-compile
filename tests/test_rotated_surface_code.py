"""Tests for patch_regions module."""

from __future__ import annotations

import pytest

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.layout import (
    AncillaFlowConstructor,
    BoundaryAncillaRetriever,
    BoundaryPathCalculator,
    BoundsCalculator,
    CoordinateGenerator,
    PatchBounds,
    PatchCoordinates,
    PipeDirectionHelper,
    RotatedSurfaceCodeLayout,
    RotatedSurfaceCodeLayoutBuilder,
    TopologicalCodeLayoutBuilder,
)
from lspattern.mytype import AxisDIRECTION2D, Coord2D, Coord3D

# Alias for shorter test code
Builder = RotatedSurfaceCodeLayoutBuilder


# =============================================================================
# Tests for PatchBounds
# =============================================================================


class TestPatchBounds:
    """Tests for PatchBounds dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic PatchBounds creation."""
        bounds = PatchBounds(x_min=0, x_max=4, y_min=0, y_max=4)
        assert bounds.x_min == 0
        assert bounds.x_max == 4
        assert bounds.y_min == 0
        assert bounds.y_max == 4


# =============================================================================
# Tests for PatchCoordinates
# =============================================================================


class TestPatchCoordinates:
    """Tests for PatchCoordinates dataclass."""

    def test_to_mutable_sets(self) -> None:
        """Test conversion to mutable sets."""
        coords = PatchCoordinates(
            data=frozenset({Coord2D(0, 0), Coord2D(2, 0)}),
            ancilla_x=frozenset({Coord2D(1, 1)}),
            ancilla_z=frozenset({Coord2D(3, 1)}),
        )

        data, x_anc, z_anc = coords.to_mutable_sets()

        assert isinstance(data, set)
        assert isinstance(x_anc, set)
        assert isinstance(z_anc, set)
        assert data == {Coord2D(0, 0), Coord2D(2, 0)}


# =============================================================================
# Tests for RotatedSurfaceCodeLayoutBuilder.cube
# =============================================================================


class TestBuilderCube:
    """Tests for RotatedSurfaceCodeLayoutBuilder.cube static method."""

    def test_cube_basic(self) -> None:
        """Test basic cube layout generation."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        coords = Builder.cube(code_distance=3, global_pos=Coord2D(0, 0), boundary=boundary)

        assert isinstance(coords, PatchCoordinates)
        assert len(coords.data) > 0
        assert len(coords.ancilla_x) > 0
        assert len(coords.ancilla_z) > 0

    def test_cube_data_at_even_coords(self) -> None:
        """Test that data qubits are at even coordinates."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        coords = Builder.cube(code_distance=3, global_pos=Coord2D(0, 0), boundary=boundary)

        for coord in coords.data:
            assert coord.x % 2 == 0
            assert coord.y % 2 == 0

    @pytest.mark.parametrize("code_distance", [2, 3, 4, 5])
    def test_cube_various_distances(self, code_distance: int) -> None:
        """Test cube layout with various code distances."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        coords = Builder.cube(code_distance=code_distance, global_pos=Coord2D(0, 0), boundary=boundary)

        assert len(coords.data) > 0
        # Number of data qubits should scale with code distance
        expected_data_count = code_distance**2
        # Allow for corner removal (up to 2 corners may be removed)
        assert len(coords.data) >= expected_data_count - 2

    @pytest.mark.parametrize(
        "boundary_config",
        [
            {
                BoundarySide.TOP: EdgeSpecValue.X,
                BoundarySide.BOTTOM: EdgeSpecValue.X,
                BoundarySide.LEFT: EdgeSpecValue.Z,
                BoundarySide.RIGHT: EdgeSpecValue.Z,
            },
            {
                BoundarySide.TOP: EdgeSpecValue.Z,
                BoundarySide.BOTTOM: EdgeSpecValue.Z,
                BoundarySide.LEFT: EdgeSpecValue.X,
                BoundarySide.RIGHT: EdgeSpecValue.X,
            },
            {
                BoundarySide.TOP: EdgeSpecValue.O,
                BoundarySide.BOTTOM: EdgeSpecValue.O,
                BoundarySide.LEFT: EdgeSpecValue.X,
                BoundarySide.RIGHT: EdgeSpecValue.Z,
            },
        ],
    )
    def test_cube_various_boundaries(
        self,
        boundary_config: dict[BoundarySide, EdgeSpecValue],
    ) -> None:
        """Test cube layout with various boundary configurations."""
        coords = Builder.cube(code_distance=3, global_pos=Coord2D(0, 0), boundary=boundary_config)

        assert isinstance(coords, PatchCoordinates)
        assert len(coords.data) > 0


# =============================================================================
# Tests for RotatedSurfaceCodeLayoutBuilder.pipe
# =============================================================================


class TestBuilderPipe:
    """Tests for RotatedSurfaceCodeLayoutBuilder.pipe static method."""

    def test_pipe_horizontal(self) -> None:
        """Test horizontal pipe layout."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.O,
            BoundarySide.BOTTOM: EdgeSpecValue.O,
            BoundarySide.LEFT: EdgeSpecValue.X,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

        source = Coord3D(0, 0, 0)
        target = Coord3D(0, 1, 0)

        coords = Builder.pipe(
            code_distance=3,
            global_pos_source=source,
            global_pos_target=target,
            boundary=boundary,
        )

        assert isinstance(coords, PatchCoordinates)
        assert len(coords.data) > 0

    def test_pipe_vertical(self) -> None:
        """Test vertical pipe layout."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.Z,
            BoundarySide.LEFT: EdgeSpecValue.O,
            BoundarySide.RIGHT: EdgeSpecValue.O,
        }

        source = Coord3D(0, 0, 0)
        target = Coord3D(1, 0, 0)

        coords = Builder.pipe(
            code_distance=3,
            global_pos_source=source,
            global_pos_target=target,
            boundary=boundary,
        )

        assert isinstance(coords, PatchCoordinates)
        assert len(coords.data) > 0


# =============================================================================
# Tests for PatchBounds center properties
# =============================================================================


class TestPatchBoundsCenter:
    """Tests for PatchBounds center_x and center_y properties."""

    def test_center_x_even(self) -> None:
        """Test center_x returns even value when center is already even."""
        bounds = PatchBounds(x_min=0, x_max=4, y_min=0, y_max=4)
        # (0 + 4) // 2 = 2, which is even
        assert bounds.center_x == 2

    def test_center_x_rounds_to_even(self) -> None:
        """Test center_x rounds to nearest even value."""
        bounds = PatchBounds(x_min=0, x_max=6, y_min=0, y_max=6)
        # (0 + 6) // 2 = 3, which is odd, so should return 4
        assert bounds.center_x == 4

    def test_center_y_even(self) -> None:
        """Test center_y returns even value when center is already even."""
        bounds = PatchBounds(x_min=0, x_max=4, y_min=0, y_max=4)
        # (0 + 4) // 2 = 2, which is even
        assert bounds.center_y == 2

    def test_center_y_rounds_to_even(self) -> None:
        """Test center_y rounds to nearest even value."""
        bounds = PatchBounds(x_min=0, x_max=6, y_min=0, y_max=6)
        # (0 + 6) // 2 = 3, which is odd, so should return 4
        assert bounds.center_y == 4

    def test_center_with_offset_bounds(self) -> None:
        """Test center calculation with non-zero offset bounds."""
        bounds = PatchBounds(x_min=10, x_max=14, y_min=20, y_max=24)
        # (10 + 14) // 2 = 12, which is even
        assert bounds.center_x == 12
        # (20 + 24) // 2 = 22, which is even
        assert bounds.center_y == 22


# =============================================================================
# Tests for RotatedSurfaceCodeLayoutBuilder.cube_boundary_path
# =============================================================================


class TestBuilderCubeBoundaryPath:
    """Tests for RotatedSurfaceCodeLayoutBuilder.cube_boundary_path."""

    @pytest.fixture
    def standard_boundary(self) -> dict[BoundarySide, EdgeSpecValue]:
        """Standard boundary configuration for tests."""
        return {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

    def test_vertical_path_top_to_bottom(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test vertical path from TOP to BOTTOM."""
        path = Builder.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.TOP,
            side_b=BoundarySide.BOTTOM,
        )
        assert len(path) > 0
        # All coordinates should have the same x (center)
        xs = {c.x for c in path}
        assert len(xs) == 1
        # Y should be increasing (TOP to BOTTOM means y_min to y_max)
        ys = [c.y for c in path]
        assert ys == sorted(ys)

    def test_vertical_path_bottom_to_top(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test vertical path from BOTTOM to TOP."""
        path = Builder.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.BOTTOM,
            side_b=BoundarySide.TOP,
        )
        assert len(path) > 0
        # All coordinates should have the same x (center)
        xs = {c.x for c in path}
        assert len(xs) == 1
        # Y should be decreasing
        ys = [c.y for c in path]
        assert ys == sorted(ys, reverse=True)

    def test_horizontal_path_left_to_right(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test horizontal path from LEFT to RIGHT."""
        path = Builder.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.LEFT,
            side_b=BoundarySide.RIGHT,
        )
        assert len(path) > 0
        # All coordinates should have the same y (center)
        ys = {c.y for c in path}
        assert len(ys) == 1
        # X should be increasing
        xs = [c.x for c in path]
        assert xs == sorted(xs)

    def test_l_shaped_path_top_to_right(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test L-shaped path from TOP to RIGHT."""
        path = Builder.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.TOP,
            side_b=BoundarySide.RIGHT,
        )
        assert len(path) > 0
        # Path should include coordinates from both segments

    def test_l_shaped_path_left_to_bottom(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test L-shaped path from LEFT to BOTTOM."""
        path = Builder.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.LEFT,
            side_b=BoundarySide.BOTTOM,
        )
        assert len(path) > 0

    def test_path_contains_only_data_qubits(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test that path contains only data qubits."""
        coords = Builder.cube(code_distance=3, global_pos=Coord2D(0, 0), boundary=standard_boundary)
        path = Builder.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.TOP,
            side_b=BoundarySide.BOTTOM,
        )
        for coord in path:
            assert coord in coords.data

    @pytest.mark.parametrize("code_distance", [2, 3, 4, 5])
    def test_path_length_scales_with_distance(
        self,
        code_distance: int,
        standard_boundary: dict[BoundarySide, EdgeSpecValue],
    ) -> None:
        """Test that path length scales with code distance."""
        path = Builder.cube_boundary_path(
            code_distance=code_distance,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.TOP,
            side_b=BoundarySide.BOTTOM,
        )
        # Path length should be approximately code_distance
        assert len(path) >= code_distance - 1
        assert len(path) <= code_distance + 1


# =============================================================================
# Tests for RotatedSurfaceCodeLayoutBuilder.pipe_boundary_path
# =============================================================================


class TestBuilderPipeBoundaryPath:
    """Tests for RotatedSurfaceCodeLayoutBuilder.pipe_boundary_path."""

    def test_horizontal_pipe_left_to_right(self) -> None:
        """Test boundary path in horizontal pipe from LEFT to RIGHT."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.O,
            BoundarySide.BOTTOM: EdgeSpecValue.O,
            BoundarySide.LEFT: EdgeSpecValue.X,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        path = Builder.pipe_boundary_path(
            code_distance=3,
            global_pos_source=Coord3D(0, 0, 0),
            global_pos_target=Coord3D(0, 1, 0),
            boundary=boundary,
            side_a=BoundarySide.LEFT,
            side_b=BoundarySide.RIGHT,
        )
        assert len(path) > 0
        # All coordinates should have the same y (center)
        ys = {c.y for c in path}
        assert len(ys) == 1

    def test_vertical_pipe_top_to_bottom(self) -> None:
        """Test boundary path in vertical pipe from TOP to BOTTOM."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.Z,
            BoundarySide.LEFT: EdgeSpecValue.O,
            BoundarySide.RIGHT: EdgeSpecValue.O,
        }
        path = Builder.pipe_boundary_path(
            code_distance=3,
            global_pos_source=Coord3D(0, 0, 0),
            global_pos_target=Coord3D(1, 0, 0),
            boundary=boundary,
            side_a=BoundarySide.TOP,
            side_b=BoundarySide.BOTTOM,
        )
        assert len(path) > 0
        # All coordinates should have the same x (center)
        xs = {c.x for c in path}
        assert len(xs) == 1

    def test_pipe_path_contains_only_data_qubits(self) -> None:
        """Test that pipe path contains only data qubits."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.O,
            BoundarySide.BOTTOM: EdgeSpecValue.O,
            BoundarySide.LEFT: EdgeSpecValue.X,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        source = Coord3D(0, 0, 0)
        target = Coord3D(0, 1, 0)

        coords = Builder.pipe(
            code_distance=3,
            global_pos_source=source,
            global_pos_target=target,
            boundary=boundary,
        )
        path = Builder.pipe_boundary_path(
            code_distance=3,
            global_pos_source=source,
            global_pos_target=target,
            boundary=boundary,
            side_a=BoundarySide.LEFT,
            side_b=BoundarySide.RIGHT,
        )
        for coord in path:
            assert coord in coords.data


# =============================================================================
# Tests for RotatedSurfaceCodeLayoutBuilder.cube_boundary_ancillas_for_side
# =============================================================================


class TestBuilderCubeBoundaryAncillasForSide:
    """Tests for RotatedSurfaceCodeLayoutBuilder.cube_boundary_ancillas_for_side."""

    @pytest.fixture
    def standard_boundary(self) -> dict[BoundarySide, EdgeSpecValue]:
        """Standard boundary configuration for tests."""
        return {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

    def test_returns_tuple_of_frozensets(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test that the method returns a tuple of frozensets."""
        x_anc, z_anc = Builder.cube_boundary_ancillas_for_side(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side=BoundarySide.TOP,
        )
        assert isinstance(x_anc, frozenset)
        assert isinstance(z_anc, frozenset)

    @pytest.mark.parametrize("side", [BoundarySide.TOP, BoundarySide.BOTTOM, BoundarySide.LEFT, BoundarySide.RIGHT])
    def test_all_sides_return_valid_coordinates(
        self,
        standard_boundary: dict[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> None:
        """Test that all 2D sides return valid coordinate results."""
        x_anc, z_anc = Builder.cube_boundary_ancillas_for_side(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side=side,
        )
        # All coordinates should be Coord2D
        for coord in x_anc | z_anc:
            assert isinstance(coord, Coord2D)

    def test_top_boundary_x_type_returns_x_ancillas(self) -> None:
        """Test that TOP boundary with X type returns X ancillas."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        x_anc, _z_anc = Builder.cube_boundary_ancillas_for_side(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
            side=BoundarySide.TOP,
        )
        # TOP boundary is X type, so should have X ancillas (not Z)
        assert len(x_anc) > 0

    def test_left_boundary_z_type_returns_z_ancillas(self) -> None:
        """Test that LEFT boundary with Z type returns Z ancillas."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        _x_anc, z_anc = Builder.cube_boundary_ancillas_for_side(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=boundary,
            side=BoundarySide.LEFT,
        )
        # LEFT boundary is Z type, so should have Z ancillas (not X)
        assert len(z_anc) > 0

    def test_union_of_all_sides_equals_cube_boundary_ancillas(
        self, standard_boundary: dict[BoundarySide, EdgeSpecValue]
    ) -> None:
        """Test that union of all 4 sides equals the boundary ancillas from cube()."""
        coords = Builder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
        )
        bulk = Builder._generate_bulk_coords(Builder._cube_bounds(3, Builder._compute_cube_offset(3, Coord2D(0, 0))))

        # Get boundary ancillas from cube() (exclude bulk ancillas)
        cube_boundary_x = coords.ancilla_x - bulk.ancilla_x
        cube_boundary_z = coords.ancilla_z - bulk.ancilla_z

        # Get union of all 2D sides (TOP, BOTTOM, LEFT, RIGHT)
        all_x: set[Coord2D] = set()
        all_z: set[Coord2D] = set()
        for side in [BoundarySide.TOP, BoundarySide.BOTTOM, BoundarySide.LEFT, BoundarySide.RIGHT]:
            x_anc, z_anc = Builder.cube_boundary_ancillas_for_side(
                code_distance=3,
                global_pos=Coord2D(0, 0),
                boundary=standard_boundary,
                side=side,
            )
            all_x.update(x_anc)
            all_z.update(z_anc)

        # Union should equal cube boundary ancillas
        assert all_x == cube_boundary_x
        assert all_z == cube_boundary_z

    def test_corner_appears_in_both_adjacent_sides(self) -> None:
        """Test that corner ancillas appear in both adjacent sides."""
        # Use open boundaries to generate corner ancillas
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.O,
            BoundarySide.BOTTOM: EdgeSpecValue.O,
            BoundarySide.LEFT: EdgeSpecValue.O,
            BoundarySide.RIGHT: EdgeSpecValue.O,
        }

        # Get all corner ancillas
        all_corners = Builder._get_corner_ancillas(
            Builder._cube_bounds(3, Builder._compute_cube_offset(3, Coord2D(0, 0))),
            boundary,
        )
        all_corner_coords = all_corners[0] | all_corners[1]

        # Check each corner appears in exactly 2 sides (among 2D sides only)
        sides_2d = [BoundarySide.TOP, BoundarySide.BOTTOM, BoundarySide.LEFT, BoundarySide.RIGHT]
        for corner in all_corner_coords:
            count = 0
            for side in sides_2d:
                x_anc, z_anc = Builder.cube_boundary_ancillas_for_side(
                    code_distance=3,
                    global_pos=Coord2D(0, 0),
                    boundary=boundary,
                    side=side,
                )
                if corner in (x_anc | z_anc):
                    count += 1
            assert count == 2, f"Corner {corner} should appear in exactly 2 sides, but appeared in {count}"

    @pytest.mark.parametrize("code_distance", [2, 3, 4, 5])
    def test_various_code_distances(
        self,
        code_distance: int,
        standard_boundary: dict[BoundarySide, EdgeSpecValue],
    ) -> None:
        """Test that the method works with various code distances."""
        for side in [BoundarySide.TOP, BoundarySide.BOTTOM, BoundarySide.LEFT, BoundarySide.RIGHT]:
            x_anc, z_anc = Builder.cube_boundary_ancillas_for_side(
                code_distance=code_distance,
                global_pos=Coord2D(0, 0),
                boundary=standard_boundary,
                side=side,
            )
            # Should return valid results (may be empty for some configurations)
            assert isinstance(x_anc, frozenset)
            assert isinstance(z_anc, frozenset)

    def test_different_global_positions(self, standard_boundary: dict[BoundarySide, EdgeSpecValue]) -> None:
        """Test that coordinates shift correctly with different global positions."""
        x_anc_0, _z_anc_0 = Builder.cube_boundary_ancillas_for_side(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side=BoundarySide.TOP,
        )
        x_anc_1, _z_anc_1 = Builder.cube_boundary_ancillas_for_side(
            code_distance=3,
            global_pos=Coord2D(1, 0),
            boundary=standard_boundary,
            side=BoundarySide.TOP,
        )

        # Coordinates should be shifted
        assert x_anc_0 != x_anc_1
        # The shift should be 2*(code_distance + 1) in x direction
        shift = 2 * (3 + 1)
        shifted_x = frozenset(Coord2D(c.x + shift, c.y) for c in x_anc_0)
        assert shifted_x == x_anc_1


# =============================================================================
# Tests for ABC inheritance and interface compliance
# =============================================================================


class TestABCInheritance:
    """Tests for ABC inheritance structure."""

    def test_rotated_surface_code_layout_inherits_combined_abc(self) -> None:
        """Test that RotatedSurfaceCodeLayout inherits from TopologicalCodeLayoutBuilder."""
        assert issubclass(RotatedSurfaceCodeLayout, TopologicalCodeLayoutBuilder)

    def test_rotated_surface_code_layout_inherits_coordinate_generator(self) -> None:
        """Test inheritance from CoordinateGenerator ABC."""
        assert issubclass(RotatedSurfaceCodeLayout, CoordinateGenerator)

    def test_rotated_surface_code_layout_inherits_bounds_calculator(self) -> None:
        """Test inheritance from BoundsCalculator ABC."""
        assert issubclass(RotatedSurfaceCodeLayout, BoundsCalculator)

    def test_rotated_surface_code_layout_inherits_boundary_path_calculator(self) -> None:
        """Test inheritance from BoundaryPathCalculator ABC."""
        assert issubclass(RotatedSurfaceCodeLayout, BoundaryPathCalculator)

    def test_rotated_surface_code_layout_inherits_boundary_ancilla_retriever(self) -> None:
        """Test inheritance from BoundaryAncillaRetriever ABC."""
        assert issubclass(RotatedSurfaceCodeLayout, BoundaryAncillaRetriever)

    def test_rotated_surface_code_layout_inherits_ancilla_flow_constructor(self) -> None:
        """Test inheritance from AncillaFlowConstructor ABC."""
        assert issubclass(RotatedSurfaceCodeLayout, AncillaFlowConstructor)

    def test_rotated_surface_code_layout_inherits_pipe_direction_helper(self) -> None:
        """Test inheritance from PipeDirectionHelper ABC."""
        assert issubclass(RotatedSurfaceCodeLayout, PipeDirectionHelper)


class TestInstanceBasedAPI:
    """Tests for the instance-based RotatedSurfaceCodeLayout API."""

    @pytest.fixture
    def layout(self) -> RotatedSurfaceCodeLayout:
        """Create a layout instance for testing."""
        return RotatedSurfaceCodeLayout()

    @pytest.fixture
    def standard_boundary(self) -> dict[BoundarySide, EdgeSpecValue]:
        """Standard boundary configuration for tests."""
        return {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }

    def test_instance_cube_matches_static(
        self,
        layout: RotatedSurfaceCodeLayout,
        standard_boundary: dict[BoundarySide, EdgeSpecValue],
    ) -> None:
        """Test that instance.cube() produces same result as static Builder.cube()."""
        instance_result = layout.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
        )
        static_result = Builder.cube(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
        )
        assert instance_result == static_result

    def test_instance_pipe_matches_static(
        self,
        layout: RotatedSurfaceCodeLayout,
    ) -> None:
        """Test that instance.pipe() produces same result as static Builder.pipe()."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.O,
            BoundarySide.BOTTOM: EdgeSpecValue.O,
            BoundarySide.LEFT: EdgeSpecValue.X,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        source = Coord3D(0, 0, 0)
        target = Coord3D(0, 1, 0)

        instance_result = layout.pipe(
            code_distance=3,
            global_pos_source=source,
            global_pos_target=target,
            boundary=boundary,
        )
        static_result = Builder.pipe(
            code_distance=3,
            global_pos_source=source,
            global_pos_target=target,
            boundary=boundary,
        )
        assert instance_result == static_result

    def test_instance_cube_bounds(
        self,
        layout: RotatedSurfaceCodeLayout,
    ) -> None:
        """Test that instance.cube_bounds() returns valid PatchBounds."""
        result = layout.cube_bounds(code_distance=3, offset=Coord2D(0, 0))
        assert isinstance(result, PatchBounds)
        # Verify the bounds have expected dimensions for code_distance=3
        # Width = 2*(d-1) + 1 = 2*d - 1 = 5 for d=3
        assert result.width == 2 * 3 - 1
        assert result.height == 2 * 3 - 1

    def test_instance_cube_boundary_path_matches_static(
        self,
        layout: RotatedSurfaceCodeLayout,
        standard_boundary: dict[BoundarySide, EdgeSpecValue],
    ) -> None:
        """Test that instance.cube_boundary_path() produces same result as static."""
        instance_result = layout.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.TOP,
            side_b=BoundarySide.BOTTOM,
        )
        static_result = Builder.cube_boundary_path(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            side_a=BoundarySide.TOP,
            side_b=BoundarySide.BOTTOM,
        )
        assert instance_result == static_result

    def test_instance_is_topological_code_layout_builder(
        self,
        layout: RotatedSurfaceCodeLayout,
    ) -> None:
        """Test that isinstance check works for TopologicalCodeLayoutBuilder."""
        assert isinstance(layout, TopologicalCodeLayoutBuilder)

    def test_instance_is_coordinate_generator(
        self,
        layout: RotatedSurfaceCodeLayout,
    ) -> None:
        """Test that isinstance check works for CoordinateGenerator."""
        assert isinstance(layout, CoordinateGenerator)

    def test_instance_is_bounds_calculator(
        self,
        layout: RotatedSurfaceCodeLayout,
    ) -> None:
        """Test that isinstance check works for BoundsCalculator."""
        assert isinstance(layout, BoundsCalculator)

    def test_construct_initial_ancilla_flow(
        self,
        layout: RotatedSurfaceCodeLayout,
        standard_boundary: dict[BoundarySide, EdgeSpecValue],
    ) -> None:
        """Test construct_initial_ancilla_flow instance method."""
        flow = layout.construct_initial_ancilla_flow(
            code_distance=3,
            global_pos=Coord2D(0, 0),
            boundary=standard_boundary,
            ancilla_type=EdgeSpecValue.X,
        )
        assert isinstance(flow, dict)
        # Flow should have Coord2D keys and set[Coord2D] values
        for key, value in flow.items():
            assert isinstance(key, Coord2D)
            assert isinstance(value, set)

    def test_pipe_offset(
        self,
        layout: RotatedSurfaceCodeLayout,
    ) -> None:
        """Test pipe_offset instance method."""
        # y increases -> BOTTOM (y_max side)
        offset = layout.pipe_offset(
            global_pos_source=Coord3D(0, 0, 0),
            global_pos_target=Coord3D(0, 1, 0),
        )
        assert offset == BoundarySide.BOTTOM

        # x increases -> RIGHT
        offset = layout.pipe_offset(
            global_pos_source=Coord3D(0, 0, 0),
            global_pos_target=Coord3D(1, 0, 0),
        )
        assert offset == BoundarySide.RIGHT

    def test_pipe_axis_from_offset(
        self,
        layout: RotatedSurfaceCodeLayout,
    ) -> None:
        """Test pipe_axis_from_offset instance method."""
        # RIGHT/LEFT -> H (horizontal)
        axis_h = layout.pipe_axis_from_offset(BoundarySide.RIGHT)
        assert axis_h == AxisDIRECTION2D.H

        # TOP/BOTTOM -> V (vertical)
        axis_v = layout.pipe_axis_from_offset(BoundarySide.BOTTOM)
        assert axis_v == AxisDIRECTION2D.V
