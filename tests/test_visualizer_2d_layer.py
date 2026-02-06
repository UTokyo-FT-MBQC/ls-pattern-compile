"""Tests for visualizer_2d layer visualization functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from lspattern.canvas_loader import CanvasCubeSpec, CanvasPipeSpec
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.layout import PatchCoordinates, RotatedSurfaceCodeLayoutBuilder
from lspattern.mytype import Coord2D, Coord3D
from lspattern.visualizer_2d import (
    _collect_blocks_at_z,
    _generate_cube_coordinates,
    _generate_pipe_coordinates,
    _merge_patch_coordinates,
    visualize_canvas_layout,
    visualize_patch_coordinates_2d,
)

# Use the design examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "design"


class TestCollectBlocksAtZ:
    """Tests for _collect_blocks_at_z helper function."""

    def test_single_cube_at_target_z(self) -> None:
        """Test filtering cubes at target z."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        cubes = [
            CanvasCubeSpec(Coord3D(0, 0, 0), "MemoryBlock", boundary, None),
            CanvasCubeSpec(Coord3D(0, 0, 1), "MemoryBlock", boundary, None),
            CanvasCubeSpec(Coord3D(1, 0, 1), "MemoryBlock", boundary, None),
            CanvasCubeSpec(Coord3D(0, 0, 2), "MemoryBlock", boundary, None),
        ]
        pipes: list[CanvasPipeSpec] = []

        cubes_at_z, pipes_at_z = _collect_blocks_at_z(cubes, pipes, target_z=1)

        assert len(cubes_at_z) == 2
        assert all(c.position.z == 1 for c in cubes_at_z)
        assert len(pipes_at_z) == 0

    def test_spatial_pipes_only(self) -> None:
        """Test that only spatial pipes (same z for start and end) are included."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        cubes: list[CanvasCubeSpec] = []
        pipes = [
            # Spatial pipe at z=1 (both start and end have z=1)
            CanvasPipeSpec(Coord3D(0, 0, 1), Coord3D(1, 0, 1), "MemoryBlock", boundary, None),
            # Temporal pipe from z=0 to z=1 (should NOT be included)
            CanvasPipeSpec(Coord3D(0, 0, 0), Coord3D(0, 0, 1), "MemoryBlock", boundary, None),
            # Spatial pipe at z=2
            CanvasPipeSpec(Coord3D(0, 0, 2), Coord3D(1, 0, 2), "MemoryBlock", boundary, None),
        ]

        cubes_at_z, pipes_at_z = _collect_blocks_at_z(cubes, pipes, target_z=1)

        assert len(cubes_at_z) == 0
        assert len(pipes_at_z) == 1
        assert pipes_at_z[0].start.z == 1
        assert pipes_at_z[0].end.z == 1

    def test_empty_layer(self) -> None:
        """Test filtering with no blocks at target z."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        cubes = [
            CanvasCubeSpec(Coord3D(0, 0, 0), "MemoryBlock", boundary, None),
        ]
        pipes: list[CanvasPipeSpec] = []

        cubes_at_z, pipes_at_z = _collect_blocks_at_z(cubes, pipes, target_z=5)

        assert len(cubes_at_z) == 0
        assert len(pipes_at_z) == 0


class TestGenerateBlockCoordinates:
    """Tests for block coordinate generation helpers."""

    def test_generate_cube_coordinates(self) -> None:
        """Test generating coordinates for a cube."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        cube = CanvasCubeSpec(Coord3D(0, 0, 0), "MemoryBlock", boundary, None)

        coords = _generate_cube_coordinates(cube, code_distance=3)

        assert isinstance(coords, PatchCoordinates)
        assert len(coords.data) > 0
        assert len(coords.ancilla_x) > 0
        assert len(coords.ancilla_z) > 0

    def test_generate_pipe_coordinates(self) -> None:
        """Test generating coordinates for a pipe."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.Z,
            BoundarySide.BOTTOM: EdgeSpecValue.Z,
            BoundarySide.LEFT: EdgeSpecValue.O,
            BoundarySide.RIGHT: EdgeSpecValue.O,
        }
        pipe = CanvasPipeSpec(Coord3D(0, 0, 1), Coord3D(1, 0, 1), "MemoryBlock", boundary, None)

        coords = _generate_pipe_coordinates(pipe, code_distance=3)

        assert isinstance(coords, PatchCoordinates)
        assert len(coords.data) > 0


class TestMergePatchCoordinates:
    """Tests for _merge_patch_coordinates helper function."""

    def test_merge_empty_list(self) -> None:
        """Test merging an empty list returns empty PatchCoordinates."""
        result = _merge_patch_coordinates([])

        assert len(result.data) == 0
        assert len(result.ancilla_x) == 0
        assert len(result.ancilla_z) == 0

    def test_merge_single_coords(self) -> None:
        """Test merging a single PatchCoordinates."""
        coords = PatchCoordinates(
            frozenset({Coord2D(0, 0), Coord2D(2, 0)}),
            frozenset({Coord2D(1, 1)}),
            frozenset({Coord2D(3, 1)}),
        )

        result = _merge_patch_coordinates([coords])

        assert result.data == coords.data
        assert result.ancilla_x == coords.ancilla_x
        assert result.ancilla_z == coords.ancilla_z

    def test_merge_multiple_coords(self) -> None:
        """Test merging multiple PatchCoordinates."""
        coords1 = PatchCoordinates(
            frozenset({Coord2D(0, 0)}),
            frozenset({Coord2D(1, 1)}),
            frozenset({Coord2D(3, 1)}),
        )
        coords2 = PatchCoordinates(
            frozenset({Coord2D(8, 0)}),
            frozenset({Coord2D(9, 1)}),
            frozenset({Coord2D(11, 1)}),
        )

        result = _merge_patch_coordinates([coords1, coords2])

        assert len(result.data) == 2
        assert Coord2D(0, 0) in result.data
        assert Coord2D(8, 0) in result.data
        assert len(result.ancilla_x) == 2
        assert len(result.ancilla_z) == 2


class TestVisualizePatchCoordinates2d:
    """Tests for visualize_patch_coordinates_2d function."""

    def test_returns_figure(self) -> None:
        """Test that the function returns a matplotlib Figure."""
        boundary = {
            BoundarySide.TOP: EdgeSpecValue.X,
            BoundarySide.BOTTOM: EdgeSpecValue.X,
            BoundarySide.LEFT: EdgeSpecValue.Z,
            BoundarySide.RIGHT: EdgeSpecValue.Z,
        }
        coords = RotatedSurfaceCodeLayoutBuilder.cube(3, Coord2D(0, 0), boundary)

        fig = visualize_patch_coordinates_2d(coords, title="Test")

        assert fig is not None
        plt.close(fig)

    def test_empty_coordinates(self) -> None:
        """Test visualization of empty coordinates."""
        empty_coords = PatchCoordinates(frozenset(), frozenset(), frozenset())

        fig = visualize_patch_coordinates_2d(empty_coords, title="Empty")

        assert fig is not None
        plt.close(fig)


class TestVisualizeLayer2d:
    """Tests for visualize_canvas_layout function."""

    def test_single_cube_layer(self) -> None:
        """Test visualization of a layer with a single cube."""
        yaml_path = EXAMPLES_DIR / "short_z_memory_canvas.yml"
        if not yaml_path.exists():
            pytest.skip("short_z_memory_canvas.yml not found")

        fig = visualize_canvas_layout(yaml_path, code_distance=3, target_z=0)

        assert fig is not None
        # Check title contains cube count
        ax = fig.axes[0]
        assert "1 cube" in ax.get_title()
        plt.close(fig)

    def test_layer_with_multiple_cubes(self) -> None:
        """Test visualization of a layer with multiple cubes."""
        yaml_path = EXAMPLES_DIR / "cnot.yml"
        if not yaml_path.exists():
            pytest.skip("cnot.yml not found")

        # z=0 has 3 cubes in cnot.yml
        fig = visualize_canvas_layout(yaml_path, code_distance=3, target_z=0)

        assert fig is not None
        ax = fig.axes[0]
        assert "3 cube" in ax.get_title()
        plt.close(fig)

    def test_layer_with_cubes_and_pipes(self) -> None:
        """Test visualization of a layer with both cubes and pipes."""
        yaml_path = EXAMPLES_DIR / "cnot.yml"
        if not yaml_path.exists():
            pytest.skip("cnot.yml not found")

        # z=1 has 3 cubes and 1 spatial pipe in cnot.yml
        fig = visualize_canvas_layout(yaml_path, code_distance=3, target_z=1)

        assert fig is not None
        ax = fig.axes[0]
        title = ax.get_title()
        assert "3 cube" in title
        assert "1 pipe" in title
        plt.close(fig)

    def test_empty_layer(self) -> None:
        """Test visualization of a layer with no blocks."""
        yaml_path = EXAMPLES_DIR / "short_z_memory_canvas.yml"
        if not yaml_path.exists():
            pytest.skip("short_z_memory_canvas.yml not found")

        # This canvas only has z=0, so z=10 should be empty
        fig = visualize_canvas_layout(yaml_path, code_distance=3, target_z=10)

        assert fig is not None
        ax = fig.axes[0]
        assert "0 cube" in ax.get_title()
        assert "0 pipe" in ax.get_title()
        plt.close(fig)

    def test_custom_figsize(self) -> None:
        """Test visualization with custom figure size."""
        yaml_path = EXAMPLES_DIR / "short_z_memory_canvas.yml"
        if not yaml_path.exists():
            pytest.skip("short_z_memory_canvas.yml not found")

        fig = visualize_canvas_layout(yaml_path, code_distance=3, target_z=0, figsize=(5, 5))

        assert fig is not None
        # Check figure size (in inches)
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(5, abs=0.1)
        assert size[1] == pytest.approx(5, abs=0.1)
        plt.close(fig)
