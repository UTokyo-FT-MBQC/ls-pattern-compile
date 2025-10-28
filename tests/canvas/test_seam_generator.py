"""Unit tests for SeamGenerator class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.canvas.seams import SeamGenerator
from lspattern.mytype import (
    NodeIdLocal,
    PatchCoordGlobal3D,
    PipeCoordGlobal3D,
    PhysCoordGlobal3D,
    QubitGroupIdGlobal,
)

if TYPE_CHECKING:
    from lspattern.blocks.cubes.base import RHGCube
    from lspattern.blocks.pipes.base import RHGPipe


class TestSeamGeneratorBasic:
    """Test basic SeamGenerator functionality."""

    def test_initialization(self) -> None:
        """Test SeamGenerator initializes correctly."""
        cubes: dict[PatchCoordGlobal3D, "RHGCube"] = {}
        pipes: dict[PipeCoordGlobal3D, "RHGPipe"] = {}
        node2coord: dict[NodeIdLocal, PhysCoordGlobal3D] = {}
        coord2node: dict[PhysCoordGlobal3D, NodeIdLocal] = {}
        allowed_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]] = set()

        generator = SeamGenerator(
            cubes=cubes,
            pipes=pipes,
            node2coord=node2coord,
            coord2node=coord2node,
            allowed_gid_pairs=allowed_pairs,
        )

        assert generator.cubes_ == cubes
        assert generator.pipes_ == pipes
        assert generator.node2coord == node2coord
        assert generator.coord2node == coord2node
        assert generator.allowed_gid_pairs == allowed_pairs


class TestSeamGeneratorPopulateCoordGid2d:
    """Test _populate_coord_gid_2d method."""

    def test_populate_coord_gid_2d_empty(self) -> None:
        """Test _populate_coord_gid_2d with no cubes or pipes."""
        generator = SeamGenerator(
            cubes={},
            pipes={},
            node2coord={},
            coord2node={},
            allowed_gid_pairs=set(),
        )

        coord_gid_2d = generator._populate_coord_gid_2d()

        assert coord_gid_2d == {}


class TestSeamGeneratorCollectBlockXYRegions:
    """Test _collect_block_xy_regions method."""

    def test_collect_block_xy_regions_empty(self) -> None:
        """Test _collect_block_xy_regions with no cubes or pipes."""
        generator = SeamGenerator(
            cubes={},
            pipes={},
            node2coord={},
            coord2node={},
            allowed_gid_pairs=set(),
        )

        cube_xy, measure_pipe_xy = generator._collect_block_xy_regions()

        assert cube_xy == set()
        assert measure_pipe_xy == set()


class TestSeamGeneratorIsMeasurePipeNode:
    """Test _is_measure_pipe_node method."""

    def test_is_measure_pipe_node_in_cube(self) -> None:
        """Test that nodes in cube region are not measure pipe nodes."""
        cube_xy_all = {(0, 0), (1, 0), (0, 1)}
        measure_pipe_xy = {(5, 5)}
        xy = (0, 0)

        result = SeamGenerator._is_measure_pipe_node(xy, cube_xy_all, measure_pipe_xy)
        assert result is False

    def test_is_measure_pipe_node_not_in_any_pipe(self) -> None:
        """Test that nodes not in any pipe are not measure pipe nodes."""
        cube_xy_all = {(0, 0)}
        measure_pipe_xy = {(5, 5)}
        xy = (10, 10)  # Not in cube, not in any pipe

        result = SeamGenerator._is_measure_pipe_node(xy, cube_xy_all, measure_pipe_xy)
        assert result is False


class TestSeamGeneratorShouldConnectNodes:
    """Test _should_connect_nodes method."""

    def test_should_not_connect_both_in_cube(self) -> None:
        """Test that nodes both in cube region are not connected."""
        generator = SeamGenerator(
            cubes={},
            pipes={},
            node2coord={},
            coord2node={},
            allowed_gid_pairs={(QubitGroupIdGlobal(0), QubitGroupIdGlobal(1))},
        )

        cube_xy_all = {(0, 0), (1, 0)}
        measure_pipe_xy: set[tuple[int, int]] = set()
        xy_u = (0, 0)
        xy_v = (1, 0)
        gid_u = QubitGroupIdGlobal(0)
        gid_v = QubitGroupIdGlobal(1)

        result = generator._should_connect_nodes(xy_u, xy_v, cube_xy_all, measure_pipe_xy, gid_u, gid_v)
        assert result is False

    def test_should_not_connect_both_in_pipe(self) -> None:
        """Test that nodes both in pipe region are not connected."""
        generator = SeamGenerator(
            cubes={},
            pipes={},
            node2coord={},
            coord2node={},
            allowed_gid_pairs={(QubitGroupIdGlobal(0), QubitGroupIdGlobal(1))},
        )

        cube_xy_all = {(0, 0)}
        measure_pipe_xy: set[tuple[int, int]] = set()
        xy_u = (2, 0)  # Not in cube
        xy_v = (3, 0)  # Not in cube
        gid_u = QubitGroupIdGlobal(0)
        gid_v = QubitGroupIdGlobal(1)

        result = generator._should_connect_nodes(xy_u, xy_v, cube_xy_all, measure_pipe_xy, gid_u, gid_v)
        # Both in pipe region, should not connect
        assert result is False

    def test_should_connect_cube_to_pipe_allowed_pair(self) -> None:
        """Test that nodes at cube-pipe boundary with allowed pair are connected."""
        allowed_pairs = {(QubitGroupIdGlobal(0), QubitGroupIdGlobal(1))}
        generator = SeamGenerator(
            cubes={},
            pipes={},
            node2coord={},
            coord2node={},
            allowed_gid_pairs=allowed_pairs,
        )

        cube_xy_all = {(0, 0)}
        measure_pipe_xy: set[tuple[int, int]] = set()
        xy_u = (0, 0)  # In cube
        xy_v = (1, 0)  # Not in cube (pipe)
        gid_u = QubitGroupIdGlobal(0)
        gid_v = QubitGroupIdGlobal(1)

        result = generator._should_connect_nodes(xy_u, xy_v, cube_xy_all, measure_pipe_xy, gid_u, gid_v)
        assert result is True

    def test_should_not_connect_disallowed_pair(self) -> None:
        """Test that nodes with disallowed tiling ID pair are not connected."""
        allowed_pairs = {(QubitGroupIdGlobal(0), QubitGroupIdGlobal(1))}
        generator = SeamGenerator(
            cubes={},
            pipes={},
            node2coord={},
            coord2node={},
            allowed_gid_pairs=allowed_pairs,
        )

        cube_xy_all = {(0, 0)}
        measure_pipe_xy: set[tuple[int, int]] = set()
        xy_u = (0, 0)  # In cube
        xy_v = (1, 0)  # Not in cube (pipe)
        gid_u = QubitGroupIdGlobal(0)
        gid_v = QubitGroupIdGlobal(2)  # Disallowed pair

        result = generator._should_connect_nodes(xy_u, xy_v, cube_xy_all, measure_pipe_xy, gid_u, gid_v)
        assert result is False
