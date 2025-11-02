"""Unit tests for CoordinateMapper class."""

from __future__ import annotations

import pytest

from lspattern.canvas.coordinates import CoordinateMapper
from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D


class TestCoordinateMapperBasic:
    """Test basic CoordinateMapper functionality."""

    def test_initialization(self) -> None:
        """Test CoordinateMapper initializes with empty mappings."""
        mapper = CoordinateMapper()
        assert mapper.node2coord == {}
        assert mapper.coord2node == {}
        assert mapper.node2role == {}

    def test_add_node_without_role(self) -> None:
        """Test adding a node without role."""
        mapper = CoordinateMapper()
        node_id = NodeIdLocal(1)
        coord = PhysCoordGlobal3D((0, 0, 0))

        mapper.add_node(node_id, coord)

        assert mapper.node2coord[node_id] == coord
        assert mapper.coord2node[coord] == node_id
        assert node_id not in mapper.node2role

    def test_add_node_with_role(self) -> None:
        """Test adding a node with role."""
        mapper = CoordinateMapper()
        node_id = NodeIdLocal(1)
        coord = PhysCoordGlobal3D((0, 0, 0))
        role = "data"

        mapper.add_node(node_id, coord, role)

        assert mapper.node2coord[node_id] == coord
        assert mapper.coord2node[coord] == node_id
        assert mapper.node2role[node_id] == role

    def test_add_multiple_nodes(self) -> None:
        """Test adding multiple nodes."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")
        mapper.add_node(NodeIdLocal(3), PhysCoordGlobal3D((2, 0, 0)), "ancilla_z")

        assert len(mapper.node2coord) == 3
        assert len(mapper.coord2node) == 3
        assert len(mapper.node2role) == 3

    def test_overwrite_node(self) -> None:
        """Test that adding the same node overwrites previous data."""
        mapper = CoordinateMapper()
        node_id = NodeIdLocal(1)
        coord1 = PhysCoordGlobal3D((0, 0, 0))
        coord2 = PhysCoordGlobal3D((1, 1, 1))

        mapper.add_node(node_id, coord1, "data")
        mapper.add_node(node_id, coord2, "ancilla_x")

        assert mapper.node2coord[node_id] == coord2
        assert mapper.coord2node[coord2] == node_id
        assert coord1 not in mapper.coord2node
        assert mapper.node2role[node_id] == "ancilla_x"


class TestCoordinateMapperGetters:
    """Test getter methods."""

    def test_get_coordinate(self) -> None:
        """Test getting coordinate for a node."""
        mapper = CoordinateMapper()
        node_id = NodeIdLocal(1)
        coord = PhysCoordGlobal3D((0, 0, 0))
        mapper.add_node(node_id, coord)

        assert mapper.get_coordinate(node_id) == coord
        assert mapper.get_coordinate(NodeIdLocal(999)) is None

    def test_get_node(self) -> None:
        """Test getting node for a coordinate."""
        mapper = CoordinateMapper()
        node_id = NodeIdLocal(1)
        coord = PhysCoordGlobal3D((0, 0, 0))
        mapper.add_node(node_id, coord)

        assert mapper.get_node(coord) == node_id
        assert mapper.get_node(PhysCoordGlobal3D((999, 999, 999))) is None

    def test_get_role(self) -> None:
        """Test getting role for a node."""
        mapper = CoordinateMapper()
        node_id = NodeIdLocal(1)
        mapper.add_node(node_id, PhysCoordGlobal3D((0, 0, 0)), "data")

        assert mapper.get_role(node_id) == "data"
        assert mapper.get_role(NodeIdLocal(999)) is None


class TestCoordinateMapperRemapping:
    """Test node remapping functionality."""

    def test_remap_nodes_identity(self) -> None:
        """Test remapping with identity mapping."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        mapper.remap_nodes({})  # Empty mapping = identity

        assert NodeIdLocal(1) in mapper.node2coord
        assert NodeIdLocal(2) in mapper.node2coord

    def test_remap_nodes_basic(self) -> None:
        """Test basic node remapping."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        node_map = {1: 10, 2: 20}
        mapper.remap_nodes(node_map)

        assert NodeIdLocal(10) in mapper.node2coord
        assert NodeIdLocal(20) in mapper.node2coord
        assert NodeIdLocal(1) not in mapper.node2coord
        assert NodeIdLocal(2) not in mapper.node2coord

        assert mapper.node2coord[NodeIdLocal(10)] == PhysCoordGlobal3D((0, 0, 0))
        assert mapper.node2coord[NodeIdLocal(20)] == PhysCoordGlobal3D((1, 0, 0))

        assert mapper.coord2node[PhysCoordGlobal3D((0, 0, 0))] == NodeIdLocal(10)
        assert mapper.coord2node[PhysCoordGlobal3D((1, 0, 0))] == NodeIdLocal(20)

        assert mapper.node2role[NodeIdLocal(10)] == "data"
        assert mapper.node2role[NodeIdLocal(20)] == "ancilla_x"

    def test_remap_nodes_partial(self) -> None:
        """Test partial remapping (only some nodes mapped)."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")
        mapper.add_node(NodeIdLocal(3), PhysCoordGlobal3D((2, 0, 0)), "ancilla_z")

        node_map = {1: 10}  # Only remap node 1
        mapper.remap_nodes(node_map)

        assert NodeIdLocal(10) in mapper.node2coord
        assert NodeIdLocal(2) in mapper.node2coord  # Unchanged
        assert NodeIdLocal(3) in mapper.node2coord  # Unchanged


class TestCoordinateMapperBounds:
    """Test coordinate bounds functionality."""

    def test_get_coordinate_bounds_basic(self) -> None:
        """Test getting coordinate bounds."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)))
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((5, 10, 15)))
        mapper.add_node(NodeIdLocal(3), PhysCoordGlobal3D((2, 3, 4)))

        xmin, xmax, ymin, ymax, zmin, zmax = mapper.get_coordinate_bounds()

        assert xmin == 0
        assert xmax == 5
        assert ymin == 0
        assert ymax == 10
        assert zmin == 0
        assert zmax == 15

    def test_get_coordinate_bounds_negative(self) -> None:
        """Test bounds with negative coordinates."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((-5, -10, -15)))
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((5, 10, 15)))

        xmin, xmax, ymin, ymax, zmin, zmax = mapper.get_coordinate_bounds()

        assert xmin == -5
        assert xmax == 5
        assert ymin == -10
        assert ymax == 10
        assert zmin == -15
        assert zmax == 15

    def test_get_coordinate_bounds_single_node(self) -> None:
        """Test bounds with single node."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((3, 4, 5)))

        xmin, xmax, ymin, ymax, zmin, zmax = mapper.get_coordinate_bounds()

        assert xmin == 3
        assert xmax == 3
        assert ymin == 4
        assert ymax == 4
        assert zmin == 5
        assert zmax == 5

    def test_get_coordinate_bounds_empty(self) -> None:
        """Test bounds with no nodes raises ValueError."""
        mapper = CoordinateMapper()

        with pytest.raises(ValueError, match="No coordinates available"):
            mapper.get_coordinate_bounds()


class TestFaceChecker:
    """Test face checker functionality."""

    def test_create_face_checker_x_plus(self) -> None:
        """Test face checker for x+ face."""
        bounds = (0, 10, 0, 10, 0, 10)
        checker = CoordinateMapper.create_face_checker("x+", bounds, [0])

        assert checker((10, 5, 5)) is True  # On x+ face
        assert checker((9, 5, 5)) is False  # Not on face
        assert checker((5, 5, 5)) is False  # Interior

    def test_create_face_checker_x_minus(self) -> None:
        """Test face checker for x- face."""
        bounds = (0, 10, 0, 10, 0, 10)
        checker = CoordinateMapper.create_face_checker("x-", bounds, [0])

        assert checker((0, 5, 5)) is True
        assert checker((1, 5, 5)) is False

    def test_create_face_checker_y_plus(self) -> None:
        """Test face checker for y+ face."""
        bounds = (0, 10, 0, 10, 0, 10)
        checker = CoordinateMapper.create_face_checker("y+", bounds, [0])

        assert checker((5, 10, 5)) is True
        assert checker((5, 9, 5)) is False

    def test_create_face_checker_y_minus(self) -> None:
        """Test face checker for y- face."""
        bounds = (0, 10, 0, 10, 0, 10)
        checker = CoordinateMapper.create_face_checker("y-", bounds, [0])

        assert checker((5, 0, 5)) is True
        assert checker((5, 1, 5)) is False

    def test_create_face_checker_z_plus(self) -> None:
        """Test face checker for z+ face."""
        bounds = (0, 10, 0, 10, 0, 10)
        checker = CoordinateMapper.create_face_checker("z+", bounds, [0])

        assert checker((5, 5, 10)) is True
        assert checker((5, 5, 9)) is False

    def test_create_face_checker_z_minus(self) -> None:
        """Test face checker for z- face."""
        bounds = (0, 10, 0, 10, 0, 10)
        checker = CoordinateMapper.create_face_checker("z-", bounds, [0])

        assert checker((5, 5, 0)) is True
        assert checker((5, 5, 1)) is False

    def test_create_face_checker_multiple_depths(self) -> None:
        """Test face checker with multiple depths."""
        bounds = (0, 10, 0, 10, 0, 10)
        checker = CoordinateMapper.create_face_checker("x+", bounds, [0, 1, 2])

        assert checker((10, 5, 5)) is True  # depth 0
        assert checker((9, 5, 5)) is True  # depth 1
        assert checker((8, 5, 5)) is True  # depth 2
        assert checker((7, 5, 5)) is False  # depth 3


class TestClassifyNodesByRole:
    """Test role-based node classification."""

    def test_classify_nodes_by_role_basic(self) -> None:
        """Test basic role classification."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")
        mapper.add_node(NodeIdLocal(3), PhysCoordGlobal3D((2, 0, 0)), "ancilla_z")

        # Checker that includes all coordinates
        def always_true(c: tuple[int, int, int]) -> bool:  # noqa: ARG001
            return True

        result = mapper.classify_nodes_by_role(always_true)

        assert len(result["data"]) == 1
        assert len(result["xcheck"]) == 1
        assert len(result["zcheck"]) == 1
        assert PhysCoordGlobal3D((0, 0, 0)) in result["data"]
        assert PhysCoordGlobal3D((1, 0, 0)) in result["xcheck"]
        assert PhysCoordGlobal3D((2, 0, 0)) in result["zcheck"]

    def test_classify_nodes_by_role_filtered(self) -> None:
        """Test role classification with face filtering."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")
        mapper.add_node(NodeIdLocal(3), PhysCoordGlobal3D((2, 0, 0)), "ancilla_z")

        # Checker that only includes x=0
        def only_x_zero(c: tuple[int, int, int]) -> bool:
            return c[0] == 0

        result = mapper.classify_nodes_by_role(only_x_zero)

        assert len(result["data"]) == 1
        assert len(result["xcheck"]) == 0
        assert len(result["zcheck"]) == 0

    def test_classify_nodes_by_role_no_roles(self) -> None:
        """Test classification when nodes have no roles (default to data)."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)))
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)))

        def always_true(c: tuple[int, int, int]) -> bool:  # noqa: ARG001
            return True

        result = mapper.classify_nodes_by_role(always_true)

        assert len(result["data"]) == 2
        assert len(result["xcheck"]) == 0
        assert len(result["zcheck"]) == 0

    def test_classify_nodes_by_role_empty(self) -> None:
        """Test classification with no nodes."""
        mapper = CoordinateMapper()

        def always_true(c: tuple[int, int, int]) -> bool:  # noqa: ARG001
            return True

        result = mapper.classify_nodes_by_role(always_true)

        assert result["data"] == []
        assert result["xcheck"] == []
        assert result["zcheck"] == []


class TestCoordinateMapperCopy:
    """Test CoordinateMapper copy functionality."""

    def test_copy_basic(self) -> None:
        """Test basic copy operation."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        copied = mapper.copy()

        assert copied.node2coord == mapper.node2coord
        assert copied.coord2node == mapper.coord2node
        assert copied.node2role == mapper.node2role

    def test_copy_independence(self) -> None:
        """Test that copied mapper is independent from original."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")

        copied = mapper.copy()
        copied.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        assert len(mapper.node2coord) == 1
        assert len(copied.node2coord) == 2


class TestCoordinateMapperMerge:
    """Test CoordinateMapper merge functionality."""

    def test_merge_basic(self) -> None:
        """Test basic merge operation."""
        mapper1 = CoordinateMapper()
        mapper1.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")

        mapper2 = CoordinateMapper()
        mapper2.add_node(NodeIdLocal(10), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        merged = mapper1.merge(mapper2, {}, {})

        assert len(merged.node2coord) == 2
        assert NodeIdLocal(1) in merged.node2coord
        assert NodeIdLocal(10) in merged.node2coord

    def test_merge_with_remapping(self) -> None:
        """Test merge with node remapping."""
        mapper1 = CoordinateMapper()
        mapper1.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")

        mapper2 = CoordinateMapper()
        mapper2.add_node(NodeIdLocal(10), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        # Remap both mappers to different ranges
        map1 = {1: 100}
        map2 = {10: 200}

        merged = mapper1.merge(mapper2, map1, map2)

        assert NodeIdLocal(100) in merged.node2coord
        assert NodeIdLocal(200) in merged.node2coord
        assert NodeIdLocal(1) not in merged.node2coord
        assert NodeIdLocal(10) not in merged.node2coord

    def test_merge_preserves_roles(self) -> None:
        """Test that merge preserves roles."""
        mapper1 = CoordinateMapper()
        mapper1.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")

        mapper2 = CoordinateMapper()
        mapper2.add_node(NodeIdLocal(10), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        merged = mapper1.merge(mapper2, {}, {})

        assert merged.node2role[NodeIdLocal(1)] == "data"
        assert merged.node2role[NodeIdLocal(10)] == "ancilla_x"


class TestCoordinateMapperClear:
    """Test clear functionality."""

    def test_clear(self) -> None:
        """Test clearing all mappings."""
        mapper = CoordinateMapper()
        mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
        mapper.add_node(NodeIdLocal(2), PhysCoordGlobal3D((1, 0, 0)), "ancilla_x")

        mapper.clear()

        assert mapper.node2coord == {}
        assert mapper.coord2node == {}
        assert mapper.node2role == {}


class TestCoordinateMapperEdgeCases:
    """Test edge cases."""

    def test_large_coordinates(self) -> None:
        """Test handling of large coordinate values."""
        mapper = CoordinateMapper()
        large_coord = PhysCoordGlobal3D((999999, 999999, 999999))
        node_id = NodeIdLocal(1)

        mapper.add_node(node_id, large_coord)

        assert mapper.get_coordinate(node_id) == large_coord

    def test_negative_coordinates(self) -> None:
        """Test handling of negative coordinates."""
        mapper = CoordinateMapper()
        neg_coord = PhysCoordGlobal3D((-100, -200, -300))
        node_id = NodeIdLocal(1)

        mapper.add_node(node_id, neg_coord)

        assert mapper.get_coordinate(node_id) == neg_coord

    def test_zero_coordinate(self) -> None:
        """Test handling of zero coordinate."""
        mapper = CoordinateMapper()
        zero_coord = PhysCoordGlobal3D((0, 0, 0))
        node_id = NodeIdLocal(0)

        mapper.add_node(node_id, zero_coord)

        assert mapper.get_coordinate(node_id) == zero_coord


class TestCoordinateCollisions:
    """Test coordinate collision scenarios."""

    def test_add_same_coordinate_different_nodes(self) -> None:
        """Test adding the same coordinate with different node IDs (last-wins)."""
        mapper = CoordinateMapper()
        coord = PhysCoordGlobal3D((0, 0, 0))
        node1 = NodeIdLocal(1)
        node2 = NodeIdLocal(2)

        # Add first node
        mapper.add_node(node1, coord, "data")
        assert mapper.coord2node[coord] == node1

        # Add second node with same coordinate (should overwrite in coord2node)
        mapper.add_node(node2, coord, "ancilla_x")
        assert mapper.coord2node[coord] == node2  # Last wins
        assert mapper.node2coord[node1] == coord  # First node still maps to coord
        assert mapper.node2coord[node2] == coord  # Second node also maps to coord

    def test_merge_with_coordinate_collision(self) -> None:
        """Test merge raises ValueError on coordinate collision."""
        mapper1 = CoordinateMapper()
        mapper2 = CoordinateMapper()

        coord = PhysCoordGlobal3D((5, 5, 5))
        mapper1.add_node(NodeIdLocal(1), coord)
        mapper2.add_node(NodeIdLocal(2), coord)

        # Merge should raise ValueError due to coordinate collision
        with pytest.raises(ValueError, match="Coordinate collision detected"):
            mapper1.merge(mapper2, {}, {})

    def test_merge_with_same_node_same_coordinate(self) -> None:
        """Test merge succeeds when same node maps to same coordinate."""
        mapper1 = CoordinateMapper()
        mapper2 = CoordinateMapper()

        coord = PhysCoordGlobal3D((5, 5, 5))
        node_id = NodeIdLocal(1)
        mapper1.add_node(node_id, coord, "data")
        mapper2.add_node(node_id, coord, "data")

        # Should succeed because same node ID maps to same coordinate
        merged = mapper1.merge(mapper2, {}, {})
        assert merged.get_coordinate(node_id) == coord

    def test_merge_with_remapped_collision(self) -> None:
        """Test merge with coordinate collision after node remapping."""
        mapper1 = CoordinateMapper()
        mapper2 = CoordinateMapper()

        coord = PhysCoordGlobal3D((10, 10, 10))
        mapper1.add_node(NodeIdLocal(1), coord)
        mapper2.add_node(NodeIdLocal(2), coord)

        # Both nodes will be remapped to different IDs, but coordinate collision remains
        node_map1 = {1: 100}
        node_map2 = {2: 200}

        with pytest.raises(ValueError, match="Coordinate collision detected"):
            mapper1.merge(mapper2, node_map1, node_map2)
