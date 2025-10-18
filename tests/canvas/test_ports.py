"""Unit tests for PortManager class."""

from __future__ import annotations

import pytest

from lspattern.canvas.ports import PortManager
from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D


class TestPortManagerBasic:
    """Test basic PortManager functionality."""

    def test_initialization(self) -> None:
        """Test PortManager initializes with empty collections."""
        manager = PortManager()
        assert manager.in_portset == {}
        assert manager.out_portset == {}
        assert manager.cout_port_groups_cube == {}
        assert manager.cout_group_lookup_cube == {}
        assert manager.in_ports == []
        assert manager.out_ports == []

    def test_add_in_ports(self) -> None:
        """Test adding input ports."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        nodes = [NodeIdLocal(1), NodeIdLocal(2), NodeIdLocal(3)]

        manager.add_in_ports(patch_pos, nodes)

        assert patch_pos in manager.in_portset
        assert manager.in_portset[patch_pos] == nodes
        assert manager.in_ports == nodes

    def test_add_out_ports(self) -> None:
        """Test adding output ports."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        nodes = [NodeIdLocal(4), NodeIdLocal(5)]

        manager.add_out_ports(patch_pos, nodes)

        assert patch_pos in manager.out_portset
        assert manager.out_portset[patch_pos] == nodes
        assert manager.out_ports == nodes

    def test_add_multiple_ports_same_patch(self) -> None:
        """Test adding multiple ports to the same patch."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))

        manager.add_in_ports(patch_pos, [NodeIdLocal(1)])
        manager.add_in_ports(patch_pos, [NodeIdLocal(2)])

        assert manager.in_portset[patch_pos] == [NodeIdLocal(1), NodeIdLocal(2)]
        assert manager.in_ports == [NodeIdLocal(1), NodeIdLocal(2)]

    def test_add_ports_different_patches(self) -> None:
        """Test adding ports to different patches."""
        manager = PortManager()
        patch1 = PatchCoordGlobal3D((0, 0, 0))
        patch2 = PatchCoordGlobal3D((1, 0, 0))

        manager.add_in_ports(patch1, [NodeIdLocal(1)])
        manager.add_in_ports(patch2, [NodeIdLocal(2)])

        assert len(manager.in_portset) == 2
        assert manager.in_portset[patch1] == [NodeIdLocal(1)]
        assert manager.in_portset[patch2] == [NodeIdLocal(2)]
        assert manager.in_ports == [NodeIdLocal(1), NodeIdLocal(2)]


class TestPortManagerCoutGroups:
    """Test cout group management functionality."""

    def test_register_cout_group_cube(self) -> None:
        """Test registering a cout group."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        nodes = [NodeIdLocal(10), NodeIdLocal(11), NodeIdLocal(12)]

        manager.register_cout_group_cube(patch_pos, nodes)

        assert patch_pos in manager.cout_port_groups_cube
        assert len(manager.cout_port_groups_cube[patch_pos]) == 1
        assert manager.cout_port_groups_cube[patch_pos][0] == nodes
        # Check lookup
        for i, node in enumerate(nodes):
            assert manager.cout_group_lookup_cube[node] == (patch_pos, 0)

    def test_register_multiple_cout_groups(self) -> None:
        """Test registering multiple cout groups for same patch."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        group1 = [NodeIdLocal(10), NodeIdLocal(11)]
        group2 = [NodeIdLocal(12), NodeIdLocal(13)]

        manager.register_cout_group_cube(patch_pos, group1)
        manager.register_cout_group_cube(patch_pos, group2)

        assert len(manager.cout_port_groups_cube[patch_pos]) == 2
        assert manager.cout_port_groups_cube[patch_pos][0] == group1
        assert manager.cout_port_groups_cube[patch_pos][1] == group2
        assert set(manager.cout_portset_cube[patch_pos]) == set(group1 + group2)
        # Check lookup indices
        assert manager.cout_group_lookup_cube[NodeIdLocal(10)] == (patch_pos, 0)
        assert manager.cout_group_lookup_cube[NodeIdLocal(12)] == (patch_pos, 1)

    def test_register_empty_cout_group(self) -> None:
        """Test that empty cout groups are not registered."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))

        manager.register_cout_group_cube(patch_pos, [])

        assert patch_pos not in manager.cout_port_groups_cube

    def test_register_cout_group_with_none_values(self) -> None:
        """Test cout group registration filters None values."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        nodes_with_none = [NodeIdLocal(10), None, NodeIdLocal(11)]  # type: ignore[list-item]

        manager.register_cout_group_cube(patch_pos, nodes_with_none)  # type: ignore[arg-type]

        expected = [NodeIdLocal(10), NodeIdLocal(11)]
        assert manager.cout_port_groups_cube[patch_pos][0] == expected

    def test_get_cout_group_by_node(self) -> None:
        """Test retrieving cout group by node ID."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        nodes = [NodeIdLocal(10), NodeIdLocal(11)]

        manager.register_cout_group_cube(patch_pos, nodes)

        result = manager.get_cout_group_by_node(NodeIdLocal(10))
        assert result is not None
        patch, group = result
        assert patch == patch_pos
        assert group == nodes

    def test_get_cout_group_by_node_not_found(self) -> None:
        """Test get_cout_group_by_node returns None for non-existent node."""
        manager = PortManager()
        result = manager.get_cout_group_by_node(NodeIdLocal(999))
        assert result is None

    def test_rebuild_cout_group_cache(self) -> None:
        """Test rebuilding cout group caches."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        group1 = [NodeIdLocal(10), NodeIdLocal(11)]
        group2 = [NodeIdLocal(12), NodeIdLocal(13)]

        manager.register_cout_group_cube(patch_pos, group1)
        manager.register_cout_group_cube(patch_pos, group2)

        # Manually corrupt caches
        manager.cout_portset_cube = {}
        manager.cout_group_lookup_cube = {}

        # Rebuild
        manager.rebuild_cout_group_cache()

        # Verify caches are restored
        assert set(manager.cout_portset_cube[patch_pos]) == set(group1 + group2)
        assert manager.cout_group_lookup_cube[NodeIdLocal(10)] == (patch_pos, 0)
        assert manager.cout_group_lookup_cube[NodeIdLocal(12)] == (patch_pos, 1)


class TestPortManagerRemapping:
    """Test node remapping functionality."""

    def test_remap_in_ports(self) -> None:
        """Test remapping input ports."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        manager.add_in_ports(patch_pos, [NodeIdLocal(1), NodeIdLocal(2)])

        node_map = {1: 10, 2: 20}
        manager.remap_ports(node_map)

        assert manager.in_portset[patch_pos] == [NodeIdLocal(10), NodeIdLocal(20)]
        assert manager.in_ports == [NodeIdLocal(10), NodeIdLocal(20)]

    def test_remap_out_ports(self) -> None:
        """Test remapping output ports."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        manager.add_out_ports(patch_pos, [NodeIdLocal(3), NodeIdLocal(4)])

        node_map = {3: 30, 4: 40}
        manager.remap_ports(node_map)

        assert manager.out_portset[patch_pos] == [NodeIdLocal(30), NodeIdLocal(40)]
        assert manager.out_ports == [NodeIdLocal(30), NodeIdLocal(40)]

    def test_remap_cout_groups(self) -> None:
        """Test remapping cout port groups."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        manager.register_cout_group_cube(patch_pos, [NodeIdLocal(10), NodeIdLocal(11)])

        node_map = {10: 100, 11: 110}
        manager.remap_ports(node_map)

        expected = [NodeIdLocal(100), NodeIdLocal(110)]
        assert manager.cout_port_groups_cube[patch_pos][0] == expected
        assert manager.cout_group_lookup_cube[NodeIdLocal(100)] == (patch_pos, 0)

    def test_remap_preserves_unmapped_nodes(self) -> None:
        """Test that nodes not in mapping are preserved."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        manager.add_in_ports(patch_pos, [NodeIdLocal(1), NodeIdLocal(2)])

        node_map = {1: 10}  # Only remap node 1
        manager.remap_ports(node_map)

        assert manager.in_portset[patch_pos] == [NodeIdLocal(10), NodeIdLocal(2)]

    def test_remap_empty_manager(self) -> None:
        """Test remapping empty port manager does not fail."""
        manager = PortManager()
        node_map = {1: 10}

        manager.remap_ports(node_map)  # Should not raise

        assert manager.in_ports == []
        assert manager.out_ports == []


class TestPortManagerCopy:
    """Test PortManager copy functionality."""

    def test_copy_creates_independent_instance(self) -> None:
        """Test that copy creates an independent instance."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        manager.add_in_ports(patch_pos, [NodeIdLocal(1)])
        manager.register_cout_group_cube(patch_pos, [NodeIdLocal(10)])

        copied = manager.copy()

        # Modify original
        manager.add_in_ports(patch_pos, [NodeIdLocal(2)])

        # Copied should be unchanged
        assert len(copied.in_portset[patch_pos]) == 1
        assert copied.in_portset[patch_pos] == [NodeIdLocal(1)]

    def test_copy_deep_copies_collections(self) -> None:
        """Test that copy performs deep copy of collections."""
        manager = PortManager()
        patch1 = PatchCoordGlobal3D((0, 0, 0))
        patch2 = PatchCoordGlobal3D((1, 0, 0))
        manager.add_in_ports(patch1, [NodeIdLocal(1)])
        manager.add_out_ports(patch2, [NodeIdLocal(2)])
        manager.register_cout_group_cube(patch1, [NodeIdLocal(10), NodeIdLocal(11)])

        copied = manager.copy()

        # Verify all collections are copied
        assert copied.in_portset == manager.in_portset
        assert copied.in_portset is not manager.in_portset
        assert copied.out_portset == manager.out_portset
        assert copied.out_portset is not manager.out_portset
        assert copied.cout_port_groups_cube == manager.cout_port_groups_cube
        assert copied.cout_port_groups_cube is not manager.cout_port_groups_cube


class TestPortManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_multiple_patches_different_ports(self) -> None:
        """Test managing ports across multiple patches."""
        manager = PortManager()
        patch1 = PatchCoordGlobal3D((0, 0, 0))
        patch2 = PatchCoordGlobal3D((1, 0, 0))
        patch3 = PatchCoordGlobal3D((2, 0, 0))

        manager.add_in_ports(patch1, [NodeIdLocal(1)])
        manager.add_out_ports(patch2, [NodeIdLocal(2)])
        manager.register_cout_group_cube(patch3, [NodeIdLocal(10)])

        assert len(manager.in_portset) == 1
        assert len(manager.out_portset) == 1
        assert len(manager.cout_port_groups_cube) == 1

    def test_large_node_ids(self) -> None:
        """Test handling of large node IDs."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        large_id = NodeIdLocal(999999)

        manager.add_in_ports(patch_pos, [large_id])

        assert manager.in_portset[patch_pos] == [large_id]
        assert manager.in_ports == [large_id]

    def test_repeated_node_in_different_port_types(self) -> None:
        """Test that same node can appear in different port types (though unusual)."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        node = NodeIdLocal(42)

        manager.add_in_ports(patch_pos, [node])
        manager.add_out_ports(patch_pos, [node])

        # This is allowed (though may be unusual in practice)
        assert node in manager.in_ports
        assert node in manager.out_ports

    def test_add_empty_in_ports(self) -> None:
        """Test adding empty in_ports list."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))

        manager.add_in_ports(patch_pos, [])

        # Empty list should not create an entry
        assert patch_pos not in manager.in_portset
        assert manager.in_ports == []

    def test_add_empty_out_ports(self) -> None:
        """Test adding empty out_ports list."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))

        manager.add_out_ports(patch_pos, [])

        # Empty list should not create an entry
        assert patch_pos not in manager.out_portset
        assert manager.out_ports == []

    def test_add_in_ports_with_none_values(self) -> None:
        """Test that None values are filtered from in_ports."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        nodes_with_none = [NodeIdLocal(1), None, NodeIdLocal(2)]  # type: ignore[list-item]

        manager.add_in_ports(patch_pos, nodes_with_none)  # type: ignore[arg-type]

        expected = [NodeIdLocal(1), NodeIdLocal(2)]
        assert manager.in_portset[patch_pos] == expected
        assert manager.in_ports == expected

    def test_add_out_ports_with_none_values(self) -> None:
        """Test that None values are filtered from out_ports."""
        manager = PortManager()
        patch_pos = PatchCoordGlobal3D((0, 0, 0))
        nodes_with_none = [NodeIdLocal(1), None, NodeIdLocal(2)]  # type: ignore[list-item]

        manager.add_out_ports(patch_pos, nodes_with_none)  # type: ignore[arg-type]

        expected = [NodeIdLocal(1), NodeIdLocal(2)]
        assert manager.out_portset[patch_pos] == expected
        assert manager.out_ports == expected


class TestPortManagerMerge:
    """Test PortManager merge functionality."""

    def test_merge_basic(self) -> None:
        """Test basic merge operation."""
        pm1 = PortManager()
        pm2 = PortManager()

        patch1 = PatchCoordGlobal3D((0, 0, 0))
        patch2 = PatchCoordGlobal3D((1, 0, 0))

        pm1.add_in_ports(patch1, [NodeIdLocal(1)])
        pm1.add_out_ports(patch1, [NodeIdLocal(2)])

        pm2.add_in_ports(patch2, [NodeIdLocal(10)])
        pm2.add_out_ports(patch2, [NodeIdLocal(20)])

        # Identity mapping (no remapping)
        node_map = {}
        merged = pm1.merge(pm2, node_map, node_map)

        # in_ports should come from pm2 (other) by default
        assert merged.in_portset == {patch2: [NodeIdLocal(10)]}
        # out_ports should be merged from both
        assert patch1 in merged.out_portset
        assert patch2 in merged.out_portset

    def test_merge_with_remapping(self) -> None:
        """Test merge with node remapping."""
        pm1 = PortManager()
        pm2 = PortManager()

        patch = PatchCoordGlobal3D((0, 0, 0))

        pm1.add_out_ports(patch, [NodeIdLocal(1)])
        pm2.add_out_ports(patch, [NodeIdLocal(10)])

        # Remap pm1's nodes to 100+, pm2's to 200+
        map1 = {1: 101}
        map2 = {10: 210}

        merged = pm1.merge(pm2, map1, map2)

        # Both out_ports should be remapped and merged
        assert NodeIdLocal(101) in merged.out_ports
        assert NodeIdLocal(210) in merged.out_ports

    def test_merge_in_ports_from_self(self) -> None:
        """Test merge with in_ports from self."""
        pm1 = PortManager()
        pm2 = PortManager()

        patch1 = PatchCoordGlobal3D((0, 0, 0))
        patch2 = PatchCoordGlobal3D((1, 0, 0))

        pm1.add_in_ports(patch1, [NodeIdLocal(1)])
        pm2.add_in_ports(patch2, [NodeIdLocal(10)])

        merged = pm1.merge(pm2, {}, {}, in_ports_from="self")

        # in_ports should come from pm1 (self)
        assert merged.in_portset == {patch1: [NodeIdLocal(1)]}

    def test_merge_in_ports_from_both(self) -> None:
        """Test merge with in_ports from both sources."""
        pm1 = PortManager()
        pm2 = PortManager()

        patch1 = PatchCoordGlobal3D((0, 0, 0))
        patch2 = PatchCoordGlobal3D((1, 0, 0))

        pm1.add_in_ports(patch1, [NodeIdLocal(1)])
        pm2.add_in_ports(patch2, [NodeIdLocal(10)])

        merged = pm1.merge(pm2, {}, {}, in_ports_from="both")

        # in_ports should come from both
        assert patch1 in merged.in_portset
        assert patch2 in merged.in_portset

    def test_merge_invalid_in_ports_from(self) -> None:
        """Test merge with invalid in_ports_from raises ValueError."""
        pm1 = PortManager()
        pm2 = PortManager()

        with pytest.raises(ValueError, match="Invalid in_ports_from"):
            pm1.merge(pm2, {}, {}, in_ports_from="invalid")  # type: ignore[arg-type]

    def test_merge_cout_groups(self) -> None:
        """Test merge with cout_port_groups."""
        pm1 = PortManager()
        pm2 = PortManager()

        patch = PatchCoordGlobal3D((0, 0, 0))

        pm1.register_cout_group_cube(patch, [NodeIdLocal(1), NodeIdLocal(2)])
        pm2.register_cout_group_cube(patch, [NodeIdLocal(10), NodeIdLocal(11)])

        merged = pm1.merge(pm2, {}, {})

        # Both groups should be present
        assert patch in merged.cout_port_groups_cube
        assert len(merged.cout_port_groups_cube[patch]) == 2
