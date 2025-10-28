"""Tests for UnitLayer implementations.

This module tests the basic functionality of different UnitLayer types:
- MemoryUnitLayer: Standard memory layer with X and Z checks
- InitPlusUnitLayer: |+⟩ initialization layer
- InitZeroUnitLayer: |0⟩ initialization layer
- EmptyUnitLayer: Empty placeholder layer
"""

from __future__ import annotations

from graphqomb.graphstate import GraphState

from lspattern.blocks.layers.empty import EmptyUnitLayer
from lspattern.blocks.layers.initialize import InitPlusUnitLayer, InitZeroUnitLayer
from lspattern.blocks.layers.measure import MeasureXUnitLayer, MeasureZUnitLayer
from lspattern.blocks.layers.memory import MemoryUnitLayer
from lspattern.consts import BoundarySide, EdgeSpecValue, NodeRole
from lspattern.tiling.template import RotatedPlanarCubeTemplate


def test_memory_unit_layer_builds_correct_structure() -> None:
    """Test that MemoryUnitLayer creates the expected graph structure."""
    # Create template with d=3
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    # Build memory unit layer
    graph = GraphState()
    memory_layer = MemoryUnitLayer()
    z_offset = 0
    layer_data = memory_layer.build_layer(graph, z_offset, template)

    # Check that layer has two z-levels (z=0 with Z-checks, z=1 with X-checks)
    assert len(layer_data.nodes_by_z) == 2
    assert z_offset in layer_data.nodes_by_z
    assert z_offset + 1 in layer_data.nodes_by_z

    # Check that nodes have correct roles
    z_layer_nodes = layer_data.nodes_by_z[z_offset]
    x_layer_nodes = layer_data.nodes_by_z[z_offset + 1]

    # Count data and ancilla nodes
    z_data_count = sum(1 for node_id in z_layer_nodes.values() if layer_data.node2role[node_id] == NodeRole.DATA)
    z_ancilla_count = sum(
        1 for node_id in z_layer_nodes.values() if layer_data.node2role[node_id] == NodeRole.ANCILLA_Z
    )
    x_data_count = sum(1 for node_id in x_layer_nodes.values() if layer_data.node2role[node_id] == NodeRole.DATA)
    x_ancilla_count = sum(
        1 for node_id in x_layer_nodes.values() if layer_data.node2role[node_id] == NodeRole.ANCILLA_X
    )

    # For d=3, we expect 9 data qubits (3x3) and some ancilla qubits
    assert z_data_count == 9
    assert x_data_count == 9
    assert z_ancilla_count > 0  # Should have some Z-check ancillas
    assert x_ancilla_count > 0  # Should have some X-check ancillas

    # Check that schedule is populated
    assert len(layer_data.schedule.schedule) > 0

    # Check that parity checks are populated
    assert len(layer_data.parity.checks) > 0


def test_init_plus_unit_layer_structure() -> None:
    """Test that InitPlusUnitLayer creates the expected structure."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    init_layer = InitPlusUnitLayer()
    z_offset = 0
    layer_data = init_layer.build_layer(graph, z_offset, template)

    # Check structure (should have 2 layers like memory)
    assert len(layer_data.nodes_by_z) == 2
    assert z_offset in layer_data.nodes_by_z
    assert z_offset + 1 in layer_data.nodes_by_z

    # Check that first layer ancillas are in dangling_parity (not deterministic)
    assert len(layer_data.parity.dangling_parity) > 0

    # Check that some coordinates are marked as ignore_dangling
    assert len(layer_data.parity.ignore_dangling) > 0


def test_init_zero_unit_layer_structure() -> None:
    """Test that InitZeroUnitLayer creates the expected thin structure."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    init_zero_layer = InitZeroUnitLayer()
    z_offset = 0
    layer_data = init_zero_layer.build_layer(graph, z_offset, template)

    # InitZero should only have one layer (z_offset + 1) with X-checks
    assert len(layer_data.nodes_by_z) == 1
    assert z_offset + 1 in layer_data.nodes_by_z

    # Check that nodes have X-check ancillas
    x_layer_nodes = layer_data.nodes_by_z[z_offset + 1]
    x_ancilla_count = sum(
        1 for node_id in x_layer_nodes.values() if layer_data.node2role[node_id] == NodeRole.ANCILLA_X
    )
    assert x_ancilla_count > 0

    # Check that all ancillas are marked as ignore_dangling
    assert len(layer_data.parity.ignore_dangling) > 0


def test_empty_unit_layer_produces_empty_data() -> None:
    """Test that EmptyUnitLayer produces completely empty layer data."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    empty_layer = EmptyUnitLayer()
    z_offset = 0
    layer_data = empty_layer.build_layer(graph, z_offset, template)

    # Check that everything is empty
    assert len(layer_data.nodes_by_z) == 0
    assert len(layer_data.node2coord) == 0
    assert len(layer_data.coord2node) == 0
    assert len(layer_data.node2role) == 0
    assert len(layer_data.schedule.schedule) == 0
    assert len(layer_data.flow.flow) == 0
    assert len(layer_data.parity.checks) == 0


def test_memory_unit_layer_parity_construction() -> None:
    """Test that MemoryUnitLayer constructs correct parity checks."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    memory_layer = MemoryUnitLayer()
    z_offset = 0
    layer_data = memory_layer.build_layer(graph, z_offset, template)

    # Check that parity checks are created for both Z and X layers
    assert len(layer_data.parity.checks) > 0

    # Check that dangling_parity is populated for connection to next layer
    assert len(layer_data.parity.dangling_parity) > 0

    # Verify that parity checks contain node IDs that exist in the graph
    for z_dict in layer_data.parity.checks.values():
        for node_set in z_dict.values():
            for node_id in node_set:
                assert int(node_id) in layer_data.node2role


def test_unit_layer_temporal_edges() -> None:
    """Test that temporal edges are created correctly between layers."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    memory_layer = MemoryUnitLayer()
    z_offset = 0
    layer_data = memory_layer.build_layer(graph, z_offset, template)

    # Check that flow accumulator has entries (temporal dependencies)
    assert len(layer_data.flow.flow) > 0

    # Verify that flow entries reference nodes in the graph
    for src, dsts in layer_data.flow.flow.items():
        assert int(src) in layer_data.node2role
        for dst in dsts:
            assert int(dst) in layer_data.node2role


def test_measure_x_unit_layer_structure() -> None:
    """Test that MeasureXUnitLayer creates the expected single-layer structure."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    measure_layer = MeasureXUnitLayer()
    z_offset = 0
    layer_data = measure_layer.build_layer(graph, z_offset, template)

    # MeasureX should only have one layer with data qubits only
    assert len(layer_data.nodes_by_z) == 1
    assert z_offset in layer_data.nodes_by_z

    # Check that all nodes are data qubits (no ancillas)
    layer_nodes = layer_data.nodes_by_z[z_offset]
    for node_id in layer_nodes.values():
        assert layer_data.node2role[node_id] == NodeRole.DATA

    # Check that we have the expected number of data qubits
    assert len(layer_nodes) == 9  # 3x3 data qubits for d=3

    # Check that parity checks are populated (detectors)
    assert len(layer_data.parity.checks) > 0

    # Check that schedule is populated
    assert len(layer_data.schedule.schedule) > 0


def test_measure_z_unit_layer_structure() -> None:
    """Test that MeasureZUnitLayer creates the expected single-layer structure."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    measure_layer = MeasureZUnitLayer()
    z_offset = 0
    layer_data = measure_layer.build_layer(graph, z_offset, template)

    # MeasureZ should only have one layer with data qubits only
    assert len(layer_data.nodes_by_z) == 1
    assert z_offset in layer_data.nodes_by_z

    # Check that all nodes are data qubits (no ancillas)
    layer_nodes = layer_data.nodes_by_z[z_offset]
    for node_id in layer_nodes.values():
        assert layer_data.node2role[node_id] == NodeRole.DATA

    # Check that we have the expected number of data qubits
    assert len(layer_nodes) == 9  # 3x3 data qubits for d=3

    # Check that parity checks are populated (detectors)
    assert len(layer_data.parity.checks) > 0

    # Check that schedule is populated
    assert len(layer_data.schedule.schedule) > 0


def test_measure_x_parity_construction() -> None:
    """Test that MeasureXUnitLayer constructs correct parity checks."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    measure_layer = MeasureXUnitLayer()
    z_offset = 0
    layer_data = measure_layer.build_layer(graph, z_offset, template)

    # Verify that parity checks contain node IDs that exist in the graph
    for z_dict in layer_data.parity.checks.values():
        for z, node_set in z_dict.items():
            assert z == z_offset  # All checks should be at z_offset
            for node_id in node_set:
                assert int(node_id) in layer_data.node2role
                assert layer_data.node2role[int(node_id)] == NodeRole.DATA


def test_measure_z_parity_construction() -> None:
    """Test that MeasureZUnitLayer constructs correct parity checks."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    template.to_tiling()

    graph = GraphState()
    measure_layer = MeasureZUnitLayer()
    z_offset = 0
    layer_data = measure_layer.build_layer(graph, z_offset, template)

    # Verify that parity checks contain node IDs that exist in the graph
    for z_dict in layer_data.parity.checks.values():
        for z, node_set in z_dict.items():
            assert z == z_offset  # All checks should be at z_offset
            for node_id in node_set:
                assert int(node_id) in layer_data.node2role
                assert layer_data.node2role[int(node_id)] == NodeRole.DATA
