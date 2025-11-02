from __future__ import annotations

from lspattern.blocks.cubes.measure import MeasureXSkeleton, MeasureZSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import NodeIdLocal


def test_measure_blocks_single_layer_optimization() -> None:
    """Test that measurement blocks use only single layer with data qubits only."""
    # Test MeasureX with d=3
    spec: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    measure_x = MeasureXSkeleton(d=3, edgespec=spec).to_block()

    # Materialize the block to build the graph
    materialized_x = measure_x.materialize()

    # Verify that only data nodes exist (no ancilla nodes)
    data_nodes = [n for n, role in materialized_x.node2role.items() if role == "data"]
    ancilla_nodes = [n for n, role in materialized_x.node2role.items() if role.startswith("ancilla")]

    # Should have d*d=9 data nodes and no ancilla nodes
    assert len(data_nodes) == 9, f"Expected 9 data nodes, got {len(data_nodes)}"
    assert len(ancilla_nodes) == 0, f"Expected 0 ancilla nodes, got {len(ancilla_nodes)}"

    # Verify all nodes are at the same time coordinate (single layer)
    z_coords = [coord[2] for coord in materialized_x.node2coord.values()]
    unique_z_coords = set(z_coords)
    assert len(unique_z_coords) == 1, f"Expected single layer, got {len(unique_z_coords)} layers"

    # Test MeasureZ with same expectations
    measure_z = MeasureZSkeleton(d=3, edgespec=spec).to_block()
    materialized_z = measure_z.materialize()

    data_nodes_z = [n for n, role in materialized_z.node2role.items() if role == "data"]
    ancilla_nodes_z = [n for n, role in materialized_z.node2role.items() if role.startswith("ancilla")]

    assert len(data_nodes_z) == 9
    assert len(ancilla_nodes_z) == 0

    z_coords_z = [coord[2] for coord in materialized_z.node2coord.values()]
    unique_z_coords_z = set(z_coords_z)
    assert len(unique_z_coords_z) == 1


def test_measure_blocks_schedule_optimization() -> None:
    """Test that measurement blocks have optimized schedule with single time slot."""
    spec: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    measure_x = MeasureXSkeleton(d=3, edgespec=spec).to_block()
    materialized_x = measure_x.materialize()

    # Should have only one time slot in the schedule
    schedule_times = list(materialized_x.schedule.schedule.keys())
    assert len(schedule_times) == 1, f"Expected 1 time slot, got {len(schedule_times)}"

    # The single time slot should contain all data nodes
    time_slot = schedule_times[0]
    scheduled_nodes = materialized_x.schedule.schedule[time_slot]
    data_node_count = len([n for n, role in materialized_x.node2role.items() if role == "data"])

    assert len(scheduled_nodes) == data_node_count


def test_measure_blocks_no_temporal_edges() -> None:
    """Test that measurement blocks have no temporal edges (since they're single layer)."""
    spec: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    measure_x = MeasureXSkeleton(d=3, edgespec=spec).to_block()
    materialized_x = measure_x.materialize()

    # Count edges between different time coordinates
    temporal_edges = 0
    for edge in materialized_x.local_graph.physical_edges:
        node1, node2 = edge
        coord1 = materialized_x.node2coord[NodeIdLocal(int(node1))]
        coord2 = materialized_x.node2coord[NodeIdLocal(int(node2))]
        if coord1[2] != coord2[2]:  # Different z-coordinates means temporal edge
            temporal_edges += 1

    assert temporal_edges == 0, f"Expected 0 temporal edges, got {temporal_edges}"


def test_measure_blocks_memory_efficiency() -> None:
    """Test memory efficiency compared to theoretical full cube implementation."""
    spec: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    measure_x = MeasureXSkeleton(d=3, edgespec=spec).to_block()
    materialized_x = measure_x.materialize()

    # Actual node count (optimized)
    actual_nodes = len(materialized_x.node2coord)

    # Theoretical cube would have 2*d layers with data + ancilla nodes
    # Each layer would have d*d data + some ancilla nodes
    # For a rough estimate: 2*d layers * d*d data nodes per layer
    d = 3
    theoretical_cube_nodes = 2 * d * d * d  # Conservative estimate

    # Our optimization should use significantly fewer nodes
    efficiency_ratio = actual_nodes / theoretical_cube_nodes
    assert efficiency_ratio < 0.5, f"Expected >50% reduction, got {efficiency_ratio:.2%} efficiency"

    # Should have exactly d*d=9 nodes for d=3
    assert actual_nodes == 9, f"Expected exactly 9 nodes, got {actual_nodes}"
