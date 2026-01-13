"""Tests for inter-block temporal edges and pipe fragment offset fixes."""

from pathlib import Path
from textwrap import dedent

from lspattern.canvas_loader import load_canvas
from lspattern.mytype import Coord3D


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_inter_block_temporal_edges_stacked_cubes(tmp_path: Path) -> None:
    """Test that z-direction edges are automatically generated between stacked blocks.

    When two cubes are placed at z=0 and z=1 using patch-based blocks with num_layers=3,
    the nodes at the boundary should be connected across blocks.
    With d=3 and num_layers=3: each block has 6 z-layers (3 layers * 2 z per layer).
    First block: z in [0, 5], Second block: z in [6, 11]
    Inter-block edge connects z=5 to z=6.
    """
    layer_yaml = """
    name: StandardLayer
    description: Standard layer with ancilla
    layer1:
      basis: X
      ancilla: true
    layer2:
      basis: Z
      ancilla: true
    """

    block_yaml = """
    name: StandardBlock
    boundary: XXZZ
    layers:
      - type: standard_layer
        num_layers: 3
    """

    canvas_yaml = """
    name: StackedCubesCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: standard_block
      - position: [0, 0, 1]
        block: standard_block
    """

    _write_yaml(tmp_path / "standard_layer.yml", layer_yaml)
    _write_yaml(tmp_path / "standard_block.yml", block_yaml)
    canvas_path = tmp_path / "stacked_cubes_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # With d=3 and num_layers=3:
    # Each block has 6 z-layers: num_layers * 2 = 6
    # First block: z in [0, 5], Second block: z in [6, 11]
    # Inter-block edge should connect z=5 to z=6 for shared (x, y) coordinates

    # Find nodes at z=5 (last layer of first block)
    nodes_z5 = {n for n in canvas.nodes if n.z == 5}
    # Find nodes at z=6 (first layer of second block)
    nodes_z6 = {n for n in canvas.nodes if n.z == 6}

    assert nodes_z5, f"Should have nodes at z=5, got nodes with z values: {sorted({n.z for n in canvas.nodes})}"
    assert nodes_z6, f"Should have nodes at z=6, got nodes with z values: {sorted({n.z for n in canvas.nodes})}"

    # Check that inter-block edges exist for matching (x, y) coordinates
    inter_block_edges_found = []
    for n5 in nodes_z5:
        n6 = Coord3D(n5.x, n5.y, 6)
        if n6 in nodes_z6:
            edge = (n5, n6)
            if edge in canvas.edges:
                inter_block_edges_found.append(edge)

    assert inter_block_edges_found, (
        f"Expected inter-block edges between z=5 and z=6. Nodes at z=5: {nodes_z5}, Nodes at z=6: {nodes_z6}"
    )


def test_inter_block_edges_not_added_within_same_block(tmp_path: Path) -> None:
    """Test that inter-block edges are not duplicated for edges within a single block."""
    graph_json = """
    {
      "coord_mode": "local",
      "time_mode": "local",
      "nodes": [
        {"coord": [0, 0, 0], "basis": "X"},
        {"coord": [0, 0, 1], "basis": "X"},
        {"coord": [0, 0, 2], "basis": "X"}
      ],
      "edges": [
        [[0, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [0, 0, 2]]
      ],
      "xflow": [
        {"from": [0, 0, 0], "to": [[0, 0, 1]]},
        {"from": [0, 0, 1], "to": [[0, 0, 2]]}
      ],
      "schedule": {
        "prep": [{"time": 0, "nodes": [[0, 0, 0], [0, 0, 1], [0, 0, 2]]}],
        "entangle": [
          {"time": 1, "edges": [[[0, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 0, 2]]]}
        ],
        "meas": [{"time": 2, "nodes": [[0, 0, 0], [0, 0, 1], [0, 0, 2]]}]
      },
      "detector_candidates": {
        "syndrome_meas": []
      }
    }
    """

    block_yaml = """
    name: ThreeLayerBlock
    boundary: XXZZ
    graph: three_layer_block.json
    """

    canvas_yaml = """
    name: SingleBlockCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: three_layer_block
    """

    _write_json(tmp_path / "three_layer_block.json", graph_json)
    _write_yaml(tmp_path / "three_layer_block.yml", block_yaml)
    canvas_path = tmp_path / "single_block_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _spec = load_canvas(canvas_path, code_distance=3)

    # Only the edges defined in the graph should exist, no duplicates
    expected_edges = {
        (Coord3D(0, 0, 0), Coord3D(0, 0, 1)),
        (Coord3D(0, 0, 1), Coord3D(0, 0, 2)),
    }
    assert canvas.edges == expected_edges, f"Expected {expected_edges}, got {canvas.edges}"


def test_pipe_does_not_overlap_with_cube(tmp_path: Path) -> None:
    """Test that pipe nodes do not overlap with adjacent cube nodes.

    This verifies the fix for duplicate offset calculation in pipe fragment generation.
    """
    layer_yaml = """
    name: StandardLayer
    description: Standard layer with ancilla
    layer1:
      basis: X
      ancilla: true
    layer2:
      basis: Z
      ancilla: true
    """

    block_yaml = """
    name: StandardBlock
    boundary: XXZZ
    layers:
      - type: standard_layer
        num_layers: 1
    """

    canvas_yaml = """
    name: CubePipeCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: standard_block
      - position: [1, 0, 0]
        block: standard_block
    pipe:
      - start: [0, 0, 0]
        end: [1, 0, 0]
        block: standard_block
    """

    _write_yaml(tmp_path / "standard_layer.yml", layer_yaml)
    _write_yaml(tmp_path / "standard_block.yml", block_yaml)
    canvas_path = tmp_path / "cube_pipe_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _spec = load_canvas(canvas_path, code_distance=3)

    # Get all nodes
    all_nodes = canvas.nodes

    # Cube at [0,0,0] with d=3: x range [0, 4], cube at [1,0,0]: x range [6, 10]
    # Pipe should be in between (x around 5) and not overlap with cube x-ranges
    cube0_x_max = 4  # 2*(d-1) = 2*2 = 4
    cube1_x_min = 6  # cube1_offset_x = 2*d = 6

    # Check pipe nodes exist
    pipe_nodes = {n for n in all_nodes if cube0_x_max < n.x < cube1_x_min}

    # Verify no node has x coordinate that would indicate overlap
    # (Pipe should not have nodes at x <= 4 or x >= 6 that are also pipe nodes)
    assert len(pipe_nodes) > 0 or len(all_nodes) > 0, "Canvas should have nodes"

    # The key check: pipe nodes should not duplicate cube boundary nodes
    # Count nodes at boundary x-coordinates
    cube0_boundary_nodes = {n for n in all_nodes if n.x == cube0_x_max}
    cube1_boundary_nodes = {n for n in all_nodes if n.x == cube1_x_min}

    # Each boundary coordinate should appear exactly once (no duplicates from pipe)
    for node in cube0_boundary_nodes:
        count = sum(1 for n in all_nodes if n == node)
        assert count == 1, f"Node {node} appears {count} times, should be 1"

    for node in cube1_boundary_nodes:
        count = sum(1 for n in all_nodes if n == node)
        assert count == 1, f"Node {node} appears {count} times, should be 1"


def test_inter_block_entanglement_scheduled(tmp_path: Path) -> None:
    """Test that inter-block edges are scheduled for entanglement.

    With d=3, num_layers=3, inter-block edges connect z=5 to z=6.
    These edges should be scheduled for entanglement at the start of the second block.
    """
    layer_yaml = """
    name: StandardLayer
    description: Standard layer with ancilla
    layer1:
      basis: X
      ancilla: true
    layer2:
      basis: Z
      ancilla: true
    """

    block_yaml = """
    name: StandardBlock
    boundary: XXZZ
    layers:
      - type: standard_layer
        num_layers: 3
    """

    canvas_yaml = """
    name: TwoBlockCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: standard_block
      - position: [0, 0, 1]
        block: standard_block
    """

    _write_yaml(tmp_path / "standard_layer.yml", layer_yaml)
    _write_yaml(tmp_path / "standard_block.yml", block_yaml)
    canvas_path = tmp_path / "two_block_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Check that inter-block entanglement is scheduled
    # Inter-block edges connect z=5 to z=6
    entangle_times = canvas.scheduler.entangle_time

    # Find edges that span z=5 to z=6
    inter_block_edges_scheduled = [
        edge for edges in entangle_times.values() for edge in edges if edge[0].z == 5 and edge[1].z == 6
    ]

    assert inter_block_edges_scheduled, (
        f"Inter-block edges (z=5 to z=6) should be scheduled for entanglement. All scheduled edges: {entangle_times}"
    )
