"""Tests for UnitLayer and YAML-based unit layer construction."""

from pathlib import Path

from lspattern.consts import NodeRole
from lspattern.new_blocks.mytype import Coord3D
from lspattern.new_blocks.unit_layer import load_unit_layer_from_yaml


def test_load_memory_unit_layer_from_yaml() -> None:
    """Test loading MemoryUnit layer from YAML configuration."""
    yaml_path = Path("lspattern/new_blocks/patch_layout/layers/memory.yml")
    unit_layer = load_unit_layer_from_yaml(yaml_path)

    assert unit_layer.name == "MemoryUnit"
    assert unit_layer.description == "This patch layout specifies a memory unit layer"
    assert unit_layer.layout_type == "rotated_surface_code"
    assert len(unit_layer.layers_config) == 2


def test_memory_unit_layer_build_metadata() -> None:
    """Test building coordinate-based metadata from MemoryUnit YAML."""
    yaml_path = Path("lspattern/new_blocks/patch_layout/layers/memory.yml")
    unit_layer = load_unit_layer_from_yaml(yaml_path)

    # Build metadata for code distance 3 at global position (0, 0, 0)
    code_distance = 3
    global_pos = Coord3D(0, 0, 0)
    metadata = unit_layer.build_metadata(code_distance, global_pos)

    # Check we have 2 z-layers (z=0 and z=1)
    assert 0 in metadata.coords_by_z
    assert 1 in metadata.coords_by_z

    # Check that both layers have coordinates
    assert len(metadata.coords_by_z[0]) > 0
    assert len(metadata.coords_by_z[1]) > 0

    # Check coord2role has entries
    assert len(metadata.coord2role) > 0

    # Verify roles are assigned correctly
    data_coords_z0 = {c for c in metadata.coords_by_z[0] if metadata.coord2role[c] == NodeRole.DATA}
    z_ancilla_coords = {c for c in metadata.coords_by_z[0] if metadata.coord2role[c] == NodeRole.ANCILLA_Z}
    x_ancilla_coords = {c for c in metadata.coords_by_z[1] if metadata.coord2role[c] == NodeRole.ANCILLA_X}

    # For d=3, we should have some data and ancilla qubits
    assert len(data_coords_z0) > 0
    assert len(z_ancilla_coords) > 0
    assert len(x_ancilla_coords) > 0

    # Check spatial edges exist (ancilla -> data connections)
    assert len(metadata.spatial_edges) > 0

    # Check temporal edges exist (data qubit vertical connections)
    assert len(metadata.temporal_edges) > 0

    # Check schedule has entries for both layers
    assert len(metadata.coord_schedule.schedule) > 0
    assert 0 in metadata.coord_schedule.schedule
    assert 1 in metadata.coord_schedule.schedule

    # Check flow has entries
    assert len(metadata.coord_flow.flow) > 0

    # Check parity has entries
    assert len(metadata.coord_parity.checks) > 0


def test_spatial_edges_diagonal_connections() -> None:
    """Test that spatial edges connect ancillas to 4 diagonal data qubits."""
    yaml_path = Path("lspattern/new_blocks/patch_layout/layers/memory.yml")
    unit_layer = load_unit_layer_from_yaml(yaml_path)

    code_distance = 3
    global_pos = Coord3D(0, 0, 0)
    metadata = unit_layer.build_metadata(code_distance, global_pos)

    # Check that each ancilla connects to its diagonal neighbors
    # Get an ancilla coordinate and verify it has edges to data qubits
    ancilla_coords = {
        c for c in metadata.coord2role if metadata.coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}
    }

    for ancilla in ancilla_coords:
        # Find edges from this ancilla
        edges_from_ancilla = [edge for edge in metadata.spatial_edges if edge[0] == ancilla]

        # Each ancilla should connect to some data qubits (up to 4)
        # Some boundary ancillas may have fewer connections
        assert len(edges_from_ancilla) <= 4

        # Verify all connections are to data qubits
        for ancilla_coord, data_coord in edges_from_ancilla:
            assert metadata.coord2role[data_coord] == NodeRole.DATA

            # Verify diagonal connection: (±1, ±1, same z)
            dx = data_coord.x - ancilla_coord.x
            dy = data_coord.y - ancilla_coord.y
            dz = data_coord.z - ancilla_coord.z

            assert abs(dx) == 1
            assert abs(dy) == 1
            assert dz == 0


def test_temporal_edges_vertical_connections() -> None:
    """Test that temporal edges connect data qubits vertically."""
    yaml_path = Path("lspattern/new_blocks/patch_layout/layers/memory.yml")
    unit_layer = load_unit_layer_from_yaml(yaml_path)

    code_distance = 3
    global_pos = Coord3D(0, 0, 0)
    metadata = unit_layer.build_metadata(code_distance, global_pos)

    # Check that temporal edges connect data qubits at same (x, y) across z-layers
    for coord_z0, coord_z1 in metadata.temporal_edges:
        # Both should be data qubits
        assert metadata.coord2role[coord_z0] == NodeRole.DATA
        assert metadata.coord2role[coord_z1] == NodeRole.DATA

        # Same (x, y) position
        assert coord_z0.x == coord_z1.x
        assert coord_z0.y == coord_z1.y

        # Different z (should be consecutive layers)
        assert coord_z1.z - coord_z0.z == 1
