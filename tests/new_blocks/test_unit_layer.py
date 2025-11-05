
from lspattern.new_blocks.mytype import Coord3D, NodeRole
from lspattern.new_blocks.unit_layer import MemoryUnitLayer


def test_memory_unit_layer_basic():
    layer = MemoryUnitLayer(Coord3D(0, 0, 0))

    # Simple 3x3 data patch
    data2d = [(0, 0), (0, 2), (2, 0), (2, 2)]
    x2d = [(1, 1)]  # single X ancilla
    z2d = [(1, 1)]  # single Z ancilla

    metadata = layer.build_metadata(0, data2d, x2d, z2d)

    # Check we have 2 z-layers
    assert 0 in metadata.coords_by_z
    assert 1 in metadata.coords_by_z

    # Check roles
    assert metadata.coord2role[Coord3D(0, 0, 0)] == NodeRole.DATA
    assert metadata.coord2role[Coord3D(1, 1, 0)] == NodeRole.ANCILLA_Z
    assert metadata.coord2role[Coord3D(1, 1, 1)] == NodeRole.ANCILLA_X

    # Check schedule exists
    assert len(metadata.coord_schedule) > 0
