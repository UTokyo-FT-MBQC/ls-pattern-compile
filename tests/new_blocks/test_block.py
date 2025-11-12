from graphqomb.graphstate import GraphState

from lspattern.new_blocks.block import RHGCube
from lspattern.new_blocks.mytype import Coord3D


def test_rhg_cube_initialization() -> None:
    cube = RHGCube(_global_pos=Coord3D(0, 0, 0), d=3)
    assert len(cube.unit_layers) == 3  # d=3 means 3 unit layers
    assert cube.global_pos == Coord3D(0, 0, 0)


def test_rhg_cube_prepare() -> None:
    cube = RHGCube(_global_pos=Coord3D(0, 0, 0), d=2)

    data2d = [(0, 0), (0, 2), (2, 0), (2, 2)]
    x2d = [(1, 1)]
    z2d = [(1, 1)]

    cube.prepare(data2d, x2d, z2d)

    # Should have metadata for 2*d = 4 physical layers
    assert len(cube.coord2role) > 0
    assert len(cube.coord_schedule.schedule) > 0


def test_rhg_cube_materialize() -> None:
    cube = RHGCube(_global_pos=Coord3D(0, 0, 0), d=2)

    data2d = [(0, 0), (0, 2), (2, 0), (2, 2)]
    x2d = [(1, 1)]
    z2d = [(1, 1)]

    cube.prepare(data2d, x2d, z2d)

    graph = GraphState()
    node_map = {}

    graph, node_map = cube.materialize(graph, node_map)

    # Should have created nodes
    assert len(node_map) > 0
    assert len(graph.physical_nodes) > 0
