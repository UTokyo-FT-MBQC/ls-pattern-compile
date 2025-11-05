import pytest

from lspattern.new_blocks.coord_utils import CoordTransform
from lspattern.new_blocks.mytype import Coord2D, Coord3D


def test_shift_coords_3d():
    coords = {Coord3D(0, 0, 0), Coord3D(1, 1, 1)}
    shifted = CoordTransform.shift_coords_3d(coords, Coord3D(10, 20, 30))
    assert shifted == {Coord3D(10, 20, 30), Coord3D(11, 21, 31)}


def test_get_neighbors_2d():
    neighbors = CoordTransform.get_neighbors_2d(Coord2D(0, 0))
    assert neighbors == {
        Coord2D(1, 0),
        Coord2D(-1, 0),
        Coord2D(0, 1),
        Coord2D(0, -1),
    }


def test_get_neighbors_3d_spatial_only():
    neighbors = CoordTransform.get_neighbors_3d(
        Coord3D(0, 0, 0), spatial_only=True
    )
    assert len(neighbors) == 4  # only spatial (x, y)
    assert Coord3D(0, 0, 1) not in neighbors  # no temporal


def test_get_neighbors_3d_with_temporal():
    neighbors = CoordTransform.get_neighbors_3d(
        Coord3D(0, 0, 0), spatial_only=False
    )
    assert len(neighbors) == 6  # spatial + temporal
    assert Coord3D(0, 0, 1) in neighbors
    assert Coord3D(0, 0, -1) in neighbors
