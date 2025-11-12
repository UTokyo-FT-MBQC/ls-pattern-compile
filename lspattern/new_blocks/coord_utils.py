"""Coordinate transformation utilities used by new block logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.new_blocks import mytype

if TYPE_CHECKING:
    from collections.abc import Mapping

Coord2D = mytype.Coord2D
Coord3D = mytype.Coord3D


class CoordTransform:
    """Stateless helpers for working with 2D and 3D coordinates."""

    @staticmethod
    def shift_coords_3d(
        coords: set[Coord3D],
        offset: Coord3D,
    ) -> set[Coord3D]:
        """Shift a set of 3D coordinates by ``offset``."""
        dx, dy, dz = offset
        return {Coord3D(x + dx, y + dy, z + dz) for x, y, z in coords}

    @staticmethod
    def neighbors_2d(coord: Coord2D) -> set[Coord2D]:
        """Return 4-connected (NESW) neighbors for a 2D coordinate."""
        x, y = coord
        return {
            Coord2D(x + 1, y + 1),  # top-right
            Coord2D(x - 1, y + 1),  # top-left
            Coord2D(x + 1, y - 1),  # bottom-right
            Coord2D(x - 1, y - 1),  # bottom-left
        }

    @staticmethod
    def neighbors_3d(
        coord: Coord3D,
        spatial_only: bool = False,
    ) -> set[Coord3D]:
        """Return 3D neighbors; include ±z unless ``spatial_only`` is true."""
        x, y, z = coord
        neighbors = {
            Coord3D(x + 1, y, z),
            Coord3D(x - 1, y, z),
            Coord3D(x, y + 1, z),
            Coord3D(x, y - 1, z),
        }
        if not spatial_only:
            neighbors.add(Coord3D(x, y, z + 1))
            neighbors.add(Coord3D(x, y, z - 1))
        return neighbors

    @staticmethod
    def extract_z_layer(coords: set[Coord3D], z: int) -> set[Coord2D]:
        """Extract 2D coordinates that lie on the given ``z`` layer."""
        return {Coord2D(x, y) for x, y, zc in coords if zc == z}

    @staticmethod
    def coords_to_node_ids(
        coords: set[Coord3D],
        coord2node: Mapping[Coord3D, int],
    ) -> set[int]:
        """Map 3D coordinates to node identifiers using ``coord2node``."""
        return {coord2node[c] for c in coords if c in coord2node}
