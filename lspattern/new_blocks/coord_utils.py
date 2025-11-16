"""Coordinate utilities for building unit layer graphs.

This module provides helper functions for coordinate transformations and edge generation
in the new coordinate-based block architecture.
"""

from __future__ import annotations

from lspattern.new_blocks.mytype import Coord3D


def get_ancilla_neighbors(ancilla_coord: Coord3D) -> list[Coord3D]:
    """Get 4 diagonal neighbors of an ancilla qubit.

    In rotated surface code, each ancilla connects to 4 data qubits at diagonal positions:
    (±1, ±1) relative to the ancilla.

    Parameters
    ----------
    ancilla_coord : Coord3D
        The 3D coordinate of the ancilla qubit (x, y, z).

    Returns
    -------
    list[Coord3D]
        List of 4 diagonal neighbor coordinates in the order:
        [(x+1, y+1, z), (x+1, y-1, z), (x-1, y+1, z), (x-1, y-1, z)]
    """
    x, y, z = ancilla_coord
    return [
        Coord3D(x + 1, y + 1, z),
        Coord3D(x + 1, y - 1, z),
        Coord3D(x - 1, y + 1, z),
        Coord3D(x - 1, y - 1, z),
    ]


def build_spatial_edges(
    data_coords: set[Coord3D],
    ancilla_coords: set[Coord3D],
) -> set[tuple[Coord3D, Coord3D]]:
    """Build spatial edges from ancillas to their 4 diagonal data neighbors.

    Each ancilla qubit connects to up to 4 data qubits at diagonal positions.
    Edges to non-existent data coordinates are excluded.

    Parameters
    ----------
    data_coords : set[Coord3D]
        Set of data qubit coordinates.
    ancilla_coords : set[Coord3D]
        Set of ancilla qubit coordinates.

    Returns
    -------
    set[tuple[Coord3D, Coord3D]]
        Set of edges as (ancilla_coord, data_coord) tuples.
        Only edges where the data coordinate exists are included.
    """
    edges: set[tuple[Coord3D, Coord3D]] = set()

    for ancilla in ancilla_coords:
        neighbors = get_ancilla_neighbors(ancilla)
        for neighbor in neighbors:
            if neighbor in data_coords:
                edges.add((ancilla, neighbor))

    return edges


def build_temporal_edges(
    data_coords_z0: set[Coord3D],
    data_coords_z1: set[Coord3D],
) -> set[tuple[Coord3D, Coord3D]]:
    """Build temporal edges between two consecutive layers.

    Temporal edges connect data qubits at the same (x, y) position
    across two consecutive z-layers.

    Parameters
    ----------
    data_coords_z0 : set[Coord3D]
        Data qubit coordinates at layer z.
    data_coords_z1 : set[Coord3D]
        Data qubit coordinates at layer z+1.

    Returns
    -------
    set[tuple[Coord3D, Coord3D]]
        Set of temporal edges as (coord_z0, coord_z1) tuples.
        Only edges where both coordinates exist are included.
    """
    edges: set[tuple[Coord3D, Coord3D]] = set()

    # Create a map of (x, y) -> coordinate for faster lookup
    z1_map: dict[tuple[int, int], Coord3D] = {(coord.x, coord.y): coord for coord in data_coords_z1}

    for coord_z0 in data_coords_z0:
        coord_z1 = z1_map.get((coord_z0.x, coord_z0.y))
        if coord_z1 is not None:
            edges.add((coord_z0, coord_z1))

    return edges
