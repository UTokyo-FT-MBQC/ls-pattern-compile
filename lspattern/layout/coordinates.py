"""Data structures for patch coordinates and bounds.

This module provides immutable data structures for representing
patch regions in rotated surface code layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lspattern.mytype import Coord2D


@dataclass(frozen=True, slots=True)
class PatchBounds:
    """Bounding box for a patch region in 2D coordinates.

    Attributes
    ----------
    x_min : int
        Minimum x coordinate (inclusive).
    x_max : int
        Maximum x coordinate (inclusive).
    y_min : int
        Minimum y coordinate (inclusive).
    y_max : int
        Maximum y coordinate (inclusive).
    """

    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def width(self) -> int:
        """Width of the bounding box (inclusive)."""
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        """Height of the bounding box (inclusive)."""
        return self.y_max - self.y_min + 1

    @property
    def center_x(self) -> int:
        """Center x coordinate (nearest even value for data qubits)."""
        raw_center = (self.x_min + self.x_max) // 2
        return raw_center if raw_center % 2 == 0 else raw_center + 1

    @property
    def center_y(self) -> int:
        """Center y coordinate (nearest even value for data qubits)."""
        raw_center = (self.y_min + self.y_max) // 2
        return raw_center if raw_center % 2 == 0 else raw_center + 1


@dataclass(frozen=True, slots=True)
class PatchCoordinates:
    """Complete coordinate sets for a patch region.

    This immutable container holds the generated coordinates for
    data qubits, X ancillas, and Z ancillas.

    Attributes
    ----------
    data : frozenset[Coord2D]
        Coordinates of data qubits.
    ancilla_x : frozenset[Coord2D]
        Coordinates of X-type ancilla qubits.
    ancilla_z : frozenset[Coord2D]
        Coordinates of Z-type ancilla qubits.
    """

    data: frozenset[Coord2D]
    ancilla_x: frozenset[Coord2D]
    ancilla_z: frozenset[Coord2D]

    def to_mutable_sets(self) -> tuple[set[Coord2D], set[Coord2D], set[Coord2D]]:
        """Convert to mutable sets for backward compatibility.

        Returns
        -------
        tuple[set[Coord2D], set[Coord2D], set[Coord2D]]
            (data_coords, x_ancilla_coords, z_ancilla_coords)
        """
        return set(self.data), set(self.ancilla_x), set(self.ancilla_z)
