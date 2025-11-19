"""The base definition for RHG blocks"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lspattern.new_blocks.mytype import Coord3D


@dataclass
class Node:
    """Node of space-time diagram."""

    _global_pos: Coord3D
    _cout_ports: set[Coord3D] = field(default_factory=set)

    @property
    def global_pos(self) -> Coord3D:
        """Return the global origin coordinate of the cube."""
        return self._global_pos

    @property
    def cout_ports(self) -> set[Coord3D]:
        """Return the set of 3D coordinates used as classical output ports.

        Notes
        -----
        Currently, there is only one cout group per block
        """
        return self._cout_ports


@dataclass
class Edge:
    """Concrete implementation of an RHG pipe block."""

    _global_edge: tuple[Coord3D, Coord3D]
    _cout_ports: set[Coord3D] = field(default_factory=set)

    @property
    def global_edge(self) -> tuple[Coord3D, Coord3D]:
        """Return the global edge coordinates of the cube."""
        return self._global_edge

    @property
    def cout_ports(self) -> set[Coord3D]:
        """Return the set of 3D coordinates used as classical output ports."""
        return self._cout_ports
