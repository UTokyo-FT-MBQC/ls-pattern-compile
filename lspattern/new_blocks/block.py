"""The base definition for RHG blocks"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.new_blocks.accumulator import (
    CoordFlowAccumulator,
    CoordParityAccumulator,
    CoordScheduleAccumulator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lspattern.new_blocks.mytype import Coord2D, Coord3D
    from lspattern.new_blocks.unit_layer import UnitLayer


class RHGBlock(ABC):
    @property
    @abstractmethod
    def global_pos(self) -> Coord3D:
        """Get the global position of the block.

        Returns
        -------
        Coord3D
            The global (x, y, z) position of the block.
        """
        ...

    @property
    @abstractmethod
    def in_ports(self) -> set[Coord2D]:
        """Get the input ports of the block.

        Returns
        -------
        set[Coord2D]
            A set of input port coordinates.
        """
        ...

    @property
    @abstractmethod
    def out_ports(self) -> set[Coord2D]:
        """Get the output ports of the block.

        Returns
        -------
        set[Coord2D]
            A set of output port coordinates.
        """
        ...

    @property
    @abstractmethod
    def cout_ports(self) -> set[Coord3D]:
        """Get the classical output ports of the block.

        Returns
        -------
        set[Coord3D]
            A set of classical output port coordinates.
        """
        ...

    @property
    @abstractmethod
    def unit_layers(self) -> list[UnitLayer]:
        """Get the unit layers comprising the block.

        Returns
        -------
        list[UnitLayer]
            A list of unit layers in the block.
        """
        ...


@dataclass
class RHGCube(RHGBlock):
    """Concrete implementation of an RHG cube block."""

    _global_pos: Coord3D
    d: int
    coord2role: dict[Coord3D, str] = field(default_factory=dict)
    coord_schedule: CoordScheduleAccumulator = field(default_factory=CoordScheduleAccumulator)
    coord_flow: CoordFlowAccumulator = field(default_factory=CoordFlowAccumulator)
    coord_parity: CoordParityAccumulator = field(default_factory=CoordParityAccumulator)
    _in_ports: set[Coord2D] = field(default_factory=set)
    _out_ports: set[Coord2D] = field(default_factory=set)
    _cout_ports: set[Coord3D] = field(default_factory=set)

    @property
    def global_pos(self) -> Coord3D:
        """Return the global origin coordinate of the cube."""
        return self._global_pos

    @property
    def in_ports(self) -> set[Coord2D]:
        """Return the set of 2D coordinates used as input ports."""
        return self._in_ports

    @property
    def out_ports(self) -> set[Coord2D]:
        """Return the set of 2D coordinates used as output ports."""
        return self._out_ports

    @property
    def cout_ports(self) -> set[Coord3D]:
        """Return the set of 3D coordinates used as classical output ports."""
        return self._cout_ports

    @classmethod
    def from_unitlayers(
        cls,
        global_pos: Coord3D,
        unit_layers: Sequence[UnitLayer],
    ) -> RHGCube:
        """Populate coordinate metadata ahead of graph materialization.

        Parameters
        ----------
        global_pos : Coord3D
            The global (x, y, z) position of the block.
        unit_layers : collections.abc.Sequence[UnitLayer]
            The sequence of unit layers comprising the block.
        """
        raise NotImplementedError
