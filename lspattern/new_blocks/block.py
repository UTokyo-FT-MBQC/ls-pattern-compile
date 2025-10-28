"""The base definition for RHG blocks"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphqomb.graphstate import GraphState

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

    @abstractmethod
    def materialize(self, graph: GraphState, node_map: Mapping[Coord3D, int]) -> tuple[GraphState, dict[Coord3D, int]]:
        """Materialize the block into the given graph.

        Parameters
        ----------
        graph : GraphState
            The graph to materialize the block into.
        node_map : dict[Coord3D, int]
            A mapping from local coordinates to node IDs.

        Returns
        -------
        tuple[GraphState, dict[Coord3D, int]]
            A tuple containing the updated graph and an updatedmapping from local coordinates to node IDs.
        """
        ...
