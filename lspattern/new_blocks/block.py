"""The base definition for RHG blocks"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphqomb.graphstate import GraphState

    from lspattern.new_blocks.mytype import Coord2D, Coord3D


class RHGBlock(ABC):
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
