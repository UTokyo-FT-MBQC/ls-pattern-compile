"""The base definition for RHG unit layers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphqomb.graphstate import GraphState

    from lspattern.new_blocks.mytype import Coord3D


class UnitLayer(ABC):
    """Abstract base class for RHG unit layers (2 physical layers)."""

    @property
    @abstractmethod
    def global_pos(self) -> Coord3D:
        """Get the global position of the unit layer.

        Returns
        -------
        Coord3D
            The global (x, y, z) position of the unit layer.
        """
        ...

    @abstractmethod
    def materialize(self, graph: GraphState, node_map: Mapping[Coord3D, int]) -> tuple[GraphState, dict[Coord3D, int]]:
        """Materialize the unit layer into the given graph.

        Parameters
        ----------
        graph : GraphState
            The graph to materialize the unit layer into.

        Returns
        -------
        tuple[GraphState, dict[Coord3D, int]]
            A tuple containing the updated graph and an updated mapping from local coordinates to node IDs.
        """
        ...
