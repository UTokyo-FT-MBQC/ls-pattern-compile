"""The base definition for RHG blocks"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphqomb.graphstate import GraphState


class RHGBlock(ABC):
    @abstractmethod
    def materialize(self, graph: GraphState) -> tuple[GraphState, dict[tuple[int, int, int], int]]:
        """Materialize the block into the given graph.

        Parameters
        ----------
        graph : GraphState
            The graph to materialize the block into.

        Returns
        -------
        tuple[GraphState, dict[tuple[int, int, int], int]]
            A tuple containing the updated graph and a mapping from local coordinates to node IDs.
        """
        ...
