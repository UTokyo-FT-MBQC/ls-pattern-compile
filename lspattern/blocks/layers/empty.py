"""Empty unit layer implementation.

Empty layers contain no nodes and serve as placeholders to advance z-coordinates
while maintaining temporal edge connections between surrounding layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.accumulator import FlowAccumulator, ParityAccumulator
from lspattern.blocks.unit_layer import LayerData, UnitLayer

if TYPE_CHECKING:
    from graphix_zx.graphstate import GraphState

    from lspattern.tiling.template import ScalableTemplate


class EmptyUnitLayer(UnitLayer):
    """Empty layer with no nodes.

    This layer consumes two z-coordinates (z and z+1) but does not place any
    nodes. It serves as a placeholder to skip measurement operations at specific
    layers while maintaining the overall structure and temporal connections.

    Temporal edges from the previous non-empty layer will connect directly to
    the next non-empty layer, skipping over this empty layer.
    """

    def build_layer(
        self,
        graph: GraphState,  # noqa: ARG002
        z_offset: int,  # noqa: ARG002
        template: ScalableTemplate,  # noqa: ARG002
    ) -> LayerData:
        """Build an empty layer (no nodes placed).

        Parameters
        ----------
        graph : GraphState
            The graph state (unchanged by this operation).
        z_offset : int
            Starting z-coordinate for this layer (consumed but unused).
        template : ScalableTemplate
            Template (unused by empty layer).

        Returns
        -------
        LayerData
            Empty layer data with no nodes, edges, or accumulators.
        """
        # Return completely empty layer data
        return LayerData(
            nodes_by_z={},
            node2coord={},
            coord2node={},
            node2role={},
            schedule=self._construct_schedule({}, {}),
            flow=FlowAccumulator(),
            parity=ParityAccumulator(),
        )
