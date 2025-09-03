from __future__ import annotations

from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from graphix_zx.graphstate import BaseGraphState, GraphState

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.geom.rhg_parity import is_data

if TYPE_CHECKING:
    from lspattern.canvas import RHGCanvas


class InitPlus(RHGBlock):
    """Initialize a logical patch in |+> on the current top z-layer.

    - Places DATA nodes only (no ancillas) according to RHG parity at z_top.
    - Registers each DATA node as both input and output with matching q_index.
    - Exposes the DATA set as the logical boundary for this logical index.
    """

    def __init__(self, logical: int, dx: int, dy: int, origin: Optional[Tuple[int, int]] = None) -> None:
        self.logical = logical
        self.dx = dx
        self.dy = dy
        self.origin = origin

    def emit(self, canvas: "RHGCanvas") -> BlockDelta:
        # Allocate or reserve a rectangle for this logical.
        if self.origin is None:
            x0, y0 = canvas.tiler.alloc(self.logical, dx=self.dx, dy=self.dy)
        else:
            x0, y0 = self.origin
            # Reserve if possible to keep tiler state coherent.
            try:
                canvas.tiler.reserve(self.logical, x0=x0, y0=y0, dx=self.dx, dy=self.dy)
            except Exception:
                pass

        z = canvas.z_top

        g: BaseGraphState = GraphState()
        node_at: Dict[Tuple[int, int], int] = {}
        node_coords: Dict[int, Tuple[int, int, int]] = {}

        # Create DATA nodes on a (2*dx-1) x (2*dy-1) grid footprint.
        for ix in range(2 * self.dx - 1):
            for iy in range(2 * self.dy - 1):
                X, Y = x0 + ix, y0 + iy
                if not is_data(X, Y, z):
                    continue
                n = g.add_physical_node()
                node_at[(X, Y)] = n
                node_coords[n] = (X, Y, z)

        # Deterministic q_index: sort by (x, y), then mirror to outputs.
        data_coords = sorted(node_at.keys())
        q_indices: Dict[Tuple[int, int], int] = {}
        for xy in data_coords:
            q_indices[xy] = g.register_input(node_at[xy])
        for xy in data_coords:
            g.register_output(node_at[xy], q_indices[xy])

        out_nodes: Set[int] = {node_at[xy] for xy in data_coords}
        out_qmap_local: Dict[int, int] = {node_at[xy]: q_indices[xy] for xy in data_coords}

        return BlockDelta(
            local_graph=g,
            in_ports={},
            out_ports={self.logical: out_nodes},
            out_qmap={self.logical: out_qmap_local},
            node_coords=node_coords,
            x_checks=[],
            z_checks=[],
            schedule_tuples=[],
            flow_local={},
        )
