from __future__ import annotations

from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from graphix_zx.graphstate import BaseGraphState, compose_sequentially

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.geom.rhg_parity import is_data

if TYPE_CHECKING:
    from lspattern.canvas import RHGCanvas


class InitPlus(RHGBlock):
    """Initialization block that prepares a |+> logical patch on the RHG canvas.

    Behavior
    --------
    * Single z-layer (no duplication).
    * Create nodes only on DATA sites (per RHG parity).
    * Register all DATA nodes as both MBQC inputs and outputs with the same q_index.
    * No ancilla nodes are created here; parity checks remain empty.
    """

    def __init__(self, logical: int, dx: int, dy: int, origin: Optional[Tuple[int, int]] = None) -> None:
        self.logical = logical
        self.dx = dx
        self.dy = dy
        self.origin = origin

    def emit(self, canvas: "RHGCanvas") -> BlockDelta:
        """Build a one-layer DATA-only patch and expose its DATA nodes as MBQC I/O."""
        if GraphState is None:  # pragma: no cover - defensive guard
            raise RuntimeError("graphix_zx is required to build GraphState for InitPlus.")

        # Placement: allocate or reserve a rectangle on the canvas tiler.
        if self.origin is None:
            x0, y0 = canvas.tiler.alloc(self.logical, dx=self.dx, dy=self.dy)
        else:
            x0, y0 = self.origin
            # Best-effort reservation; ignore collision if the caller already placed it.
            try:
                canvas.tiler.reserve(self.logical, x0=x0, y0=y0, dx=self.dx, dy=self.dy)
            except Exception:
                pass

        z = canvas.z_top

        g: BaseGraphState = GraphState()
        node_at: Dict[Tuple[int, int], int] = {}
        node_coords: Dict[int, Tuple[int, int, int]] = {}

        # Create DATA nodes on this layer.
        for ix in range(2 * self.dx - 1):
            for iy in range(2 * self.dy - 1):
                X, Y = x0 + ix, y0 + iy
                if not is_data(X, Y, z):
                    continue
                n = g.add_physical_node()
                node_at[(X, Y)] = n
                node_coords[n] = (X, Y, z)

        # MBQC: register all DATA nodes as both inputs and outputs with identical q_index.
        data_coords = sorted(node_at.keys())
        q_indices: Dict[Tuple[int, int], int] = {}
        for xy in data_coords:
            q_indices[xy] = g.register_input(node_at[xy])
        for xy in data_coords:
            g.register_output(node_at[xy], q_indices[xy])

        # Logical boundary: DATA nodes of this layer.
        out_set: Set[int] = {node_at[xy] for xy in data_coords}
        out_qmap_l: Dict[int, int] = {node_at[xy]: q_indices[xy] for xy in data_coords}

        return BlockDelta(
            local_graph=g,
            in_ports={},  # MBQC ports are carried within GraphState I/O
            out_ports={self.logical: out_set},
            out_qmap={self.logical: out_qmap_l},
            node_coords=node_coords,
            x_checks=[],
            z_checks=[],
            schedule_tuples=[],  # no measurements in initialization
            flow_local={},
        )
