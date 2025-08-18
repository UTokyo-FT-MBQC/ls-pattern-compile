
from __future__ import annotations

from typing import Dict, Set, Tuple, Optional, List

try:
    from graphix_zx.graphstate import GraphState
except Exception:  # pragma: no cover
    GraphState = None  # type: ignore

from .base import BlockDelta, RHGBlock, GraphStateLike
from ..geom.rhg_parity import is_allowed, is_data, is_ancilla_x, is_ancilla_z


class InitPlus(RHGBlock):
    """Initialization block that prepares a |+> logical patch on RHG canvas.

    - Single z-layer (no duplication).
    - Build nodes only at *allowed* parities.
    - MBQC ports are *data* nodes only (all data nodes are inputs and outputs with same q_index).
    - Ancilla nodes are measured immediately (basis = X for ANCILLA_X, Z for ANCILLA_Z).
    """

    def __init__(self, logical: int, dx: int, dy: int, origin: Optional[Tuple[int, int]] = None):
        self.logical = logical
        self.dx = dx
        self.dy = dy
        self.origin = origin

    def emit(self, canvas) -> BlockDelta:  # canvas: RHGCanvas
        if GraphState is None:
            raise RuntimeError("graphix_zx is required to build GraphState for InitPlus.")

        # placement
        if self.origin is None:
            x0, y0 = canvas.tiler.alloc(self.logical, dx=self.dx, dy=self.dy)
        else:
            x0, y0 = self.origin
            try:
                canvas.tiler.reserve(self.logical, x0=x0, y0=y0, dx=self.dx, dy=self.dy)
            except Exception:
                pass

        z = canvas.z_top

        g: GraphStateLike = GraphState()
        node_at: Dict[Tuple[int, int], int] = {}
        node_coords: Dict[int, Tuple[int, int, int]] = {}

        # Create allowed nodes on this layer
        for ix in range(2*self.dx - 1):
            for iy in range(2*self.dy - 1):
                X, Y = x0 + ix, y0 + iy
                if not is_data(X, Y, z):
                    continue
                n = g.add_physical_node()
                node_at[(X, Y)] = n
                node_coords[n] = (X, Y, z)

        # DATA nodes become MBQC inputs/outputs (same nodes)
        data_coords = sorted([ (X,Y) for (X,Y) in node_at.keys() if is_data(X,Y,z) ])
        q_indices: Dict[Tuple[int,int], int] = {}
        for xy in data_coords:
            q_indices[xy] = g.register_input(node_at[xy])
        for xy in data_coords:
            g.register_output(node_at[xy], q_indices[xy])

        out_set: Set[int] = { node_at[xy] for xy in data_coords }  # logical boundary = data nodes
        out_qmap_l: Dict[int,int] = { node_at[xy]: q_indices[xy] for xy in data_coords }

        return BlockDelta(
            local_graph=g,
            in_ports={},                           # MBQC ports are in GraphState
            out_ports={ self.logical: out_set },   # boundary carries *data* nodes
            out_qmap={ self.logical: out_qmap_l },
            node_coords=node_coords,
            x_checks=[],
            z_checks=[],
            measure_groups=[ out_set ],  # MBQC inputs/outputs are the same nodes
            flow_local={},
        )
