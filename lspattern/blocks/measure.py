
from __future__ import annotations

from typing import Dict, Set, Tuple, List

from graphix_zx.graphstate import GraphState
from graphix_zx.common import Plane, PlannerMeasBasis

from lspattern.blocks.base import BlockDelta, RHGBlock, GraphStateLike
from lspattern.geom.rhg_parity import is_allowed, is_data, is_ancilla_x, is_ancilla_z


class _MeasureBase(RHGBlock):
    """Multi-port MBQC measurement block with RHG parity.

    Behavior:
      - Determine the last data layer's (x,y) footprint from the canvas logical boundary.
      - On a new z-slice (z_in = z_last + 1), create nodes *only at allowed parities*.
      - Choose as many allowed positions as needed (>= data count) to serve as MBQC inputs.
        Prefer the same (x,y) when allowed; otherwise, assign remaining allowed sites in order.
      - For each input, create a readout node, connect input->readout, and assign measurement basis.
      - Register outputs on the *same input nodes* (q_index preserved) to satisfy canonical form.
      - `out_ports` is empty: this block consumes the logical boundary.
    """

    def __init__(self, logical: int, basis: str):
        self.logical = logical
        self.basis = PlannerMeasBasis(Plane.XY, 0.0) if basis == 'X' else PlannerMeasBasis(Plane.ZX, 0.0)

    def emit(self, canvas) -> BlockDelta:

        lidx = self.logical
        boundary = canvas.logical_registry.require_boundary(lidx)

        # Get boundary coords and footprint
        xs: List[int] = []
        ys: List[int] = []
        zs: List[int] = []
        for (x, y, z), nid in canvas.coord_to_node.items():
            if nid in boundary:
                xs.append(x); ys.append(y); zs.append(z)
        if not xs:
            raise ValueError("Measure.emit: could not find coordinates for boundary nodes.")

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z0 = max(zs)  # last data layer
        
        g: GraphStateLike = GraphState()
        node_at_layer: Dict[int, Dict[Tuple[int,int], int]] = {}
        node_coords: Dict[int, Tuple[int,int,int]] = {}
        
        layer_map: Dict[Tuple[int,int], int] = {}
        for X in range(x_min, x_max + 1):
            for Y in range(y_min, y_max + 1):
                if is_data(X, Y, z0):
                    n = g.add_physical_node()
                    g.assign_meas_basis(n, self.basis)
                    layer_map[(X, Y)] = n
                    node_coords[n] = (X, Y, z0)
        
        q_indices: Dict[Tuple[int,int], int] = {}
        prev_qmap = canvas.logical_registry.boundary_qidx.get(lidx, {})
        if not prev_qmap:
            raise ValueError("Memory.emit: boundary_qidx is missing for logical {}".format(lidx))
        inv_coord = { nid: coord for coord, nid in canvas.coord_to_node.items() }
        prev_xy_order: List[Tuple[int,int]] = []
        for nid, _ in sorted(prev_qmap.items(), key=lambda kv: kv[1]):
            x, y, _ = inv_coord[nid]
            prev_xy_order.append((x, y))
        for xy in prev_xy_order:
            q = g.register_input(layer_map[xy])
            g.register_output(layer_map[xy], q)

        # in_ports/out_ports for canvas bookkeeping
        in_port_nodes: Set[int] = g.physical_nodes
        # out_ports empty -> logical consumed in canvas._merge_delta
        
        last_x = canvas.parity_layers.get_last(lidx, 'X')
        caps = []
        if last_x:
            for (xc, yc), center_global in last_x.by_xy.items():
                locals4 = []
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nid = layer_map.get((xc+dx, yc+dy))   # 同じ z0 のローカルデータ
                    if nid is not None:
                        locals4.append(nid)
                if locals4:
                    caps.append((center_global, locals4))

        return BlockDelta(
            local_graph=g,
            in_ports={ lidx: in_port_nodes },
            out_ports={},
            node_coords=node_coords,
            x_checks=[],
            z_checks=[],
            schedule_tuples=[(0, in_port_nodes)],  # MBQC inputs are the same nodes
            flow_local={},
            parity_x_prev_global_curr_local=caps,
        )


class MeasureX(_MeasureBase):
    def __init__(self, logical: int):
        super().__init__(logical, 'X')


class MeasureZ(_MeasureBase):
    def __init__(self, logical: int):
        super().__init__(logical, 'Z')
