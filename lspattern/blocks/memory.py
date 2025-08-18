
from __future__ import annotations

from typing import Dict, Set, Tuple, Optional, List

from graphix_zx.graphstate import GraphState
from graphix_zx.common import Plane, PlannerMeasBasis

from .base import BlockDelta, RHGBlock, GraphStateLike
from ..geom.rhg_parity import is_allowed, is_data, is_ancilla_x, is_ancilla_z


class Memory(RHGBlock):
    """Extend the logical patch upward by `rounds` time-slices on RHG lattice.

    - Each new z-slice creates nodes *only at allowed parities*.
    - MBQC inputs = first slice's DATA nodes; outputs = last slice's DATA nodes (same (x,y) order).
    - All non-output nodes are assigned a measurement basis:
        * ancilla X-parity -> 'X'
        * ancilla Z-parity -> 'Z'
        * data (but not output slice) -> 'X' (teleport-like default to satisfy canonical form)
    - Edges: 4-neighbor in-plane + vertical between same (x,y) on adjacent slices when both allowed.
    """

    def __init__(self, logical: int, rounds: int):
        self.logical = logical
        self.rounds = rounds

    def emit(self, canvas) -> BlockDelta:

        lidx = self.logical
        boundary = canvas.logical_registry.require_boundary(lidx)

        # Recover bounding box footprint from existing boundary coords (use min..max ranges)
        xs, ys, zs = [], [], []
        for (x, y, z), nid in canvas.coord_to_node.items():
            if nid in boundary:
                xs.append(x); ys.append(y); zs.append(z)
        if not xs:
            raise ValueError("Memory.emit: could not find coordinates for boundary nodes.")

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z0 = max(zs)  # continue upward

        g: GraphStateLike = GraphState()
        node_at_layer: Dict[int, Dict[Tuple[int,int], int]] = {}
        node_coords: Dict[int, Tuple[int,int,int]] = {}
        
        x_parity_check_groups: List[Set[int]] = []
        z_parity_check_groups: List[Set[int]] = []
        
        grouping: list[Set[int]] = []

        # Build layers z0 .. z0+rounds with allowed parities only
        f = dict()
        for t in range(2 * self.rounds + 1):
            z = z0 + t
            layer_map: Dict[Tuple[int,int], int] = {}
            group1: Set[int] = set()
            group2: Set[int] = set()
            for X in range(x_min, x_max + 1):
                for Y in range(y_min, y_max + 1):
                    if is_data(X, Y, z):
                        n = g.add_physical_node()
                        if t != 2 * self.rounds:
                            g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                            group2.add(n)  # last layer's DATA nodes
                        layer_map[(X, Y)] = n
                        node_coords[n] = (X, Y, z)
                    elif is_ancilla_x(X, Y, z) or is_ancilla_z(X, Y, z):
                        if t != 2 * self.rounds:  # no ancillas on last layer
                            n = g.add_physical_node()
                            g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                            layer_map[(X, Y)] = n
                            node_coords[n] = (X, Y, z)
                            group1.add(n)
                    
            node_at_layer[t] = layer_map
            if group1:
                grouping.append(group1)
            if group2:
                grouping.append(group2)

            # in-plane edges
            for (X, Y), u in layer_map.items():
                for dX, dY in [(1,0),(0,1)]:
                    v = layer_map.get((X + dX, Y + dY))
                    if v is not None:
                        g.add_physical_edge(u, v)

            # vertical edges to previous layer where both coords allowed
            if t > 0:
                prev = node_at_layer[t - 1]
                for xy, u in layer_map.items():
                    v = prev.get(xy)
                    if v is not None:
                        g.add_physical_edge(u, v)
                        f[v] = {u}

        # MBQC inputs/outputs on DATA nodes only, aligned by (x,y)
        q_indices: Dict[Tuple[int,int], int] = {}
        if self.rounds >= 1:
            first_map = node_at_layer[0]
            prev_qmap = canvas.logical_registry.boundary_qidx.get(lidx, {})
            if not prev_qmap:
                raise ValueError("Memory.emit: boundary_qidx is missing for logical {}".format(lidx))
            inv_coord = { nid: coord for coord, nid in canvas.coord_to_node.items() }
            prev_xy_order: List[Tuple[int,int]] = []
            for nid, q in sorted(prev_qmap.items(), key=lambda kv: kv[1]):
                x, y, _ = inv_coord[nid]
                prev_xy_order.append((x, y))
            for xy in prev_xy_order:
                q_indices[xy] = g.register_input(first_map[xy])
            # outputs on last slice with same (x,y) keys (must be present; if missing, skip)
            last_map = node_at_layer[2 * self.rounds]
            for xy in prev_xy_order:
                if xy in last_map:
                    g.register_output(last_map[xy], q_indices[xy])
                    
        first_layer = node_at_layer[0]
        first_x_local = { (X,Y): n for (X,Y), n in first_layer.items() if is_ancilla_x(X,Y,z0) }
        first_z_local = { (X,Y): n for (X,Y), n in first_layer.items() if is_ancilla_z(X,Y,z0) }
        if not first_x_local:
            second_layer = node_at_layer[1]
            first_x_local = { (X,Y): n for (X,Y), n in second_layer.items() if is_ancilla_x(X,Y,z0+1) }
        if not first_z_local:
            second_layer = node_at_layer[1]
            first_z_local = { (X,Y): n for (X,Y), n in second_layer.items() if is_ancilla_z(X,Y,z0+1) }
        

        last_anc_t = 2*self.rounds-1
        last_layer = node_at_layer[last_anc_t]
        seam_last_x  = { (X,Y): n for (X,Y), n in last_layer.items() if is_ancilla_x(X,Y,z0+last_anc_t) }
        seam_last_z  = { (X,Y): n for (X,Y), n in last_layer.items() if is_ancilla_z(X,Y,z0+last_anc_t) }
        if not seam_last_x:
            last_anc_t = 2*self.rounds-2
            last_layer = node_at_layer[last_anc_t]
            seam_last_x  = { (X,Y): n for (X,Y), n in last_layer.items() if is_ancilla_x(X,Y,z0+last_anc_t) }
        if not seam_last_z:
            last_anc_t = 2*self.rounds-2
            last_layer = node_at_layer[last_anc_t]
            seam_last_z  = { (X,Y): n for (X,Y), n in last_layer.items() if is_ancilla_z(X,Y,z0+last_anc_t) }
            
        last_x = canvas.parity_layers.get_last(lidx, 'X')  # ParityLast or None
        last_z = canvas.parity_layers.get_last(lidx, 'Z')
        
        parity_x = []
        parity_z = []
        if last_x:
            for xy, n_local in first_x_local.items():
                if xy in last_x.by_xy:
                    parity_x.append((last_x.by_xy[xy], [n_local]))  # ← 長さ1のリストで seam を表現
        if last_z:
            for xy, n_local in first_z_local.items():
                if xy in last_z.by_xy:
                    parity_z.append((last_z.by_xy[xy], [n_local]))  # ← 同上

        # 下端の単独 {u}（Z 側）。前回Zが無いときだけ追加（これは通常の parity_check として）
        if not last_z:
            for n_local in first_z_local.values():
                z_parity_check_groups.append({ n_local })
            
                    
        x_parity_check_groups: list[set[int]] = []
        z_parity_check_groups: list[set[int]] = []
        
        coord2node = {(x, y, z): u for u, (x, y, z) in node_coords.items()}

        for u, (x, y, z) in node_coords.items():
            next_ancilla = coord2node.get((x, y, z + 2), None)

            if is_ancilla_x(x, y, z):
                if next_ancilla:
                    x_parity_check_groups.append({u, next_ancilla})
            elif is_ancilla_z(x, y, z):
                if next_ancilla:
                    z_parity_check_groups.append({u, next_ancilla})

        # Logical boundary becomes the last slice's DATA nodes
        out_boundary: Set[int] = set()
        if self.rounds >= 1:
            last_map = node_at_layer[2 * self.rounds]
            for (X, Y), n in last_map.items():
                if is_data(X, Y, z0 + self.rounds):
                    out_boundary.add(n)

        # in_ports/out_ports for canvas bookkeeping (data qubits only)
        in_ports = {}
        if self.rounds >= 1:
            first_map = node_at_layer[z0]
            in_ports = { lidx: { first_map[xy] for xy in first_map if is_data(xy[0], xy[1], z0 + 1) } }

        return BlockDelta(
            local_graph=g,
            in_ports=in_ports,
            out_ports={ lidx: out_boundary } if out_boundary else {},
            node_coords=node_coords,
            x_checks= x_parity_check_groups,
            z_checks= z_parity_check_groups,
            measure_groups= grouping,
            flow_local= f,
            parity_x_prev_global_curr_local=parity_x,
            parity_z_prev_global_curr_local=parity_z,
            seam_last_x=seam_last_x,   seam_last_z=seam_last_z,
        )
