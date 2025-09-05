"""Memory blocks for cube-based patterns."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import BaseGraphState, GraphState

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.geom.rhg_parity import is_ancilla_x, is_ancilla_z, is_data

if TYPE_CHECKING:
    from lspattern.canvas import RHGCanvas


class Memory(RHGBlock):
    """Extend a logical patch upward by `rounds` time-slices on the RHG lattice.

    Behavior
    --------
    * For each new z-slice, create nodes only at allowed parities.
    * MBQC inputs = first slice's DATA nodes; outputs = last slice's DATA nodes (keeps (x, y) order).
    * Measurement bases:
        - Ancilla X-parity  -> X basis (XY plane, angle 0).
        - Ancilla Z-parity  -> Z-equivalent (XY plane, angle 0) here as a placeholder.
        - DATA except the last slice -> X basis (to satisfy canonical form).
    * Edges:
        - In-plane 4-neighbor.
        - Vertical edges between the same (x, y) across adjacent slices when allowed.
    * Scheduling:
        - Each layer contributes two measurement time bins:
          (2*t) for ancillas, (2*t+1) for DATA (except the last DATA layer).
    * Parity checks:
        - Intra-block 2-body checks along time (ancilla to next ancilla two z apart).
        - Lower/upper closures are provided via parity directives and seams to the canvas.
    """

    def __init__(self, logical: int, rounds: int) -> None:
        self.logical = logical
        self.rounds = rounds

    def emit(self, canvas: RHGCanvas) -> BlockDelta:  # noqa: C901, PLR0912, PLR0914, PLR0915
        """Emit the memory block to the canvas.

        Raises
        ------
        ValueError
            If boundary conditions are invalid.
        """
        lidx = self.logical
        boundary = canvas.logical_registry.require_boundary(lidx)

        # Derive the footprint (x, y) and base z from the current logical boundary.
        xs: list[int] = []
        ys: list[int] = []
        zs: list[int] = []
        for (x, y, z), nid in canvas.coord_to_node.items():
            if nid in boundary:
                xs.append(x)
                ys.append(y)
                zs.append(z)
        if not xs:
            msg = "Memory.emit: could not find coordinates for boundary nodes."
            raise ValueError(msg)

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z0 = max(zs)  # continue upward from the last data layer

        g: BaseGraphState = GraphState()
        node_at_layer: dict[int, dict[tuple[int, int], int]] = {}
        node_coords: dict[int, tuple[int, int, int]] = {}

        x_parity_check_groups: list[set[int]] = []
        z_parity_check_groups: list[set[int]] = []

        # (t_local, LOCAL-node-set) pairs for scheduling; t_local starts at 0 in this block.
        schedule_tuples: list[tuple[int, set[int]]] = []

        # Local X-flow accumulator (LOCAL ids).
        f: dict[int, set[int]] = {}

        # Build layers z0 . z0 + rounds with allowed parities only.
        for t in range(2 * self.rounds + 1):
            z = z0 + t
            layer_map: dict[tuple[int, int], int] = {}
            anc_group: set[int] = set()
            data_group: set[int] = set()

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    if is_data(x, y, z):
                        n = g.add_physical_node()
                        if t != 2 * self.rounds:
                            # All DATA except the last slice get a measurement basis.
                            g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                            data_group.add(n)
                        layer_map[x, y] = n
                        node_coords[n] = (x, y, z)
                    elif is_ancilla_x(x, y, z) or is_ancilla_z(x, y, z):
                        if t != 2 * self.rounds:  # no ancillas on the very last (pure DATA) slice
                            n = g.add_physical_node()
                            g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                            layer_map[x, y] = n
                            node_coords[n] = (x, y, z)
                            anc_group.add(n)

            node_at_layer[t] = layer_map
            if anc_group:
                schedule_tuples.append((2 * t, anc_group))
            if data_group:
                schedule_tuples.append((2 * t + 1, data_group))

            # In-plane edges.
            for (x, y), u in layer_map.items():
                for dx, dy in [(1, 0), (0, 1)]:
                    v = layer_map.get((x + dx, y + dy))
                    if v is not None:
                        g.add_physical_edge(u, v)

            # Vertical edges to previous layer where both coords exist.
            if t > 0:
                prev = node_at_layer[t - 1]
                for xy, u in layer_map.items():
                    v = prev.get(xy)
                    if v is not None:
                        g.add_physical_edge(u, v)
                        # Minimal X-flow exemplar: correction from previous to current.
                        f[v] = {u}

        # MBQC inputs/outputs on DATA nodes only, aligned by (x, y).
        q_indices: dict[tuple[int, int], int] = {}
        if self.rounds >= 1:
            first_map = node_at_layer[0]
            prev_qmap = canvas.logical_registry.boundary_qidx.get(lidx, {})
            if not prev_qmap:
                msg = f"Memory.emit: boundary_qidx is missing for logical {lidx}"
                raise ValueError(msg)
            inv_coord = {nid: coord for coord, nid in canvas.coord_to_node.items()}
            prev_xy_order: list[tuple[int, int]] = []
            for nid, _q in sorted(prev_qmap.items(), key=operator.itemgetter(1)):
                x, y, _ = inv_coord[nid]
                prev_xy_order.append((x, y))
            for xy in prev_xy_order:
                q_indices[xy] = g.register_input(first_map[xy])
            # Outputs on the last slice with the same (x, y) keys.
            last_map = node_at_layer[2 * self.rounds]
            for xy in prev_xy_order:
                if xy in last_map:
                    g.register_output(last_map[xy], q_indices[xy])

        # Determine the first ancilla layer (X/Z) to stitch with previous block, if any.
        first_layer = node_at_layer[0]
        first_x_local = {(X, Y): n for (X, Y), n in first_layer.items() if is_ancilla_x(X, Y, z0)}
        first_z_local = {(X, Y): n for (X, Y), n in first_layer.items() if is_ancilla_z(X, Y, z0)}
        if not first_x_local:
            second_layer = node_at_layer[1]
            first_x_local = {(X, Y): n for (X, Y), n in second_layer.items() if is_ancilla_x(X, Y, z0 + 1)}
        if not first_z_local:
            second_layer = node_at_layer[1]
            first_z_local = {(X, Y): n for (X, Y), n in second_layer.items() if is_ancilla_z(X, Y, z0 + 1)}

        # Determine the last ancilla layer to expose as seam_last_*.
        last_anc_t = 2 * self.rounds - 1
        last_layer = node_at_layer[last_anc_t]
        seam_last_x = {(X, Y): n for (X, Y), n in last_layer.items() if is_ancilla_x(X, Y, z0 + last_anc_t)}
        seam_last_z = {(X, Y): n for (X, Y), n in last_layer.items() if is_ancilla_z(X, Y, z0 + last_anc_t)}
        if not seam_last_x:
            last_anc_t = 2 * self.rounds - 2
            last_layer = node_at_layer[last_anc_t]
            seam_last_x = {(X, Y): n for (X, Y), n in last_layer.items() if is_ancilla_x(X, Y, z0 + last_anc_t)}
        if not seam_last_z:
            last_anc_t = 2 * self.rounds - 2
            last_layer = node_at_layer[last_anc_t]
            seam_last_z = {(X, Y): n for (X, Y), n in last_layer.items() if is_ancilla_z(X, Y, z0 + last_anc_t)}

        # Build parity directives that pair previous GLOBAL centers with current LOCAL nodes.
        last_x = canvas.parity_layers.get_last(lidx, "X")
        last_z = canvas.parity_layers.get_last(lidx, "Z")

        parity_x: list[tuple[int, list[int]]] = []
        parity_z: list[tuple[int, list[int]]] = []
        if last_x:
            for xy, n_local in first_x_local.items():
                if xy in last_x.by_xy:
                    # Seam pair: (prev_global_center, [curr_local_node])
                    parity_x.append((last_x.by_xy[xy], [n_local]))
        if last_z:
            for xy, n_local in first_z_local.items():
                if xy in last_z.by_xy:
                    parity_z.append((last_z.by_xy[xy], [n_local]))

        # Lower boundary singleton checks when there was no previous X-layer.
        if not last_x:
            x_parity_check_groups.extend({n_local} for n_local in first_x_local.values())

        # Intra-block 2-body ancilla parity checks along z (separated by 2).
        coord2node = {(x, y, z): u for u, (x, y, z) in node_coords.items()}
        for u, (x, y, z) in node_coords.items():
            next_ancilla = coord2node.get((x, y, z + 2))
            if is_ancilla_x(x, y, z):
                if next_ancilla:
                    x_parity_check_groups.append({u, next_ancilla})
            elif is_ancilla_z(x, y, z) and next_ancilla:
                z_parity_check_groups.append({u, next_ancilla})

        # Logical boundary becomes the last slice's DATA nodes.
        out_boundary: set[int] = set()
        if self.rounds >= 1:
            last_map = node_at_layer[2 * self.rounds]
            for (x, y), n in last_map.items():
                if is_data(x, y, z0 + self.rounds):
                    out_boundary.add(n)

        # in_ports/out_ports for canvas bookkeeping (DATA only).
        in_ports: dict[int, set[int]] = {}
        if self.rounds >= 1:
            # NOTE: Keep the original logic intact.
            first_map_for_ports = node_at_layer[z0]
            in_ports = {lidx: {first_map_for_ports[xy] for xy in first_map_for_ports if is_data(xy[0], xy[1], z0 + 1)}}

        return BlockDelta(
            local_graph=g,
            in_ports=in_ports,
            out_ports={lidx: out_boundary} if out_boundary else {},
            node_coords=node_coords,
            x_checks=x_parity_check_groups,
            z_checks=z_parity_check_groups,
            schedule_tuples=schedule_tuples,
            flow_local=f,
            parity_x_prev_global_curr_local=parity_x,
            parity_z_prev_global_curr_local=parity_z,
            seam_last_x=seam_last_x,
            seam_last_z=seam_last_z,
        )
