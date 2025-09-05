from __future__ import annotations

from typing import TYPE_CHECKING

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import BaseGraphState, GraphState

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.geom.rhg_parity import is_data

if TYPE_CHECKING:
    from lspattern.canvas import RHGCanvas


class _MeasureBase(RHGBlock):
    """MBQC measurement block on the latest DATA layer (RHG parity-aware).

    Behavior
    --------
    - Determine the latest DATA layer footprint from the canvas logical boundary.
    - Create readout nodes only on DATA sites at that z-layer.
    - Register each readout node as both MBQC input and output (same q_index).
    - `out_ports` is empty: this block consumes the logical boundary.
    - Provide X-cap parity directives that close the top with the previous X layer.
    """

    def __init__(self, logical: int, basis: str) -> None:
        self.logical = logical
        self.basis = (
            PlannerMeasBasis(Plane.XY, 0.0)
            if basis == "X"
            else PlannerMeasBasis(Plane.ZX, 0.0)
        )

    def emit(self, canvas: RHGCanvas) -> BlockDelta:
        lidx = self.logical
        boundary = canvas.logical_registry.require_boundary(lidx)

        # Recover the footprint from current boundary coordinates.
        xs: list[int] = []
        ys: list[int] = []
        zs: list[int] = []
        for (x, y, z), nid in canvas.coord_to_node.items():
            if nid in boundary:
                xs.append(x)
                ys.append(y)
                zs.append(z)
        if not xs:
            raise ValueError(
                "Measure.emit: could not find coordinates for boundary nodes."
            )

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z0 = max(zs)  # measure on the latest DATA layer

        g: BaseGraphState = GraphState()
        layer_map: dict[tuple[int, int], int] = {}
        node_coords: dict[int, tuple[int, int, int]] = {}

        # Create DATA readout nodes at z0 and assign the measurement basis.
        for X in range(x_min, x_max + 1):
            for Y in range(y_min, y_max + 1):
                if is_data(X, Y, z0):
                    n = g.add_physical_node()
                    g.assign_meas_basis(n, self.basis)
                    layer_map[X, Y] = n
                    node_coords[n] = (X, Y, z0)

        # Preserve q_index order using the previous boundary's q_map.
        prev_qmap = canvas.logical_registry.boundary_qidx.get(lidx, {})
        if not prev_qmap:
            raise ValueError(
                f"Measure.emit: boundary_qidx is missing for logical {lidx}"
            )

        inv_coord = {nid: coord for coord, nid in canvas.coord_to_node.items()}
        prev_xy_order: list[tuple[int, int]] = []
        for nid, _ in sorted(prev_qmap.items(), key=lambda kv: kv[1]):
            x, y, _ = inv_coord[nid]
            prev_xy_order.append((x, y))

        for xy in prev_xy_order:
            q = g.register_input(layer_map[xy])
            g.register_output(layer_map[xy], q)

        # Consume the logical: in_ports populated, out_ports empty.
        in_port_nodes: set[int] = set(g.physical_nodes)

        # Build X-cap parity directives using the last X layer's centers.
        last_x = canvas.parity_layers.get_last(lidx, "X")
        caps: list[tuple[int, list[int]]] = []
        if last_x:
            for (xc, yc), center_global in last_x.by_xy.items():
                locals4: list[int] = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nid = layer_map.get((xc + dx, yc + dy))
                    if nid is not None:
                        locals4.append(nid)
                if locals4:
                    caps.append((center_global, locals4))

        return BlockDelta(
            local_graph=g,
            in_ports={lidx: in_port_nodes},
            out_ports={},
            node_coords=node_coords,
            x_checks=[],
            z_checks=[],
            schedule_tuples=[(0, in_port_nodes)],  # all readouts measured together
            flow_local={},
            parity_x_prev_global_curr_local=caps,
        )


class MeasureX(_MeasureBase):
    """Measure a logical block in the X basis."""

    def __init__(self, logical: int) -> None:
        super().__init__(logical, "X")
