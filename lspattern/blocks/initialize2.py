from __future__ import annotations

from typing import Dict, List, Set, Tuple, Union

from graphix_zx.graphstate import BaseGraphState, GraphState
from graphix_zx.common import Plane, PlannerMeasBasis

from lspattern.blocks.base import RHGBlockSkeleton, RHGBlock
from lspattern.mytype import BlockKindstr
from lspattern.template.base import RotatedPlanarTemplate, ScalableTemplate
from mytype import *
from mytype import PhysCoordLocal2D
from lspattern.consts.consts import DIRECTIONS2D, DIRECTIONS3D
from lspattern.utils import __tuple_sum


class InitPlusSkeleton(RHGBlockSkeleton):
    name: str = __qualname__
    d: int
    kind: BlockKindstr


class InitPlus(RHGBlock):
    name: str = __qualname__

    def materialize(self):
        # Build 2d-rotated planar tiling across 2*d time steps plus final data slice.
        # - Even z: X-ancillas present (except final slice)
        # - Odd z:  Z-ancillas present (except final slice)
        # - Every z: DATA present; last slice DATA are outputs and not measured
        if self.template.kind != self.kind:
            raise ValueError("Template kind mismatch")

        tiling = self.template.to_tiling()
        data_indices = self.template.get_data_indices()

        g = GraphState()
        coord2node: dict[PhysCoordLocal3D, NodeIdLocal] = {}
        node2coord: dict[NodeIdLocal, PhysCoordGlobal3D] = {}
        node2role: dict[NodeIdLocal, str] = {}

        # Build nodes per layer z in [0, 2*d]
        max_t = 2 * self.d
        nodes_by_z: dict[int, dict[PhysCoordLocal2D, NodeIdLocal]] = {}
        for z in range(0, max_t + 1):
            timeslice: dict[PhysCoordLocal2D, NodeIdLocal] = {}

            # DATA on every slice
            for x, y in tiling["data"]:
                # 2d + 1でちょっと飛び出ている
                n = g.add_physical_node()
                if z != max_t:
                    g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                else:
                    g.register_output(n, data_indices[(x, y)])
                coord = (x, y, z)
                coord2node[coord] = n
                node2coord[n] = coord
                node2role[n] = "data"
                timeslice[(x, y)] = n

            # Ancillas except on the final slice
            if z != max_t:
                if z % 2 == 0:
                    # X ancillas on even layers
                    for x, y in tiling["X"]:
                        n = g.add_physical_node()
                        g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                        coord = (x, y, z)
                        coord2node[coord] = n
                        node2coord[n] = coord
                        node2role[n] = "ancilla_x"
                        timeslice[(x, y)] = n

                else:
                    # Z ancillas on odd layers
                    for x, y in tiling["Z"]:
                        n = g.add_physical_node()
                        g.assign_meas_basis(n, PlannerMeasBasis(Plane.XY, 0.0))
                        coord = (x, y, z)
                        coord2node[coord] = n
                        node2coord[n] = coord
                        node2role[n] = "ancilla_z"
                        timeslice[(x, y)] = n

            nodes_by_z[z] = timeslice

        # In-plane edges (diagonal neighbors per consts) within each layer
        for z, timeslice in nodes_by_z.items():
            for (x, y), src in timeslice.items():
                for dx, dy, dz in DIRECTIONS3D:
                    xy_tgt = (x + dx, y + dy)
                    tgt = timeslice.get(xy_tgt)
                    if tgt is not None and tgt > src:
                        g.add_physical_edge(src, tgt)

        # Vertical edges between same (x, y) across adjacent layers; also build flow
        flow_local: FlowLocal = {}
        for z in range(1, max_t + 1):
            cur = nodes_by_z[z]
            prev = nodes_by_z[z - 1]
            for xy, u in cur.items():
                v = prev.get(xy)
                if v is not None:
                    g.add_physical_edge(u, v)
                    # Minimal X-flow: correction from previous to current
                    flow_local[v] = {u}

        # Parity checks along time for ancillas (2-body separated by delta z = 2)
        x_checks: List[NodeSetLocal] = []
        z_checks: List[NodeSetLocal] = []
        # Fast lookup by coord
        for n, (x, y, z) in node2coord.items():
            if node2role[n] == "ancilla_x":
                nxt = coord2node.get((x, y, z + 2))
                if nxt is not None and node2role.get(nxt) == "ancilla_x":
                    x_checks.append({n, nxt})
            elif node2role[n] == "ancilla_z":
                nxt = coord2node.get((x, y, z + 2))
                if nxt is not None and node2role.get(nxt) == "ancilla_z":
                    z_checks.append({n, nxt})

        # Measurement schedule: (2*z) ancillas, (2*z+1) data except last slice
        schedule_local: ScheduleTuplesLocal = []
        for z, timeslice in nodes_by_z.items():
            anc_group: Set[int] = set()
            data_group: Set[int] = set()
            for n in timeslice.values():
                role = node2role[n]
                if role.startswith("ancilla"):
                    anc_group.add(n)
                elif role == "data":
                    if z != max_t:
                        data_group.add(n)
            if anc_group:
                schedule_local.append((2 * z, anc_group))
            if data_group:
                schedule_local.append((2 * z + 1, data_group))

        # Out ports are DATA nodes on the final slice
        out_ports: NodeSetLocal = set()
        for (x, y), n in nodes_by_z[max_t].items():
            if node2role.get(n) == "data":
                out_ports.add(n)

        # Bind to block fields
        self.graph_local = g
        self.node2coords = node2coord
        self.coords2node = {coord: nid for nid, coord in node2coord.items()}
        self.node2role = node2role
        self.in_ports = set()
        self.out_ports = out_ports
        self.cout_ports = []
        self.x_checks = x_checks
        self.z_checks = z_checks
        self.schedule_local = schedule_local
        self.flow_local = flow_local


if __name__ == "__main__":
    # Debug visualization referencing template/base visualization style.
    # Uses d=3 and kind="XZX" as requested.
    import matplotlib.pyplot as plt

    d = 3
    kind = ("X", "Z", "X")  # "XZX"

    template = RotatedPlanarTemplate(d=d, kind=kind)
    tiling = template.to_tiling()
    print("Tiling counts:", {k: len(v) for k, v in tiling.items()})

    fig, ax = plt.subplots(figsize=(6, 6))
    template.visualize_tiling(ax=ax, show=False, title_suffix="InitPlus debug")
    fig.suptitle(f"Rotated Planar Tiling d={d} kind={kind}")
    fig.tight_layout()
    plt.show()
