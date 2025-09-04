from __future__ import annotations

from typing import List, Set

from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState
from lspattern.blocks.base import RHGBlock, RHGBlockSkeleton
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.mytype import (
    BlockKindstr,
    FlowLocal,
    NodeIdLocal,
    NodeSetLocal,
    PhysCoordGlobal3D,
    PhysCoordLocal2D,
    PhysCoordLocal3D,
    ScheduleTuplesLocal,
)
from lspattern.template.base import RotatedPlanarTemplate


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
                if z == max_t:
                    g.register_output(n, data_indices[(x, y)])
                if z == 0:
                    # mark initial data slice as input nodes
                    # g.register_input(n)
                    pass
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
        self.node2coord = node2coord
        self.coord2node = {coord: nid for nid, coord in node2coord.items()}
        self.node2role = node2role
        self.in_ports = set()
        self.out_ports = out_ports
        self.cout_ports = []
        self.x_checks = x_checks
        self.z_checks = z_checks
        self.schedule_local = schedule_local
        self.flow_local = flow_local


if __name__ == "__main__":
    # 次にInitpluseのRHGBlockを定義して、materializeして、そのgraph_localをtemplateと同様の色づけてvisualizeして。edgeは黒線でかいて、3Dプロットを使って
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, ensure 3D is registered

    # Hardcoded options (edit here as needed)
    d = 3
    kind = ("X", "Z", "X")  # e.g., "XZX"
    ANCILLA_MODE = "both"  # "both" | "x" | "z"
    EDGE_WIDTH = 0.5  # thicker black edges
    INTERACTIVE = True  # interactive plot

    # Build template and block
    template = RotatedPlanarTemplate(d=d, kind=kind)
    _ = template.to_tiling()  # populate internal coords for indices

    block = InitPlus(d=d, kind=kind, template=template)
    block.materialize()

    g = block.graph_local
    node2coord = block.node2coord
    node2role = block.node2role

    # Prepare colored point clouds (match template colors)
    # data: white, X ancilla: green, Z ancilla: blue
    color_map = {
        "data": {
            "face": "white",
            "edge": "black",
            "size": 40,
        },
        "ancilla_x": {
            "face": "#2ecc71",
            "edge": "#1e8449",
            "size": 36,
        },
        "ancilla_z": {
            "face": "#3498db",
            "edge": "#1f618d",
            "size": 36,
        },
    }

    groups = {k: {"x": [], "y": [], "z": []} for k in color_map}
    for n, (x, y, z) in node2coord.items():
        role = node2role.get(n, "data")
        if role not in groups:
            role = "data"
        groups[role]["x"].append(x)
        groups[role]["y"].append(y)
        groups[role]["z"].append(z)

    # 3D plot
    if INTERACTIVE:
        plt.ion()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"InitPlus RHGBlock d={d} kind={kind}")

    # Plot nodes by role with template-like colors
    for role, pts in list(groups.items()):
        # Filter by ancilla option
        if ANCILLA_MODE == "x" and role == "ancilla_z":
            continue
        if ANCILLA_MODE == "z" and role == "ancilla_x":
            continue
        if not pts["x"]:
            continue
        spec = color_map[role]
        ax.scatter(
            pts["x"],
            pts["y"],
            pts["z"],
            s=spec["size"],
            c=spec["face"],
            edgecolors=spec["edge"],
            depthshade=True,
            label=role,
            linewidths=0.8,
        )

    # Draw edges in black (thicker)
    for u, v in g.physical_edges:
        x1, y1, z1 = node2coord[u]
        x2, y2, z2 = node2coord[v]
        ax.plot(
            [x1, x2], [y1, y2], [z1, z2], color="black", linewidth=EDGE_WIDTH, alpha=0.9
        )

    # Overlay input and output nodes with black fill
    in_nodes = set(g.input_node_indices.keys())
    out_nodes = set(g.output_node_indices.keys())
    if in_nodes:
        xin = [node2coord[n][0] for n in in_nodes]
        yin = [node2coord[n][1] for n in in_nodes]
        zin = [node2coord[n][2] for n in in_nodes]
        ax.scatter(
            xin, yin, zin, s=60, c="black", edgecolors="black", label="input", zorder=5
        )
    if out_nodes:
        xout = [node2coord[n][0] for n in out_nodes]
        yout = [node2coord[n][1] for n in out_nodes]
        zout = [node2coord[n][2] for n in out_nodes]
        ax.scatter(
            xout,
            yout,
            zout,
            s=60,
            c="black",
            edgecolors="black",
            label="output",
            zorder=6,
        )

    # Nice framing
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
