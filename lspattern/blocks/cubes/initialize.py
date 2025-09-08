"""Initialization block(s) for cube-shaped RHG structures."""

from __future__ import annotations

from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.tiling.template import RotatedPlanarCubeTemplate


class InitPlusCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for initialization blocks in cube-shaped RHG structures."""

    name: str = __qualname__

    def to_block(self) -> RHGCube:
        """
        Return a template-holding block (no local graph state).

        Returns
        -------
            RHGBlock: A block containing the template with no local graph state.
        """
        for direction in ["LEFT", "RIGHT", "TOP", "BOTTOM"]:
            if self.edgespec[direction] == "O":
                self.trim_spatial_boundary(direction)
        self.template.to_tiling()

        block = InitPlus(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )

        # Init 系は最終層は測定せず開放(O)
        block.final_layer = "O"

        return block


class InitPlus(RHGCube):
    name: str = __qualname__

    def set_in_ports(self):
        # Init plus sets no input ports
        return super().set_in_ports()

    def set_out_ports(self):
        # Init: 最終スライス(z+)の data を出力ポート(テンプレートの data 全インデックス)とみなす
        idx_map = self.template.get_data_indices()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self):
        # sets no classical output ports
        return super().set_cout_ports()


if __name__ == "__main__":
    # NOTE: Interactive 3D preview code omitted for brevity

    # Hardcoded options (edit here as needed)
    d = 3
    edgespec = {
        "TOP": "X",
        "BOTTOM": "Z",
        "LEFT": "X",
        "RIGHT": "Z",
    }  # e.g., {"TOP":"X","BOTTOM":"Z",...}
    ANCILLA_MODE = "both"  # "both" | "x" | "z"
    EDGE_WIDTH = 0.5  # thicker black edges
    INTERACTIVE = True  # interactive plot

    # Build template and block
    template = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec)
    _ = template.to_tiling()  # populate internal coords for indices

    block = InitPlus(d=d, template=template)

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

    # groups = {k: {"x": [], "y": [], "z": []} for k in color_map}
    # for n, (x, y, z) in node2coord.items():
    #     role = node2role.get(n, "data")
    #     if role not in groups:
    #         role = "data"
    #     groups[role]["x"].append(x)
    #     groups[role]["y"].append(y)
    #     groups[role]["z"].append(z)

    # # 3D plot
    # if INTERACTIVE:
    #     plt.ion()
    # fig = plt.figure(figsize=(8, 7))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_box_aspect((1, 1, 1))
    # ax.set_title(f"InitPlus RHGBlock d={d} edgespec={edgespec}")

    # # Plot nodes by role with template-like colors
    # for role, pts in list(groups.items()):
    #     # Filter by ancilla option
    #     if ANCILLA_MODE == "x" and role == "ancilla_z":
    #         continue
    #     if ANCILLA_MODE == "z" and role == "ancilla_x":
    #         continue
    #     if not pts["x"]:
    #         continue
    #     spec = color_map[role]
    #     ax.scatter(
    #         pts["x"],
    #         pts["y"],
    #         pts["z"],
    #         s=spec["size"],
    #         c=spec["face"],
    #         edgecolors=spec["edge"],
    #         depthshade=True,
    #         label=role,
    #         linewidths=0.8,
    #     )

    # # Draw edges in black (thicker)
    # for u, v in g.physical_edges:
    #     x1, y1, z1 = node2coord[u]
    #     x2, y2, z2 = node2coord[v]
    #     ax.plot([x1, x2], [y1, y2], [z1, z2], color="black", linewidth=EDGE_WIDTH, alpha=0.9)

    # # Overlay input and output nodes with black fill
    # in_nodes = set(g.input_node_indices.keys())
    # out_nodes = set(g.output_node_indices.keys())
    # if in_nodes:
    #     xin = [node2coord[n][0] for n in in_nodes]
    #     yin = [node2coord[n][1] for n in in_nodes]
    #     zin = [node2coord[n][2] for n in in_nodes]
    #     ax.scatter(xin, yin, zin, s=60, c="black", edgecolors="black", label="input", zorder=5)
    # if out_nodes:
    #     xout = [node2coord[n][0] for n in out_nodes]
    #     yout = [node2coord[n][1] for n in out_nodes]
    #     zout = [node2coord[n][2] for n in out_nodes]
    #     ax.scatter(
    #         xout,
    #         yout,
    #         zout,
    #         s=60,
    #         c="black",
    #         edgecolors="black",
    #         label="output",
    #         zorder=6,
    #     )

    # # Nice framing
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # ax.legend(loc="upper left")
    # plt.tight_layout()
    # plt.show()
