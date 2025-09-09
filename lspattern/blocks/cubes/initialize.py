"""Initialization block(s) for cube-shaped RHG structures."""

from __future__ import annotations

from typing import ClassVar, Literal

from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.tiling.template import RotatedPlanarCubeTemplate


class InitPlusCubeSkeleton(RHGCubeSkeleton):
    """Skeleton for initialization blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "InitPlusCubeSkeleton"

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
    name: ClassVar[str] = "InitPlus"

    def set_in_ports(self) -> None:
        # Init plus sets no input ports
        return super().set_in_ports()

    def set_out_ports(self) -> None:
        # Init: 最終スライス(z+)の data を出力ポート(テンプレートの data 全インデックス)とみなす
        idx_map = self.template.get_data_indices()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self) -> None:
        # sets no classical output ports
        return super().set_cout_ports()


if __name__ == "__main__":
    # NOTE: Interactive 3D preview code omitted for brevity

    # Hardcoded options (edit here as needed)
    d = 3
    edgespec: dict[str, Literal["X", "Z", "O"]] = {
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
