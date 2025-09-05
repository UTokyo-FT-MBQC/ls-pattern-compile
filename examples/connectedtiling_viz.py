"""
T10 可視化デモ: 2 ブロック + 1 パイプをレイヤに配置し、
ConnectedTiling の 2D 表示を行う。

Run: python examples/connectedtiling_viz.py
"""

from __future__ import annotations

from lspattern.blocks.cubes.initialize import InitPlusBlockSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton

from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.tiling.visualize import plot_layer_tiling


def main() -> None:
    d = 5
    block_spec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

    # ブロック2つ
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    skel_a = InitPlusBlockSkeleton(d=d, edgespec=block_spec)
    skel_b = InitPlusBlockSkeleton(d=d, edgespec=block_spec)

    # Pipe (RIGHT direction). edgespec uses default values based on direction in Skeleton side
    p_skel = InitPlusPipeSkeleton(logical=0, d=d)

    canvas = RHGCanvasSkeleton("T10Viz")
    canvas.add_block(a, skel_a)
    canvas.add_block(b, skel_b)
    canvas.add_pipe(a, b, p_skel)

    # Canvas -> 層生成
    canvas2 = canvas.to_canvas()
    layers = canvas2.to_temporal_layers()
    layer0 = layers[0]

    # 2D ConnectedTiling を描画
    plot_layer_tiling(layer0, anchor="inner", show=True, title="T10 ConnectedTiling (RIGHT)")


if __name__ == "__main__":
    main()
