from __future__ import annotations

"""
T10 可視化デモ: 2 ブロック + 1 パイプをレイヤに配置し、
ConnectedTiling の 2D 表示を行う。

Run: python examples/connectedtiling_viz.py
"""

import os
import sys


def _ensure_paths() -> None:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    GX = os.path.join(ROOT, "src", "graphix_zx")
    if GX not in sys.path:
        sys.path.insert(0, GX)


def main() -> None:
    _ensure_paths()

    from lspattern.consts.consts import PIPEDIRECTION
    from lspattern.mytype import PatchCoordGlobal3D
    from lspattern.blocks.initialize import InitPlusSkeleton
    from lspattern.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.tiling.visualize import plot_layer_tiling

    d = 5
    block_spec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

    # ブロック2つ
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    skel_a = InitPlusSkeleton(d=d, edgespec=block_spec)
    skel_b = InitPlusSkeleton(d=d, edgespec=block_spec)

    # パイプ（RIGHT）。edgespec は Skeleton 側で方向に応じて既定値を使う
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
