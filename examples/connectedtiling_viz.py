from __future__ import annotations

import sys
from pathlib import Path

"""
T10 可視化デモ: 2 ブロック + 1 パイプをレイヤに配置し、
ConnectedTiling の 2D 表示を行う。

Run: python examples/connectedtiling_viz.py
"""


def _ensure_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    gx = root / "src" / "graphix_zx"
    gx_str = str(gx)
    if gx_str not in sys.path:
        sys.path.insert(0, gx_str)


def main() -> None:
    _ensure_paths()

    from lspattern.blocks.initialize import InitPlusSkeleton
    from lspattern.pipes.initialize import InitPlusPipeSkeleton

    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.mytype import PatchCoordGlobal3D
    from lspattern.tiling.visualize import plot_layer_tiling

    d = 5
    block_spec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

    # ブロック2つ
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    skel_a = InitPlusSkeleton(d=d, edgespec=block_spec)
    skel_b = InitPlusSkeleton(d=d, edgespec=block_spec)

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
