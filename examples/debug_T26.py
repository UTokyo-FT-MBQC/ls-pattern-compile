#!/usr/bin/env python3
"""T26: パイプのアンカー不整合による負座標ノード検出のデバッグ。

2キューブ+1パイプ（水平）と、2キューブ+1パイプ（垂直）の layer0 を構築し、
node2coord の (x,y) で負値が存在しないことを検証する。
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter


# Ensure local import paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GRAPHIX = SRC / "graphix_zx"
for p in (ROOT, SRC, SRC_GRAPHIX):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def build_layer_with_pipe_horizontal():
    from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.mytype import PatchCoordGlobal3D

    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_pipe_h = {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"}

    sk = RHGCanvasSkeleton("T26 horiz")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_h))

    canvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    return layers[0]


def build_layer_with_pipe_vertical():
    from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.mytype import PatchCoordGlobal3D

    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_pipe_v = {"TOP": "X", "BOTTOM": "Z", "LEFT": "O", "RIGHT": "O"}

    sk = RHGCanvasSkeleton("T26 vert")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 1, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_v))

    canvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    return layers[0]


def stat_negatives(layer) -> dict:
    node2coord = getattr(layer, "node2coord", {}) or {}
    xs = [x for (_, (x, y, z)) in node2coord.items()]
    ys = [y for (_, (x, y, z)) in node2coord.items()]
    cnt_x_neg = sum(1 for x in xs if x < 0)
    cnt_y_neg = sum(1 for y in ys if y < 0)
    return {
        "total": len(node2coord),
        "x_min": min(xs) if xs else None,
        "y_min": min(ys) if ys else None,
        "x_neg": cnt_x_neg,
        "y_neg": cnt_y_neg,
    }


def main() -> None:
    horiz = build_layer_with_pipe_horizontal()
    vert = build_layer_with_pipe_vertical()

    s1 = stat_negatives(horiz)
    s2 = stat_negatives(vert)

    print({"horizontal": s1})
    print({"vertical": s2})

    assert s1["x_neg"] == 0 and s1["y_neg"] == 0, "horizontal: negative XY coords found"
    assert s2["x_neg"] == 0 and s2["y_neg"] == 0, "vertical: negative XY coords found"
    print("[T26] OK: no negative XY coordinates for pipes with inner anchor")


if __name__ == "__main__":
    main()

