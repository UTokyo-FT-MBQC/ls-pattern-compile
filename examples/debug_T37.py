#!/usr/bin/env python3
"""T37: TemporalLayer 内の cube↔pipe シームで CZ 辺が生成されない不具合の再現と可視化用統計。

目的・使い方・入出力:
- 目的: 層内（同じ z）でキューブとパイプの境界に CZ 辺（物理エッジ）が張られていないことを検出し、数を報告します。
- 使い方: `python examples/debug_T37.py` を実行すると、水平方向と垂直方向の最小例を構築し、シーム交差エッジ数を出力します。
- 入出力: 標準出力に辞書形式の統計を表示（ファイル出力なし）。

(1) タスクの目的・コードの説明:
- 2 キューブ + 1 パイプのレイヤーを構築し、テンプレートの 2D 座標からキューブ領域とパイプ領域の XY を絶対座標に変換します。
- `TemporalLayer.local_graph.physical_edges` を走査し、片端がパイプ XY、もう片端がキューブ XY（左右/上下いずれか）に属するエッジ数をカウントします。
- 期待値は > 0 ですが、現在は 0（＝シーム CZ 欠落）となることを確認します。

(2) 実行例の std out（抜粋例）:
    {'horizontal': {'edges_total': 1234, 'seam_edges': 0}}
    {'vertical':   {'edges_total': 1234, 'seam_edges': 0}}
    [T37] NG: seam CZ edges are missing between cube and pipe
"""

from __future__ import annotations

import pathlib
import sys
from typing import Iterable


# Ensure local import paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GRAPHIX = SRC / "graphix_zx"
for p in (ROOT, SRC, SRC_GRAPHIX):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _abs_xy_set_for_cube(d: int, pos, blk) -> set[tuple[int, int]]:
    from lspattern.tiling.template import cube_offset_xy, offset_tiling

    dx, dy = cube_offset_xy(d, pos)
    t = offset_tiling(blk.template, dx, dy)
    xy = set()
    for L in (t.data_coords, t.x_coords, t.z_coords):
        xy.update((int(x), int(y)) for (x, y) in (L or []))
    return xy


def _abs_xy_set_for_pipe(d: int, source, sink, pipe) -> set[tuple[int, int]]:
    from lspattern.utils import get_direction
    from lspattern.tiling.template import pipe_offset_xy, offset_tiling

    direction = get_direction(source, sink)
    dx, dy = pipe_offset_xy(d, source, sink, direction)
    t = offset_tiling(pipe.template, dx, dy)
    xy = set()
    for L in (t.data_coords, t.x_coords, t.z_coords):
        xy.update((int(x), int(y)) for (x, y) in (L or []))
    return xy


def _count_seam_edges(layer) -> int:
    """Count edges whose endpoints lie across cube vs pipe XY regions (same layer)."""
    # Build absolute XY sets for cubes and pipes from the layer
    cube_regions: list[set[tuple[int, int]]] = []
    for pos, blk in layer.cubes_.items():
        cube_regions.append(_abs_xy_set_for_cube(int(blk.d), pos, blk))
    pipe_regions: list[set[tuple[int, int]]] = []
    for (u, v), pipe in layer.pipes_.items():
        pipe_regions.append(_abs_xy_set_for_pipe(int(pipe.d), u, v, pipe))

    cube_xy_all: set[tuple[int, int]] = set().union(*cube_regions) if cube_regions else set()
    pipe_xy_all: set[tuple[int, int]] = set().union(*pipe_regions) if pipe_regions else set()

    n2c = getattr(layer, "node2coord", {}) or {}
    g = getattr(layer, "local_graph", None)
    if g is None:
        return 0

    seam = 0
    try:
        edges = getattr(g, "physical_edges", []) or []
    except Exception:
        edges = []
    for (u, v) in edges:
        cu = n2c.get(u)
        cv = n2c.get(v)
        if not cu or not cv:
            continue
        xu, yu = int(cu[0]), int(cu[1])
        xv, yv = int(cv[0]), int(cv[1])
        # across cube vs pipe
        if ((xu, yu) in pipe_xy_all and (xv, yv) in cube_xy_all) or (
            (xv, yv) in pipe_xy_all and (xu, yu) in cube_xy_all
        ):
            seam += 1
    return seam


def _build_layer_with_pipe_horizontal():
    from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.mytype import PatchCoordGlobal3D

    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_pipe_h = {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"}

    sk = RHGCanvasSkeleton("T37 horiz")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_h))

    canvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    return layers[0]


def _build_layer_with_pipe_vertical():
    from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.mytype import PatchCoordGlobal3D

    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_pipe_v = {"TOP": "X", "BOTTOM": "Z", "LEFT": "O", "RIGHT": "O"}

    sk = RHGCanvasSkeleton("T37 vert")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 1, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_v))

    canvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    return layers[0]


def _stat(layer):
    g = getattr(layer, "local_graph", None)
    edges_total = len(getattr(g, "physical_edges", []) or []) if g else 0
    seam = _count_seam_edges(layer)
    return {"edges_total": edges_total, "seam_edges": seam}


def main() -> None:
    horiz = _build_layer_with_pipe_horizontal()
    vert = _build_layer_with_pipe_vertical()

    s1 = _stat(horiz)
    s2 = _stat(vert)

    print({"horizontal": s1})
    print({"vertical": s2})

    if s1["seam_edges"] == 0 or s2["seam_edges"] == 0:
        print("[T37] NG: seam CZ edges are missing between cube and pipe")
    else:
        print("[T37] OK: seam CZ edges detected between cube and pipe")


if __name__ == "__main__":
    main()

