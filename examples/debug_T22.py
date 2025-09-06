#!/usr/bin/env python3
"""T22: TemporalLayer.compile の CZ 接続（allowed_pairs）検証デモ。

2 つのキューブとそれらを接続する 1 本のパイプを用意し、
パイプ有無で layer0 のエッジ数が変わること（=許可された組み合わせのみ
CZ が張られること）を確認します。
"""

from __future__ import annotations

import pathlib
import sys


# Ensure repo-local imports (lspattern, graphix_zx)
_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SRC_GRAPHIX = _SRC / "graphix_zx"
for _p in (_ROOT, _SRC, _SRC_GRAPHIX):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


def build_layer(with_pipe: bool):
    from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.mytype import PatchCoordGlobal3D

    d = 3

    # キューブ 2 個（x 方向に隣接）
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    sk = RHGCanvasSkeleton("T22 demo")
    cpos_a = PatchCoordGlobal3D((0, 0, 0))
    cpos_b = PatchCoordGlobal3D((1, 0, 0))
    sk.add_cube(cpos_a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(cpos_b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))

    # キューブ間パイプ（x 方向の隣接なので、TOP/BOTTOM を O にして水平パイプ）
    if with_pipe:
        edgespec_pipe = {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"}
        sk.add_pipe(cpos_a, cpos_b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe))

    canvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    layer0 = layers.get(0)
    if layer0 is None:
        raise RuntimeError("layer 0 not created")
    # compile は to_temporal_layers() 内で呼ばれているが、明示的に再実行しても良い
    # layer0.compile()
    g = layer0.local_graph
    nnodes = len(getattr(g, "physical_nodes", []) or []) if g else 0
    nedges = len(getattr(g, "physical_edges", []) or []) if g else 0
    return {
        "with_pipe": with_pipe,
        "nodes": nnodes,
        "edges": nedges,
    }


def main() -> None:
    r_no = build_layer(with_pipe=False)
    r_yes = build_layer(with_pipe=True)

    print("[T22] no pipe:", r_no)
    print("[T22] with pipe:", r_yes)

    # パイプがある方が（同一 gid となり）境界を跨ぐ CZ が張られるため、
    # エッジ数が増えることを期待
    assert r_yes["edges"] > r_no["edges"], "Expected more edges when pipe is present"
    print("[T22] OK: allowed_pairs（=パイプ）により CZ が有効化されました")


if __name__ == "__main__":
    main()

