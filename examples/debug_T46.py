"""
目的: T46（ゲーティング常時ON）の検証用デバッグスクリプト。

使い方:
- Python Interactive Window でセル毎に実行（'# %%' セル区切り）。
- 空間パイプ有/無、時間方向パイプ有/無の最小ケースを構築し、
  コンパイル結果のノード/エッジ統計の差分を標準出力に表示する。

入出力:
- 入力: なし（スクリプト内で最小構成を生成）
- 出力: stdout に統計ログ。必要に応じ `fig_T46_*.png` を保存（省略可）。

(1) 目的・コードの説明:
- 層内（same-z）の spatial pipe 有無で CZ spanning（物理エッジ）の件数差が出ること。
- 層間（prev→next）の temporal pipe 有無で seam 物理エッジが追加されること。
- 本IssueではゲーティングON/OFFのトグルはなく、常時ONの前提で構成差を確認する。

(2) 実行例（標準出力の抜粋）:
Spatial ON  edges=XXXX, nodes=YYYY
Spatial OFF edges=AAAA, nodes=BBBB
Temporal ON  edges=CCCC, nodes=DDDD
Temporal OFF edges=EEEE, nodes=FFFF
（行数が100行を超える場合は省略）
"""

# %%

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GRAPHIX = SRC / "graphix_zx"
for p in (SRC, SRC_GRAPHIX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvas, RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D

# %%
d = 3
r = 3


def visualizer_connection():
    canvass = RHGCanvasSkeleton("Memory X")

    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_trimmed = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}
    # tmpl = RotatedPlanarTemplate(d=3, edgespec=edgespec)
    # _ = tmpl.to_tiling()
    blocks = [
        (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=3, edgespec=edgespec)),
        (PatchCoordGlobal3D((2, 2, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
    ]
    pipes = [(PatchCoordGlobal3D((0, 0, 0)), PatchCoordGlobal3D((0, 0, 1)))]

    for block in blocks:
        # RHGCanvasSkeleton は skeleton を受け取り、to_canvas() で block 化します
        canvass.add_cube(*block)
    for pipe in pipes:
        canvass.add_pipe(*pipe)

    canvas = canvass.to_canvas()
    temporal_layer = canvas.to_temporal_layers()

    compiled_canvas: CompiledRHGCanvas = canvas.compile()
    nnodes = (
        len(getattr(compiled_canvas.global_graph, "physical_nodes", []) or [])
        if compiled_canvas.global_graph
        else 0
    )
    print(
        {
            "layers": len(temporal_layer),
            "nodes": nnodes,
            "coord_map": len(compiled_canvas.coord2node),
        }
    )


def visualizer_noconnection():
    canvass = RHGCanvasSkeleton("Memory X")

    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_trimmed = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}
    # tmpl = RotatedPlanarTemplate(d=3, edgespec=edgespec)
    # _ = tmpl.to_tiling()
    blocks = [
        (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=3, edgespec=edgespec)),
        (PatchCoordGlobal3D((2, 2, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
    ]
    pipes = []  # No temporal pipe

    for block in blocks:
        # RHGCanvasSkeleton は skeleton を受け取り、to_canvas() で block 化します
        canvass.add_cube(*block)
    for pipe in pipes:
        canvass.add_pipe(*pipe)

    canvas = canvass.to_canvas()
    temporal_layer = canvas.to_temporal_layers()

    compiled_canvas: CompiledRHGCanvas = canvas.compile()
    nnodes = (
        len(getattr(compiled_canvas.global_graph, "physical_nodes", []) or [])
        if compiled_canvas.global_graph
        else 0
    )
    print(
        {
            "layers": len(temporal_layer),
            "nodes": nnodes,
            "coord_map": len(compiled_canvas.coord2node),
        }
    )
