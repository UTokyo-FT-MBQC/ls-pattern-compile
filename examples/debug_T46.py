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
from __future__ import annotations

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D


def build_spatial_case(pipe_on: bool = True, d: int = 3):
    sk = RHGCanvasSkeleton("T46 spatial case")
    a: PatchCoordGlobal3D = (0, 0, 0)
    b: PatchCoordGlobal3D = (1, 0, 0)

    # Horizontal connect の参考 edgespec（Agents 付録）
    edgespec_cube_1 = {"LEFT": "X", "RIGHT": "O", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_cube_2 = {"LEFT": "O", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_pipe_3 = {"LEFT": "O", "RIGHT": "O", "TOP": "Z", "BOTTOM": "Z"}

    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube_1))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube_2))
    if pipe_on:
        sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_3))
    canvas = sk.to_canvas()
    cgraph = canvas.compile()
    g = cgraph.global_graph
    edges = len(getattr(g, "physical_edges", []) or [])
    nodes = len(getattr(g, "physical_nodes", []) or [])
    return cgraph, nodes, edges


def build_temporal_case(pipe_on: bool = True, d: int = 3):
    sk = RHGCanvasSkeleton("T46 temporal case")
    a0: PatchCoordGlobal3D = (0, 0, 0)
    a1: PatchCoordGlobal3D = (0, 0, 1)

    # キューブは上下のみを開き、左右は閉じる例（Agents 付録を参考）
    edgespec_top_open = {"LEFT": "X", "RIGHT": "X", "TOP": "O", "BOTTOM": "Z"}
    edgespec_bottom_open = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "O"}
    # パイプ（時間方向）は左右 X を維持し、上下を Open
    edgespec_pipe_t = {"LEFT": "X", "RIGHT": "X", "TOP": "O", "BOTTOM": "O"}

    sk.add_cube(a0, InitPlusCubeSkeleton(d=d, edgespec=edgespec_top_open))
    sk.add_cube(a1, InitPlusCubeSkeleton(d=d, edgespec=edgespec_bottom_open))
    if pipe_on:
        sk.add_pipe(a0, a1, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_t))
    canvas = sk.to_canvas()
    cgraph = canvas.compile()
    g = cgraph.global_graph
    edges = len(getattr(g, "physical_edges", []) or [])
    nodes = len(getattr(g, "physical_nodes", []) or [])
    return cgraph, nodes, edges


# %%
if __name__ == "__main__":
    # Spatial
    cg_on, n_on, e_on = build_spatial_case(pipe_on=True)
    cg_off, n_off, e_off = build_spatial_case(pipe_on=False)
    print(f"Spatial ON  edges={e_on}, nodes={n_on}")
    print(f"Spatial OFF edges={e_off}, nodes={n_off}")

    # Temporal
    ct_on, n2_on, e2_on = build_temporal_case(pipe_on=True)
    ct_off, n2_off, e2_off = build_temporal_case(pipe_on=False)
    print(f"Temporal ON  edges={e2_on}, nodes={n2_on}")
    print(f"Temporal OFF edges={e2_off}, nodes={n2_off}")

    # 簡易差分（参考表示）
    print("-- Diff summary --")
    print(f"Spatial edge diff: {e_on - e_off}")
    print(f"Temporal edge diff: {e2_on - e2_off}")

