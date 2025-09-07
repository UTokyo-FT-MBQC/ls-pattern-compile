"""
T43 デバッグスクリプト（CompiledRHGCanvas 可視化: Matplotlib/Plotly）

目的:
- 複数の時間層を含む CompiledRHGCanvas を生成し、Matplotlib/Plotly 双方で可視化できることを確認する。

使い方:
- リポジトリ直下で `python examples/debug_T43.py` を実行。
- VS Code の Python Interactive Window でもセル単位で実行可能（# %% セルを使用）。

入出力:
- 入力: なし（最小ケースを内部で構築）
- 標準出力: レイヤ統計（zlist, global_nodes）と画像保存ログ
- 画像出力: `fig_T43_compiled_mpl.png` を保存（Matplotlib）。
"""

# 実行結果（例: 抜粋）
# compiled: global_nodes=XXXX, zlist=[0, 1]
# Saved: fig_T43_compiled_mpl.png

# %% [markdown]
# (1) 準備と最小ケース構築（z=0: InitPlusCube, z=1: MemoryCube）

# %%
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D

edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
d = 3

sk = RHGCanvasSkeleton("T43 compiled viz")
a = PatchCoordGlobal3D((0, 0, 0))
b = PatchCoordGlobal3D((0, 0, 1))
sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec))
sk.add_cube(b, MemoryCubeSkeleton(d=d, edgespec=edgespec))

# cross-time pipe の挙動確認をする場合はコメント解除
sk.add_pipe(a, b, MemoryPipeSkeleton(d=d))

# %% [markdown]
# (2) compile と基本統計

# %%
canvas = sk.to_canvas()
cgraph = canvas.compile()
global_nodes = len(getattr(cgraph.global_graph, "physical_nodes", []) or [])
zlist = getattr(cgraph, "zlist", [])
print(f"compiled: global_nodes={global_nodes}, zlist={zlist}")

# %% [markdown]
# (3) Matplotlib ビジュアライザで保存

# %%
from lspattern.visualizers import visualize_compiled_canvas

fig = visualize_compiled_canvas(
    cgraph,
    save_path="fig_T43_compiled_mpl.png",
    show=False,
    show_axes=True,
    show_grid=True,
    show_edges=True,
)
print("Saved: fig_T43_compiled_mpl.png")

# %% [markdown]
# (4) Plotly ビジュアライザ（表示のみ）

# %%
from lspattern.visualizers import visualize_compiled_canvas_plotly

_fig = visualize_compiled_canvas_plotly(
    cgraph,
    show_edges=True,
    reverse_axes=False,
    show_axes=True,
    show_grid=True,
)
# Notebook では `_fig.show()` 可能。ここでは生成のみ。
