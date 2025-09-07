"""
T41 デバッグスクリプト（全ブロック×edgespec×d 可視化マトリクス）

目的・使い方・入出力
- 目的: InitPlus/Memory（Cube/Pipe）の各ブロックについて、複数の edgespec と d∈{3,5,7}
  を走査し、materialize() の基本性と入出力ノード強調可視化の成立を確認する。
- 使い方: リポジトリ直下で実行。必要に応じて `pip install -r requirements.txt`。
  ノートブック版は `examples/visualize_T41.ipynb` を参照。
- 入出力: 標準出力に各構成の in/out と z± data 数を記録。代表ケースで PNG を保存。

実行例（stdout 抜粋: 代表ケース）
Block=InitPlusCube d=3 spec=A: in=9 out=9 z- data=9 z+ data=9
Block=InitPlusPipe d=3 spec=H1 dir=RIGHT: in=3 out=3 z- data=3 z+ data=3
Block=MemoryCube d=5 spec=B: in=25 out=25 z- data=25 z+ data=25
Block=MemoryPipe d=7 spec=V2 dir=TOP: in=7 out=7 z- data=7 z+ data=7
"""

# %% [markdown]
# (1) import とパス設定

# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GZX = SRC / "graphix_zx"
for p in (SRC, SRC_GZX):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# %%
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton

from lspattern.visualizers.temporallayer import visualize_temporal_layer
from lspattern.visualizers.plotly_temporallayer import (
    visualize_temporal_layer_plotly,
)


# %% [markdown]
# (2) 補助: Block を TemporalLayer 風に包む / 入出力ノードを求める

# %%
class _LayerView:
    def __init__(self, block, z=0):
        self.node2coord = block.node2coord
        self.local_graph = block.local_graph
        self.node2role = block.node2role
        self.z = z


def layer_from_block(block):
    z_vals = [c[2] for c in block.node2coord.values()] if block.node2coord else [0]
    z0 = min(z_vals) if z_vals else 0
    return _LayerView(block, z0)


def io_nodes_from_graph(block):
    g = block.local_graph
    ins = set(g.input_node_indices.keys()) if hasattr(g, "input_node_indices") else set()
    outs = set(g.output_node_indices.keys()) if hasattr(g, "output_node_indices") else set()
    return ins, outs


# %% [markdown]
# (3) edgespec と走査対象 d

# %%
CUBE_SPECS = {
    "A": {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"},
    "B": {"LEFT": "Z", "RIGHT": "Z", "TOP": "X", "BOTTOM": "X"},
    "AllX": {"LEFT": "X", "RIGHT": "X", "TOP": "X", "BOTTOM": "X"},
    "AllZ": {"LEFT": "Z", "RIGHT": "Z", "TOP": "Z", "BOTTOM": "Z"},
}

# Pipe: 水平（RIGHT）= TOP/BOTTOM は O、左右を X/Z（もしくは逆）
PIPE_SPECS_H = {
    "H1": {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"},
    "H2": {"LEFT": "Z", "RIGHT": "X", "TOP": "O", "BOTTOM": "O"},
}
# Pipe: 垂直（TOP）= LEFT/RIGHT は O、上下を X/Z（もしくは逆）
PIPE_SPECS_V = {
    "V1": {"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"},
    "V2": {"LEFT": "O", "RIGHT": "O", "TOP": "Z", "BOTTOM": "X"},
}

DISTANCES = [3, 5, 7]


# %% [markdown]
# (4) 代表ケースを可視化保存（各ブロック×d=3×1 edgespec）

# %%
import pathlib

def summarize_block(title: str, block) -> None:
    minus = block.get_boundary_nodes(face="z-")
    plus = block.get_boundary_nodes(face="z+")
    print(
        f"{title}: in={len(block.in_ports)} out={len(block.out_ports)} "
        f"z- data={len(minus['data'])} z+ data={len(plus['data'])}"
    )


def save_figs(tag: str, block) -> None:
    layer = layer_from_block(block)
    ins, outs = io_nodes_from_graph(block)
    out_png = pathlib.Path(".").resolve().with_name(f"fig_T41_{tag}.png")
    visualize_temporal_layer(
        layer, save_path=str(out_png), show=False, show_axes=True, show_grid=True,
        input_nodes=ins, output_nodes=outs,
    )
    print("Saved:", out_png)
    # Plotly は interactive。show() 呼び出しのみ。
    _ = visualize_temporal_layer_plotly(
        layer, aspectmode="cube", reverse_axes=True, show_axes=True, show_grid=True,
        input_nodes=ins, output_nodes=outs,
    )


# 代表: InitPlusCube d=3 spec=A
cube = InitPlusCubeSkeleton(d=3, edgespec=CUBE_SPECS["A"]) .to_block().materialize()
summarize_block("Block=InitPlusCube d=3 spec=A", cube)
save_figs("initcube_d3_A", cube)

# 代表: InitPlusPipe d=3 spec=H1 RIGHT
pipe_h = InitPlusPipeSkeleton(d=3, edgespec=PIPE_SPECS_H["H1"]).to_block(
    source=(0, 0, 0), sink=(1, 0, 0)
).materialize()
summarize_block("Block=InitPlusPipe d=3 spec=H1 dir=RIGHT", pipe_h)
save_figs("initpipe_d3_H1_right", pipe_h)

# 代表: MemoryCube d=3 spec=B
mcube = MemoryCubeSkeleton(d=3, edgespec=CUBE_SPECS["B"]).to_block().materialize()
summarize_block("Block=MemoryCube d=3 spec=B", mcube)
save_figs("memcube_d3_B", mcube)

# 代表: MemoryPipe d=3 spec=V2 TOP
mpipe_v = MemoryPipeSkeleton(d=3, edgespec=PIPE_SPECS_V["V2"]).to_block(
    source=(0, 0, 0), sink=(0, 1, 0)
).materialize()
summarize_block("Block=MemoryPipe d=3 spec=V2 dir=TOP", mpipe_v)
save_figs("mempipe_d3_V2_top", mpipe_v)


# %% [markdown]
# (5) マトリクス全走査（ログのみ）

# %%
def build_all_and_log():
    for d in DISTANCES:
        # Cubes
        for name, spec in CUBE_SPECS.items():
            blk = InitPlusCubeSkeleton(d=d, edgespec=spec).to_block().materialize()
            summarize_block(f"Block=InitPlusCube d={d} spec={name}", blk)
        for name, spec in CUBE_SPECS.items():
            blk = MemoryCubeSkeleton(d=d, edgespec=spec).to_block().materialize()
            summarize_block(f"Block=MemoryCube d={d} spec={name}", blk)

        # Pipes (horizontal/right)
        for name, spec in PIPE_SPECS_H.items():
            blk = InitPlusPipeSkeleton(d=d, edgespec=spec).to_block(
                source=(0, 0, 0), sink=(1, 0, 0)
            ).materialize()
            summarize_block(f"Block=InitPlusPipe d={d} spec={name} dir=RIGHT", blk)
        for name, spec in PIPE_SPECS_H.items():
            blk = MemoryPipeSkeleton(d=d, edgespec=spec).to_block(
                source=(0, 0, 0), sink=(1, 0, 0)
            ).materialize()
            summarize_block(f"Block=MemoryPipe d={d} spec={name} dir=RIGHT", blk)

        # Pipes (vertical/top)
        for name, spec in PIPE_SPECS_V.items():
            blk = InitPlusPipeSkeleton(d=d, edgespec=spec).to_block(
                source=(0, 0, 0), sink=(0, 1, 0)
            ).materialize()
            summarize_block(f"Block=InitPlusPipe d={d} spec={name} dir=TOP", blk)
        for name, spec in PIPE_SPECS_V.items():
            blk = MemoryPipeSkeleton(d=d, edgespec=spec).to_block(
                source=(0, 0, 0), sink=(0, 1, 0)
            ).materialize()
            summarize_block(f"Block=MemoryPipe d={d} spec={name} dir=TOP", blk)


if __name__ == "__main__":
    build_all_and_log()

