"""
T38 デバッグスクリプト

目的:
- add_temporal_layer の堅牢化（compose_sequentially を第一候補に使用）と、
- Memory 系ブロックを含む最小構成で to_temporal_layers()/compile が例外なく動くこと、
- TemporalLayer の in/out が空でないこと
を確認する。

使い方:
- Python で本スクリプトを単体実行する（VS Code の Python Interactive Window でも可）。
- 主要な統計（レイヤ z、qubit_count、in/out 個数）と簡単な層情報を stdout に出力する。

入出力:
- 入力: なし（スクリプト内で最小ケースを構築）
- 出力: 層ごとの統計出力（標準出力）
"""

# 実行結果（例: 実行環境により数は変動し得ます）
#
# z=0: qubits=XX, in=YY, out=ZZ
# z=1: qubits=AA, in=BB, out=CC
# compile: global_nodes=GG, layers=2

# %% 準備: インポート
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton  # 任意（compile 検証用）
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D


# %% 最小構成の組み立て（Memory を z=1 に配置）
d = 3
edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

sk = RHGCanvasSkeleton("T38 minimal")
a = PatchCoordGlobal3D((0, 0, 0))
b = PatchCoordGlobal3D((0, 0, 1))
sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
sk.add_cube(b, MemoryCubeSkeleton(d=d, edgespec=edgespec_cube))

# cross-time pipe の挙動確認をする場合のみ（任意）
# sk.add_pipe(a, b, MemoryPipeSkeleton(d=d))


# %% TemporalLayer の生成と基本統計
canvas = sk.to_canvas()
layers = canvas.to_temporal_layers()

def layer_stats(layer):
    return dict(
        z=layer.z,
        qubits=layer.qubit_count,
        in_count=len(layer.in_ports or []),
        out_count=len(layer.out_ports or []),
    )

for z, layer in sorted(layers.items()):
    st = layer_stats(layer)
    print(f"z={st['z']}: qubits={st['qubits']}, in={st['in_count']}, out={st['out_count']}")


# %% compile の Smoke テスト（compose_sequentially の動作確認）
cgraph = canvas.compile()
global_nodes = len(getattr(cgraph.global_graph, "physical_nodes", []) or [])
zlist = getattr(cgraph, "zlist", [])
print(f"compile: global_nodes={global_nodes}, layers={len(cgraph.layers)}, zlist={zlist}")

# 主要 API が例外なく終われば OK（詳細検証は T41/T39 を参照）
