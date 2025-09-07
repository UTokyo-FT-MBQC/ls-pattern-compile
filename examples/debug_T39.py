"""
T39 デバッグスクリプト（Memory クラス: Cube/Pipe）

目的
- InitPlus→Memory→Measure の中間段として、MemoryCube / MemoryPipe の最小仕様
  （in/out/cout ポートの意味付けとテンプレート駆動の材質化）を確認する。

使い方
- リポジトリ直下で実行。
- 事前に `pip install -r requirements.txt` を推奨。
- `src/` を `sys.path` に追加して、同梱の `graphix_zx` を解決する。

入出力
- 入力: なし
- 出力: stdout に各ブロックの in/out ポート数と z± 側 data ノード数を表示。

実行例（stdout, d=3 for Cube, d=5 for Pipe）
Cube: in=9, out=9, z- data=9, z+ data=9
Pipe(RIGHT): in=5, out=5, z- data=5, z+ data=5
"""

# %% [markdown]
# (1) 目的とコードの説明
# - MemoryCube/MemoryPipe を最小構成で materialize し、
#   in/out の非空性と z- / z+ の data ノード数を観察する。
# - 本版ではテンプレートに基づき data の全インデックスを in/out に割当てる。

# %%
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GZX = SRC / "graphix_zx"
for p in (SRC, SRC_GZX):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# %%
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton


def summarize_block(block, title: str) -> None:
    b = block.materialize()
    minus = b.get_boundary_nodes(face="z-")
    plus = b.get_boundary_nodes(face="z+")
    print(
        f"{title}: in={len(b.in_ports)}, out={len(b.out_ports)}, "
        f"z- data={len(minus['data'])}, z+ data={len(plus['data'])}"
    )


def main() -> None:
    # Cube: d=3（9 データ）
    cube_spec = {"TOP": "Z", "BOTTOM": "Z", "LEFT": "X", "RIGHT": "X"}
    cube = MemoryCubeSkeleton(d=3, edgespec=cube_spec).to_block()
    summarize_block(cube, "Cube")

    # Pipe: d=5（5 データ） 右方向を想定
    pipe = MemoryPipeSkeleton(
        d=5, edgespec={"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"}
    )
    block = pipe.to_block(source=(0, 0, 0), sink=(1, 0, 0))
    summarize_block(block, "Pipe(RIGHT)")


# %% (2) 実行ログ（stdout）は上の実行例参照
if __name__ == "__main__":
    main()
