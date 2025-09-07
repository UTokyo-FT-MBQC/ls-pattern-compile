"""
T42 デバッグスクリプト（Cross-time Pipe と allowed_pairs ゲーティング）

目的・使い方・入出力
- 目的: zシーム（prev_z→next_z）の seam_pairs を計測し、
  allowed_pairs=seam_pairs を Accumulator.update_at に適用した場合の
  Flow/Parity 統計の変化を観察する。
- 使い方: リポジトリ直下で単体実行。環境変数で挙動を切替。
  - OFF: `LSPATTERN_ACC_USE_SEAM_PAIRS=0`（デフォルト）
  - ON : `LSPATTERN_ACC_USE_SEAM_PAIRS=1`
  本スクリプトは内部で ON/OFF を切替えて比較出力する。
- 入出力: 入力なし。標準出力に seam_pairs_count、Flow/Parity の統計、zlist を表示。

実行ログ例（抜粋・環境により数値は変動）
# with_pipe=False, gating=OFF: seam_pairs=0, xflow=.., zflow=.., parity_nodes=..
# with_pipe=False, gating=ON : seam_pairs=0, xflow=.., zflow=.., parity_nodes=..
# with_pipe=True,  gating=OFF: seam_pairs>0, xflow=.., zflow=.., parity_nodes=..
# with_pipe=True,  gating=ON : seam_pairs>0, xflow(ON) != xflow(OFF) or parity differs
"""

# %% [markdown]
# (1) import とパス設定

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
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D


# %% [markdown]
# (2) 最小ケースの構築関数（z=0: InitPlusCube, z=1: MemoryCube）

# %%
def build_canvas(with_pipe: bool, d: int = 3):
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    sk = RHGCanvasSkeleton("T42 minimal")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 0, 1))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, MemoryCubeSkeleton(d=d, edgespec=edgespec_cube))
    if with_pipe:
        # cross-time pipe: a(z=0) -> b(z=1)
        sk.add_pipe(a, b, MemoryPipeSkeleton(d=d))
    return sk.to_canvas()


# %% [markdown]
# (3) コンパイル＆統計算出ヘルパ

# %%
def compile_and_stats(canvas, gating: bool):
    # Preserve original env and toggle
    key = "LSPATTERN_ACC_USE_SEAM_PAIRS"
    old = os.environ.get(key)
    try:
        os.environ[key] = "1" if gating else "0"
        cgraph = canvas.compile()
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old

    # seam_pairs_count を構築（prev(zmax)↔next(zmin) の同一XYに対するエッジ数）
    try:
        prev_layer = cgraph.layers[-2]
        next_layer = cgraph.layers[-1]
        prev_last_z = max(c[2] for c in prev_layer.node2coord.values()) if prev_layer.node2coord else None
        next_first_z = min(c[2] for c in next_layer.node2coord.values()) if next_layer.node2coord else None
        seam_pairs_count = 0
        if prev_last_z is not None and next_first_z is not None:
            prev_xy_to_node = { (x,y): n for n,(x,y,z) in prev_layer.node2coord.items() if z == prev_last_z }
            next_xy_to_node = { (x,y): n for n,(x,y,z) in next_layer.node2coord.items() if z == next_first_z }
            g = cgraph.global_graph
            edge_set = set()
            if g is not None and hasattr(g, 'physical_edges'):
                edge_set = { (min(u,v), max(u,v)) for (u,v) in g.physical_edges }
            for xy,u in prev_xy_to_node.items():
                v = next_xy_to_node.get(xy)
                if v is not None and (min(u,v), max(u,v)) in edge_set:
                    seam_pairs_count += 1
    except Exception:
        seam_pairs_count = 0

    # Flow/Parity の集計（全体）
    xflow_size = sum(len(v) for v in cgraph.flow.xflow.values())
    zflow_size = sum(len(v) for v in cgraph.flow.zflow.values())
    xpar_size = sum(len(g) for g in cgraph.parity.x_checks)
    zpar_size = sum(len(g) for g in cgraph.parity.z_checks)

    # 追加: シーム直下（next_layer の z-）アンシラを起点とする Flow のみを抽出
    anchors_x = 0
    anchors_z = 0
    try:
        # 最終レイヤ（next_layer 相当）の z- 境界アンシラを収集
        last = cgraph.layers[-1]
        bn = last.get_boundary_nodes(face="z-", depth=[0])
        anchor_ids: list[int] = []
        for c in bn.get("xcheck", []):
            nid = last.coord2node.get(c)
            if nid is not None:
                anchor_ids.append(nid)
        for c in bn.get("zcheck", []):
            nid = last.coord2node.get(c)
            if nid is not None:
                anchor_ids.append(nid)

        # FlowAccumulator からアンカー起点のみをカウント
        xf = cgraph.flow.xflow
        zf = cgraph.flow.zflow
        anchors_x = sum(len(xf.get(a, set())) for a in anchor_ids)
        anchors_z = sum(len(zf.get(a, set())) for a in anchor_ids)
    except Exception:
        pass

    return {
        "cgraph": cgraph,
        "seam_pairs_count": seam_pairs_count,
        "xflow_size": xflow_size,
        "zflow_size": zflow_size,
        "xpar_size": xpar_size,
        "zpar_size": zpar_size,
        "anchors_xflow": anchors_x,
        "anchors_zflow": anchors_z,
        "zlist": list(getattr(cgraph, "zlist", [])),
    }


# %% [markdown]
# (4) 比較実行: with_pipe in {False, True} × gating in {OFF, ON}

# %%
def run_compare():
    for with_pipe in (False, True):
        canvas = build_canvas(with_pipe=with_pipe, d=3)
        stats_off = compile_and_stats(canvas, gating=False)
        stats_on = compile_and_stats(canvas, gating=True)

        def line(s):
            return (
                f"seam_pairs={s['seam_pairs_count']}, "
                f"xflow={s['xflow_size']}, zflow={s['zflow_size']}, "
                f"xpar_nodes={s['xpar_size']}, zpar_nodes={s['zpar_size']}, "
                f"anchors_xflow={s['anchors_xflow']}, anchors_zflow={s['anchors_zflow']}, "
                f"zlist={s['zlist']}"
            )

        print(f"with_pipe={with_pipe}, gating=OFF: {line(stats_off)}")
        print(f"with_pipe={with_pipe}, gating=ON : {line(stats_on)}")


if __name__ == "__main__":
    run_compare()
