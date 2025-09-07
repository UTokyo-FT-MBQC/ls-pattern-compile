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

ROOT = pathlib.Path("./").resolve().parents[1]
SRC = ROOT / "src"
SRC_GRAPHIX = SRC / "graphix_zx"
for p in (ROOT, SRC, SRC_GRAPHIX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvas, RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D, PhysCoordGlobal3D, QubitGroupIdGlobal
from lspattern.utils import is_allowed_pair

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
    pipes = [(PatchCoordGlobal3D((0, 0, 0)), PatchCoordGlobal3D((0, 0, 1)), MemoryPipeSkeleton(d=3, edgespec=edgespec))]

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
    nedges = (
        len(getattr(compiled_canvas.global_graph, "physical_edges", []) or [])
        if compiled_canvas.global_graph
        else 0
    )
    print(
        {
            "layers": len(temporal_layer),
            "nodes": nnodes,
            "edges": nedges,
            "coord_map": len(compiled_canvas.coord2node),
        }
    )

    # --- 追加: 確認用出力（allowed_gid_pairs と seam の gid 比較） ---
    try:
        # 前後レイヤの把握
        last_z = max(temporal_layer.keys())
        prev_z = last_z - 1
        prev_layer = temporal_layer[prev_z]
        next_layer = temporal_layer[last_z]

        # Pipe 由来の allowed_gid_pairs（prev レイヤの cube と next レイヤの cube）
        allowed: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]] = set()
        for (u, v), _p in canvas.pipes_.items():
            if u[2] == prev_z and v[2] == last_z:
                gu = QubitGroupIdGlobal(prev_layer.cubes_[u].get_tiling_id())
                gv = QubitGroupIdGlobal(next_layer.cubes_[v].get_tiling_id())
                allowed.add((min(gu, gv), max(gu, gv)))
        print({"allowed_gid_pairs": sorted(tuple(map(int, p)) for p in allowed)})

        # 合成に使う coord2gid を近似再構成（prev layer + next layer の順で上書き）
        new_coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal] = {}
        for _pos, cube in [*prev_layer.cubes_.items(), *next_layer.cubes_.items()]:
            new_coord2gid.update(cube.coord2gid)
        for _pos, pipe in [*prev_layer.pipes_.items(), *next_layer.pipes_.items()]:
            new_coord2gid.update(pipe.coord2gid)

        # seam = next_layer の z- 境界。サンプルを 6 件まで表示
        samples = next_layer.get_boundary_nodes(face="z-", depth=[-1])["data"]
        print(f"seam gid comparison (first 6): count={len(samples)}")
        shown = 0
        for source in sorted(samples):
            if shown >= 6:
                break
            sink = (source[0], source[1], source[2] - 1)
            sgid = new_coord2gid.get(PhysCoordGlobal3D(source))
            tgid = new_coord2gid.get(PhysCoordGlobal3D(sink))
            ok = is_allowed_pair(sgid, tgid, allowed) if (sgid and tgid) else False
            print(f"  ({int(sgid) if sgid else None}, {int(tgid) if tgid else None}) at {source}->{sink} | allowed={ok}")
            shown += 1
    except Exception as e:
        print("[warn] seam/allowed gid probe failed:", repr(e))


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
    # for pipe in pipes:
        # canvass.add_pipe(*pipe)

    canvas = canvass.to_canvas()
    temporal_layer = canvas.to_temporal_layers()

    compiled_canvas: CompiledRHGCanvas = canvas.compile()
    nnodes = (
        len(getattr(compiled_canvas.global_graph, "physical_nodes", []) or [])
        if compiled_canvas.global_graph
        else 0
    )
    nedges = (
        len(getattr(compiled_canvas.global_graph, "physical_edges", []) or [])
        if compiled_canvas.global_graph
        else 0
    )
    print(
        {
            "layers": len(temporal_layer),
            "nodes": nnodes,
            "edges": nedges,
            "coord_map": len(compiled_canvas.coord2node),
        }
    )


def acceptance_check():
    """Temporal Pipe ON/OFF でのエッジ数差分 (d^2) を検証して出力する。"""
    d_local = 3
    canvass = RHGCanvasSkeleton("Memory X (acceptance)")
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 0, 1))
    canvass.add_cube(a, InitPlusCubeSkeleton(d=d_local, edgespec=edgespec))
    canvass.add_cube(b, MemoryCubeSkeleton(d=d_local, edgespec=edgespec))
    canvass_with = RHGCanvasSkeleton("Memory X (with pipe)")
    canvass_with.cubes_ = dict(canvass.cubes_)
    canvass_with.add_pipe(a, b, MemoryPipeSkeleton(d=d_local))

    # OFF
    canvas_off = canvass.to_canvas()
    cgraph_off: CompiledRHGCanvas = canvas_off.compile()
    edges_off = (
        len(getattr(cgraph_off.global_graph, "physical_edges", []) or [])
        if cgraph_off.global_graph
        else 0
    )

    # ON
    canvas_on = canvass_with.to_canvas()
    cgraph_on: CompiledRHGCanvas = canvas_on.compile()
    edges_on = (
        len(getattr(cgraph_on.global_graph, "physical_edges", []) or [])
        if cgraph_on.global_graph
        else 0
    )

    delta = edges_on - edges_off
    expected = d_local * d_local
    print({"edges_off": edges_off, "edges_on": edges_on, "delta": delta, "expected(d^2)": expected, "ok": (delta == expected)})

# %%
print("Temporal ON")
visualizer_connection()

# %%
print("Temporal OFF")
visualizer_noconnection()

# %%
print("Acceptance check (edges delta == d^2)")
acceptance_check()
