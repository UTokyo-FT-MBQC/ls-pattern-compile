#!/usr/bin/env python3
"""T31: RHGBlock.materialize() 内での RHG 構築の動作確認デモ。

単一のキューブ Block を直接 materialize() し、local_graph と座標写像が
正しく生成されているかを簡易に検査する。
"""

from __future__ import annotations

import pathlib
import sys


# Ensure repo-local imports (lspattern, graphix_zx)
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GRAPHIX = SRC / "graphix_zx"
for p in (ROOT, SRC, SRC_GRAPHIX):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def main() -> None:
    from lspattern.blocks.cubes.base import RHGCube

    # キューブ1個を直接生成して materialize()
    d = 3
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    blk = RHGCube(d=d, edge_spec=edgespec)
    blk.materialize()

    g = blk.local_graph
    node2coord = blk.node2coord
    node2role = blk.node2role

    nnodes = len(getattr(g, "physical_nodes", []) or []) if g else 0
    nedges = len(getattr(g, "physical_edges", []) or []) if g else 0

    print("[T31] block materialized:")
    print({
        "d": d,
        "nodes": nnodes,
        "edges": nedges,
        "coords": len(node2coord),
        "roles": len(node2role),
    })

    # 簡単な健全性
    assert g is not None, "local_graph is None"
    assert nnodes == len(node2coord), "node2coord size must match node count"
    # 役割情報（data/ancilla_x/ancilla_z）が少なくとも一部に付与されているはず
    assert any(r in ("ancilla_x", "ancilla_z") for r in node2role.values()), (
        "ancilla roles not found; check interleaving parity logic"
    )

    print("[T31] OK: RHG graph built within RHGBlock.materialize()")


if __name__ == "__main__":
    main()

