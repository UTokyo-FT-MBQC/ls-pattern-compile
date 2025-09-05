# 以下のコードが動くようにtemporal connectionを実行するコードを書いて
# Example

#!/usr/bin/env python3
"""T17 debug: build a small RHG canvas and compile to CompiledRHGCanvas.

Runs without external libs (stim/pymatching) and only uses local graphix_zx.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Tuple


# Ensure repo-local graphix_zx is importable
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SRC_GX = SRC / "graphix_zx"
for p in (REPO_ROOT, SRC, SRC_GX):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D


def build_skeleton() -> RHGCanvasSkeleton:
    skel = RHGCanvasSkeleton("T17 Minimal Canvas")
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_open = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}

    # minimal 1-layer, 3 cubes in a diagonal arrangement
    cubes: Dict[PatchCoordGlobal3D, InitPlusCubeSkeleton] = {
        PatchCoordGlobal3D((0, 0, 0)): InitPlusCubeSkeleton(d=3, edgespec=edgespec),
        PatchCoordGlobal3D((1, 1, 0)): InitPlusCubeSkeleton(d=3, edgespec=edgespec_open),
        PatchCoordGlobal3D((1, 1, 1)): InitPlusCubeSkeleton(d=3, edgespec=edgespec_open),
        PatchCoordGlobal3D((1, 1, 2)): InitPlusCubeSkeleton(d=3, edgespec=edgespec_open),
        PatchCoordGlobal3D((2, 2, 0)): InitPlusCubeSkeleton(d=3, edgespec=edgespec),
    }
    for pos, cube in cubes.items():
        skel.add_cube(pos, cube)
    return skel


def main() -> None:
    skel = build_skeleton()
    canvas = skel.to_canvas()

    layers = canvas.to_temporal_layers()
    # Just grab z=0
    layer0 = layers.get(0)
    if layer0 is None:
        raise RuntimeError("No layer 0 was produced")

    # force materialization (idempotent)
    layer0.materialize()
    nm = layer0.get_node_maps()
    print("Layer[0] node_maps sizes:", {k: len(v) for k, v in nm.items()})

    compiled = canvas.compile()
    nnodes = len(compiled.global_graph.physical_nodes) if compiled.global_graph else 0
    print(f"Compiled canvas: layers={len(compiled.layers)}, nodes={nnodes}")

    # quick sanity assertions
    assert compiled.global_graph is not None, "Global graph must exist"
    assert len(compiled.layers) >= 1, "At least one layer expected"
    assert nnodes > 0, "Non-empty graph expected"


if __name__ == "__main__":
    main()
