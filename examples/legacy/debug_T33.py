#!/usr/bin/env python3
"""T33: TemporalLayer.compile uses compose_in_parallel on pre-materialized blocks.

Build a tiny canvas with two cubes and one spatial pipe, compile, and
print basic stats to verify the layer graph comes from parallel composition
without allocating nodes inside compile().
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
    from lspattern.canvas import RHGCanvas
    from lspattern.blocks.cubes.base import RHGCube
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton

    # Two adjacent cubes at z=0, connected by a spatial pipe (x-direction)
    d = 3
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

    c1 = RHGCube(d=d, edge_spec=edgespec)
    c2 = RHGCube(d=d, edge_spec=edgespec)
    psk = InitPlusPipeSkeleton(d=d)
    pipe = psk.to_block((0, 0, 0), (1, 0, 0))

    canvas = RHGCanvas(name="T33-canvas")
    canvas.add_cube((0, 0, 0), c1)
    canvas.add_cube((1, 0, 0), c2)
    canvas.add_pipe((0, 0, 0), (1, 0, 0), pipe)

    compiled = canvas.compile()

    g = compiled.global_graph
    nnodes = len(getattr(g, "physical_nodes", []) or []) if g else 0
    nedges = len(getattr(g, "physical_edges", []) or []) if g else 0
    print(
        "[T33] compiled:",
        {
            "layers": [tl.z for tl in compiled.layers],
            "nodes": nnodes,
            "edges": nedges,
            "coord2node": len(compiled.coord2node),
        },
    )


if __name__ == "__main__":
    main()

