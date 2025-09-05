#!/usr/bin/env python3
"""T16 debug runner: build a small canvas and reach CompiledRHGCanvas.

This uses lspattern.canvas primitives directly (TemporalLayer pipeline).
"""

from __future__ import annotations

import pathlib
import sys

# Ensure repo root on sys.path for local imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lspattern.tiling.template import RotatedPlanarBlockTemplate
from lspattern.canvas import (
    CompiledRHGCanvas,
    RHGCanvasSkeleton,
    TemporalLayer,
    add_temporal_layer,
)
from lspattern.mytype import PatchCoordGlobal3D


def to_temporal_layers_from_skeleton(sk: RHGCanvasSkeleton) -> dict[int, TemporalLayer]:
    """Materialize TemporalLayers from a skeleton (blocks-only path).

    - Groups blocks by z
    - Converts skeleton blocks to blocks via to_block()
    - Calls layer.materialize() to build local graphs
    """
    if not sk.blocks_:
        return {}

    max_z = max(pos[2] for pos in sk.blocks_.keys())
    layers: dict[int, TemporalLayer] = {}
    for z in range(max_z + 1):
        layer = TemporalLayer(z)

        # Blocks at this z (minimal block-like: d + template only)
        blocks_z = {}
        for pos, skblk in sk.blocks_.items():
            if pos[2] != z:
                continue
            d = int(getattr(skblk, "d", 3))
            edgespec = getattr(skblk, "edgespec", {})
            tmpl = RotatedPlanarBlockTemplate(d=d, edgespec=edgespec)
            _ = tmpl.to_tiling()
            blocks_z[pos] = type("_B", (), {"d": d, "template": tmpl})()

        if blocks_z:
            # Directly assign to layer internals to avoid extra requirements
            layer.blocks_ = blocks_z  # type: ignore[attr-defined]

        # Build local graph/state
        layer.materialize()

        layers[z] = layer
    return layers


def main() -> None:
    d = 3
    # edgespec examples
    edgespec_all = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_open = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}

    sk = RHGCanvasSkeleton("T16 debug")
    # Place a few init blocks on z=0
    # Store skeleton-like entries directly (only d and edgespec are used here)
    B = type("_SK", (), {})
    sk.blocks_[PatchCoordGlobal3D((0, 0, 0))] = B()
    sk.blocks_[PatchCoordGlobal3D((1, 1, 0))] = B()
    sk.blocks_[PatchCoordGlobal3D((2, 2, 0))] = B()
    sk.blocks_[PatchCoordGlobal3D((0, 0, 0))].d = d
    sk.blocks_[PatchCoordGlobal3D((0, 0, 0))].edgespec = edgespec_all
    sk.blocks_[PatchCoordGlobal3D((1, 1, 0))].d = d
    sk.blocks_[PatchCoordGlobal3D((1, 1, 0))].edgespec = edgespec_open
    sk.blocks_[PatchCoordGlobal3D((2, 2, 0))].d = d
    sk.blocks_[PatchCoordGlobal3D((2, 2, 0))].edgespec = edgespec_all

    # Build layers and compile
    layers = to_temporal_layers_from_skeleton(sk)

    cgraph = CompiledRHGCanvas(
        layers=[],
        global_graph=None,
        coord2node={},
        in_portset={},
        out_portset={},
        cout_portset={},
    )
    for z in sorted(layers.keys()):
        cgraph = add_temporal_layer(cgraph, layers[z], pipes=[])

    # Report
    nnodes = len(getattr(cgraph.global_graph, "physical_nodes", []) or []) if cgraph.global_graph else 0
    print("[T16] Layers:", len(cgraph.layers))
    print("[T16] Nodes:", nnodes)
    print("[T16] Coord map size:", len(cgraph.coord2node))


if __name__ == "__main__":
    main()
