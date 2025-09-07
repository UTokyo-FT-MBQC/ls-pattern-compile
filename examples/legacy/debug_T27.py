#!/usr/bin/env python3
"""T27: Visualizer XY aspect equality check (Matplotlib/Plotly, schedule slices).

Build a tiny canvas and first temporal layer, populate schedule via update_at,
and produce 2D XY scatter (schedule slices) with enforced equal aspect in both
Matplotlib and Plotly visualizers.
"""

from __future__ import annotations

import pathlib
import sys


def _ensure_paths() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    gzx = src / "graphix_zx"
    for p in (src, gzx):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


_ensure_paths()

from lspattern.canvas import RHGCanvas, RHGCanvasSkeleton
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers.accumulators import (
    visualize_schedule_mpl,
    visualize_schedule_plotly,
)


def build_min_layer():
    # Two cubes connected by a pipe ensures non-empty schedule groups
    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    sk = RHGCanvasSkeleton("T27 schedule slices")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=None))
    canvas: RHGCanvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    layer0 = layers[min(layers.keys())]
    layer0.compile()

    # Populate schedule by updating at ancilla nodes
    ancillas = [n for n, r in layer0.node2role.items() if str(r).startswith("ancilla")]
    for a_n in ancillas:
        layer0.schedule.update_at(a_n, layer0)
    return layer0


def main() -> None:
    layer = build_min_layer()

    # Matplotlib: schedule slices equal XY
    out_png = pathlib.Path(__file__).with_name("fig_T27_schedule_slices_mpl.png")
    visualize_schedule_mpl(layer, mode="slices", show=False, save_path=str(out_png))
    print({"mpl_saved": str(out_png.name)})

    # Plotly: schedule slices equal XY (saved as HTML for portability)
    try:
        fig = visualize_schedule_plotly(layer, mode="slices")
        out_html = pathlib.Path(__file__).with_name("fig_T27_schedule_slices_plotly.html")
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        print({"plotly_saved": str(out_html.name)})
    except Exception as e:
        # Plotly may be unavailable in some environments; treat as optional
        print({"plotly_skipped": True, "reason": str(e)})


if __name__ == "__main__":
    main()

