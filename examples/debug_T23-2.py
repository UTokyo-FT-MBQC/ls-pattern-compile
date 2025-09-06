#!/usr/bin/env python3
"""T23-2 debug: Matplotlib/Plotly visualizers for accumulators.

Build a small TemporalLayer, run accumulators, and render:
- parity/flow/schedule/detectors (Matplotlib)
- 2x2 overview (Matplotlib)
- parity/flow/schedule/detectors (Plotly) to HTML
"""

from __future__ import annotations

import pathlib
import sys


def _ensure_paths() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    src_graphix = src / "graphix_zx"
    for p in (src, src_graphix):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_ensure_paths()

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import (
    visualize_parity_mpl,
    visualize_flow_mpl,
    visualize_schedule_mpl,
    visualize_detectors_mpl,
    visualize_temporal_layer_2x2_mpl,
    visualize_parity_plotly,
    visualize_flow_plotly,
    visualize_schedule_plotly,
    visualize_detectors_plotly,
)


def build_min_layer():
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_o = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}
    skel = RHGCanvasSkeleton("T23-2-Min")
    skel.add_cube(PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec))
    skel.add_cube(PatchCoordGlobal3D((1, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec_o))
    canvas = skel.to_canvas()
    layer = canvas.to_temporal_layers()[0]
    layer.compile()

    # Run accumulators (update_at sweep) once
    ancillas = [n for n, r in layer.node2role.items() if str(r).startswith("ancilla")]
    for a in ancillas:
        layer.schedule.update_at(a, layer)
        layer.parity.update_at(a, layer)
        layer.flow.update_at(a, layer)
    return layer


def main() -> int:
    outdir = pathlib.Path("figures")
    outdir.mkdir(exist_ok=True)

    layer = build_min_layer()

    # Matplotlib outputs
    visualize_parity_mpl(layer, show=False, save_path=str(outdir / "t23-2_parity.png"))
    visualize_flow_mpl(layer, show=False, save_path=str(outdir / "t23-2_flow.png"))
    visualize_schedule_mpl(layer, mode="hist", show=False, save_path=str(outdir / "t23-2_schedule.png"))
    visualize_detectors_mpl(layer, show=False, save_path=str(outdir / "t23-2_detectors.png"))
    visualize_temporal_layer_2x2_mpl(layer, show=False, save_path=str(outdir / "t23-2_overview.png"))

    # Plotly outputs (HTML)
    try:
        fig = visualize_parity_plotly(layer)
        fig.write_html(str(outdir / "t23-2_parity.html"))
        fig = visualize_flow_plotly(layer)
        fig.write_html(str(outdir / "t23-2_flow.html"))
        fig = visualize_schedule_plotly(layer, mode="hist")
        fig.write_html(str(outdir / "t23-2_schedule.html"))
        fig = visualize_detectors_plotly(layer)
        fig.write_html(str(outdir / "t23-2_detectors.html"))
    except Exception as e:
        print(f"Plotly not available or failed: {e}")

    print({
        "ok": True,
        "saved": [p.name for p in outdir.iterdir()],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

