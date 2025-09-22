#!/usr/bin/env python3
"""New-API demo: RHG memory experiment (InitPlus -> Memory -> MeasureX).

This example builds a single-logical memory line by stacking blocks on a growing canvas.

Usage:
  python examples/rhg_memory.py
"""

# %%
import pathlib
import sys
from typing import Literal

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.base import RHGPipeSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvas, RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D

# %%
d = 3
r = 3

canvass = RHGCanvasSkeleton("Memory X")

edgespec: dict[str, Literal["X", "Z", "O"]] = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
edgespec_trimmed: dict[str, Literal["X", "Z", "O"]] = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}
# tmpl = RotatedPlanarTemplate(d=3, edgespec=edgespec)
# _ = tmpl.to_tiling()
blocks = [
    (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
    (PatchCoordGlobal3D((1, 1, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec_trimmed)),
    (PatchCoordGlobal3D((2, 2, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
]
pipes: list[tuple[PatchCoordGlobal3D, PatchCoordGlobal3D, RHGPipeSkeleton]] = []

for block in blocks:
    # RHGCanvasSkeleton は skeleton を受け取り、to_canvas() で block 化します
    canvass.add_cube(*block)
for pipe in pipes:
    canvass.add_pipe(*pipe)

canvas = canvass.to_canvas()
temporal_layer = canvas.to_temporal_layers()

# TODO: make a visualizer of the temporal_layer[0] like the notebooks in the examples/* directory
# Make a separate ipynb: debug_temporal_layers.ipynb (T16.md)

# Compile canvas to global graph; downstream stim compile is optional
compiled_canvas: CompiledRHGCanvas = canvas.compile()
nnodes = len(getattr(compiled_canvas.global_graph, "physical_nodes", []) or []) if compiled_canvas.global_graph else 0
print({
    "layers": len(temporal_layer),
    "nodes": nnodes,
    "coord_map": len(compiled_canvas.coord2node),
})

# %%
sys.exit(0)  # Skip stim/pymatching path below


# %%

if False:  # Optional stim/pymatching path (requires a defined Pattern)
    noise = 0.001
    circuit = create_circuit(pattern, noise)  # type: ignore  # create_circuit and pattern are not defined in this demo
    print(f"num_qubits: {circuit.num_qubits}")

    dem = circuit.detector_error_model(decompose_errors=True)
    print(dem)

    matching = pymatching.Matching.from_detector_error_model(dem)  # type: ignore  # pymatching not imported
    print(matching)
    err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
    print(len(err))
    print(err)
    svg = dem.diagram(type="match-graph-svg")
    pathlib.Path("figures/rhg_memory_dem.svg").write_text(str(svg), encoding="utf-8")

# %%
