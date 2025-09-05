#!/usr/bin/env python3
"""New-API demo: RHG memory experiment (InitPlus -> Memory -> MeasureX).

This example builds a single-logical memory line by stacking blocks on a growing canvas.

Usage:
  python examples/rhg_memory.py
"""

# %%
import pathlib
import sys

import pymatching
import stim

from graphix_zx.pattern import Pattern
from graphix_zx.stim_compiler import stim_compile
from lspattern.blocks.initialize import InitPlusBlockSkeleton

sys.path.append(r"C:\Users\qipe\Documents\GitHub\ls-pattern-compile")

from lspattern.canvas2 import CompiledRHGCanvas, RHGCanvas2
from lspattern.mytype import PatchCoordGlobal3D

# %%
d = 3
r = 3

canvas = RHGCanvas2("Memory X")

edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
edgespec_trimmed = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}
# tmpl = RotatedPlanarTemplate(d=3, edgespec=edgespec)
# _ = tmpl.to_tiling()
blocks = [
    (PatchCoordGlobal3D((0, 0, 0)), InitPlusBlockSkeleton(d=3, edgespec=edgespec)),
    (PatchCoordGlobal3D((1, 1, 0)), InitPlusBlockSkeleton(d=3, edgespec=edgespec_trimmed)),
    (PatchCoordGlobal3D((2, 2, 0)), InitPlusBlockSkeleton(d=3, edgespec=edgespec)),
]
pipes = []

for block in blocks:
    canvas.add_block(*block)
for pipe in pipes:
    canvas.add_pipe(*pipe)

compiled_canvas = CompiledRHGCanvas(
    layers=layers,
)
pattern = compiled_canvas.compile()
# %%
stim_str = stim_compile(
    pattern,
    logical_observables,
    after_clifford_depolarization=0,
    before_measure_flip_probability=0,
)
print(stim_str)

# %%


def create_circuit(pattern: Pattern, noise: float) -> stim.Circuit:
    logical_observables = {0: {i for i in range(d)}}
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)


# %%

noise = 0.001
circuit = create_circuit(pattern, noise)
print(f"num_qubits: {circuit.num_qubits}")

dem = circuit.detector_error_model(decompose_errors=True)
print(dem)

# %%

matching = pymatching.Matching.from_detector_error_model(dem)
print(matching)
# matching.draw()


# %%
err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(len(err))
print(err)

# %%
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures/rhg_memory_dem.svg").write_text(str(svg), encoding="utf-8")

# %%
