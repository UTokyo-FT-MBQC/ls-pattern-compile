#!/usr/bin/env python3
"""New-API demo: RHG memory experiment (InitPlus -> Memory -> MeasureX).

This example builds a single-logical memory line by stacking blocks on a growing canvas.

Usage:
  python examples/rhg_memory.py
"""

# %%
import pathlib

import pymatching
import stim
from graphix_zx.stim_compiler import stim_compile
from graphix_zx.pattern import Pattern, print_pattern
import sys

sys.path.append(r"C:\Users\qipe\Documents\GitHub\ls-pattern-compile")

from lspattern.canvas2 import RHGCanvas2, CompiledRHGCanvas
from lspattern.blocks import InitPlus, Memory, MeasureX
from lspattern.visualize import visualize_canvas
from lspattern.mytype import PatchCoordGlobal3D

# %%
d = 2
r = 1

canvas = RHGCanvas2("Memory X")

blocks = [
    (PatchCoordGlobal3D(0, 0, 0), InitPlus(kind="ZXX")),
    (PatchCoordGlobal3D(0, 0, 1), MeasureX(kind="ZXX")),
]
pipes = [
    (PatchCoordGlobal3D(0, 0, 0), PatchCoordGlobal3D(0, 0, 1), Memory(kind="ZXO")),
]

for block in blocks:
    canvas.add_block(*block)
for pipe in pipes:
    canvas.add_pipe(*pipe)

layers = canvas.to_temporal_layers()

compiled_canvas = CompiledRHGCanvas(
    layers=layers,
)

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
