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
from lspattern.canvas import RHGCanvas
from lspattern.blocks import InitPlus, Memory, MeasureX
from lspattern.visualize import visualize_canvas

# %%
d = 2
r = 1

canvas = RHGCanvas()
canvas.append(InitPlus(logical=0, dx=d, dy=d))
visualize_canvas(
    canvas,
    show=True,
)

# %%
canvas = RHGCanvas()
canvas.append(InitPlus(logical=0, dx=d, dy=d))
canvas.append(Memory(logical=0, rounds=r))
visualize_canvas(
    canvas,
    save_path="figures/rhg_lattice.png",
    show=True,
)

# %%
canvas = RHGCanvas()
canvas.append(InitPlus(logical=0, dx=d, dy=d))
canvas.append(Memory(logical=0, rounds=r))
canvas.append(Memory(logical=0, rounds=r))
canvas.append(MeasureX(logical=0))
visualize_canvas(
    canvas,
    save_path="figures/rhg_lattice.png",
    show=True,
)

# %%
d = 5
r = 5

canvas = RHGCanvas()
canvas.append(InitPlus(logical=0, dx=d, dy=d))
canvas.append(Memory(logical=0, rounds=r))
canvas.append(Memory(logical=0, rounds=r))
canvas.append(MeasureX(logical=0))

for group in canvas.schedule_accum.measure_groups:
    print(f"group: {group}")

logical = set(i for i in range(d))
print(f"logical X: {logical}")
logical_observables = {0: logical}

# %%
pattern = canvas.compile()
print_pattern(pattern)

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
