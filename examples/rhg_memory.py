#!/usr/bin/env python3
"""New-API demo: RHG memory experiment (InitPlus -> Memory -> MeasureX).

This example builds a single-logical memory line using the current codebase structure
with RHGCanvas, TemporalLayer composition, and compilation.
"""

# %%
import pathlib

import pymatching
import stim
from graphix_zx.stim_compiler import stim_compile
from graphix_zx.pattern import print_pattern

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# # %%
# # Demo 1: Create simple RHG memory canvas (InitPlus -> Memory)
# d = 3

# skeleton = RHGCanvasSkeleton(name="Simple RHG Memory Canvas")

# # Add InitPlus cube at position (0,0,0)
# edgespec = {"TOP": "X", "BOTTOM": "X", "LEFT": "Z", "RIGHT": "Z"}
# init_skeleton = InitPlusCubeSkeleton(d=d, edgespec=edgespec)
# skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

# # Add memory pipe from (0,0,0) to (0,0,1) - temporal connection
# memory_skeleton = MemoryPipeSkeleton(d=d, edgespec=edgespec)
# skeleton.add_pipe(PatchCoordGlobal3D((0, 0, 0)), PatchCoordGlobal3D((0, 0, 1)), memory_skeleton)

# simple_canvas = skeleton.to_canvas()
# print(f"Created simple canvas with {len(simple_canvas.cubes_)} cubes and {len(simple_canvas.pipes_)} pipes")

# # %%
# # Demo 2: Visualize the simple canvas
# compiled_simple = simple_canvas.compile()

# fig = visualize_compiled_canvas_plotly(compiled_simple, width=800, height=600)
# fig.update_layout(title="Simple RHG Memory Canvas")
# pathlib.Path("figures").mkdir(exist_ok=True)
# fig.write_html("figures/simple_rhg_lattice_plotly.html")
# fig.show()
# print("Plotly visualization completed and saved to figures/simple_rhg_lattice_plotly.html")

# %%
# Demo 3: Create extended memory canvas with multiple rounds
d = 3
r = 3  # number of memory rounds

skeleton = RHGCanvasSkeleton(name="Extended RHG Memory Canvas")

# Define edge specification
edgespec = {"TOP": "X", "BOTTOM": "X", "LEFT": "Z", "RIGHT": "Z"}

# Add InitPlus cube at the beginning
init_skeleton = InitPlusCubeSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

# Add memory rounds
for i in range(r):
    z_current = i
    z_next = i + 1

    memory_skeleton = MemoryPipeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_pipe(PatchCoordGlobal3D((0, 0, z_current)), PatchCoordGlobal3D((0, 0, z_next)), memory_skeleton)

measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, z_next)), measure_skeleton)

extended_canvas = skeleton.to_canvas()
print(f"Created extended canvas with {len(extended_canvas.cubes_)} cubes and {len(extended_canvas.pipes_)} pipes")

# # %%
# # Demo 4: Compile and visualize the extended canvas
compiled_canvas = extended_canvas.compile()

print(f"Compiled canvas has {len(compiled_canvas.layers)} temporal layers")
print(f"Global graph has {getattr(compiled_canvas.global_graph, 'num_qubits', 'unknown')} qubits")

print(f"Schedule has {len(compiled_canvas.schedule.schedule)} time slots")
for t, nodes in compiled_canvas.schedule.schedule.items():
    print(f"Time {t}: {len(nodes)} nodes")

# fig = visualize_compiled_canvas_plotly(compiled_canvas, width=800, height=600)
# fig.update_layout(title=f"Extended RHG Memory Canvas (d={d}, r={r})")
# pathlib.Path("figures").mkdir(exist_ok=True)
# fig.write_html("figures/extended_rhg_lattice_plotly.html")
# fig.show()
# print("Extended canvas plotly visualization completed and saved to figures/extended_rhg_lattice_plotly.html")


# %%
# Demo 5: Generate pattern from compiled canvas
xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}
x_parity = [{int(node) for node in group} for group in compiled_canvas.parity.checks]
print(f"X flow: {xflow}")
print(f"X parity: {x_parity}")

pattern = compile_canvas(
    compiled_canvas.global_graph,
    xflow=xflow,
    x_parity=x_parity,
    z_parity=[],
    # z_parity=[{int(node) for node in group} for group in compiled_canvas.parity.z_checks],
)
print("Pattern compilation successful")
# print_pattern(pattern)

logical = set(range(d))
print(f"Logical X: {logical}")
logical_observables = {0: logical}

# %%
# Demo 6: Circuit creation
def create_circuit(pattern, noise):
    print(f"Using logical observables: {logical_observables}")
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)

noise = 0.001
circuit = create_circuit(pattern, noise)
print(f"num_qubits: {circuit.num_qubits}")
# print(circuit)

# %%
# Demo 7: Error correction simulation
dem = circuit.detector_error_model(decompose_errors=True)
print(dem)

matching = pymatching.Matching.from_detector_error_model(dem)
print(matching)

err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(len(err))
print(err)

# %%
# Demo 8: Visualization export
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/rhg_memory_dem.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/rhg_memory_dem.svg")

# %%
