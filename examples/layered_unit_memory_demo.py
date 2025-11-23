#!/usr/bin/env python3
"""Visualization demo: RHG memory with layered unit construction.

This example demonstrates [InitPlusUnitLayer *1, MemoryUnitLayer*(d-1), MeasureX *1]
using the LayeredRHGCube objects from layered.py that build layer-by-layer from
UnitLayer sequences.
"""

# %%
import pathlib

import pymatching

from lspattern.blocks.cubes.layered import LayeredInitPlusCubeSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_to_stim
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.utils import to_edgespec
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
# Create canvas with [InitPlusUnitLayer *1, MemoryUnitLayer*(d-1), MeasureX *1] structure
d = 3

skeleton = RHGCanvasSkeleton(name=f"Layered RHG Memory (d={d})")

# Define edge specification
edgespec: dict[BoundarySide, EdgeSpecValue] = to_edgespec("ZZXX")

# Add InitPlus cube at z=0 using LayeredInitPlusCubeSkeleton
init_skeleton = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

# Add MeasureX cube
measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), measure_skeleton)

canvas = skeleton.to_canvas()
print(f"Created canvas with {len(canvas.cubes_)} cubes and {len(canvas.pipes_)} pipes")
print(f"Structure: [LayeredInitPlus(d=1) *1, LayeredMemory(d=1)*{d-1}, MeasureX *1]")

# %%
# Compile the canvas
compiled_canvas = canvas.compile()
print(f"\nCompiled canvas has {len(compiled_canvas.layers)} temporal layers")
print(f"Global graph has {getattr(compiled_canvas.global_graph, 'num_qubits', len(compiled_canvas.global_graph.physical_nodes))} qubits")

# Print schedule information
schedule = compiled_canvas.schedule.compact()
print(f"\nSchedule has {len(schedule.schedule)} time slots")
for t, nodes in schedule.schedule.items():
    print(f"Time {t}: {len(nodes)} nodes")

# %%
# Visualize the compiled canvas
fig = visualize_compiled_canvas_plotly(compiled_canvas, width=1000, height=800)
title = f"Layered RHG Memory (d={d}): [InitPlusUnitLayer *1, MemoryUnitLayer*{d-1}, MeasureX *1]"
fig.update_layout(title=title)
pathlib.Path("figures").mkdir(exist_ok=True)
output_path = "figures/layered_unit_memory_plotly.html"
fig.write_html(output_path)
fig.show()
print(f"\nVisualization saved to {output_path}")

# %%
noise = 0.001
circuit = compile_to_stim(
    compiled_canvas, logical_observable_coords={0: [PatchCoordGlobal3D((0, 0, 1))]}, p_before_meas_flip=noise,
)
print(f"\nnum_qubits: {circuit.num_qubits}")
print(circuit)

# %%
# Error correction simulation
dem = circuit.detector_error_model(decompose_errors=True)
print("\nDetector Error Model:")
print(dem)

matching = pymatching.Matching.from_detector_error_model(dem)
print(f"\nMatching object: {matching}")

err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(f"\nShortest graphlike error length: {len(err)}")
print(err)

# %%
# Visualization export
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/layered_unit_memory_dem.svg").write_text(str(svg), encoding="utf-8")
print("\nSVG diagram saved to figures/layered_unit_memory_dem.svg")
