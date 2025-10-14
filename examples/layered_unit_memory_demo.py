#!/usr/bin/env python3
"""Visualization demo: RHG memory with layered unit construction.

This example demonstrates [InitPlusUnitLayer *1, MemoryUnitLayer*(d-1), MeasureX *1]
using the LayeredRHGCube objects from layered.py that build layer-by-layer from
UnitLayer sequences.
"""

# %%
import pathlib

from graphix_zx.pattern import print_pattern
from graphix_zx.scheduler import Scheduler

from lspattern.blocks.cubes.layered import LayeredInitPlusCubeSkeleton, LayeredMemoryCubeSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
# Create canvas with [InitPlusUnitLayer *1, MemoryUnitLayer*(d-1), MeasureX *1] structure
d = 3

skeleton = RHGCanvasSkeleton(name=f"Layered RHG Memory (d={d})")

# Define edge specification
edgespec: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
}

# Add InitPlus cube at z=0 using LayeredInitPlusCubeSkeleton (d=1 for 1 unit layer)
init_skeleton = LayeredInitPlusCubeSkeleton(d=1, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

# Add Memory cubes at z=1 to z=(d-1) using LayeredMemoryCubeSkeleton (d=1 each for 1 unit layer)
for i in range(1, d):
    memory_skeleton = LayeredMemoryCubeSkeleton(d=1, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, i)), memory_skeleton)

# Add MeasureX cube at z=d
measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, d)), measure_skeleton)

canvas = skeleton.to_canvas()
print(f"Created canvas with {len(canvas.cubes_)} cubes and {len(canvas.pipes_)} pipes")
print(f"Structure: [LayeredInitPlus(d=1) *1, LayeredMemory(d=1)*{d-1}, MeasureX *1]")

# %%
# Compile the canvas
compiled_canvas = canvas.compile()
print(f"\nCompiled canvas has {len(compiled_canvas.layers)} temporal layers")
print(f"Global graph has {getattr(compiled_canvas.global_graph, 'num_qubits', 'unknown')} qubits")

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
# Generate pattern from compiled canvas
if compiled_canvas.global_graph is None:
    raise ValueError("Global graph is None")

xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}

x_parity = []
for group_dict in compiled_canvas.parity.checks.values():
    for group in group_dict.values():
        x_parity.append({int(node) for node in group})

print(f"\nX flow has {len(xflow)} entries")
print(f"X parity has {len(x_parity)} checks")

output_indices = compiled_canvas.global_graph.output_node_indices or {}
print(f"Output qubits: {output_indices}")

# Create scheduler
scheduler = Scheduler(compiled_canvas.global_graph, xflow=xflow)

# Set up timing based on compiled_canvas.schedule
compact_schedule = compiled_canvas.schedule.compact()

# Initialize prepare_time and measure_time dictionaries
prep_time = {}
meas_time = {}

# Set input nodes to have preparation at time 0
input_nodes = set(compiled_canvas.global_graph.input_node_indices.keys())
for node in compiled_canvas.global_graph.physical_nodes:
    if node not in input_nodes:
        prep_time[node] = 0  # Non-input nodes prepared at time 0

# Set measurement times based on schedule
output_nodes = set(output_indices.keys())
for node in compiled_canvas.global_graph.physical_nodes:
    if node not in output_nodes:
        # Find when this node is scheduled for measurement
        meas_time[node] = 1  # Default measurement time
        for time_slot, nodes in compact_schedule.schedule.items():
            if node in nodes:
                meas_time[node] = time_slot + 1  # Shift by 1 to account for preparation at time 0
                break

# Configure scheduler with manual timing
scheduler.manual_schedule(prepare_time=prep_time, measure_time=meas_time)

pattern = compile_canvas(
    compiled_canvas.global_graph,
    xflow=xflow,
    x_parity=x_parity,
    z_parity=[],
    scheduler=scheduler,
)
print("\nPattern compilation successful")
print_pattern(pattern)

# Set logical observables
cout_portmap = compiled_canvas.cout_portset
coord2logical_group = {0: PatchCoordGlobal3D((0, 0, d))}  # MeasureX cube is at position (0, 0, d)
logical_observables = {i: cout_portmap[coord] for i, coord in coord2logical_group.items()}
print(f"\nUsing logical observables: {logical_observables}")

print("\n" + "="*80)
print(f"Demo completed: [InitPlusUnitLayer *1, MemoryUnitLayer*{d-1}, MeasureX *1]")
print(f"Total temporal layers: {len(compiled_canvas.layers)}")
print(f"Total qubits: {getattr(compiled_canvas.global_graph, 'num_qubits', 'unknown')}")
print("="*80)
