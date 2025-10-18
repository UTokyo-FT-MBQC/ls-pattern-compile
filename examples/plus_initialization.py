#!/usr/bin/env python3
"""New-API demo: RHG memory experiment (InitPlus -> Memory -> MeasureX).

This example builds a single-logical memory line using the current codebase structure
with RHGCanvas, TemporalLayer composition, and compilation.
"""

# %%
import pathlib

import pymatching
import stim
from graphqomb.pattern import Pattern, print_pattern
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

from lspattern.blocks.cubes.initialize import InitPlusCubeThinLayerSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
# Demo 3: Create extended memory canvas with multiple rounds
d = 3

skeleton = RHGCanvasSkeleton(name="Extended RHG Memory Canvas")

# Define edge specification
edgespec: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X, BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.Z}

# Add InitPlus cube at the beginning
# init_skeleton = InitPlusCubeSkeleton(d=d, edgespec=edgespec)
init_skeleton = InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

memory_skeleton = MemoryCubeSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), memory_skeleton)

measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 2)), measure_skeleton)

extended_canvas = skeleton.to_canvas()
print(f"Created extended canvas with {len(extended_canvas.cubes_)} cubes and {len(extended_canvas.pipes_)} pipes")

# # %%
# Demo 4: Compile and visualize the extended canvas
compiled_canvas = extended_canvas.compile()
print(f"Compiled canvas has {len(compiled_canvas.layers)} temporal layers")
print(f"Global graph has {getattr(compiled_canvas.global_graph, 'num_qubits', 'unknown')} qubits")

schedule = compiled_canvas.schedule.compact()
print(f"Schedule has {len(schedule.schedule)} time slots")
for t, nodes in schedule.schedule.items():
    print(f"Time {t}: {nodes}")

fig = visualize_compiled_canvas_plotly(compiled_canvas, width=800, height=600)
fig.update_layout(title=f"Extended RHG Memory Canvas (d={d})")
pathlib.Path("figures").mkdir(exist_ok=True)
fig.write_html("figures/extended_rhg_lattice_plotly.html")
fig.show()
print("Extended canvas plotly visualization completed and saved to figures/extended_rhg_lattice_plotly.html")


# %%
# Demo 5: Generate pattern from compiled canvas
xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}
parity = []
for group_dict in compiled_canvas.parity.checks.values():
    for group in group_dict.values():
        parity.append({int(node) for node in group})
print(f"X flow: {xflow}")
print("X parity")
for coord, group_list in compiled_canvas.parity.checks.items():  # type: ignore[assignment]
    print(f"  {coord}: {group_list}")


output_indices_main = compiled_canvas.global_graph.output_node_indices or {}  # type: ignore[union-attr]
print(f"output qubits: {output_indices_main}")

# Create scheduler
scheduler = Scheduler(compiled_canvas.global_graph, xflow=xflow)

# Set up timing based on compiled_canvas.schedule
compact_schedule = compiled_canvas.schedule.compact()
print(f"Schedule has {len(compact_schedule.schedule)} time slots")

# Initialize prepare_time and measure_time dictionaries
prep_time = {}
meas_time = {}

# Set input nodes to have no preparation time (None)
if compiled_canvas.global_graph is not None:
    input_nodes = set(compiled_canvas.global_graph.input_node_indices.keys())
    for node in compiled_canvas.global_graph.physical_nodes:
        if node not in input_nodes:
            prep_time[node] = 0  # Non-input nodes prepared at time 0

    # Set measurement times based on schedule
    output_indices = compiled_canvas.global_graph.output_node_indices or {}
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
    parity=parity,
    scheduler=scheduler,
)
print("Pattern compilation successful")
print_pattern(pattern)

# set logical observables
cout_portmap = compiled_canvas.cout_portset
coord2logical_group = {0: PatchCoordGlobal3D((0, 0, 2))}
logical_observables = {i: cout_portmap[coord] for i, coord in coord2logical_group.items()}
print(f"Using logical observables: {logical_observables}")


# %%
# Demo 6: Circuit creation
def create_circuit(pattern: Pattern, noise: float) -> stim.Circuit:
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
print(circuit)

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
