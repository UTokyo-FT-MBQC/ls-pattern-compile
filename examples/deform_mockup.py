# %%
import pathlib

import pymatching
import stim
from graphqomb.pattern import Pattern, print_pattern
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

from lspattern.blocks.cubes.initialize import (
    InitZeroCubeThinLayerSkeleton,
)
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton, MeasureXSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly, visualize_compiled_canvas

from lspattern.blocks.cubes.initialize import InitPlusCubeThinLayerSkeleton

d = 3

canvass = RHGCanvasSkeleton("Patch Translate via Deform (extend+shrink)")

# Keep the same edge types throughout (rough/smooth unchanged)
edgespec = {
    BoundarySide.LEFT: EdgeSpecValue.X,
    BoundarySide.RIGHT: EdgeSpecValue.X,
    BoundarySide.TOP: EdgeSpecValue.Z,
    BoundarySide.BOTTOM: EdgeSpecValue.Z,
}

edgespec1 = {
    BoundarySide.LEFT: EdgeSpecValue.X,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.Z,
    BoundarySide.BOTTOM: EdgeSpecValue.Z,
}

edgespec2 = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.X,
    BoundarySide.TOP: EdgeSpecValue.Z,
    BoundarySide.BOTTOM: EdgeSpecValue.Z,
}

# Helper edgespecs for temporary trimming during merge/split
# - During extend, we open (O) the touching sides so two patches can merge
# - During shrink, we use all-O for the measurement pipe across the seam
edgespec_trimmed = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.Z,
    BoundarySide.BOTTOM: EdgeSpecValue.Z,
}
edgespec_measure_trimmed = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.O,
    BoundarySide.BOTTOM: EdgeSpecValue.O,
}

# Layout plan (x, y, z):
# z=0: init |+> at (0,0)
# z=1: memory at (0,0)
# z=2: extend to the right by merging with (1,0) (temporary memory patch appears at x=1)
# z=3: shrink by splitting with MeasureX pipe, leaving the right patch (x=1)
# z=4: measure Z at (1,0)
blocks = [
    # Initialize one patch in |+> at (0,0) on z=0
    (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)),

    # Keep it alive at z=1
    (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
    (PatchCoordGlobal3D((1, 0, 1)), InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)),

    # For the extend operation at z=2:
    # - Keep the original at (0,0)
    # - Prepare a neighbor patch at (1,0) with complementary trimming so they can merge
    (PatchCoordGlobal3D((0, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=edgespec1)),
    (PatchCoordGlobal3D((1, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=edgespec2)),

    # After extend, keep both alive at z=3 so we can perform the shrink (split) operation
    (PatchCoordGlobal3D((0, 0, 3)), MeasureXSkeleton(d=d, edgespec=edgespec)),
    (PatchCoordGlobal3D((1, 0, 3)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),

    # Finally, measure Z at the new location (1,0) on z=4
    (PatchCoordGlobal3D((1, 0, 4)), MeasureXSkeleton(d=d, edgespec=edgespec)),
]

pipes = [
    # z=2 extend to the right: merge (0,0,2) with (1,0,2)
    (
        PatchCoordGlobal3D((0, 0, 2)),
        PatchCoordGlobal3D((1, 0, 2)),
        InitPlusPipeSkeleton(d=d, edgespec=edgespec_trimmed),
    ),
    # z=3 shrink: split along X seam so the right patch remains (use all-O on the seam)
    (
        PatchCoordGlobal3D((0, 0, 3)),
        PatchCoordGlobal3D((1, 0, 3)),
        MeasureXPipeSkeleton(d=d, edgespec=edgespec_measure_trimmed),
    ),
]

for coord, cube in blocks:
    canvass.add_cube(coord, cube)
for s, t, pipe in pipes:
    canvass.add_pipe(s, t, pipe)

canvas = canvass.to_canvas()
compiled_canvas: CompiledRHGCanvas = canvas.compile()

# Quick summary
nnodes = (
    len(getattr(compiled_canvas.global_graph, "physical_nodes", []) or [])
    if compiled_canvas.global_graph
    else 0
)
nedges = (
    len(getattr(compiled_canvas.global_graph, "physical_edges", []) or [])
    if compiled_canvas.global_graph
    else 0
)

print(
    {
        "layers": len(compiled_canvas.layers),
        "nodes": nnodes,
        "edges": nedges,
        "coord_map": len(compiled_canvas.coord2node),
    }
)

# Visualization
fig3d = visualize_compiled_canvas_plotly(compiled_canvas, show_edges=True)
fig3d.show()

xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}
parity = []
for group_dict in compiled_canvas.parity.checks.values():
    for group in group_dict.values():
        parity.append({int(node) for node in group})

# Print X flow organized by schedule if available
print("X flow:")
compact_schedule = compiled_canvas.schedule.compact()
for t, nodes in compact_schedule.schedule.items():
    flows_at_time = []
    for node in nodes:
        if node in xflow:
            flows_at_time.append(f"{node} -> {xflow[node]}")
    if flows_at_time:
        print(f"  Time {t}: {', '.join(flows_at_time)}")
# Print any remaining flows not in schedule
scheduled_nodes = set()
for nodes in compact_schedule.schedule.values():
    scheduled_nodes.update(nodes)
remaining_flows = {
    src: dsts for src, dsts in xflow.items() if src not in scheduled_nodes
}
if remaining_flows:
    remaining_flow_strs = [f"{src} -> {dsts}" for src, dsts in remaining_flows.items()]
    print(f"  Unscheduled flows: {', '.join(remaining_flow_strs)}")

print("X parity")
for coord, group_list in compiled_canvas.parity.checks.items():
    print(f"  {coord}: {group_list}")

# Print detailed flow information for visualization
print("\nDetailed Flow Information:")
print(f"Total flow edges: {sum(len(dsts) for dsts in xflow.values())}")
print(f"Flow sources: {len(xflow)}")
print(
    f"Flow coverage: {len(set().union(*xflow.values()) if xflow else set())} unique destinations"
)

# Print detector/stabilizer information
print("\nDetector Information:")
total_detectors = sum(
    len(group_dict) for group_dict in compiled_canvas.parity.checks.values()
)
print(f"Total detectors: {total_detectors}")
for coord, group_dict in compiled_canvas.parity.checks.items():
    print(f"  Patch {coord}: {len(group_dict)} detector groups")

# classical outs
cout_portmap = compiled_canvas.cout_portset_cube
cout_portmap_pipe = compiled_canvas.cout_portset_pipe
print(f"Classical output ports: {cout_portmap}")
print(f"Classical output ports (pipes): {cout_portmap_pipe}")

# Pattern generation
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
                    meas_time[node] = (
                        time_slot + 1
                    )  # Shift by 1 to account for preparation at time 0
                    break

# Configure scheduler with manual timing
scheduler.manual_schedule(prepare_time=prep_time, measure_time=meas_time)

pattern = compile_canvas(
    compiled_canvas.global_graph,
    flow=xflow,
    parity=parity,
    scheduler=scheduler,
)
print("Pattern compilation successful")
print_pattern(pattern)

# set logical observables
coord2logical_group = {
    0: {PatchCoordGlobal3D((1, 0, 4)), PatchCoordGlobal3D((0, 0, 3)), PipeCoordGlobal3D((PatchCoordGlobal3D((0, 0, 3)), PatchCoordGlobal3D((1, 0, 3))))},  # Output patch, MeasureX patch + MeasureX pipe
}
logical_observables = {}
for i, group in coord2logical_group.items():
    nodes = []
    for coord in group:
        # PipeCoordGlobal3D is a 2-tuple of PatchCoordGlobal3D (nested tuples)
        # PatchCoordGlobal3D is a 3-tuple of ints
        if isinstance(coord, tuple) and len(coord) == 2 and all(isinstance(c, tuple) for c in coord):  # isinstance cannot be used with NewType
            # This is a PipeCoordGlobal3D
            nodes.extend(cout_portmap_pipe[coord])
        elif isinstance(coord, tuple) and len(coord) == 3:
            # This is a PatchCoordGlobal3D
            nodes.extend(cout_portmap[coord])
        else:
            msg = f"Unknown coord type: {type(coord)}"
            raise TypeError(msg)

    logical_observables[i] = set(nodes)

fig3d = visualize_compiled_canvas_plotly(compiled_canvas, show_edges=True, hilight_nodes=logical_observables[0])
fig3d.show()

def create_circuit(pattern: Pattern, noise: float) -> stim.Circuit:
    print(f"Using logical observables: {logical_observables}")
    stim_str = stim_compile(
        pattern,
        logical_observables,
        p_depol_after_clifford=0,
        p_before_meas_flip=noise,
    )
    return stim.Circuit(stim_str)


noise = 0.001
circuit = create_circuit(pattern, noise)
print(f"num_qubits: {circuit.num_qubits}")
# print(circuit)

try:
    dem = circuit.detector_error_model(decompose_errors=True)
    print(dem)
except ValueError as e:
    print(f"Error creating DEM with decompose_errors=True: {e}")
    print("Retrying without decompose_errors...")
    dem = circuit.detector_error_model(decompose_errors=False)
    print(dem)

matching = pymatching.Matching.from_detector_error_model(dem)
print(matching)

err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(f"Shortest error length: {len(err)}")
print(err)

svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/deform_dem.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/deform_dem.svg")
