"""
Merge and Split
"""

# %%
import hashlib
import pathlib

import pymatching
import stim
from graphqomb.pattern import Pattern, print_pattern
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

from lspattern.blocks.cubes.initialize import InitPlusCubeThinLayerSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitZeroPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.blocks.pipes.measure import MeasureZPipeSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
d = 3


canvass = RHGCanvasSkeleton("Merge and Split XX")

edgespec: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec1: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec2: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec_trimmed: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec_measure_trimmed: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.O,
    BoundarySide.BOTTOM: EdgeSpecValue.O,
}
blocks = [
    (
        PatchCoordGlobal3D((0, 0, 0)),
        InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 0)),
        InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 2)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec1),
    ),
    (
        PatchCoordGlobal3D((1, 0, 2)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec2),
    ),
    (
        PatchCoordGlobal3D((0, 0, 3)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 3)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 4)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 4)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
]
pipes = [
    (
        PatchCoordGlobal3D((0, 0, 2)),
        PatchCoordGlobal3D((1, 0, 2)),
        InitZeroPipeSkeleton(d=d, edgespec=edgespec_trimmed),
    ),
    (
        PatchCoordGlobal3D((0, 0, 3)),
        PatchCoordGlobal3D((1, 0, 3)),
        MeasureZPipeSkeleton(d=d, edgespec=edgespec_measure_trimmed),
    ),
]

for block in blocks:
    canvass.add_cube(*block)
for pipe in pipes:
    canvass.add_pipe(*pipe)

canvas = canvass.to_canvas()

compiled_canvas: CompiledRHGCanvas = canvas.compile()
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
output_indices = compiled_canvas.global_graph.output_node_indices or {}  # type: ignore[union-attr]
print(f"output qubits: {output_indices}")

fig3d = visualize_compiled_canvas_plotly(
    compiled_canvas,
    show_edges=True,
)
fig3d.show()

# %%

# Print flow and parity information
xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}
x_parity = []
for group_dict in compiled_canvas.parity.checks.values():
    for group in group_dict.values():
        x_parity.append({int(node) for node in group})

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
cout_portmap = compiled_canvas.cout_portset
print(f"Classical output ports: {cout_portmap}")


# %%
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
    xflow=xflow,
    x_parity=x_parity,
    z_parity=[],
    scheduler=scheduler,
)
print("Pattern compilation successful")
print_pattern(pattern)

# set logical observables
coord2logical_group = {
    # 0: {PatchCoordGlobal3D((0, 0, 3)), PatchCoordGlobal3D((1, 0, 3))},  # First output patch
    0: {PatchCoordGlobal3D((0, 0, 4))},
    # 1: {PatchCoordGlobal3D((1, 0, 3))},  # Second output patch
}
logical_observables = {}
for i, group in coord2logical_group.items():
    nodes = []
    for coord in group:
        if coord in cout_portmap:
            nodes.extend(cout_portmap[coord])
    logical_observables[i] = set(nodes)


# %%
# Circuit creation
def create_circuit(pattern: Pattern, noise: float) -> stim.Circuit:
    print(f"Using logical observables: {logical_observables}")
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=0,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)


noise = 0.001
circuit = create_circuit(pattern, noise)
fingerprint = hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()
print(f"circuit fingerprint: {fingerprint}")
print(f"num_qubits: {circuit.num_qubits}")
# print(circuit)

# %%
# Error correction simulation
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

# %%
# Visualization export
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/merge_split_dem_xx.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/merge_split_xx_dem.svg")


# %%
