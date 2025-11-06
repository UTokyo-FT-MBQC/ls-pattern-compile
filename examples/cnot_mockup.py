"""
Merge and Split
"""

# %%
import pathlib

import pymatching
from lspattern.blocks.cubes.initialize import (
    InitPlusCubeThinLayerSkeleton,
    InitZeroCubeThinLayerSkeleton,
    InitPlusCubeSkeleton,
)
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import (
    InitPlusPipeSkeleton,
    InitZeroPipeSkeleton,
    InitZeroPipeThinLayerSkeleton,
)
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton, MeasureZPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton, MeasureXSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.compile import compile_to_stim
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.utils import to_edgespec
from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
d = 3

canvass = RHGCanvasSkeleton("CNOT")

edgespec: dict[BoundarySide, EdgeSpecValue] = to_edgespec("ZZXX")
blocks = [
    # Clock 0 (init Zero)
    (
        PatchCoordGlobal3D((0, 0, 0)),
        InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 0)),
        InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    # (
    #     PatchCoordGlobal3D((1, 1, 0)),
    #     InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    # ),
    # Clock 1 (Memory)
    (
        PatchCoordGlobal3D((0, 0, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    # (
    #     PatchCoordGlobal3D((1, 1, 1)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # Clock 2 (Merge ZZ)
    (
        PatchCoordGlobal3D((0, 0, 2)),
        MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZOX")),
    ),
    (
        PatchCoordGlobal3D((0, 1, 2)),
        MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZXO")),
    ),
    # (
    #     PatchCoordGlobal3D((1, 1, 2)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    (
        PatchCoordGlobal3D((0, 0, 3)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 3)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    # (
    #     PatchCoordGlobal3D((1, 1, 3)),
    #     MeasureXSkeleton(d=d, edgespec=edgespec),
    # )
    # # Clock 3 (Split and Merge XX)
    # (
    #     PatchCoordGlobal3D((0, 0, 3)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((0, 1, 3)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 1, 3)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # # Clock 4 (Split and Memory)
    # (
    #     PatchCoordGlobal3D((0, 0, 4)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((0, 1, 4)),
    #     MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZOXX")),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 1, 4)),
    #     MemoryCubeSkeleton(d=d, edgespec=to_edgespec("OZXX")),
    # ),
    # # Clock 5 (Memory)
    # (
    #     PatchCoordGlobal3D((0, 0, 5)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((0, 1, 5)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 1, 5)),
    #     MemoryCubeSkeleton(d=d, edgespec=edgespec),
    # ),
    # # Clock 6 (Measure Z all)
    # (
    #     PatchCoordGlobal3D((0, 0, 6)),
    #     MeasureZSkeleton(d=d, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((0, 1, 6)),
    #     MeasureXSkeleton(d=d, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 1, 6)),
    #     MeasureXSkeleton(d=d, edgespec=edgespec),
    # ),
]
pipes = [
    # Clock 2 (Merge ZZ -> Split ZZ)
    (
        PatchCoordGlobal3D((0, 0, 2)),
        PatchCoordGlobal3D((0, 1, 2)),
        InitPlusPipeSkeleton(d=d, edgespec=to_edgespec("ZZOO")),
    ),
    # # Clock 3 (Split XX)
    (
        PatchCoordGlobal3D((0, 0, 3)),
        PatchCoordGlobal3D((0, 1, 3)),
        MeasureXPipeSkeleton(d=d, edgespec=to_edgespec("OOOO")),
    ),
    # # Clock 4
    # (
    #     PatchCoordGlobal3D((0, 1, 4)),
    #     PatchCoordGlobal3D((1, 1, 4)),
    #     InitZeroPipeSkeleton(d=d, edgespec=to_edgespec("OOXX")),
    # ),
    # # Clock 5
    # (
    #     PatchCoordGlobal3D((0, 1, 5)),
    #     PatchCoordGlobal3D((1, 1, 5)),
    #     MeasureZPipeSkeleton(d=d, edgespec=to_edgespec("OOOO")),
    # ),
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
included = 390 in compiled_canvas.global_graph.physical_nodes
print(f"Node 390 included: {included}")
included = 393 in compiled_canvas.global_graph.physical_nodes
print(f"Node 393 included: {included}")
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
    compiled_canvas, show_edges=True
)
fig3d.show()

# %%

# Print flow and parity information
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


# %%
# Circuit creation using compile_to_stim
noise = 0.001
circuit = compile_to_stim(
    compiled_canvas,
    logical_observable_coords={
        0: [
            # PatchCoordGlobal3D((0, 1, 6)),
            # PipeCoordGlobal3D(
            #     (PatchCoordGlobal3D((0, 0, 3)), PatchCoordGlobal3D((1, 0, 3)))
            # ),
        ],
        # 1: [PatchCoordGlobal3D((1, 1, 4))],  # Second output patch
        # 2: [PatchCoordGlobal3D((1, 1, 5))],  # Third output patch
        # 3: [PipeCoordGlobal3D((PatchCoordGlobal3D((0, 0, 3)), PatchCoordGlobal3D((0, 1, 3))))],  # InitPlus pipe
        # 4: [PipeCoordGlobal3D((PatchCoordGlobal3D((0, 1, 4)), PatchCoordGlobal3D((1, 1, 4))))],  # InitPlus pipe
    },
    p_before_meas_flip=noise,
)
print(f"num_qubits: {circuit.num_qubits}")
# print(circuit)

# %%
# Error correction simulation
try:
    dem = circuit.detector_error_model(decompose_errors=True)
    # print(dem)
except ValueError as e:
    print(f"Error creating DEM with decompose_errors=True: {e}")
    print("Retrying without decompose_errors...")
    dem = circuit.detector_error_model(decompose_errors=False)
    # print(dem)

matching = pymatching.Matching.from_detector_error_model(dem)
print(matching)

err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(f"Shortest error length: {len(err)}")
print(err)

# %%
# Visualization export
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/merge_split_dem.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/merge_split_dem.svg")


# %%
