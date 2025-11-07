"""
Merge and Split
"""

# %%
import pathlib

import pymatching
from lspattern.blocks.cubes.initialize import (
    InitZeroCubeThinLayerSkeleton,
    InitPlusCubeThinLayerSkeleton,
)
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton, MeasureZPipeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton, MeasureXSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.compile import compile_to_stim
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
d = 3

canvass = RHGCanvasSkeleton("Merge and Split")

edgespec: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
edgespec_inv: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}

edgespec_LEFT0: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
edgespec_RIGHT0: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}

edgespec_LEFT0_BOTTOMX: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.X}

edgespec_LEFT0_BOTTOMX_RIGHTZ: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.X}

edgespec_meas2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.X}

edgespec_pipe: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
edgespec_pipe_measure: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.O}

blocks = [

    (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)),
    
    (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
    (PatchCoordGlobal3D((1, 0, 1)), InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)),
    
    (PatchCoordGlobal3D((0, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=edgespec_RIGHT0)),
    (PatchCoordGlobal3D((1, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=edgespec_LEFT0)),
    
    (PatchCoordGlobal3D((0, 0, 3)), MemoryCubeSkeleton(d=d, edgespec=edgespec_RIGHT0)),
    (PatchCoordGlobal3D((1, 0, 3)), MemoryCubeSkeleton(d=d, edgespec=edgespec_LEFT0_BOTTOMX)),
    
    (PatchCoordGlobal3D((0, 0, 4)), MemoryCubeSkeleton(d=d, edgespec=edgespec_RIGHT0)),
    (PatchCoordGlobal3D((1, 0, 4)), MemoryCubeSkeleton(d=d, edgespec=edgespec_LEFT0_BOTTOMX_RIGHTZ)),
    
    (PatchCoordGlobal3D((0, 0, 5)), MeasureXSkeleton(d=d, edgespec=edgespec)),
    (PatchCoordGlobal3D((1, 0, 5)), MeasureXSkeleton(d=d, edgespec=edgespec_meas2, disable_cout=True)),
]

pipes = [
    (
        PatchCoordGlobal3D((0, 0, 2)),
        PatchCoordGlobal3D((1, 0, 2)),
        InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe),
    ),
    (
        PatchCoordGlobal3D((0, 0, 3)),
        PatchCoordGlobal3D((1, 0, 3)),
        MemoryPipeSkeleton(d=d, edgespec=edgespec_pipe),
    ),
    (
        PatchCoordGlobal3D((0, 0, 4)),
        PatchCoordGlobal3D((1, 0, 4)),
        MemoryPipeSkeleton(d=d, edgespec=edgespec_pipe),
    ),
    (
        PatchCoordGlobal3D((0, 0, 5)),
        PatchCoordGlobal3D((1, 0, 5)),
        MeasureXPipeSkeleton(d=d, edgespec=edgespec_pipe_measure),
    )
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

fig3d = visualize_compiled_canvas_plotly(compiled_canvas, show_edges=True)
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
coord2logical_group = {
    0: {PatchCoordGlobal3D((0, 0, 5)), PipeCoordGlobal3D((PatchCoordGlobal3D((0, 0, 5)), PatchCoordGlobal3D((1, 0, 5))))}
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

    logical_observables[i] = set(nodes).union({697})
    
circuit = compile_to_stim(
    compiled_canvas,
    logical_observable_coords=coord2logical_group,
    p_before_meas_flip=noise,
)
print(f"num_qubits: {circuit.num_qubits}")
# print(circuit)

fig3d = visualize_compiled_canvas_plotly(compiled_canvas, show_edges=True, hilight_nodes=logical_observables[0])
fig3d.show()

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
pathlib.Path("figures/deform_plus_dem.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/deform_plus_dem.svg")


# %%
