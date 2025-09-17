# %%
"""
Merge and Split
"""

# %%
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
d = 3


canvass = RHGCanvasSkeleton("Merge and Split")

edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
edgespec1 = {"LEFT": "X", "RIGHT": "O", "TOP": "Z", "BOTTOM": "Z"}
edgespec2 = {"LEFT": "O", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
edgespec_trimmed = {"LEFT": "O", "RIGHT": "O", "TOP": "Z", "BOTTOM": "Z"}
blocks = [
    (
        PatchCoordGlobal3D((0, 0, 0)),
        InitPlusCubeSkeleton(d=3, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 0)),
        InitPlusCubeSkeleton(d=3, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 1)),
        MemoryCubeSkeleton(d=3, edgespec=edgespec1),
    ),
    (
        PatchCoordGlobal3D((1, 0, 1)),
        MemoryCubeSkeleton(d=3, edgespec=edgespec2),
    ),
    # (
    #     PatchCoordGlobal3D((0, 0, 2)),
    #     MemoryCubeSkeleton(d=3, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 0, 2)),
    #     MemoryCubeSkeleton(d=3, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((0, 0, 3)),
    #     MeasureXSkeleton(d=3, edgespec=edgespec),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 0, 3)),
    #     MeasureXSkeleton(d=3, edgespec=edgespec),
    # )
]
pipes = [
    # (
    #     PatchCoordGlobal3D((0, 0, 1)),
    #     PatchCoordGlobal3D((1, 0, 1)),
    #     MemoryPipeSkeleton(d=3, edgespec=edgespec_trimmed),
    # ),
    # (
    #     PatchCoordGlobal3D((0, 0, 0)),
    #     PatchCoordGlobal3D((0, 0, 1)),
    #     MemoryPipeSkeleton(d=3, edgespec=edgespec_trimmed),
    # ),
]

for block in blocks:
    canvass.add_cube(*block)
for pipe in pipes:
    canvass.add_pipe(*pipe)

canvas = canvass.to_canvas()

compiled_canvas: CompiledRHGCanvas = canvas.compile()
nnodes = len(getattr(compiled_canvas.global_graph, "physical_nodes", []) or []) if compiled_canvas.global_graph else 0
nedges = len(getattr(compiled_canvas.global_graph, "physical_edges", []) or []) if compiled_canvas.global_graph else 0
print(
    {
        "layers": len(compiled_canvas.layers),
        "nodes": nnodes,
        "edges": nedges,
        "coord_map": len(compiled_canvas.coord2node),
    }
)

# Print flow and parity information
xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}
x_parity = []
for group_list in compiled_canvas.parity.checks.values():
    x_parity.extend(group_list)

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
remaining_flows = {src: dsts for src, dsts in xflow.items() if src not in scheduled_nodes}
if remaining_flows:
    remaining_flow_strs = [f"{src} -> {dsts}" for src, dsts in remaining_flows.items()]
    print(f"  Unscheduled flows: {', '.join(remaining_flow_strs)}")

print("X parity")
for coord, group_list in compiled_canvas.parity.checks.items():
    print(f"  {coord}: {group_list}")


# %%

fig3d = visualize_compiled_canvas_plotly(compiled_canvas, show_edges=True)
fig3d.show()

# %%
