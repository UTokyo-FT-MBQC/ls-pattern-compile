# %%
"""
Merge and Splitのmockup

* Measure Xが未実装．現状InitPlus, Memoryのみ
* detector, prity, xflowも未実装
* temporal pipeが一部動かない．
"""

# %%
import matplotlib.pyplot as plt

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas, visualize_compiled_canvas_plotly

# %%
d = 3
r = 3


canvass = RHGCanvasSkeleton("Memory X")

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
    (
        PatchCoordGlobal3D((0, 0, 2)),
        MemoryCubeSkeleton(d=3, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 2)),
        MemoryCubeSkeleton(d=3, edgespec=edgespec),
    ),
]
pipes = [
    (
        PatchCoordGlobal3D((0, 0, 1)),
        PatchCoordGlobal3D((1, 0, 1)),
        MemoryPipeSkeleton(d=3, edgespec=edgespec_trimmed),
    ),
    (
        PatchCoordGlobal3D((0, 0, 0)),
        PatchCoordGlobal3D((0, 0, 1)),
        MemoryPipeSkeleton(d=3, edgespec=edgespec_trimmed),
    ),
    # (
    #     PatchCoordGlobal3D((0, 0, 1)),
    #     PatchCoordGlobal3D((0, 0, 2)),
    #     MemoryPipeSkeleton(d=3, edgespec=edgespec_trimmed),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 0, 0)),
    #     PatchCoordGlobal3D((1, 0, 1)),
    #     MemoryPipeSkeleton(d=3, edgespec=edgespec_trimmed),
    # ),
    # (
    #     PatchCoordGlobal3D((1, 0, 1)),
    #     PatchCoordGlobal3D((1, 0, 2)),
    #     MemoryPipeSkeleton(d=3, edgespec=edgespec_trimmed),
    # ),
]

for block in blocks:
    # RHGCanvasSkeleton は skeleton を受け取り、to_canvas() で block 化します
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

# %%

fig = visualize_compiled_canvas(compiled_canvas, show=True, show_edges=True)
# fig  # This would display the figure in Jupyter

# %%

vals = compiled_canvas.coord2node
vals2d = set((x, y) for (x, y, z) in vals)
plt.scatter(*[list(t) for t in zip(*vals2d, strict=False)], s=1)
plt.gca().set_aspect("equal", "box")
plt.show()

# %%

fig3d = visualize_compiled_canvas_plotly(compiled_canvas, show_edges=True)
fig3d.show()

# %%
