"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.mytype import NodeRole
from lspattern.new_blocks.visualizer import visualize_canvas_plotly

spec_name = "memory_canvas.yml"
canvas, spec = load_canvas(spec_name)
fig = visualize_canvas_plotly(canvas)
print(f"Loaded canvas '{spec.name}' (d={spec.code_distance}) with {len(spec.cubes)} cubes")
fig.show()

# %%
# Debug info
# edge
print(f"Edges: {canvas.edges}")
# detect dangling edges
for edge in canvas.edges:
    if len(edge) != 2:
        print(f"Dangling edge detected: {edge}")
