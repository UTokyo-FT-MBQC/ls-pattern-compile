"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.visualizer import visualize_canvas_plotly
from lspattern.new_blocks.visualizer_2d import visualize_canvas_matplotlib_2d

spec_name = "memory_canvas.yml"
canvas, spec = load_canvas(spec_name)
fig = visualize_canvas_plotly(canvas)
print(f"Loaded canvas '{spec.name}' (d={spec.code_distance}) with {len(spec.cubes)} cubes")
fig.show()

# %%
# 2D Matplotlib visualization at target z=0
fig = visualize_canvas_matplotlib_2d(canvas, target_z=0)
