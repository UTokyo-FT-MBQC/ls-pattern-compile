"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.visualizer import visualize_canvas_plotly
from lspattern.new_blocks.visualizer_2d import visualize_canvas_matplotlib_2d
from lspattern.new_blocks.layout.rotated_surface_code import boundary_data_path_cube
from lspattern.new_blocks.mytype import Coord3D
from lspattern.consts import BoundarySide

spec_name = "memory_canvas.yml"
canvas, spec = load_canvas(spec_name)
fig = visualize_canvas_plotly(canvas)
print(f"Loaded canvas '{spec.name}' (d={spec.code_distance}) with {len(spec.cubes)} cubes")
fig.show()


# %%
# boundary path verification
boundary_path = boundary_data_path_cube(
    canvas.config.d,
    Coord3D(0, 0, 0),
    canvas.cube_config[Coord3D(0, 0, 0)].boundary,
    BoundarySide.BOTTOM,
    BoundarySide.TOP,
)
print(f"Boundary path for cube at (0,0,0): {boundary_path}")

# %%
# 2D Matplotlib visualization at target z=0
highlight_nodes = {Coord3D(coord.x, coord.y, 0) for coord in boundary_path}
fig = visualize_canvas_matplotlib_2d(canvas, target_z=0, highlight_nodes=highlight_nodes)

# %%
