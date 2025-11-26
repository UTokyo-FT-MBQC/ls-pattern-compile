"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState
from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.detector import construct_detector, remove_non_deterministic_det
from lspattern.new_blocks.visualizer import visualize_canvas_plotly, visualize_detectors_plotly
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
# Logical observables from YAML spec and computed couts
print("\n=== Logical Observables ===")
print("Cube logical observable specs:")
for cube in spec.cubes:
    lo = cube.logical_observable
    print(f"  {cube.position}: {lo.token if lo else 'None'}")

print("\nComputed couts (physical coordinates):")
for pos, coords in canvas.couts.items():
    print(f"  {pos}: {len(coords)} coordinates")
    for coord in sorted(coords, key=lambda c: (c.x, c.y, c.z)):
        print(f"    - {coord}")


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
# Detector construction and visualization
# 1) build coord->node index mapping via GraphState
_graph, node_map = GraphState.from_graph(
    nodes=canvas.nodes,
    edges=canvas.edges,
    meas_bases={coord: AxisMeasBasis(canvas.pauli_axes[coord], Sign.PLUS) for coord in canvas.nodes},
)

# 2) build detectors (Coord3D -> set[Coord3D]) then convert to node indices
det_acc = remove_non_deterministic_det(canvas)
coord2det_coords = construct_detector(det_acc)
coord2det_nodes: dict[Coord3D, set[int]] = {}
for det_coord, involved_coords in coord2det_coords.items():
    mapped = {node_map[c] for c in involved_coords if c in node_map}
    if mapped:
        coord2det_nodes[det_coord] = mapped

print(f"Constructed {len(coord2det_nodes)} detectors")

# 3) visualize detectors with hover showing node indices
fig_det = visualize_detectors_plotly(
    coord2det_nodes,
    canvas=canvas,
    show_node_indices_on_hover=True,
    show_canvas_edges=True,
)
fig_det.show()

# %%
