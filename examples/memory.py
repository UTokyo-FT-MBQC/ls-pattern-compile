"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

import pathlib

import stim
from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

from lspattern.consts import BoundarySide
from lspattern.canvas_loader import load_canvas
from lspattern.compiler import compile_canvas_to_stim
from lspattern.detector import construct_detector
from lspattern.layout import RotatedSurfaceCodeLayoutBuilder
from lspattern.mytype import Coord2D, Coord3D
from lspattern.visualizer import visualize_canvas_plotly, visualize_detectors_plotly
from lspattern.visualizer_2d import visualize_canvas_matplotlib_2d

spec_name = "memory_canvas.yml"
code_distance = 3
canvas, spec = load_canvas(spec_name, code_distance=code_distance)
fig = visualize_canvas_plotly(canvas)
print(f"Loaded canvas '{spec.name}' (d={code_distance}) with {len(spec.cubes)} cubes")
fig.show()

# # %%
# # edge order
# entangle_time = canvas.scheduler.entangle_time
# for time in sorted(entangle_time.keys()):
#     edges = entangle_time[time]
#     print(f"Time {time}:")
#     for edge in edges:
#         print(f"  Edge between {edge[0]} and {edge[1]}")

# %%
# Logical observables from YAML spec and computed couts
print("\n=== Logical Observables ===")
print("Cube logical observable specs:")
for cube in spec.cubes:
    lo = cube.logical_observables
    print(f"  {cube.position}: {[obs.token for obs in lo] if lo else 'None'}")

print("\nComputed couts (physical coordinates):")
for pos, coords in canvas.couts.items():
    print(f"  {pos}: {len(coords)} coordinates")
    for coord in sorted(coords, key=lambda c: (c.x, c.y, c.z) if isinstance(c, Coord3D) else (0, 0, 0)):
        print(f"    - {coord}")


# %%
# boundary path verification
boundary_path = RotatedSurfaceCodeLayoutBuilder.cube_boundary_path(
    canvas.config.d,
    Coord2D(0, 0),
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
node_index_to_coord = {idx: coord for coord, idx in node_map.items()}

# 2) build detectors (Coord3D -> set[Coord3D]) then convert to node indices
coord2det_coords = construct_detector(canvas.parity_accumulator)
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
    node_index_to_coord=node_index_to_coord,
    show_node_indices_on_hover=True,
    show_canvas_edges=True,
)
fig_det.show()

# %%
# Stim circuit compilation
print("\n=== Stim Circuit Compilation ===")
print(f"Logical observables: {canvas.logical_observables}")

noise = 0.001
circuit_str = compile_canvas_to_stim(
    canvas,
    p_depol_after_clifford=noise,
    p_before_meas_flip=noise,
)
circuit = stim.Circuit(circuit_str)
print(f"Stim circuit: num_qubits={circuit.num_qubits}")
print(circuit)

# %%
# Detector error model analysis
dem = circuit.detector_error_model(decompose_errors=True)

err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(f"\nShortest graphlike error length: {len(err)}")
print(err)

# %%
# Export SVG diagram
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/memory_canvas_dem.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/memory_canvas_dem.svg")

# %%
