"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import stim
from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

from lspattern.canvas_loader import load_canvas
from lspattern.compiler import compile_canvas_to_stim
from lspattern.detector import construct_detector, remove_non_deterministic_det
from lspattern.visualizer import visualize_canvas_plotly, visualize_detectors_plotly

if TYPE_CHECKING:
    from lspattern.mytype import Coord3D

spec_name = "design/cnot.yml"
code_distance = 3
canvas, spec = load_canvas(spec_name, code_distance=code_distance)

# %%
# Logical observables from YAML spec and computed couts
print("\n=== Logical Observables ===")
print("Cube logical observable specs:")
for cube in spec.cubes:
    lo = cube.logical_observables
    if lo:
        tokens = [obs.token for obs in lo]
        print(f"  {cube.position}: {tokens}")
    else:
        print(f"  {cube.position}: None")

print("\nComputed cube couts (physical coordinates):")
for pos, label_coords in canvas.couts.items():
    print(f"  {pos}:")
    for label, coords in label_coords.items():
        print(f"    [{label}]: {len(coords)} coordinates")
        for coord in sorted(coords, key=lambda c: (c.x, c.y, c.z)):
            print(f"      - {coord}")

# # collect logical obs
idx = 0
logical_observables_spec = canvas.logical_observables[idx]
logical_obs_coords: set[Coord3D] = set()
for cube_coord in logical_observables_spec.cubes:
    for coords in canvas.couts[cube_coord].values():
        logical_obs_coords |= coords
for pipe_coord in logical_observables_spec.pipes:
    for coords in canvas.pipe_couts[pipe_coord].values():
        logical_obs_coords |= coords
# logical_obs_coords: set[Coord3D] = set()

# %%
fig = visualize_canvas_plotly(canvas, highlight_nodes=logical_obs_coords)
print(f"Loaded canvas '{spec.name}' (d={code_distance}) with {len(spec.cubes)} cubes")
fig.show()

# %%
# Detector construction and visualization
# 1) build coord->node index mapping via GraphState
_graph, node_map = GraphState.from_graph(
    nodes=canvas.nodes,
    edges=canvas.edges,
    meas_bases={coord: AxisMeasBasis(canvas.pauli_axes[coord], Sign.PLUS) for coord in canvas.nodes},
)
node_index_to_coord = {idx: coord for coord, idx in node_map.items()}

nodeidx_to_highlight = {}
inv_node_map = {v: k for k, v in node_map.items()}
highlight_node = set()
for idx in nodeidx_to_highlight:
    if idx in inv_node_map:
        highlight_node.add(inv_node_map[idx])

# 2) build detectors (Coord3D -> set[Coord3D]) then convert to node indices
det_acc = remove_non_deterministic_det(canvas)
coord2det_coords = construct_detector(det_acc)
coord2det_nodes: dict[Coord3D, set[int]] = {}
for det_coord, involved_coords in coord2det_coords.items():
    mapped = {node_map[c] for c in involved_coords if c in node_map}
    if mapped:
        coord2det_nodes[det_coord] = mapped

print(f"Constructed {len(coord2det_nodes)} detectors")

# %%
fig = visualize_canvas_plotly(canvas, highlight_nodes=highlight_node)
print(f"Loaded canvas '{spec.name}' (d={code_distance}) with {len(spec.cubes)} cubes")
fig.show()

# %%

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
