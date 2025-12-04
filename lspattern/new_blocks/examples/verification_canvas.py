"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

import stim
from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.compiler import compile_canvas_to_stim
from lspattern.new_blocks.detector import construct_detector, remove_non_deterministic_det
from lspattern.new_blocks.mytype import Coord3D
from lspattern.new_blocks.visualizer import visualize_canvas_plotly, visualize_detectors_plotly

spec_name = "design/merge_split.yml"
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

print("\nComputed cube couts (physical coordinates):")
for pos, coords in canvas.couts.items():
    print(f"  {pos}: {len(coords)} coordinates")
    for coord in sorted(coords, key=lambda c: (c.x, c.y, c.z)):
        print(f"    - {coord}")

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
