"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

import pathlib

import stim
from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.compiler import compile_canvas_to_stim
from lspattern.new_blocks.detector import construct_detector, remove_non_deterministic_det
from lspattern.new_blocks.mytype import Coord3D
from lspattern.new_blocks.visualizer import visualize_canvas_plotly, visualize_detectors_plotly

spec_name = "hadamard_canvas.yml"
canvas, spec = load_canvas(spec_name)
# fig = visualize_canvas_plotly(canvas, highlight_nodes=selected_coords)
# print(f"Loaded canvas '{spec.name}' (d={spec.code_distance}) with {len(spec.cubes)} cubes")
# fig.show()


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
# Detector construction and visualization
# 1) build coord->node index mapping via GraphState
_graph, node_map = GraphState.from_graph(
    nodes=canvas.nodes,
    edges=canvas.edges,
    meas_bases={coord: AxisMeasBasis(canvas.pauli_axes[coord], Sign.PLUS) for coord in canvas.nodes},
)
node_index_to_coord = {idx: coord for coord, idx in node_map.items()}

# collect selected coordinates
selected = {
    131,
    132,
    4,
    15,
    16,
    145,
    18,
    144,
    23,
    154,
    30,
    33,
    37,
    38,
    47,
    53,
    54,
    62,
    63,
    67,
    68,
    74,
    75,
    77,
    87,
    90,
    96,
    98,
    101,
    107,
    108,
    111,
    112,
    118,
}
selected_coords = {node_index_to_coord[n] for n in selected}
fig = visualize_canvas_plotly(canvas, highlight_nodes=selected_coords)
fig.show()
collapsed_stabilizer_x = {4, 33, 53, 74, 87, 101, 118}
collapsed_stabilizer_z = {4, 45, 53, 87, 118, 137}
# remap to coordinates
collapsed_stabilizer_x_coords = {node_index_to_coord[n] for n in collapsed_stabilizer_x}
collapsed_stabilizer_z_coords = {node_index_to_coord[n] for n in collapsed_stabilizer_z}
print("\nCollapsed stabilizers:")
print("X-type:")
for c in collapsed_stabilizer_x_coords:
    print(f"  - {c}")
print("Z-type:")
for c in collapsed_stabilizer_z_coords:
    print(f"  - {c}")

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
# Build logical_observables dict from couts keys
# Use the cube positions that have logical observables defined
logical_observables: dict[int, set[Coord3D]] = {}
for idx, (pos, _) in enumerate(canvas.couts.items()):
    logical_observables[idx] = {pos}

print("\n=== Stim Circuit Compilation ===")
print(f"Logical observables: {logical_observables}")

noise = 0.001
circuit_str = compile_canvas_to_stim(
    canvas,
    logical_observables,
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
# print schedule
prep_time = canvas.scheduler.prep_time
meas_time = canvas.scheduler.meas_time
entangle_time = canvas.scheduler.entangle_time

for t in sorted(prep_time.keys() | meas_time.keys() | entangle_time.keys()):
    prep_nodes = prep_time.get(t, [])
    meas_nodes = meas_time.get(t, [])
    entangle_edges = entangle_time.get(t, [])

    print(f"\nTime step {t}:")
    if prep_nodes:
        print(f"  Preparation on nodes: {prep_nodes}")
    if entangle_edges:
        print(f"  Entangling edges: {entangle_edges}")
    if meas_nodes:
        print(f"  Measurement on nodes: {meas_nodes}")
