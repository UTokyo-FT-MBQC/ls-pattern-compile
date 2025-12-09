"""Build and visualize a memory canvas using packaged YAML specs."""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import stim
from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.compiler import compile_canvas_to_stim
from lspattern.new_blocks.detector import construct_detector, remove_non_deterministic_det
from lspattern.new_blocks.visualizer import visualize_canvas_plotly, visualize_detectors_plotly

if TYPE_CHECKING:
    from lspattern.new_blocks.mytype import Coord3D

spec_name = "design/cnot.yml"
canvas, spec = load_canvas(spec_name)

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

# collect logical obs
idx = 0
logical_observables_spec = canvas.logical_observables[idx]
logical_obs_coords: set[Coord3D] = set()
for cube_coord in logical_observables_spec.cubes:
    logical_obs_coords |= canvas.couts[cube_coord]
for pipe_coord in logical_observables_spec.pipes:
    logical_obs_coords |= canvas.pipe_couts[pipe_coord]

# %%
fig = visualize_canvas_plotly(canvas, highlight_nodes=logical_obs_coords)
print(f"Loaded canvas '{spec.name}' (d={spec.code_distance}) with {len(spec.cubes)} cubes")
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

nodeidx_to_highlight = {
    2,
    515,
    1029,
    1035,
    525,
    16,
    1044,
    23,
    26,
    540,
    29,
    1056,
    1058,
    35,
    549,
    551,
    1066,
    43,
    561,
    50,
    566,
    65,
    577,
    72,
    586,
    81,
    82,
    597,
    91,
    96,
    102,
    106,
    107,
    620,
    626,
    627,
    119,
    633,
    123,
    637,
    640,
    130,
    643,
    133,
    134,
    650,
    141,
    654,
    143,
    653,
    150,
    663,
    151,
    665,
    156,
    672,
    677,
    678,
    170,
    172,
    693,
    695,
    185,
    189,
    702,
    706,
    710,
    206,
    207,
    721,
    211,
    728,
    730,
    220,
    232,
    746,
    238,
    751,
    759,
    248,
    769,
    771,
    262,
    775,
    268,
    793,
    805,
    293,
    311,
    823,
    314,
    317,
    832,
    834,
    324,
    846,
    339,
    854,
    345,
    346,
    866,
    356,
    873,
    880,
    881,
    882,
    372,
    373,
    892,
    382,
    386,
    901,
    903,
    393,
    398,
    399,
    911,
    910,
    922,
    926,
    416,
    933,
    423,
    936,
    937,
    431,
    944,
    945,
    433,
    952,
    958,
    450,
    453,
    971,
    974,
    975,
    977,
    468,
    984,
    472,
    988,
    993,
    489,
    490,
    494,
    1007,
    1008,
    1012,
    502,
}
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
print(f"Loaded canvas '{spec.name}' (d={spec.code_distance}) with {len(spec.cubes)} cubes")
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
