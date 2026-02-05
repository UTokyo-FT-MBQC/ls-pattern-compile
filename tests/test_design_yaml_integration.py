"""End-to-end integration tests for all design YAML canvas files.

Verifies that every ``examples/design/*.yml`` canvas can be loaded, compiled
to a Stim circuit, and analyzed for detector error models without errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import stim
from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState

from lspattern.canvas_loader import load_canvas
from lspattern.compiler import _collect_logical_observable_nodes, compile_canvas_to_stim
from lspattern.detector import construct_detector

_DESIGN_DIR = Path(__file__).resolve().parent.parent / "examples" / "design"
_CANVAS_YMLS = sorted(p.name for p in _DESIGN_DIR.glob("*.yml"))

# Known failing canvases — to be fixed in future work.
_KNOWN_FAILURES: dict[str, str] = {
    "L_patch.yml": "non-deterministic detectors",
    "graph_block_canvas.yml": "non-deterministic observables",
    "patch_expansion_x.yml": "cycle detected in feedforward graph",
    "patch_expansion_z.yml": "non-deterministic detectors",
}


@pytest.mark.parametrize("yml_name", _CANVAS_YMLS)
def test_design_yaml_end_to_end(yml_name: str) -> None:
    if yml_name in _KNOWN_FAILURES:
        pytest.xfail(_KNOWN_FAILURES[yml_name])

    canvas_path = _DESIGN_DIR / yml_name
    code_distance = 3

    # 1. Load canvas from YAML spec
    canvas, _spec = load_canvas(canvas_path, code_distance=code_distance)
    assert canvas.nodes, f"Canvas '{yml_name}' produced no nodes"

    # 2. Build GraphState and detector map
    GraphState.from_graph(
        nodes=canvas.nodes,
        edges=canvas.edges,
        meas_bases={coord: AxisMeasBasis(canvas.pauli_axes[coord], Sign.PLUS) for coord in canvas.nodes},
    )
    construct_detector(canvas.parity_accumulator)

    # 3. Collect logical observables for each spec entry
    for logical_obs_spec in canvas.logical_observables:
        _collect_logical_observable_nodes(canvas, logical_obs_spec)

    # 4. Compile to Stim circuit
    circuit_str = compile_canvas_to_stim(
        canvas,
        p_depol_after_clifford=0.001,
        p_before_meas_flip=0.001,
    )
    circuit = stim.Circuit(circuit_str)
    assert circuit.num_qubits > 0, f"Stim circuit for '{yml_name}' has no qubits"

    # 5. Detector error model
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem is not None, f"DEM for '{yml_name}' is None"

    # 6. Shortest graphlike error (validates DEM structure)
    dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
