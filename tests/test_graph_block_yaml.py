"""Tests for graph-based block YAML support."""

from pathlib import Path
from textwrap import dedent

import pytest
from graphqomb.graphstate import GraphState
from graphqomb.ptn_format import load as load_ptn
from graphqomb.scheduler import Scheduler

from lspattern.canvas_loader import load_canvas
from lspattern.compiler import (
    _delay_measurements_for_flow,
    compile_canvas_to_pattern,
    compile_canvas_to_stim,
    export_canvas_to_ptn,
)
from lspattern.mytype import Coord2D, Coord3D


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _write_graph_canvas_fixture(tmp_path: Path) -> Path:
    graph_json = """
    {
      "coord_mode": "local",
      "time_mode": "local",
      "nodes": [
        {"coord": [0, 0, 0], "basis": "X"},
        {"coord": [2, 0, 0], "basis": "Z"}
      ],
      "edges": [
        [[0, 0, 0], [2, 0, 0]]
      ],
      "xflow": [],
      "schedule": {
        "prep": [{"time": 0, "nodes": [[0, 0, 0], [2, 0, 0]]}],
        "entangle": [{"time": 1, "edges": [[[0, 0, 0], [2, 0, 0]]]}],
        "meas": [{"time": 2, "nodes": [[0, 0, 0], [2, 0, 0]]}]
      },
      "detector_candidates": {
        "syndrome_meas": [
          {"id": [1, 1], "rounds": [{"z": 0, "nodes": [[0, 0, 0]]}]}
        ]
      }
    }
    """

    block_yaml = """
    name: GraphBlock
    boundary: XXZZ
    graph: graph_block.json
    """

    canvas_yaml = """
    name: GraphCanvas
    layout: rotated_surface_code
    cube:
      - position: [1, 0, 2]
        block: graph_block
        logical_observables:
          nodes:
            - [0, 0, 0]
    pipe: []
    logical_observables:
      - cube: [[1, 0, 2]]
    """

    _write_json(tmp_path / "graph_block.json", graph_json)
    _write_yaml(tmp_path / "graph_block.yml", block_yaml)
    canvas_path = tmp_path / "graph_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)
    return canvas_path


def test_delay_measurements_for_flow_splits_equal_time_dependencies() -> None:
    graph, node_map = GraphState.from_graph(
        nodes=["input", "middle", "output"],
        edges=[("input", "middle"), ("middle", "output")],
        inputs=["input"],
        outputs=["output"],
    )
    input_node = node_map["input"]
    middle_node = node_map["middle"]
    output_node = node_map["output"]

    flow = {input_node: {middle_node}, middle_node: {output_node}}
    scheduler = Scheduler(graph, flow)
    scheduler.manual_schedule(
        prepare_time={middle_node: 0, output_node: 0},
        measure_time={input_node: 1, middle_node: 1},
    )

    with pytest.raises(ValueError, match="DAG violation"):
        scheduler.validate_schedule()

    _delay_measurements_for_flow(scheduler)

    assert scheduler.measure_time[input_node] == 1
    assert scheduler.measure_time[middle_node] == 2
    scheduler.validate_schedule()


def test_load_canvas_with_graph_block(tmp_path: Path) -> None:
    canvas_path = _write_graph_canvas_fixture(tmp_path)

    canvas, _spec = load_canvas(canvas_path, code_distance=3)

    # local -> global translation at cube position [1,0,2] with d=3
    expected_nodes = {
        Coord3D(8, 0, 12),
        Coord3D(10, 0, 12),
    }
    assert canvas.nodes == expected_nodes
    assert canvas.edges == {(Coord3D(8, 0, 12), Coord3D(10, 0, 12))}

    # couts now use label-indexed dict structure
    assert canvas.couts[Coord3D(1, 0, 2)] == {"0": {Coord3D(8, 0, 12)}}

    # Scheduler time offset should follow the same convention as patch-based cubes.
    assert canvas.scheduler.prep_time[72] == expected_nodes
    assert canvas.scheduler.meas_time[74] == expected_nodes
    assert canvas.scheduler.entangle_time[73] == {(Coord3D(8, 0, 12), Coord3D(10, 0, 12))}

    # Detector candidates (syndrome measurements) should be translated alongside node coordinates.
    assert canvas.parity_accumulator.syndrome_meas[Coord2D(9, 1)][12] == {Coord3D(8, 0, 12)}

    # End-to-end sanity: compile to stim should succeed with user-provided schedule/xflow/detector candidates.
    circuit = compile_canvas_to_stim(canvas, p_depol_after_clifford=0.0, p_before_meas_flip=0.0)
    assert circuit
    assert "OBSERVABLE_INCLUDE" in circuit


def test_export_canvas_to_ptn_roundtrips_graphqomb_pattern(tmp_path: Path) -> None:
    canvas_path = _write_graph_canvas_fixture(tmp_path)
    canvas, _spec = load_canvas(canvas_path, code_distance=3)

    pattern = compile_canvas_to_pattern(canvas)
    output_path = tmp_path / "graph_canvas.ptn"
    export_canvas_to_ptn(canvas, str(output_path))

    ptn_text = output_path.read_text(encoding="utf-8")
    loaded = load_ptn(output_path)

    assert "# GraphQOMB Pattern Format v1" in ptn_text
    assert "N 0 8.0 0.0 12.0" in ptn_text
    assert "N 1 10.0 0.0 12.0" in ptn_text
    assert loaded.coordinates == pattern.coordinates
    assert loaded.pauli_frame.xflow == pattern.pauli_frame.xflow
    assert loaded.pauli_frame.zflow == pattern.pauli_frame.zflow
    assert loaded.pauli_frame.parity_check_group == pattern.pauli_frame.parity_check_group
    assert loaded.pauli_frame.logical_observables == pattern.pauli_frame.logical_observables
