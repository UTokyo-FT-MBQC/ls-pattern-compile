"""Tests for layer YAML flags affecting syndrome measurement registration."""

from pathlib import Path
from textwrap import dedent

from lspattern.accumulator import CoordParityAccumulator
from lspattern.canvas_loader import load_canvas
from lspattern.detector import construct_detector
from lspattern.mytype import Coord2D, Coord3D


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_disable_syndrome_meas_without_ancilla(tmp_path: Path) -> None:
    layer_yaml = """
    name: NoSyndromeMeas
    description: Disable inferred syndrome measurements when ancilla=false
    layer1:
      basis: X
      ancilla: false
      syndrome_meas_without_ancilla: false
    layer2:
      basis: X
      ancilla: false
      syndrome_meas_without_ancilla: false
    """

    block_yaml = """
    name: NoSyndromeBlock
    boundary: XXZZ
    layers:
      - type: no_syndrome_meas
        num_layers: 1
    """

    canvas_yaml = """
    name: NoSyndromeCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: no_syndrome_block
    """

    _write_yaml(tmp_path / "no_syndrome_meas.yml", layer_yaml)
    _write_yaml(tmp_path / "no_syndrome_block.yml", block_yaml)
    canvas_path = tmp_path / "no_syndrome_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _spec = load_canvas(canvas_path, code_distance=3)

    assert canvas.nodes, "Sanity: cube should materialize nodes"
    assert canvas.parity_accumulator.syndrome_meas == {}


def test_enable_syndrome_meas_without_ancilla(tmp_path: Path) -> None:
    """Test that syndrome_meas is registered when flag is enabled (default)."""
    layer_yaml = """
    name: WithSyndromeMeas
    description: Enable inferred syndrome measurements when ancilla=false (default)
    layer1:
      basis: X
      ancilla: false
    layer2:
      basis: X
      ancilla: false
    """

    block_yaml = """
    name: WithSyndromeBlock
    boundary: XXZZ
    layers:
      - type: with_syndrome_meas
        num_layers: 1
    """

    canvas_yaml = """
    name: WithSyndromeCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: with_syndrome_block
    """

    _write_yaml(tmp_path / "with_syndrome_meas.yml", layer_yaml)
    _write_yaml(tmp_path / "with_syndrome_block.yml", block_yaml)
    canvas_path = tmp_path / "with_syndrome_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _spec = load_canvas(canvas_path, code_distance=3)

    assert canvas.nodes, "Sanity: cube should materialize nodes"
    # With default flag (syndrome_meas_without_ancilla=true), syndrome measurements
    # should be registered even without ancilla qubits
    assert canvas.parity_accumulator.syndrome_meas != {}, (
        "syndrome_meas should not be empty when flag is enabled"
    )


def test_parity_reset_with_empty_syndrome_meas() -> None:
    """Test that empty syndrome_meas signals parity reset in detector construction."""
    parity = CoordParityAccumulator()
    xy = Coord2D(0, 0)

    # Normal syndrome measurements at z=0 and z=2
    parity.add_syndrome_measurement(xy, 0, {Coord3D(0, 1, 0), Coord3D(1, 0, 0)})
    parity.add_remaining_parity(xy, 0, {Coord3D(0, 0, 0)})

    # Empty syndrome at z=1 signals parity reset
    parity.add_syndrome_measurement(xy, 1, set())

    # Normal syndrome measurement at z=2 (after reset)
    parity.add_syndrome_measurement(xy, 2, {Coord3D(0, 1, 2), Coord3D(1, 0, 2)})
    parity.add_remaining_parity(xy, 2, {Coord3D(0, 0, 2)})

    detectors = construct_detector(parity)

    # Detector at z=0 should use symmetric_difference with empty previous_meas
    assert Coord3D(0, 0, 0) in detectors
    assert detectors[Coord3D(0, 0, 0)] == {Coord3D(0, 1, 0), Coord3D(1, 0, 0)}

    # No detector at z=1 (empty syndrome = reset, skipped)
    assert Coord3D(0, 0, 1) not in detectors

    # Detector at z=2 should use symmetric_difference with empty previous_meas
    # (because z=1 reset cleared the previous state)
    assert Coord3D(0, 0, 2) in detectors
    assert detectors[Coord3D(0, 0, 2)] == {Coord3D(0, 1, 2), Coord3D(1, 0, 2)}
