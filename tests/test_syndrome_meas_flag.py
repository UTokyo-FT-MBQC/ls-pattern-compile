"""Tests for layer YAML flags affecting syndrome measurement registration."""

from pathlib import Path
from textwrap import dedent

from lspattern.canvas_loader import load_canvas


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
