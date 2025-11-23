"""Tests for resolving user-provided YAML search paths in canvas workflow."""

from pathlib import Path
from textwrap import dedent

from lspattern.new_blocks.canvas_loader import load_canvas


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_load_canvas_with_local_block_and_layer(tmp_path: Path) -> None:
    """Ensure canvas->block->layer workflow finds sibling YAML files."""

    layer_yaml = """
    name: CustomMemory
    description: Local memory-like layer
    layer1:
      basis: X
      ancilla: false
    layer2:
      basis: Z
      ancilla: true
    """

    block_yaml = """
    name: CustomBlock
    boundary: XXZZ
    layers:
      - type: custom_memory
        num_layers: 2
    """

    canvas_yaml = """
    name: LocalCanvas
    code_distance: 3
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: custom_block
    """

    _write_yaml(tmp_path / "custom_memory.yml", layer_yaml)
    _write_yaml(tmp_path / "custom_block.yml", block_yaml)
    canvas_path = tmp_path / "custom_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path)

    # With a single block we should at least materialize one cube worth of nodes.
    assert canvas.nodes, "Canvas failed to materialize nodes from local YAML layers"
