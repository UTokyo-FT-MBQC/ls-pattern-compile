"""Tests for invert_ancilla_order feature.

This feature allows inverting the ancilla placement order:
- Default: layer1 (even z) = Z-ancilla, layer2 (odd z) = X-ancilla
- Inverted: layer1 (even z) = X-ancilla, layer2 (odd z) = Z-ancilla

Note: boundary and invert_ancilla_order are specified at the Canvas YAML level
(cube/pipe), not in Block YAML.
"""

from pathlib import Path
from textwrap import dedent

from lspattern.canvas_loader import load_canvas
from lspattern.mytype import Coord3D, NodeRole


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_default_ancilla_order(tmp_path: Path) -> None:
    """Default behavior: Z-ancilla in layer1, X-ancilla in layer2."""
    block_yaml = """
    name: DefaultOrderBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: DefaultOrderCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: default_order_block
        boundary: XXZZ
    """

    _write_yaml(tmp_path / "default_order_block.yml", block_yaml)
    canvas_path = tmp_path / "default_order_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Check roles at z=0 (layer1) and z=1 (layer2)
    z0_ancillas = [
        c for c in canvas.nodes if c.z == 0 and canvas.coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}
    ]
    z1_ancillas = [
        c for c in canvas.nodes if c.z == 1 and canvas.coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}
    ]

    # layer1 should have Z-ancilla
    assert all(canvas.coord2role[c] == NodeRole.ANCILLA_Z for c in z0_ancillas), "layer1 should have Z-ancillas"
    # layer2 should have X-ancilla
    assert all(canvas.coord2role[c] == NodeRole.ANCILLA_X for c in z1_ancillas), "layer2 should have X-ancillas"


def test_inverted_ancilla_order_layer1(tmp_path: Path) -> None:
    """With invert_ancilla_order=true, layer1 gets X-ancilla."""
    block_yaml = """
    name: InvertedOrderBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: InvertedOrderCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: inverted_order_block
        boundary: XXZZ
        invert_ancilla_order: true
    """

    _write_yaml(tmp_path / "inverted_order_block.yml", block_yaml)
    canvas_path = tmp_path / "inverted_order_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Check roles at z=0 (layer1) - should have X-ancilla instead of Z-ancilla
    z0_ancillas = [
        c for c in canvas.nodes if c.z == 0 and canvas.coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}
    ]
    assert all(canvas.coord2role[c] == NodeRole.ANCILLA_X for c in z0_ancillas), (
        "inverted layer1 should have X-ancillas"
    )


def test_inverted_ancilla_order_layer2(tmp_path: Path) -> None:
    """With invert_ancilla_order=true, layer2 gets Z-ancilla."""
    block_yaml = """
    name: InvertedOrderBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: InvertedOrderCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: inverted_order_block
        boundary: XXZZ
        invert_ancilla_order: true
    """

    _write_yaml(tmp_path / "inverted_order_block.yml", block_yaml)
    canvas_path = tmp_path / "inverted_order_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Check roles at z=1 (layer2) - should have Z-ancilla instead of X-ancilla
    z1_ancillas = [
        c for c in canvas.nodes if c.z == 1 and canvas.coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}
    ]
    assert all(canvas.coord2role[c] == NodeRole.ANCILLA_Z for c in z1_ancillas), (
        "inverted layer2 should have Z-ancillas"
    )


def test_inverted_cout_x_observable(tmp_path: Path) -> None:
    """X observable goes to layer1 (z=0) when inverted."""
    block_yaml = """
    name: InvertedCoutBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: InvertedCoutCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: inverted_cout_block
        boundary: XXZZ
        invert_ancilla_order: true
        logical_observables:
          - token: X
            label: x_obs
    """

    _write_yaml(tmp_path / "inverted_cout_block.yml", block_yaml)
    canvas_path = tmp_path / "inverted_cout_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    cube_pos = Coord3D(0, 0, 0)
    x_obs_coords = canvas.couts[cube_pos]["x_obs"]

    # With invert, X-ancilla is in layer1 (z=0)
    assert all(c.z == 0 for c in x_obs_coords), "X observable should be at z=0 when inverted"


def test_inverted_cout_z_observable(tmp_path: Path) -> None:
    """Z observable goes to layer2 (z=1) when inverted."""
    block_yaml = """
    name: InvertedCoutBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: InvertedCoutCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: inverted_cout_block
        boundary: XXZZ
        invert_ancilla_order: true
        logical_observables:
          - token: Z
            label: z_obs
    """

    _write_yaml(tmp_path / "inverted_cout_block.yml", block_yaml)
    canvas_path = tmp_path / "inverted_cout_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    cube_pos = Coord3D(0, 0, 0)
    z_obs_coords = canvas.couts[cube_pos]["z_obs"]

    # With invert, Z-ancilla is in layer2 (z=1)
    assert all(c.z == 1 for c in z_obs_coords), "Z observable should be at z=1 when inverted"


def test_explicit_sublayer_overrides_inversion(tmp_path: Path) -> None:
    """Explicit sublayer specification takes precedence over invert_ancilla_order."""
    block_yaml = """
    name: OverrideBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: OverrideCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: override_block
        boundary: XXZZ
        invert_ancilla_order: true
        logical_observables:
          - token: X
            sublayer: 2
            label: x_explicit
          - token: Z
            sublayer: 1
            label: z_explicit
    """

    _write_yaml(tmp_path / "override_block.yml", block_yaml)
    canvas_path = tmp_path / "override_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    cube_pos = Coord3D(0, 0, 0)
    x_explicit_coords = canvas.couts[cube_pos]["x_explicit"]
    z_explicit_coords = canvas.couts[cube_pos]["z_explicit"]

    # Explicit sublayer=2 means z=1 (regardless of inversion)
    assert all(c.z == 1 for c in x_explicit_coords), "Explicit sublayer=2 should put X at z=1"
    # Explicit sublayer=1 means z=0 (regardless of inversion)
    assert all(c.z == 0 for c in z_explicit_coords), "Explicit sublayer=1 should put Z at z=0"


def test_invert_ancilla_order_default_false(tmp_path: Path) -> None:
    """invert_ancilla_order defaults to False when not specified."""
    block_yaml = """
    name: NoFlagBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: NoFlagCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: no_flag_block
        boundary: XXZZ
        logical_observables:
          - token: X
            label: x_obs
          - token: Z
            label: z_obs
    """

    _write_yaml(tmp_path / "no_flag_block.yml", block_yaml)
    canvas_path = tmp_path / "no_flag_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    cube_pos = Coord3D(0, 0, 0)
    x_obs_coords = canvas.couts[cube_pos]["x_obs"]
    z_obs_coords = canvas.couts[cube_pos]["z_obs"]

    # Default: X at layer2 (z=1), Z at layer1 (z=0)
    assert all(c.z == 1 for c in x_obs_coords), "X should be at z=1 by default"
    assert all(c.z == 0 for c in z_obs_coords), "Z should be at z=0 by default"


def test_inverted_pipe_cout(tmp_path: Path) -> None:
    """Pipe couts respect invert_ancilla_order flag."""
    block_yaml = """
    name: InvertedPipeBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    cube_block_yaml = """
    name: CubeBlock
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: InvertedPipeCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: cube_block
        boundary: XXZZ
      - position: [1, 0, 0]
        block: cube_block
        boundary: XXZZ
    pipe:
      - start: [0, 0, 0]
        end: [1, 0, 0]
        block: inverted_pipe_block
        boundary: ZZOO
        invert_ancilla_order: true
        logical_observables:
          - token: X
            label: pipe_x
          - token: Z
            label: pipe_z
    """

    _write_yaml(tmp_path / "inverted_pipe_block.yml", block_yaml)
    _write_yaml(tmp_path / "cube_block.yml", cube_block_yaml)
    canvas_path = tmp_path / "inverted_pipe_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    pipe_edge = (Coord3D(0, 0, 0), Coord3D(1, 0, 0))
    pipe_x_coords = canvas.pipe_couts[pipe_edge]["pipe_x"]
    pipe_z_coords = canvas.pipe_couts[pipe_edge]["pipe_z"]

    # With invert on pipe: X at layer1 (z=0), Z at layer2 (z=1)
    assert all(c.z == 0 for c in pipe_x_coords), "Pipe X should be at z=0 when inverted"
    assert all(c.z == 1 for c in pipe_z_coords), "Pipe Z should be at z=1 when inverted"
