"""Tests for logical observable features: labels, negative layer indices, sublayers, and error handling."""

from pathlib import Path
from textwrap import dedent

import pytest

from lspattern.canvas_loader import CompositeLogicalObservableSpec, load_canvas
from lspattern.compiler import _collect_logical_observable_nodes
from lspattern.mytype import Coord3D


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_multiple_observables_with_labels(tmp_path: Path) -> None:
    """Multiple logical observables with different labels are correctly stored."""
    block_yaml = """
    name: MultiObsBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 2
    """

    canvas_yaml = """
    name: MultiObsCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: multi_obs_block
        logical_observables:
          - token: TB
            label: obs_z
          - token: X
            label: obs_x
    """

    _write_yaml(tmp_path / "multi_obs_block.yml", block_yaml)
    canvas_path = tmp_path / "multi_obs_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    cube_pos = Coord3D(0, 0, 0)
    assert cube_pos in canvas.couts
    assert "obs_z" in canvas.couts[cube_pos]
    assert "obs_x" in canvas.couts[cube_pos]
    assert len(canvas.couts[cube_pos]) == 2

    # obs_z (TB token) should be at sublayer 0 (z=0), obs_x (X token) at sublayer 1 (z=1)
    obs_z_coords = canvas.couts[cube_pos]["obs_z"]
    obs_x_coords = canvas.couts[cube_pos]["obs_x"]
    assert all(c.z == 0 for c in obs_z_coords)
    assert all(c.z == 1 for c in obs_x_coords)


def test_negative_layer_index(tmp_path: Path) -> None:
    """Negative layer indices are correctly resolved to positive indices."""
    block_yaml = """
    name: NegLayerBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 3
    """

    canvas_yaml = """
    name: NegLayerCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: neg_layer_block
        logical_observables:
          - token: TB
            layer: -1
            label: last_layer
          - token: TB
            layer: 2
            label: explicit_last
    """

    _write_yaml(tmp_path / "neg_layer_block.yml", block_yaml)
    canvas_path = tmp_path / "neg_layer_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    cube_pos = Coord3D(0, 0, 0)
    # layer=-1 should resolve to layer 2 (3 unit layers, so 2 is last)
    # Both should produce the same z coordinates
    last_layer_coords = canvas.couts[cube_pos]["last_layer"]
    explicit_last_coords = canvas.couts[cube_pos]["explicit_last"]
    assert last_layer_coords == explicit_last_coords


def test_negative_layer_index_out_of_range(tmp_path: Path) -> None:
    """Negative layer index beyond valid range raises ValueError with improved message."""
    block_yaml = """
    name: OutOfRangeBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 2
    """

    canvas_yaml = """
    name: OutOfRangeCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: out_of_range_block
        logical_observables:
          - token: TB
            layer: -3
    """

    _write_yaml(tmp_path / "out_of_range_block.yml", block_yaml)
    canvas_path = tmp_path / "out_of_range_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    with pytest.raises(ValueError, match=r"layer index -3 \(resolved to -1\) out of range"):
        load_canvas(canvas_path, code_distance=3)


def test_sublayer_specification(tmp_path: Path) -> None:
    """Sublayer specification correctly offsets z coordinate."""
    block_yaml = """
    name: SublayerBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: SublayerCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: sublayer_block
        logical_observables:
          - token: TB
            sublayer: 1
            label: sub1
          - token: TB
            sublayer: 2
            label: sub2
    """

    _write_yaml(tmp_path / "sublayer_block.yml", block_yaml)
    canvas_path = tmp_path / "sublayer_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    cube_pos = Coord3D(0, 0, 0)
    sub1_coords = canvas.couts[cube_pos]["sub1"]
    sub2_coords = canvas.couts[cube_pos]["sub2"]

    # sublayer=1 -> offset 0, sublayer=2 -> offset 1
    assert all(c.z == 0 for c in sub1_coords)
    assert all(c.z == 1 for c in sub2_coords)


def test_label_based_cout_collection(tmp_path: Path) -> None:
    """Compiler correctly collects couts by specific label."""
    block_yaml = """
    name: LabelCollectBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: LabelCollectCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: label_collect_block
        logical_observables:
          - token: TB
            label: target
          - token: X
            label: other
    logical_observables:
      - cube: [[0, 0, 0]]
        label: target
    """

    _write_yaml(tmp_path / "label_collect_block.yml", block_yaml)
    canvas_path = tmp_path / "label_collect_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Get the composite observable
    composite_obs = canvas.logical_observables[0]
    assert composite_obs.label == "target"

    # Collect nodes using the compiler function
    nodes = _collect_logical_observable_nodes(canvas, composite_obs)

    # Should only contain nodes from "target" label, not "other"
    target_coords = canvas.couts[Coord3D(0, 0, 0)]["target"]
    other_coords = canvas.couts[Coord3D(0, 0, 0)]["other"]

    assert nodes == target_coords
    assert not nodes.intersection(other_coords)


def test_missing_label_error(tmp_path: Path) -> None:
    """Missing label in cout raises KeyError with helpful message."""
    block_yaml = """
    name: MissingLabelBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: MissingLabelCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: missing_label_block
        logical_observables:
          - token: TB
            label: existing
    """

    _write_yaml(tmp_path / "missing_label_block.yml", block_yaml)
    canvas_path = tmp_path / "missing_label_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Create a composite observable with a non-existent label
    composite_obs = CompositeLogicalObservableSpec(
        cubes=(Coord3D(0, 0, 0),),
        pipes=(),
        label="nonexistent",
    )

    with pytest.raises(KeyError, match=r"Label 'nonexistent' not found.*Available labels: \['existing'\]"):
        _collect_logical_observable_nodes(canvas, composite_obs)


def test_pipe_missing_label_error(tmp_path: Path) -> None:
    """Missing label in pipe cout raises KeyError with helpful message."""
    block_yaml = """
    name: PipeLabelBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: PipeLabelCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: pipe_label_block
      - position: [1, 0, 0]
        block: pipe_label_block
    pipe:
      - start: [0, 0, 0]
        end: [1, 0, 0]
        block: pipe_label_block
        boundary: ZZOO
        logical_observables:
          - token: LR
            label: pipe_obs
    """

    _write_yaml(tmp_path / "pipe_label_block.yml", block_yaml)
    canvas_path = tmp_path / "pipe_label_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Create a composite observable with a non-existent label for pipe
    pipe_edge = (Coord3D(0, 0, 0), Coord3D(1, 0, 0))
    composite_obs = CompositeLogicalObservableSpec(
        cubes=(),
        pipes=(pipe_edge,),
        label="wrong_label",
    )

    with pytest.raises(KeyError, match=r"Label 'wrong_label' not found in pipe.*Available labels: \['pipe_obs'\]"):
        _collect_logical_observable_nodes(canvas, composite_obs)


def test_label_none_collects_all(tmp_path: Path) -> None:
    """When label is None, all couts from the cube/pipe are combined."""
    block_yaml = """
    name: AllCoutsBlock
    boundary: XXZZ
    layers:
      - type: MemoryUnit
        num_layers: 1
    """

    canvas_yaml = """
    name: AllCoutsCanvas
    layout: rotated_surface_code
    cube:
      - position: [0, 0, 0]
        block: all_couts_block
        logical_observables:
          - token: TB
            label: obs_a
          - token: X
            label: obs_b
    logical_observables:
      - cube: [[0, 0, 0]]
    """

    _write_yaml(tmp_path / "all_couts_block.yml", block_yaml)
    canvas_path = tmp_path / "all_couts_canvas.yml"
    _write_yaml(canvas_path, canvas_yaml)

    canvas, _ = load_canvas(canvas_path, code_distance=3)

    # Get the composite observable (no label specified)
    composite_obs = canvas.logical_observables[0]
    assert composite_obs.label is None

    # Collect nodes using the compiler function
    nodes = _collect_logical_observable_nodes(canvas, composite_obs)

    # Should contain nodes from both labels
    obs_a_coords = canvas.couts[Coord3D(0, 0, 0)]["obs_a"]
    obs_b_coords = canvas.couts[Coord3D(0, 0, 0)]["obs_b"]

    assert obs_a_coords.issubset(nodes)
    assert obs_b_coords.issubset(nodes)
    assert nodes == obs_a_coords | obs_b_coords
