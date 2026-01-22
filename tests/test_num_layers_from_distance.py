"""Tests for num_layers_from_distance: rest specification."""

from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from lspattern.canvas_loader import _analyze_layer_specs, _resolve_num_layers, load_block_config_from_name


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


class TestAnalyzeLayerSpecs:
    """Tests for the _analyze_layer_specs helper function."""

    def test_all_fixed_layers(self) -> None:
        """When all layers have fixed num_layers, return their sum."""
        layers = [
            {"type": "A", "num_layers": 2},
            {"type": "B", "num_layers": 3},
        ]
        total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
        assert total_fixed == 5
        assert rest_index is None

    def test_rest_layer_at_end(self) -> None:
        """When rest is at the end, fixed layers before it are counted."""
        layers: list[dict[str, Any]] = [
            {"type": "A", "num_layers": 1},
            {"type": "B", "num_layers_from_distance": "rest"},
        ]
        total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
        assert total_fixed == 1
        assert rest_index == 1

    def test_rest_layer_in_middle(self) -> None:
        """When rest is in the middle, fixed layers before AND after are counted."""
        layers: list[dict[str, Any]] = [
            {"type": "A", "num_layers": 1},
            {"type": "B", "num_layers_from_distance": "rest"},
            {"type": "C", "num_layers": 1},
        ]
        total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
        # Both num_layers=1 before and after rest are counted
        assert total_fixed == 2
        assert rest_index == 1

    def test_multiple_rest_layers_raises_error(self) -> None:
        """Having multiple rest specifications should raise ValueError."""
        layers = [
            {"type": "A", "num_layers_from_distance": "rest"},
            {"type": "B", "num_layers_from_distance": "remaining"},
        ]
        with pytest.raises(ValueError, match="Multiple layers cannot use 'rest' specification"):
            _analyze_layer_specs(layers, distance=5)

    def test_rest_synonyms(self) -> None:
        """Test that 'remaining' and 'fill' work as synonyms for 'rest'."""
        for synonym in ["rest", "remaining", "fill", "REST", "Remaining", "FILL"]:
            layers: list[dict[str, Any]] = [
                {"type": "A", "num_layers": 2},
                {"type": "B", "num_layers_from_distance": synonym},
            ]
            total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
            assert total_fixed == 2
            assert rest_index == 1

    def test_scale_offset_spec(self) -> None:
        """Test num_layers_from_distance with scale and offset."""
        layers = [
            {"type": "A", "num_layers_from_distance": {"scale": 2, "offset": -1}},
        ]
        # scale * distance + offset = 2 * 5 + (-1) = 9
        total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
        assert total_fixed == 9
        assert rest_index is None

    def test_numeric_spec(self) -> None:
        """Test num_layers_from_distance with a plain numeric value."""
        layers = [
            {"type": "A", "num_layers_from_distance": 3},
        ]
        total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
        assert total_fixed == 3
        assert rest_index is None

    def test_default_layer_count(self) -> None:
        """Layers without explicit num_layers should default to 1."""
        layers: list[dict[str, Any]] = [
            {"type": "A"},  # No num_layers specified
            {"type": "B", "num_layers": 2},
        ]
        total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
        # Default 1 + explicit 2 = 3
        assert total_fixed == 3
        assert rest_index is None

    def test_params_num_layers(self) -> None:
        """Test num_layers inside params dictionary."""
        layers = [
            {"type": "A", "params": {"num_layers": 4}},
        ]
        total_fixed, rest_index = _analyze_layer_specs(layers, distance=5)
        assert total_fixed == 4
        assert rest_index is None


class TestResolveNumLayers:
    """Tests for the _resolve_num_layers function."""

    def test_explicit_num_layers(self) -> None:
        """Explicit num_layers should be returned directly."""
        cfg = {"type": "A", "num_layers": 5}
        assert _resolve_num_layers(cfg, distance=10, total_fixed_layers=3) == 5

    def test_rest_calculation(self) -> None:
        """Rest should return distance - total_fixed_layers."""
        cfg = {"type": "A", "num_layers_from_distance": "rest"}
        # distance=5, total_fixed=2 -> rest should be 3
        assert _resolve_num_layers(cfg, distance=5, total_fixed_layers=2) == 3

    def test_rest_returns_zero_when_exceeded(self) -> None:
        """Rest should return 0 when fixed layers exceed distance."""
        cfg = {"type": "A", "num_layers_from_distance": "rest"}
        # distance=3, total_fixed=5 -> rest should be 0 (not negative)
        assert _resolve_num_layers(cfg, distance=3, total_fixed_layers=5) == 0


class TestLoadBlockConfigWithRest:
    """Integration tests for load_block_config_from_name with rest specification."""

    def test_rest_after_fixed_layer(self, tmp_path: Path) -> None:
        """Test basic rest calculation with preceding fixed layer."""
        layer_yaml = """
        name: TestLayer
        description: Test layer
        layer1:
          basis: X
          ancilla: false
        layer2:
          basis: Z
          ancilla: true
        """

        block_yaml = """
        name: TestBlock
        boundary: XXZZ
        layers:
          - type: test_layer
            num_layers: 1
          - type: test_layer
            num_layers_from_distance: rest
        """

        _write_yaml(tmp_path / "test_layer.yml", layer_yaml)
        _write_yaml(tmp_path / "test_block.yml", block_yaml)

        # distance=5: 1 fixed + 4 rest = 5 total
        block = load_block_config_from_name(
            tmp_path / "test_block.yml",
            code_distance=5,
        )
        assert len(block) == 5

    def test_rest_with_following_fixed_layer(self, tmp_path: Path) -> None:
        """Main bug fix test: rest should account for layers AFTER it."""
        layer_yaml = """
        name: TestLayer
        description: Test layer
        layer1:
          basis: X
          ancilla: false
        layer2:
          basis: Z
          ancilla: true
        """

        block_yaml = """
        name: TestBlock
        boundary: XXZZ
        layers:
          - type: test_layer
            num_layers: 1
          - type: test_layer
            num_layers_from_distance: rest
          - type: test_layer
            num_layers: 1
        """

        _write_yaml(tmp_path / "test_layer.yml", layer_yaml)
        _write_yaml(tmp_path / "test_block.yml", block_yaml)

        # distance=3: 1 + rest + 1 = 3 -> rest should be 1
        block = load_block_config_from_name(
            tmp_path / "test_block.yml",
            code_distance=3,
        )
        # Before fix: would be 1 + 2 + 1 = 4
        # After fix: should be 1 + 1 + 1 = 3
        assert len(block) == 3

    def test_rest_with_multiple_following_fixed_layers(self, tmp_path: Path) -> None:
        """Rest should account for multiple fixed layers after it."""
        layer_yaml = """
        name: TestLayer
        description: Test layer
        layer1:
          basis: X
          ancilla: false
        layer2:
          basis: Z
          ancilla: true
        """

        block_yaml = """
        name: TestBlock
        boundary: XXZZ
        layers:
          - type: test_layer
            num_layers: 1
          - type: test_layer
            num_layers_from_distance: rest
          - type: test_layer
            num_layers: 2
          - type: test_layer
            num_layers: 1
        """

        _write_yaml(tmp_path / "test_layer.yml", layer_yaml)
        _write_yaml(tmp_path / "test_block.yml", block_yaml)

        # distance=7: 1 + rest + 2 + 1 = 7 -> rest should be 3
        # Fixed layers: 1 + 2 + 1 = 4, rest = 7 - 4 = 3
        block = load_block_config_from_name(
            tmp_path / "test_block.yml",
            code_distance=7,
        )
        assert len(block) == 7

    def test_rest_becomes_zero(self, tmp_path: Path) -> None:
        """When fixed layers equal or exceed distance, rest should be 0."""
        layer_yaml = """
        name: TestLayer
        description: Test layer
        layer1:
          basis: X
          ancilla: false
        layer2:
          basis: Z
          ancilla: true
        """

        block_yaml = """
        name: TestBlock
        boundary: XXZZ
        layers:
          - type: test_layer
            num_layers: 2
          - type: test_layer
            num_layers_from_distance: rest
          - type: test_layer
            num_layers: 2
        """

        _write_yaml(tmp_path / "test_layer.yml", layer_yaml)
        _write_yaml(tmp_path / "test_block.yml", block_yaml)

        # distance=3: fixed=4 (2+2), rest should be 0
        # Total: 2 + 0 + 2 = 4 (exceeds distance, but that's allowed)
        block = load_block_config_from_name(
            tmp_path / "test_block.yml",
            code_distance=3,
        )
        assert len(block) == 4

    def test_multiple_rest_raises_error(self, tmp_path: Path) -> None:
        """Having multiple rest specifications should raise ValueError."""
        layer_yaml = """
        name: TestLayer
        description: Test layer
        layer1:
          basis: X
          ancilla: false
        layer2:
          basis: Z
          ancilla: true
        """

        block_yaml = """
        name: TestBlock
        boundary: XXZZ
        layers:
          - type: test_layer
            num_layers_from_distance: rest
          - type: test_layer
            num_layers_from_distance: remaining
        """

        _write_yaml(tmp_path / "test_layer.yml", layer_yaml)
        _write_yaml(tmp_path / "test_block.yml", block_yaml)

        with pytest.raises(ValueError, match="Multiple layers cannot use 'rest' specification"):
            load_block_config_from_name(
                tmp_path / "test_block.yml",
                code_distance=5,
            )
