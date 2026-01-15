"""Tests for exporter.studio module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lspattern.exporter.studio import (
    ExportConfig,
    axis_to_meas_basis,
    coord_to_dict,
    coord_to_id,
    export_to_studio,
    normalize_edge_id,
    save_to_studio_json,
)
from lspattern.mytype import Coord3D

if TYPE_CHECKING:
    pass


# =============================================================================
# Tests for helper functions
# =============================================================================


class TestCoordToId:
    """Tests for coord_to_id function."""

    def test_basic_conversion(self) -> None:
        """Test basic coordinate to ID conversion."""
        coord = Coord3D(1, 2, 3)
        assert coord_to_id(coord) == "1_2_3"

    def test_zero_coords(self) -> None:
        """Test conversion with zero coordinates."""
        coord = Coord3D(0, 0, 0)
        assert coord_to_id(coord) == "0_0_0"

    def test_negative_coords(self) -> None:
        """Test conversion with negative coordinates."""
        coord = Coord3D(-1, -2, -3)
        assert coord_to_id(coord) == "-1_-2_-3"


class TestCoordToDict:
    """Tests for coord_to_dict function."""

    def test_basic_conversion(self) -> None:
        """Test basic coordinate to dict conversion."""
        coord = Coord3D(1, 2, 3)
        result = coord_to_dict(coord)
        assert result == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_returns_float(self) -> None:
        """Test that values are returned as floats."""
        coord = Coord3D(1, 2, 3)
        result = coord_to_dict(coord)
        assert isinstance(result["x"], float)
        assert isinstance(result["y"], float)
        assert isinstance(result["z"], float)


class TestNormalizeEdgeId:
    """Tests for normalize_edge_id function."""

    def test_already_sorted(self) -> None:
        """Test with already sorted IDs."""
        assert normalize_edge_id("0_0_0", "1_1_1") == "0_0_0-1_1_1"

    def test_reverse_order(self) -> None:
        """Test with reverse ordered IDs."""
        assert normalize_edge_id("1_1_1", "0_0_0") == "0_0_0-1_1_1"

    def test_same_result_regardless_of_order(self) -> None:
        """Test that order doesn't affect result."""
        id_a = "2_3_4"
        id_b = "1_2_3"
        assert normalize_edge_id(id_a, id_b) == normalize_edge_id(id_b, id_a)


class TestAxisToMeasBasis:
    """Tests for axis_to_meas_basis function."""

    def test_x_axis(self) -> None:
        """Test X axis conversion."""
        from graphqomb.common import Axis

        result = axis_to_meas_basis(Axis.X)
        assert result == {"type": "axis", "axis": "X", "sign": "PLUS"}

    def test_y_axis(self) -> None:
        """Test Y axis conversion."""
        from graphqomb.common import Axis

        result = axis_to_meas_basis(Axis.Y)
        assert result == {"type": "axis", "axis": "Y", "sign": "PLUS"}

    def test_z_axis(self) -> None:
        """Test Z axis conversion."""
        from graphqomb.common import Axis

        result = axis_to_meas_basis(Axis.Z)
        assert result == {"type": "axis", "axis": "Z", "sign": "PLUS"}

    def test_custom_sign(self) -> None:
        """Test with custom sign."""
        from graphqomb.common import Axis

        result = axis_to_meas_basis(Axis.X, sign="MINUS")
        assert result == {"type": "axis", "axis": "X", "sign": "MINUS"}


# =============================================================================
# Tests for ExportConfig
# =============================================================================


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ExportConfig()
        assert config.input_nodes == set()
        assert config.output_nodes == set()
        assert config.qubit_index_map == {}

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        input_nodes = {Coord3D(0, 0, 0)}
        output_nodes = {Coord3D(1, 1, 1)}
        qubit_map = {Coord3D(0, 0, 0): 0}
        config = ExportConfig(
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            qubit_index_map=qubit_map,
        )
        assert config.input_nodes == input_nodes
        assert config.output_nodes == output_nodes
        assert config.qubit_index_map == qubit_map


# =============================================================================
# Tests for export_to_studio
# =============================================================================


class TestExportToStudio:
    """Tests for export_to_studio function."""

    def test_basic_export_schema(self) -> None:
        """Test that exported data has correct schema."""
        from graphqomb.common import Axis

        from lspattern.canvas import Canvas, CanvasConfig

        # Create minimal canvas
        config = CanvasConfig(name="test", description="test", d=3, tiling="rotated_surface_code")
        canvas = Canvas(config)

        # Add a node manually (accessing private attribute for testing)
        coord = Coord3D(0, 0, 0)
        canvas._Canvas__nodes.add(coord)  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord] = Axis.Z  # type: ignore[attr-defined]

        result = export_to_studio(canvas, "test_project")

        assert result["$schema"] == "graphqomb-studio/v1"
        assert result["name"] == "test_project"
        assert "nodes" in result
        assert "edges" in result
        assert "flow" in result
        assert "schedule" in result

    def test_flow_has_auto_zflow(self) -> None:
        """Test that flow always has zflow='auto'."""
        from graphqomb.common import Axis

        from lspattern.canvas import Canvas, CanvasConfig

        config = CanvasConfig(name="test", description="test", d=3, tiling="rotated_surface_code")
        canvas = Canvas(config)

        coord = Coord3D(0, 0, 0)
        canvas._Canvas__nodes.add(coord)  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord] = Axis.Z  # type: ignore[attr-defined]

        result = export_to_studio(canvas, "test")

        assert result["flow"]["zflow"] == "auto"
        assert isinstance(result["flow"]["xflow"], dict)

    def test_all_nodes_intermediate_by_default(self) -> None:
        """Test that all nodes are intermediate by default."""
        from graphqomb.common import Axis

        from lspattern.canvas import Canvas, CanvasConfig

        config = CanvasConfig(name="test", description="test", d=3, tiling="rotated_surface_code")
        canvas = Canvas(config)

        coord1 = Coord3D(0, 0, 0)
        coord2 = Coord3D(1, 1, 1)
        canvas._Canvas__nodes.add(coord1)  # type: ignore[attr-defined]
        canvas._Canvas__nodes.add(coord2)  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord1] = Axis.Z  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord2] = Axis.X  # type: ignore[attr-defined]

        result = export_to_studio(canvas, "test")

        # All nodes should be intermediate by default
        input_nodes = [n for n in result["nodes"] if n["role"] == "input"]
        output_nodes = [n for n in result["nodes"] if n["role"] == "output"]
        intermediate_nodes = [n for n in result["nodes"] if n["role"] == "intermediate"]

        assert len(input_nodes) == 0
        assert len(output_nodes) == 0
        assert len(intermediate_nodes) == 2

    def test_explicit_input_output_nodes(self) -> None:
        """Test explicit input/output node specification via ExportConfig."""
        from graphqomb.common import Axis

        from lspattern.canvas import Canvas, CanvasConfig

        config = CanvasConfig(name="test", description="test", d=3, tiling="rotated_surface_code")
        canvas = Canvas(config)

        coord1 = Coord3D(0, 0, 0)
        coord2 = Coord3D(1, 1, 1)
        coord3 = Coord3D(2, 2, 2)
        canvas._Canvas__nodes.add(coord1)  # type: ignore[attr-defined]
        canvas._Canvas__nodes.add(coord2)  # type: ignore[attr-defined]
        canvas._Canvas__nodes.add(coord3)  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord1] = Axis.Z  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord2] = Axis.X  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord3] = Axis.Y  # type: ignore[attr-defined]

        # Explicitly specify input and output nodes
        export_config = ExportConfig(
            input_nodes={coord1},
            output_nodes={coord2},
        )
        result = export_to_studio(canvas, "test", config=export_config)

        input_nodes = [n for n in result["nodes"] if n["role"] == "input"]
        output_nodes = [n for n in result["nodes"] if n["role"] == "output"]
        intermediate_nodes = [n for n in result["nodes"] if n["role"] == "intermediate"]

        assert len(input_nodes) == 1
        assert input_nodes[0]["id"] == "0_0_0"
        assert "measBasis" in input_nodes[0]
        assert "qubitIndex" in input_nodes[0]

        assert len(output_nodes) == 1
        assert output_nodes[0]["id"] == "1_1_1"
        assert "measBasis" not in output_nodes[0] or output_nodes[0].get("measBasis") is None
        assert "qubitIndex" in output_nodes[0]

        assert len(intermediate_nodes) == 1
        assert intermediate_nodes[0]["id"] == "2_2_2"


# =============================================================================
# Tests for save_to_studio_json
# =============================================================================


class TestSaveToStudioJson:
    """Tests for save_to_studio_json function."""

    def test_creates_valid_json_file(self, tmp_path: Path) -> None:
        """Test that valid JSON file is created."""
        from graphqomb.common import Axis

        from lspattern.canvas import Canvas, CanvasConfig

        config = CanvasConfig(name="test", description="test", d=3, tiling="rotated_surface_code")
        canvas = Canvas(config)

        coord = Coord3D(0, 0, 0)
        canvas._Canvas__nodes.add(coord)  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord] = Axis.Z  # type: ignore[attr-defined]

        output_path = tmp_path / "test.json"
        save_to_studio_json(canvas, "test_project", output_path)

        assert output_path.exists()

        # Verify it's valid JSON
        with output_path.open() as f:
            data = json.load(f)
        assert data["$schema"] == "graphqomb-studio/v1"
        assert data["name"] == "test_project"

    def test_custom_indent(self, tmp_path: Path) -> None:
        """Test custom indentation."""
        from graphqomb.common import Axis

        from lspattern.canvas import Canvas, CanvasConfig

        config = CanvasConfig(name="test", description="test", d=3, tiling="rotated_surface_code")
        canvas = Canvas(config)

        coord = Coord3D(0, 0, 0)
        canvas._Canvas__nodes.add(coord)  # type: ignore[attr-defined]
        canvas._Canvas__pauli_axes[coord] = Axis.Z  # type: ignore[attr-defined]

        output_path = tmp_path / "test.json"
        save_to_studio_json(canvas, "test", output_path, indent=4)

        content = output_path.read_text()
        # Check for 4-space indentation
        assert "    " in content
