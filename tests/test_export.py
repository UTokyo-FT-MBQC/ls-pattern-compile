"""Tests for GraphQOMB Studio JSON export functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from graphqomb.common import Axis

from lspattern.canvas import Canvas, CanvasConfig
from lspattern.export import (
    _axis_to_string,
    _convert_detectors,
    _convert_edges,
    _convert_logical_observables,
    _convert_nodes,
    _convert_schedule,
    _convert_xflow,
    _coord_to_node_id,
    _normalize_edge_id,
    canvas_to_graphqomb_studio_dict,
    export_canvas_to_graphqomb_studio,
)
from lspattern.mytype import Coord3D


@pytest.fixture
def empty_canvas() -> Canvas:
    """Create an empty canvas for testing."""
    config = CanvasConfig(name="test", description="test canvas", d=3, tiling="rotated_surface_code")
    return Canvas(config)


@pytest.fixture
def simple_canvas() -> Canvas:
    """Create a simple canvas with basic nodes and edges for testing."""
    config = CanvasConfig(name="simple", description="simple test canvas", d=3, tiling="rotated_surface_code")
    canvas = Canvas(config)

    # Manually add some nodes and edges for testing
    coords = [Coord3D(0, 0, 0), Coord3D(1, 0, 0), Coord3D(0, 1, 0), Coord3D(1, 1, 0)]

    # Access private members for testing (adding nodes directly)
    canvas._Canvas__nodes.update(coords)  # type: ignore[attr-defined]
    canvas._Canvas__pauli_axes[coords[0]] = Axis.X  # type: ignore[attr-defined]
    canvas._Canvas__pauli_axes[coords[1]] = Axis.Y  # type: ignore[attr-defined]
    canvas._Canvas__pauli_axes[coords[2]] = Axis.Z  # type: ignore[attr-defined]
    canvas._Canvas__pauli_axes[coords[3]] = Axis.Z  # type: ignore[attr-defined]

    # Add some edges
    canvas._Canvas__edges.add((coords[0], coords[1]))  # type: ignore[attr-defined]
    canvas._Canvas__edges.add((coords[0], coords[2]))  # type: ignore[attr-defined]
    canvas._Canvas__edges.add((coords[1], coords[3]))  # type: ignore[attr-defined]
    canvas._Canvas__edges.add((coords[2], coords[3]))  # type: ignore[attr-defined]

    # Add flow
    canvas.flow.add_flow(coords[0], coords[1])
    canvas.flow.add_flow(coords[0], coords[2])

    # Add scheduler entries
    canvas.scheduler.add_prep_at_time(0, coords)
    canvas.scheduler.add_entangle_at_time(1, [(coords[0], coords[1]), (coords[2], coords[3])])
    canvas.scheduler.add_meas_at_time(2, [coords[0], coords[2]])
    canvas.scheduler.add_meas_at_time(3, [coords[1], coords[3]])

    # Add couts (logical observables)
    canvas.couts[Coord3D(0, 0, 0)] = {"obs_X": {coords[0], coords[1]}, "obs_Z": {coords[2], coords[3]}}

    return canvas


class TestCoordToNodeId:
    """Tests for _coord_to_node_id function."""

    def test_basic_conversion(self) -> None:
        """Test basic coordinate to node ID conversion."""
        coord = Coord3D(1, 2, 3)
        assert _coord_to_node_id(coord) == "n_1_2_3"

    def test_zero_coordinates(self) -> None:
        """Test conversion with zero coordinates."""
        coord = Coord3D(0, 0, 0)
        assert _coord_to_node_id(coord) == "n_0_0_0"

    def test_negative_coordinates(self) -> None:
        """Test conversion with negative coordinates."""
        coord = Coord3D(-1, -2, 3)
        assert _coord_to_node_id(coord) == "n_-1_-2_3"

    def test_large_coordinates(self) -> None:
        """Test conversion with large coordinates."""
        coord = Coord3D(100, 200, 300)
        assert _coord_to_node_id(coord) == "n_100_200_300"


class TestNormalizeEdgeId:
    """Tests for _normalize_edge_id function."""

    def test_source_less_than_target(self) -> None:
        """Test when source ID is alphabetically less than target."""
        assert _normalize_edge_id("n_0_0_0", "n_1_0_0") == "n_0_0_0-n_1_0_0"

    def test_source_greater_than_target(self) -> None:
        """Test when source ID is alphabetically greater than target."""
        assert _normalize_edge_id("n_1_0_0", "n_0_0_0") == "n_0_0_0-n_1_0_0"

    def test_equal_ids(self) -> None:
        """Test when source and target IDs are equal (edge case)."""
        assert _normalize_edge_id("n_1_1_1", "n_1_1_1") == "n_1_1_1-n_1_1_1"


class TestAxisToString:
    """Tests for _axis_to_string function."""

    def test_axis_x(self) -> None:
        """Test Axis.X conversion."""
        assert _axis_to_string(Axis.X) == "X"

    def test_axis_y(self) -> None:
        """Test Axis.Y conversion."""
        assert _axis_to_string(Axis.Y) == "Y"

    def test_axis_z(self) -> None:
        """Test Axis.Z conversion."""
        assert _axis_to_string(Axis.Z) == "Z"


class TestConvertNodes:
    """Tests for _convert_nodes function."""

    def test_empty_canvas(self, empty_canvas: Canvas) -> None:
        """Test node conversion for empty canvas."""
        nodes = _convert_nodes(empty_canvas)
        assert nodes == []

    def test_simple_canvas_nodes(self, simple_canvas: Canvas) -> None:
        """Test node conversion for simple canvas."""
        nodes = _convert_nodes(simple_canvas)
        assert len(nodes) == 4

        # Check structure of first node
        node0 = next(n for n in nodes if n["id"] == "n_0_0_0")
        assert node0["coordinate"] == {"x": 0, "y": 0, "z": 0}
        assert node0["role"] == "intermediate"
        assert node0["measBasis"]["type"] == "axis"
        assert node0["measBasis"]["axis"] == "X"
        assert node0["measBasis"]["sign"] == "PLUS"

    def test_nodes_sorted(self, simple_canvas: Canvas) -> None:
        """Test that nodes are sorted by coordinate."""
        nodes = _convert_nodes(simple_canvas)
        ids = [n["id"] for n in nodes]
        assert ids == sorted(ids)


class TestConvertEdges:
    """Tests for _convert_edges function."""

    def test_empty_canvas(self, empty_canvas: Canvas) -> None:
        """Test edge conversion for empty canvas."""
        edges = _convert_edges(empty_canvas)
        assert edges == []

    def test_simple_canvas_edges(self, simple_canvas: Canvas) -> None:
        """Test edge conversion for simple canvas."""
        edges = _convert_edges(simple_canvas)
        assert len(edges) == 4

        # Check that all edges have required fields
        for edge in edges:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge

    def test_edges_normalized(self, simple_canvas: Canvas) -> None:
        """Test that edge IDs are normalized (source < target)."""
        edges = _convert_edges(simple_canvas)
        for edge in edges:
            # Verify source comes before target alphabetically
            assert edge["source"] <= edge["target"]
            # Verify ID matches format
            assert edge["id"] == f"{edge['source']}-{edge['target']}"

    def test_no_duplicate_edges(self, simple_canvas: Canvas) -> None:
        """Test that duplicate edges are not created."""
        edges = _convert_edges(simple_canvas)
        edge_ids = [e["id"] for e in edges]
        assert len(edge_ids) == len(set(edge_ids))


class TestConvertXflow:
    """Tests for _convert_xflow function."""

    def test_empty_canvas(self, empty_canvas: Canvas) -> None:
        """Test xflow conversion for empty canvas."""
        xflow = _convert_xflow(empty_canvas)
        assert xflow == {}

    def test_simple_canvas_xflow(self, simple_canvas: Canvas) -> None:
        """Test xflow conversion for simple canvas."""
        xflow = _convert_xflow(simple_canvas)
        assert "n_0_0_0" in xflow
        assert sorted(xflow["n_0_0_0"]) == ["n_0_1_0", "n_1_0_0"]


class TestConvertSchedule:
    """Tests for _convert_schedule function."""

    def test_empty_canvas(self, empty_canvas: Canvas) -> None:
        """Test schedule conversion for empty canvas."""
        schedule = _convert_schedule(empty_canvas)
        assert "prepareTime" in schedule
        assert "measureTime" in schedule
        assert "entangleTime" in schedule
        assert "timeline" in schedule
        assert schedule["timeline"] == []

    def test_simple_canvas_schedule(self, simple_canvas: Canvas) -> None:
        """Test schedule conversion for simple canvas."""
        schedule = _convert_schedule(simple_canvas)

        # Check prepareTime
        assert schedule["prepareTime"]["n_0_0_0"] == 0

        # Check measureTime
        assert schedule["measureTime"]["n_0_0_0"] == 2
        assert schedule["measureTime"]["n_1_1_0"] == 3

        # Check entangleTime
        assert "n_0_0_0-n_1_0_0" in schedule["entangleTime"]
        assert schedule["entangleTime"]["n_0_0_0-n_1_0_0"] == 1

    def test_timeline_structure(self, simple_canvas: Canvas) -> None:
        """Test timeline structure in schedule."""
        schedule = _convert_schedule(simple_canvas)
        timeline = schedule["timeline"]

        assert len(timeline) == 4  # times 0, 1, 2, 3

        # Check time 0 (preparation)
        time0 = next(t for t in timeline if t["time"] == 0)
        assert len(time0["prepareNodes"]) == 4
        assert time0["entangleEdges"] == []
        assert time0["measureNodes"] == []

        # Check time 1 (entanglement)
        time1 = next(t for t in timeline if t["time"] == 1)
        assert time1["prepareNodes"] == []
        assert len(time1["entangleEdges"]) == 2
        assert time1["measureNodes"] == []


class TestConvertDetectors:
    """Tests for _convert_detectors function."""

    def test_empty_canvas(self, empty_canvas: Canvas) -> None:
        """Test detector conversion for empty canvas."""
        detectors = _convert_detectors(empty_canvas)
        assert detectors == []


class TestConvertLogicalObservables:
    """Tests for _convert_logical_observables function."""

    def test_empty_canvas(self, empty_canvas: Canvas) -> None:
        """Test logical observable conversion for empty canvas."""
        observables = _convert_logical_observables(empty_canvas)
        assert observables == {}

    def test_simple_canvas_observables(self, simple_canvas: Canvas) -> None:
        """Test logical observable conversion for simple canvas."""
        observables = _convert_logical_observables(simple_canvas)
        assert "obs_X" in observables
        assert "obs_Z" in observables
        assert sorted(observables["obs_X"]) == ["n_0_0_0", "n_1_0_0"]
        assert sorted(observables["obs_Z"]) == ["n_0_1_0", "n_1_1_0"]


class TestExportCanvasToGraphqombStudio:
    """Tests for export_canvas_to_graphqomb_studio function."""

    def test_export_creates_valid_json(self, simple_canvas: Canvas) -> None:
        """Test that export creates valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            output_path = Path(f.name)

        try:
            export_canvas_to_graphqomb_studio(simple_canvas, output_path, name="Test Project")

            # Read and validate JSON
            with output_path.open(encoding="utf-8") as f:
                data = json.load(f)

            assert data["$schema"] == "graphqomb-studio/v1"
            assert data["name"] == "Test Project"
            assert "nodes" in data
            assert "edges" in data
            assert "flow" in data
            assert "ftqc" in data
            assert "schedule" in data
        finally:
            output_path.unlink(missing_ok=True)

    def test_export_with_string_path(self, simple_canvas: Canvas) -> None:
        """Test export with string path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            output_path = f.name

        try:
            export_canvas_to_graphqomb_studio(simple_canvas, output_path)
            assert Path(output_path).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestCanvasToGraphqombStudioDict:
    """Tests for canvas_to_graphqomb_studio_dict function."""

    def test_returns_dict(self, simple_canvas: Canvas) -> None:
        """Test that function returns a dictionary."""
        result = canvas_to_graphqomb_studio_dict(simple_canvas)
        assert isinstance(result, dict)

    def test_dict_structure(self, simple_canvas: Canvas) -> None:
        """Test the structure of returned dictionary."""
        result = canvas_to_graphqomb_studio_dict(simple_canvas, name="Dict Test")

        assert result["$schema"] == "graphqomb-studio/v1"
        assert result["name"] == "Dict Test"
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)
        assert isinstance(result["flow"], dict)
        assert result["flow"]["zflow"] == "auto"
        assert isinstance(result["ftqc"], dict)
        assert "parityCheckGroup" in result["ftqc"]
        assert "logicalObservableGroup" in result["ftqc"]
        assert isinstance(result["schedule"], dict)

    def test_serializable_to_json(self, simple_canvas: Canvas) -> None:
        """Test that returned dict can be serialized to JSON."""
        result = canvas_to_graphqomb_studio_dict(simple_canvas)
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed == result
