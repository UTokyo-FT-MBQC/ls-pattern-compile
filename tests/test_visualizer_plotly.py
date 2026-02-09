"""Tests for Plotly 3D visualizer axis scaling options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from lspattern.mytype import Coord3D, NodeRole
from lspattern.visualizer import (
    render_canvas_z_window_plotly_figure,
    visualize_canvas_plotly,
    visualize_detectors_plotly,
)


@dataclass
class DummyCanvas:
    """Minimal canvas-like object for visualizer tests."""

    nodes: set[Coord3D]
    edges: set[tuple[Coord3D, Coord3D]]
    coord2role: dict[Coord3D, NodeRole]
    pauli_axes: dict[Coord3D, Any]


def _make_dummy_canvas() -> DummyCanvas:
    a = Coord3D(0, 0, 0)
    b = Coord3D(0, 1, 4)
    return DummyCanvas(
        nodes={a, b},
        edges={(a, b)},
        coord2role={a: NodeRole.DATA, b: NodeRole.ANCILLA_X},
        pauli_axes={},
    )


class TestVisualizeCanvasPlotlyAspectRatio:
    """Tests for aspect-ratio control in visualize_canvas_plotly."""

    def test_default_uses_data_aspectmode(self) -> None:
        fig = visualize_canvas_plotly(_make_dummy_canvas())

        assert fig.layout.scene.aspectmode == "data"
        assert fig.layout.scene.aspectratio.to_plotly_json() == {}

    def test_manual_aspect_ratio_is_applied(self) -> None:
        fig = visualize_canvas_plotly(_make_dummy_canvas(), aspect_ratio=(1.0, 1.0, 0.2))

        assert fig.layout.scene.aspectmode == "manual"
        assert fig.layout.scene.aspectratio.x == 1.0
        assert fig.layout.scene.aspectratio.y == 1.0
        assert fig.layout.scene.aspectratio.z == 0.2

    def test_non_positive_aspect_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            visualize_canvas_plotly(_make_dummy_canvas(), aspect_ratio=(1.0, 0.0, 1.0))


class TestVisualizeDetectorsPlotlyAspectRatio:
    """Tests for aspect-ratio control in visualize_detectors_plotly."""

    def test_manual_aspect_ratio_is_applied(self) -> None:
        fig = visualize_detectors_plotly(
            detectors={Coord3D(1, 2, 6): [1, 2]},
            aspect_ratio=(1.0, 1.0, 0.1),
        )

        assert fig.layout.scene.aspectmode == "manual"
        assert fig.layout.scene.aspectratio.x == 1.0
        assert fig.layout.scene.aspectratio.y == 1.0
        assert fig.layout.scene.aspectratio.z == 0.1


class TestVisualizeCanvasPlotlySizing:
    """Tests for node/edge size scaling in visualize_canvas_plotly."""

    def test_node_and_edge_scale_are_applied(self) -> None:
        fig = visualize_canvas_plotly(
            _make_dummy_canvas(),
            node_size_scale=2.0,
            edge_width_scale=1.5,
        )

        data_trace = next(trace for trace in fig.data if trace.name == "Data")
        edge_trace = next(trace for trace in fig.data if trace.mode == "lines")
        assert data_trace.marker.size == pytest.approx(16.0)
        assert edge_trace.line.width == pytest.approx(4.5)

    def test_non_positive_node_size_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="node_size_scale"):
            visualize_canvas_plotly(_make_dummy_canvas(), node_size_scale=0.0)


class TestRenderCanvasZWindowPlotlyFigure:
    """Tests for sliding-window frame rendering helper."""

    def test_current_layer_is_highlighted_and_tail_is_dimmed(self) -> None:
        a = Coord3D(0, 0, 0)
        b = Coord3D(1, 0, 1)
        canvas = DummyCanvas(
            nodes={a, b},
            edges={(a, b)},
            coord2role={a: NodeRole.DATA, b: NodeRole.ANCILLA_X},
            pauli_axes={},
        )

        fig = render_canvas_z_window_plotly_figure(
            canvas,
            current_z=1,
            z_window=2,
            node_size_scale=2.0,
            edge_width_scale=2.0,
            tail_alpha=0.2,
            current_alpha=0.95,
            highlight_size_scale=1.5,
        )

        data_trace = next(trace for trace in fig.data if trace.name == "Data")
        highlight_trace = next(trace for trace in fig.data if trace.name == "Highlighted")
        edge_trace = next(trace for trace in fig.data if trace.mode == "lines")
        assert data_trace.marker.opacity == pytest.approx(0.2)
        assert highlight_trace.marker.opacity == pytest.approx(0.95)
        assert highlight_trace.marker.size == pytest.approx(33.0)
        assert edge_trace.line.width == pytest.approx(6.0)

    def test_invalid_z_window_raises(self) -> None:
        with pytest.raises(ValueError, match="z_window"):
            render_canvas_z_window_plotly_figure(_make_dummy_canvas(), current_z=0, z_window=0)
