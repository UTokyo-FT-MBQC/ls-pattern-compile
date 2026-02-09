"""Tests for Plotly 3D visualizer axis scaling options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from lspattern.mytype import Coord3D, NodeRole
from lspattern.visualizer import visualize_canvas_plotly, visualize_detectors_plotly


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
