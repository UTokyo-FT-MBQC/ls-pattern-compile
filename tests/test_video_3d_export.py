"""Tests for 3D z-sweep MP4 export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import pytest

from lspattern import video_3d
from lspattern.mytype import Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass
class DummyCanvas:
    """Minimal canvas-like object for video export tests."""

    nodes: set[Coord3D]
    edges: set[tuple[Coord3D, Coord3D]]
    coord2role: dict[Coord3D, NodeRole]
    pauli_axes: dict[Coord3D, Any]


class _DummyWriter:
    """In-memory writer stub."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray[Any, np.dtype[np.uint8]]] = []
        self.closed = False

    def append_data(self, frame: np.ndarray[Any, np.dtype[np.uint8]]) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class _DummyImageIOModule:
    """Stub imageio module exposing only get_writer."""

    def __init__(self, writer: _DummyWriter) -> None:
        self.writer = writer
        self.uri: str | None = None
        self.kwargs: dict[str, object] = {}

    def get_writer(self, uri: str, **kwargs: object) -> _DummyWriter:
        self.uri = uri
        self.kwargs = kwargs
        return self.writer


def _make_dummy_canvas() -> DummyCanvas:
    z0 = Coord3D(0, 0, 0)
    z1 = Coord3D(0, 0, 1)
    z2 = Coord3D(0, 0, 2)
    return DummyCanvas(
        nodes={z0, z1, z2},
        edges={(z0, z1), (z1, z2)},
        coord2role={z0: NodeRole.DATA, z1: NodeRole.ANCILLA_X, z2: NodeRole.ANCILLA_Z},
        pauli_axes={},
    )


def test_export_sweeps_all_z_layers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    canvas = _make_dummy_canvas()
    writer = _DummyWriter()
    imageio_stub = _DummyImageIOModule(writer)

    render_calls: list[int] = []

    def fake_render(
        _canvas: DummyCanvas,
        *,
        current_z: int,
        **_kwargs: object,
    ) -> go.Figure:
        render_calls.append(current_z)
        return go.Figure()

    def fake_frame(_fig: go.Figure, *, width: int, height: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return np.zeros((height, width, 3), dtype=np.uint8)

    monkeypatch.setattr(video_3d, "_get_imageio_v2", lambda: imageio_stub)
    monkeypatch.setattr(video_3d, "render_canvas_z_window_plotly_figure", fake_render)
    monkeypatch.setattr(video_3d, "_figure_to_rgb_array", fake_frame)

    output = video_3d.export_canvas_z_sweep_3d_mp4(
        canvas,
        tmp_path / "movie.mp4",
        fps=12,
        z_window=2,
        width=8,
        height=4,
        codec="libx264",
        crf=18,
        preset="fast",
    )

    assert output.suffix == ".mp4"
    assert render_calls == [0, 1, 2]
    assert len(writer.frames) == 3
    assert writer.closed
    assert imageio_stub.uri is not None
    assert imageio_stub.uri.endswith("movie.mp4")
    assert imageio_stub.kwargs["fps"] == 12
    assert imageio_stub.kwargs["codec"] == "libx264"
    assert imageio_stub.kwargs["ffmpeg_params"] == ["-crf", "18", "-preset", "fast"]


@pytest.mark.parametrize(
    ("invalid_call", "match"),
    [
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, fps=0),
            "fps",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, z_window=0),
            "z_window",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, node_size_scale=0.0),
            "node_size_scale",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, edge_width_scale=0.0),
            "edge_width_scale",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, tail_alpha=1.1),
            "tail_alpha",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, current_alpha=-0.1),
            "current_alpha",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, width=0),
            "width",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, height=0),
            "height",
        ),
        (
            lambda canvas, path: video_3d.export_canvas_z_sweep_3d_mp4(canvas, path, crf=-1),
            "crf",
        ),
    ],
)
def test_invalid_parameters_raise(
    invalid_call: Callable[[DummyCanvas, Path], Path],
    match: str,
    tmp_path: Path,
) -> None:
    canvas = _make_dummy_canvas()

    with pytest.raises(ValueError, match=match):
        invalid_call(canvas, tmp_path / "movie.mp4")


def test_non_mp4_extension_raises(tmp_path: Path) -> None:
    canvas = _make_dummy_canvas()

    with pytest.raises(ValueError, match=r"\.mp4"):
        video_3d.export_canvas_z_sweep_3d_mp4(canvas, tmp_path / "movie.gif")


def test_empty_canvas_raises(tmp_path: Path) -> None:
    canvas = DummyCanvas(nodes=set(), edges=set(), coord2role={}, pauli_axes={})

    with pytest.raises(ValueError, match="empty canvas"):
        video_3d.export_canvas_z_sweep_3d_mp4(canvas, tmp_path / "movie.mp4")
