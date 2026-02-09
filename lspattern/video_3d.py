"""3D z-sweep MP4 export utilities for compiled canvases."""

from __future__ import annotations

import importlib
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from lspattern.visualizer import CanvasLike, render_canvas_z_window_plotly_figure

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _FrameWriter(Protocol):
    """Minimal writer protocol used by imageio ffmpeg backend."""

    def append_data(self, frame: NDArray[np.uint8]) -> None:
        """Append one RGB frame."""

    def close(self) -> None:
        """Close the writer and flush output."""


class _ImageIOModule(Protocol):
    """Protocol for the subset of imageio.v2 APIs used by this module."""

    def get_writer(self, uri: str, **kwargs: object) -> _FrameWriter:
        """Create a writer for the target URI."""

    def imread(self, uri: BytesIO, **kwargs: object) -> NDArray[np.generic]:
        """Read an image from an in-memory stream."""


def _get_imageio_v2() -> _ImageIOModule:
    """Import imageio.v2 lazily with a clear dependency error."""

    try:
        imageio = importlib.import_module("imageio.v2")
    except ImportError as exc:  # pragma: no cover - environment-dependent
        msg = "imageio and imageio-ffmpeg are required for MP4 export. Install both packages first."
        raise RuntimeError(msg) from exc

    return cast("_ImageIOModule", imageio)


def _validate_positive_int(value: int, *, name: str) -> int:
    """Validate that an integer argument is positive."""

    numeric = int(value)
    if numeric <= 0:
        msg = f"{name} must be positive."
        raise ValueError(msg)
    return numeric


def _validate_positive_float(value: float, *, name: str) -> float:
    """Validate that a numeric argument is positive."""

    numeric = float(value)
    if numeric <= 0:
        msg = f"{name} must be positive."
        raise ValueError(msg)
    return numeric


def _validate_alpha(value: float, *, name: str) -> float:
    """Validate opacity-like values in [0, 1]."""

    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        msg = f"{name} must be between 0.0 and 1.0."
        raise ValueError(msg)
    return numeric


def _figure_to_rgb_array(
    fig: go.Figure,
    *,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    """Convert a Plotly figure into an RGB uint8 frame."""
    rgb_channels = 3

    try:
        image_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=1)
    except ValueError as exc:  # pragma: no cover - environment-dependent
        msg = "Plotly static export requires 'kaleido'. Install kaleido to enable MP4 export."
        raise RuntimeError(msg) from exc

    imageio = _get_imageio_v2()
    frame = np.asarray(imageio.imread(BytesIO(image_bytes), format="png"))

    if frame.ndim != rgb_channels or frame.shape[2] < rgb_channels:
        msg = "Unexpected image shape returned from Plotly frame rendering."
        raise RuntimeError(msg)

    rgb = frame[:, :, :rgb_channels]
    if rgb.dtype == np.uint8:
        return cast("NDArray[np.uint8]", rgb)

    rgb_float = np.asarray(rgb, dtype=np.float64)
    if np.issubdtype(rgb.dtype, np.floating):
        scaled = np.clip(rgb_float * 255.0, 0.0, 255.0)
    else:
        scaled = np.clip(rgb_float, 0.0, 255.0)
    return cast("NDArray[np.uint8]", scaled.astype(np.uint8))


def export_canvas_z_sweep_3d_mp4(
    canvas: CanvasLike,
    output_path: str | Path,
    *,
    fps: int = 24,
    z_window: int = 6,
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    tail_alpha: float = 0.25,
    current_alpha: float = 1.0,
    highlight_size_scale: float = 1.4,
    width: int = 1280,
    height: int = 720,
    reverse_axes: bool = True,
    aspect_ratio: tuple[float, float, float] | None = None,
    codec: str = "libx264",
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Export a 3D z-sweep MP4 movie from z=0 up to max-z.

    Frames are rendered with a sliding z-window ``[z_current-z_window+1, z_current]``
    and the current layer emphasized with larger highlighted markers.
    """

    output = Path(output_path)
    if output.suffix.lower() != ".mp4":
        msg = "output_path must use .mp4 extension."
        raise ValueError(msg)

    fps_value = _validate_positive_int(fps, name="fps")
    z_window_value = _validate_positive_int(z_window, name="z_window")
    width_value = _validate_positive_int(width, name="width")
    height_value = _validate_positive_int(height, name="height")
    node_size_scale_value = _validate_positive_float(node_size_scale, name="node_size_scale")
    edge_width_scale_value = _validate_positive_float(edge_width_scale, name="edge_width_scale")
    highlight_size_scale_value = _validate_positive_float(highlight_size_scale, name="highlight_size_scale")
    tail_alpha_value = _validate_alpha(tail_alpha, name="tail_alpha")
    current_alpha_value = _validate_alpha(current_alpha, name="current_alpha")

    if crf < 0:
        msg = "crf must be non-negative."
        raise ValueError(msg)

    z_values = sorted({coord.z for coord in canvas.nodes})
    if not z_values:
        msg = "Cannot export video from an empty canvas."
        raise ValueError(msg)

    output.parent.mkdir(parents=True, exist_ok=True)

    imageio = _get_imageio_v2()
    ffmpeg_params = ["-crf", str(int(crf)), "-preset", preset]
    writer = imageio.get_writer(
        str(output),
        fps=fps_value,
        codec=codec,
        ffmpeg_params=ffmpeg_params,
    )

    try:
        for current_z in z_values:
            fig = render_canvas_z_window_plotly_figure(
                canvas,
                current_z=current_z,
                z_window=z_window_value,
                node_size_scale=node_size_scale_value,
                edge_width_scale=edge_width_scale_value,
                tail_alpha=tail_alpha_value,
                current_alpha=current_alpha_value,
                highlight_size_scale=highlight_size_scale_value,
                width=width_value,
                height=height_value,
                reverse_axes=reverse_axes,
                aspect_ratio=aspect_ratio,
            )
            frame = _figure_to_rgb_array(fig, width=width_value, height=height_value)
            writer.append_data(frame)
    finally:
        writer.close()

    return output
