from __future__ import annotations

import operator
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import plotly.graph_objects as go

from lspattern.mytype import Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from graphqomb.common import Axis

type NodeIndex = int | Coord3D


class CanvasLike(Protocol):
    """Structural protocol for canvas-like objects used by visualizers."""

    nodes: set[Coord3D]
    edges: set[tuple[Coord3D, Coord3D]]
    coord2role: dict[Coord3D, NodeRole]
    pauli_axes: dict[Coord3D, Axis | None]


class NodeStyleSpec(TypedDict):
    """Style specification for node visualization.

    Attributes
    ----------
    color : str
        Fill color of the node marker.
    line_color : str
        Border color of the node marker.
    size : int
        Size of the node marker in pixels.
    label : str
        Display label for the node type in the legend.
    """

    color: str
    line_color: str
    size: int
    label: str


class NodeGroup(TypedDict):
    """Grouped node data for visualization.

    Attributes
    ----------
    x : list[int]
        List of x coordinates.
    y : list[int]
        List of y coordinates.
    z : list[int]
        List of z coordinates.
    coords : list[Coord3D]
        List of Coord3D objects.
    """

    x: list[int]
    y: list[int]
    z: list[int]
    coords: list[Coord3D]


_COLOR_MAP: dict[NodeRole, NodeStyleSpec] = {
    NodeRole.DATA: {"color": "white", "line_color": "black", "size": 8, "label": "Data"},
    NodeRole.ANCILLA_X: {"color": "green", "line_color": "darkgreen", "size": 7, "label": "X ancilla"},
    NodeRole.ANCILLA_Z: {"color": "blue", "line_color": "darkblue", "size": 7, "label": "Z ancilla"},
}

# Constants for edge rendering
_EDGE_SEPARATOR = float("nan")


def _axis_to_str(axis: Axis | None) -> str:
    """Return compact string for measurement axis."""

    return axis.name if axis is not None else "-"


def _node_hover_label(coord: Coord3D, role_label: str, axis: Axis | None) -> str:
    """Compose hover label for a single node."""

    return f"{role_label} ({coord.x}, {coord.y}, {coord.z})<br>axis: {_axis_to_str(axis)}"


def _group_nodes(
    nodes: Iterable[Coord3D],
    coord2role: Mapping[Coord3D, NodeRole],
) -> dict[NodeRole, NodeGroup]:
    """Group nodes by their role for separate visualization traces.

    Parameters
    ----------
    nodes : Iterable[Coord3D]
        Iterable of 3D coordinates representing nodes in the canvas.
    coord2role : Mapping[Coord3D, NodeRole]
        Mapping from coordinates to their roles. Nodes not in the mapping
        default to NodeRole.DATA.

    Returns
    -------
    dict[NodeRole, NodeGroup]
        Dictionary mapping each node role to its grouped coordinate data.
        Each NodeGroup contains separate lists for x, y, z coordinates
        and the original Coord3D objects.
    """
    groups: dict[NodeRole, NodeGroup] = {role: {"x": [], "y": [], "z": [], "coords": []} for role in _COLOR_MAP}
    for coord in nodes:
        role = coord2role.get(coord, NodeRole.DATA)
        groups[role]["x"].append(coord.x)
        groups[role]["y"].append(coord.y)
        groups[role]["z"].append(coord.z)
        groups[role]["coords"].append(coord)
    return groups


def _edge_coordinates(
    edges: Iterable[tuple[Coord3D, Coord3D]],
    valid_nodes: set[Coord3D],
) -> tuple[list[float], list[float], list[float]]:
    """Extract coordinates for edges whose endpoints both exist on the canvas."""

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_z: list[float] = []

    for start, end in edges:
        if start not in valid_nodes or end not in valid_nodes:
            continue
        edge_x.extend((float(start.x), float(end.x), _EDGE_SEPARATOR))
        edge_y.extend((float(start.y), float(end.y), _EDGE_SEPARATOR))
        edge_z.extend((float(start.z), float(end.z), _EDGE_SEPARATOR))

    return edge_x, edge_y, edge_z


def _format_node_labels(
    nodes: Iterable[NodeIndex],
    index_to_coord: Mapping[int, Coord3D] | None,
) -> list[str]:
    """Format node identifiers for hover text, preferring 3D coordinates."""

    sortable: list[tuple[tuple[int, int, int, int], str]] = []
    for node in nodes:
        if isinstance(node, Coord3D):
            coord = node
            key = (0, coord.x, coord.y, coord.z)
            label = f"({coord.x}, {coord.y}, {coord.z})"
        elif index_to_coord is not None and isinstance(node, int) and node in index_to_coord:
            coord = index_to_coord[node]
            key = (0, coord.x, coord.y, coord.z)
            label = f"({coord.x}, {coord.y}, {coord.z})"
        else:
            try:
                num = int(node)
            except (TypeError, ValueError):
                num = None
            key = (1, num, 0, 0) if num is not None else (2, 0, 0, 0)
            label = str(node)
        sortable.append((key, label))

    sortable.sort(key=operator.itemgetter(0))

    labels: list[str] = []
    seen: set[str] = set()
    for _, label in sortable:
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _build_plotly_scene(
    *,
    reverse_axes: bool,
    aspect_ratio: tuple[float, float, float] | None,
) -> dict[str, object]:
    """Build 3D scene configuration with optional manual axis scaling."""

    manual_aspect = _normalize_manual_aspect_ratio(aspect_ratio)

    scene: dict[str, object] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
    }

    if manual_aspect is None:
        scene["aspectmode"] = "data"
    else:
        scene["aspectmode"] = "manual"
        scene["aspectratio"] = manual_aspect

    if reverse_axes:
        scene["xaxis"] = {"autorange": "reversed"}
        scene["yaxis"] = {"autorange": "reversed"}

    return scene


def _normalize_manual_aspect_ratio(
    aspect_ratio: tuple[float, float, float] | None,
) -> dict[str, float] | None:
    """Normalize optional aspect ratio tuple into Plotly-compatible mapping."""

    if aspect_ratio is None:
        return None

    try:
        x_scale, y_scale, z_scale = aspect_ratio
        x = float(x_scale)
        y = float(y_scale)
        z = float(z_scale)
    except (TypeError, ValueError) as exc:
        msg = "aspect_ratio must be a tuple of three positive numbers."
        raise ValueError(msg) from exc

    if x <= 0 or y <= 0 or z <= 0:
        msg = "aspect_ratio values must be positive."
        raise ValueError(msg)

    return {"x": x, "y": y, "z": z}


def _validate_positive_float(value: float, *, name: str) -> float:
    """Validate that a value is a strictly positive float."""

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


def _validate_non_negative_float(value: float, *, name: str) -> float:
    """Validate that a numeric value is non-negative."""

    numeric = float(value)
    if numeric < 0:
        msg = f"{name} must be non-negative."
        raise ValueError(msg)
    return numeric


def _validate_projection_type(value: str) -> str:
    """Validate camera projection type used by Plotly."""

    projection = str(value).strip().lower()
    if projection not in {"perspective", "orthographic"}:
        msg = "projection_type must be either 'perspective' or 'orthographic'."
        raise ValueError(msg)
    return projection


def _compute_axis_range(
    values: list[float],
    *,
    reverse: bool,
    padding: float,
) -> tuple[float, float]:
    """Compute a fixed axis range with optional padding and reversal."""

    low = min(values) - padding
    high = max(values) + padding
    if low == high:
        low -= 0.5
        high += 0.5
    return (high, low) if reverse else (low, high)


def _compute_window_z_axis_range(
    *,
    current_z: int,
    z_window: int,
    padding: float,
) -> tuple[float, float]:
    """Compute a z-axis range centered on the current sliding z-window."""

    low = float(int(current_z) - int(z_window) + 1) - padding
    high = float(int(current_z)) + padding
    if low == high:
        low -= 0.5
        high += 0.5
    return (low, high)


def _compute_scene_axis_ranges(
    nodes: set[Coord3D],
    *,
    reverse_axes: bool,
    padding: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute deterministic x/y/z axis ranges from full-canvas nodes."""

    if not nodes:
        msg = "Cannot compute scene ranges from an empty canvas."
        raise ValueError(msg)

    x_values = [float(node.x) for node in nodes]
    y_values = [float(node.y) for node in nodes]
    z_values = [float(node.z) for node in nodes]
    return (
        _compute_axis_range(x_values, reverse=reverse_axes, padding=padding),
        _compute_axis_range(y_values, reverse=reverse_axes, padding=padding),
        _compute_axis_range(z_values, reverse=False, padding=padding),
    )


def visualize_canvas_plotly(
    canvas: CanvasLike,
    *,
    show_edges: bool = True,
    edge_width: float = 3.0,
    edge_width_scale: float = 1.0,
    edge_color: str = "rgba(60, 60, 60, 0.7)",
    node_size_scale: float = 1.0,
    node_alpha: float = 0.9,
    highlight_nodes: Iterable[Coord3D] | None = None,
    highlight_color: str = "red",
    highlight_line_color: str = "darkred",
    highlight_size: float = 11.0,
    highlight_alpha: float = 0.98,
    width: int = 900,
    height: int = 700,
    reverse_axes: bool = True,
    aspect_ratio: tuple[float, float, float] | None = None,
) -> go.Figure:
    """Create an interactive 3D visualization of a Canvas using Plotly.

    This function renders nodes colored by their role (data, X ancilla, Z ancilla)
    and optionally displays edges connecting them. The visualization uses 3D scatter
    plots with customizable styling.

    Parameters
    ----------
    canvas : Canvas
        The Canvas object to visualize, containing nodes, edges, and role information.
    show_edges : bool, optional
        Whether to display edges between nodes, by default True.
    edge_width : float, optional
        Width of edge lines in pixels, by default 3.0.
    edge_width_scale : float, optional
        Global multiplier for `edge_width`. Must be positive. Default 1.0.
    edge_color : str, optional
        RGBA color string for edges, by default "rgba(60, 60, 60, 0.7)".
    node_size_scale : float, optional
        Global marker-size multiplier for all node traces. Must be positive.
        Default 1.0.
    node_alpha : float, optional
        Marker opacity for non-highlighted nodes. Must be in [0.0, 1.0].
        Default 0.9.
    highlight_nodes : Iterable[Coord3D] | None, optional
        Optional iterable of node coordinates to emphasize. Highlighted nodes are
        drawn in red on top of the regular markers. Default None.
    highlight_color : str, optional
        Fill color for highlighted nodes. Default "red".
    highlight_line_color : str, optional
        Outline color for highlighted nodes. Default "darkred".
    highlight_size : float, optional
        Marker size for highlighted nodes. Default 11.
    highlight_alpha : float, optional
        Marker opacity for highlighted nodes. Must be in [0.0, 1.0].
        Default 0.98.
    width : int, optional
        Figure width in pixels, by default 900.
    height : int, optional
        Figure height in pixels, by default 700.
    reverse_axes : bool, optional
        Reverse X and Y axes to match quantum circuit layout convention,
        by default True.
    aspect_ratio : tuple[float, float, float] | None, optional
        Manual display scaling ratio for X/Y/Z axes. For example,
        ``(1.0, 1.0, 0.25)`` visually compresses Z. If None, Plotly
        auto-scales with ``aspectmode="data"`` (default).

    Returns
    -------
    go.Figure
        Plotly Figure object ready for display or further customization.

    Examples
    --------
    >>> fig = visualize_canvas_plotly(canvas)
    >>> fig.show()
    """
    node_size_scale_value = _validate_positive_float(node_size_scale, name="node_size_scale")
    edge_width_scale_value = _validate_positive_float(edge_width_scale, name="edge_width_scale")
    node_alpha_value = _validate_alpha(node_alpha, name="node_alpha")
    highlight_alpha_value = _validate_alpha(highlight_alpha, name="highlight_alpha")

    nodes = canvas.nodes
    coord2role = canvas.coord2role
    pauli_axes = canvas.pauli_axes
    highlight_set = set(highlight_nodes) if highlight_nodes is not None else set()
    groups = _group_nodes(nodes, coord2role)

    fig = go.Figure()

    for role, pts in groups.items():
        if not pts["coords"]:  # Check if any nodes exist for this role
            continue
        spec = _COLOR_MAP[role]
        coords: list[Coord3D] = pts["coords"]  # narrow type for mypy/pyright
        fig.add_trace(
            go.Scatter3d(
                x=pts["x"],
                y=pts["y"],
                z=pts["z"],
                mode="markers",
                marker={
                    "size": spec["size"] * node_size_scale_value,
                    "color": spec["color"],
                    "line": {"color": spec["line_color"], "width": 1.5},
                    "opacity": node_alpha_value,
                },
                name=spec["label"],
                text=[_node_hover_label(c, spec["label"], pauli_axes.get(c)) for c in coords],
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

    # overlay highlighted nodes (if any)
    if highlight_set:
        highlight_coords = [c for c in nodes if c in highlight_set]
        if highlight_coords:
            fig.add_trace(
                go.Scatter3d(
                    x=[c.x for c in highlight_coords],
                    y=[c.y for c in highlight_coords],
                    z=[c.z for c in highlight_coords],
                    mode="markers",
                    marker={
                        "size": float(highlight_size) * node_size_scale_value,
                        "color": highlight_color,
                        "line": {"color": highlight_line_color, "width": 2},
                        "opacity": highlight_alpha_value,
                        "symbol": "diamond",
                    },
                    name="Highlighted",
                    text=[_node_hover_label(c, "Highlighted", pauli_axes.get(c)) for c in highlight_coords],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    showlegend=True,
                )
            )

    edges = canvas.edges
    if show_edges and edges:
        edge_x, edge_y, edge_z = _edge_coordinates(edges, nodes)
        if edge_x:  # only add trace when at least one valid edge remains
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line={"color": edge_color, "width": edge_width * edge_width_scale_value},
                    name="Edges",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    scene = _build_plotly_scene(reverse_axes=reverse_axes, aspect_ratio=aspect_ratio)

    fig.update_layout(
        scene=scene,
        width=width,
        height=height,
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        template="plotly_white",
    )

    return fig


def render_canvas_z_window_plotly_figure(
    canvas: CanvasLike,
    *,
    current_z: int,
    z_window: int = 6,
    node_size_scale: float = 1.0,
    edge_width_scale: float = 1.0,
    tail_alpha: float = 0.25,
    non_current_alpha: float | None = None,
    current_alpha: float = 1.0,
    highlight_size_scale: float = 1.0,
    highlight_current_layer: bool = False,
    edge_color: str = "rgba(60, 60, 60, 0.7)",
    width: int = 900,
    height: int = 700,
    reverse_axes: bool = True,
    aspect_ratio: tuple[float, float, float] | None = None,
    lock_view: bool = True,
    axis_padding: float = 1.0,
    camera_eye: tuple[float, float, float] | None = (1.8, 1.8, 0.9),
    projection_type: str = "orthographic",
) -> go.Figure:
    """Render one z-sweep frame with a sliding z-window.

    Parameters
    ----------
    canvas : Canvas
        Canvas to render.
    current_z : int
        Current z-layer to highlight.
    z_window : int, optional
        Number of recent z-layers to keep visible, by default 6.
    node_size_scale : float, optional
        Global node marker size multiplier, by default 1.0.
    edge_width_scale : float, optional
        Global edge width multiplier, by default 1.0.
    tail_alpha : float, optional
        Opacity used for non-highlighted nodes in the visible z-window,
        by default 0.25.
    non_current_alpha : float | None, optional
        Explicit opacity for nodes outside ``current_z``. When provided, this
        value overrides ``tail_alpha``. Must be in [0.0, 1.0].
    current_alpha : float, optional
        Opacity used for highlighted current-layer nodes when
        ``highlight_current_layer`` is True, by default 1.0.
    highlight_size_scale : float, optional
        Multiplier applied to the regular node size baseline when
        ``highlight_current_layer`` is True, by default 1.0.
    highlight_current_layer : bool, optional
        Whether to highlight the current z-layer. Default False.
    edge_color : str, optional
        Edge color string, by default "rgba(60, 60, 60, 0.7)".
    width : int, optional
        Figure width in pixels, by default 900.
    height : int, optional
        Figure height in pixels, by default 700.
    reverse_axes : bool, optional
        Whether to reverse X/Y axes, by default True.
    aspect_ratio : tuple[float, float, float] | None, optional
        Manual display scaling ratio for X/Y/Z axes.
    lock_view : bool, optional
        Keep camera view and axis ranges fixed across frames. X/Y use full-canvas
        ranges, while Z uses the current sliding ``z_window`` range, by default True.
    axis_padding : float, optional
        Extra margin added to fixed axis ranges when ``lock_view`` is True.
        Must be non-negative. Default 1.0.
    camera_eye : tuple[float, float, float] | None, optional
        Fixed Plotly camera eye when ``lock_view`` is True.
        Set to None to keep Plotly default camera.
    projection_type : str, optional
        Plotly camera projection type. Use ``"orthographic"`` to suppress
        perspective size changes over depth. Default ``"orthographic"``.

    Returns
    -------
    go.Figure
        Plotly figure for a single z-window frame.
    """
    if z_window < 1:
        msg = "z_window must be at least 1."
        raise ValueError(msg)

    tail_alpha_value = _validate_alpha(tail_alpha, name="tail_alpha")
    if non_current_alpha is None:
        non_current_alpha_value = tail_alpha_value
    else:
        non_current_alpha_value = _validate_alpha(non_current_alpha, name="non_current_alpha")
    axis_padding_value = _validate_non_negative_float(axis_padding, name="axis_padding")
    projection_type_value = _validate_projection_type(projection_type)

    z_min = int(current_z) - z_window + 1
    visible_nodes = {node for node in canvas.nodes if z_min <= node.z <= current_z}
    visible_edges = {
        (start, end)
        for start, end in canvas.edges
        if start in visible_nodes and end in visible_nodes
    }

    window_canvas = cast(
        "CanvasLike",
        SimpleNamespace(
        nodes=visible_nodes,
        edges=visible_edges,
        coord2role=canvas.coord2role,
        pauli_axes=canvas.pauli_axes,
        ),
    )

    if highlight_current_layer:
        highlight_size_scale_value = _validate_positive_float(highlight_size_scale, name="highlight_size_scale")
        current_layer_nodes = {node for node in visible_nodes if node.z == current_z}
        base_highlight_size = max(spec["size"] for spec in _COLOR_MAP.values())
        highlight_size = max(float(base_highlight_size) * highlight_size_scale_value, 1.0)
        fig = visualize_canvas_plotly(
            window_canvas,
            show_edges=True,
            edge_width_scale=edge_width_scale,
            edge_color=edge_color,
            node_size_scale=node_size_scale,
            node_alpha=non_current_alpha_value,
            highlight_nodes=current_layer_nodes,
            highlight_size=highlight_size,
            highlight_alpha=current_alpha,
            width=width,
            height=height,
            reverse_axes=reverse_axes,
            aspect_ratio=aspect_ratio,
        )
    else:
        fig = visualize_canvas_plotly(
            window_canvas,
            show_edges=True,
            edge_width_scale=edge_width_scale,
            edge_color=edge_color,
            node_size_scale=node_size_scale,
            node_alpha=non_current_alpha_value,
            width=width,
            height=height,
            reverse_axes=reverse_axes,
            aspect_ratio=aspect_ratio,
        )

    if lock_view:
        x_range, y_range, _ = _compute_scene_axis_ranges(
            canvas.nodes,
            reverse_axes=reverse_axes,
            padding=axis_padding_value,
        )
        z_range = _compute_window_z_axis_range(
            current_z=current_z,
            z_window=z_window,
            padding=axis_padding_value,
        )
        x_span = abs(x_range[1] - x_range[0])
        y_span = abs(y_range[1] - y_range[0])
        z_span = abs(z_range[1] - z_range[0])
        manual_aspect = _normalize_manual_aspect_ratio(aspect_ratio)
        if manual_aspect is None:
            max_span = max(x_span, y_span, z_span)
            aspect_ratio_fixed = {"x": x_span / max_span, "y": y_span / max_span, "z": z_span / max_span}
        else:
            aspect_ratio_fixed = manual_aspect
        scene_update: dict[str, object] = {
            "xaxis": {"range": [x_range[0], x_range[1]], "autorange": False},
            "yaxis": {"range": [y_range[0], y_range[1]], "autorange": False},
            "zaxis": {"range": [z_range[0], z_range[1]], "autorange": False},
            "aspectmode": "manual",
            "aspectratio": aspect_ratio_fixed,
        }
        camera_update: dict[str, object] = {
            "projection": {"type": projection_type_value},
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        }
        if camera_eye is not None:
            try:
                eye_x, eye_y, eye_z = camera_eye
            except (TypeError, ValueError) as exc:
                msg = "camera_eye must be a tuple of three numbers or None."
                raise ValueError(msg) from exc
            camera_update["eye"] = {"x": float(eye_x), "y": float(eye_y), "z": float(eye_z)}
        scene_update["camera"] = camera_update
        fig.update_layout(scene=scene_update)

    fig.update_layout(title=f"Canvas Z-Sweep (z={current_z})")
    return fig


def visualize_detectors_plotly(
    detectors: Mapping[Coord3D, Iterable[NodeIndex]],
    *,
    canvas: CanvasLike | None = None,
    show_canvas_nodes: bool = True,
    show_canvas_edges: bool = True,
    show_node_indices_on_hover: bool = True,
    node_index_to_coord: Mapping[int, Coord3D] | None = None,
    highlight_nodes: Iterable[Coord3D] | None = None,
    highlight_color: str = "red",
    highlight_line_color: str = "darkred",
    highlight_size: int = 11,
    detector_color: str = "red",
    detector_line_color: str = "darkred",
    detector_marker_size: int = 9,
    edge_width: float = 3.0,
    edge_color: str = "rgba(60, 60, 60, 0.6)",
    width: int = 900,
    height: int = 700,
    reverse_axes: bool = True,
    aspect_ratio: tuple[float, float, float] | None = None,
) -> go.Figure:
    """Visualize detectors in 3D with Plotly and optionally show involved node indices on hover.

    Parameters
    ----------
    detectors : Mapping[Coord3D, Iterable[NodeIndex]]
        Mapping from detector coordinates to the set/list of node identifiers included
        in each detector. Identifiers may be raw integer node IDs or Coord3D values.
    canvas : Canvas | None, optional
        Canvas to render as background (nodes/edges). If None, only detectors are shown.
    show_canvas_nodes : bool, optional
        Whether to draw canvas nodes when `canvas` is provided. Default True.
    show_canvas_edges : bool, optional
        Whether to draw canvas edges when `canvas` is provided. Default True.
    show_node_indices_on_hover : bool, optional
        Show the list of node identifiers in hover text. Default True.
    node_index_to_coord : Mapping[int, Coord3D] | None, optional
        Optional lookup to convert integer node IDs back to Coord3D when formatting
        hover text. If provided, integers found in this mapping are displayed using
        their corresponding coordinates.
    highlight_nodes : Iterable[Coord3D] | None, optional
        Coordinates to highlight (plotted in red diamond markers) on the canvas background.
        Only used when `canvas` is provided. Default None.
    highlight_color : str, optional
        Fill color for highlighted nodes. Default "red".
    highlight_line_color : str, optional
        Outline color for highlighted nodes. Default "darkred".
    highlight_size : int, optional
        Marker size for highlighted nodes. Default 11.
    detector_color : str, optional
        Marker fill color for detectors. Default "red".
    detector_line_color : str, optional
        Marker outline color for detectors. Default "darkred".
    detector_marker_size : int, optional
        Marker size in pixels. Default 9.
    edge_width : float, optional
        Line width for canvas edges. Default 3.0.
    edge_color : str, optional
        Color for canvas edges. Default "rgba(60, 60, 60, 0.6)".
    width : int, optional
        Figure width in pixels. Default 900.
    height : int, optional
        Figure height in pixels. Default 700.
    reverse_axes : bool, optional
        Reverse X/Y axes to mimic circuit-style layout. Default True.
    aspect_ratio : tuple[float, float, float] | None, optional
        Manual display scaling ratio for X/Y/Z axes. For example,
        ``(1.0, 1.0, 0.25)`` visually compresses Z. If None, Plotly
        auto-scales with ``aspectmode="data"`` (default).

    Returns
    -------
    go.Figure
        Plotly Figure object ready for `fig.show()` or `fig.write_html(...)`.
    """

    fig = go.Figure()

    # Optionally draw canvas nodes/edges as background
    if canvas is not None and show_canvas_nodes:
        pauli_axes = canvas.pauli_axes
        highlight_set = set(highlight_nodes) if highlight_nodes is not None else set()
        groups = _group_nodes(canvas.nodes, canvas.coord2role)
        for role, pts in groups.items():
            if not pts["coords"]:
                continue
            spec = _COLOR_MAP[role]
            coords: list[Coord3D] = pts["coords"]
            fig.add_trace(
                go.Scatter3d(
                    x=pts["x"],
                    y=pts["y"],
                    z=pts["z"],
                    mode="markers",
                    marker={
                        "size": spec["size"],
                        "color": spec["color"],
                        "line": {"color": spec["line_color"], "width": 1.2},
                        "opacity": 0.45,
                    },
                    name=spec["label"],
                    text=[_node_hover_label(c, spec["label"], pauli_axes.get(c)) for c in coords],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    showlegend=False,
                )
            )

        if highlight_set:
            highlight_coords = [c for c in canvas.nodes if c in highlight_set]
            if highlight_coords:
                fig.add_trace(
                    go.Scatter3d(
                        x=[c.x for c in highlight_coords],
                        y=[c.y for c in highlight_coords],
                        z=[c.z for c in highlight_coords],
                        mode="markers",
                        marker={
                            "size": highlight_size,
                            "color": highlight_color,
                            "line": {"color": highlight_line_color, "width": 2},
                            "opacity": 0.98,
                            "symbol": "diamond",
                        },
                        name="Highlighted",
                        text=[_node_hover_label(c, "Highlighted", pauli_axes.get(c)) for c in highlight_coords],
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        showlegend=True,
                    )
                )

    if canvas is not None and show_canvas_edges:
        edge_x, edge_y, edge_z = _edge_coordinates(canvas.edges, canvas.nodes)
        if edge_x:
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line={"color": edge_color, "width": edge_width},
                    name="Edges",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    # Draw detectors
    det_x: list[int] = []
    det_y: list[int] = []
    det_z: list[int] = []
    det_hover: list[str] = []

    for coord, nodes in detectors.items():
        det_x.append(coord.x)
        det_y.append(coord.y)
        det_z.append(coord.z)

        if show_node_indices_on_hover:
            node_labels = _format_node_labels(nodes, node_index_to_coord)
            node_str = ", ".join(node_labels)
            det_hover.append(f"Detector ({coord.x}, {coord.y}, {coord.z})<br>nodes: [{node_str}]")
        else:
            det_hover.append(f"Detector ({coord.x}, {coord.y}, {coord.z})")

    fig.add_trace(
        go.Scatter3d(
            x=det_x,
            y=det_y,
            z=det_z,
            mode="markers",
            marker={
                "size": detector_marker_size,
                "color": detector_color,
                "line": {"color": detector_line_color, "width": 2},
                "symbol": "diamond",
                "opacity": 0.95,
            },
            name="Detector",
            text=det_hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    scene = _build_plotly_scene(reverse_axes=reverse_axes, aspect_ratio=aspect_ratio)

    fig.update_layout(
        scene=scene,
        width=width,
        height=height,
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        template="plotly_white",
    )

    return fig
