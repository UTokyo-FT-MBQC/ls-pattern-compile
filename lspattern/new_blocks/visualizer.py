from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, TypedDict, TypeAlias

import plotly.graph_objects as go

from graphqomb.common import Axis

from lspattern.new_blocks.mytype import Coord3D, NodeRole

if TYPE_CHECKING:
    from lspattern.new_blocks.canvas import Canvas

NodeIndex: TypeAlias = int | Coord3D


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
                num = int(node)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                num = None
            key = (1, num, 0, 0) if num is not None else (2, 0, 0, 0)
            label = str(node)
        sortable.append((key, label))

    sortable.sort(key=lambda item: item[0])

    labels: list[str] = []
    seen: set[str] = set()
    for _, label in sortable:
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def visualize_canvas_plotly(
    canvas: Canvas,
    *,
    show_edges: bool = True,
    edge_width: float = 3.0,
    edge_color: str = "rgba(60, 60, 60, 0.7)",
    width: int = 900,
    height: int = 700,
    reverse_axes: bool = True,
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
    edge_color : str, optional
        RGBA color string for edges, by default "rgba(60, 60, 60, 0.7)".
    width : int, optional
        Figure width in pixels, by default 900.
    height : int, optional
        Figure height in pixels, by default 700.
    reverse_axes : bool, optional
        Reverse X and Y axes to match quantum circuit layout convention,
        by default True.

    Returns
    -------
    go.Figure
        Plotly Figure object ready for display or further customization.

    Examples
    --------
    >>> fig = visualize_canvas_plotly(canvas)
    >>> fig.show()
    """

    nodes = canvas.nodes
    coord2role = canvas.coord2role
    pauli_axes = canvas.pauli_axes
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
                    "size": spec["size"],
                    "color": spec["color"],
                    "line": {"color": spec["line_color"], "width": 1.5},
                    "opacity": 0.9,
                },
                name=spec["label"],
                text=[_node_hover_label(c, spec["label"], pauli_axes.get(c)) for c in coords],
                hovertemplate="<b>%{text}</b><extra></extra>",
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
                    line={"color": edge_color, "width": edge_width},
                    name="Edges",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    scene: dict[str, object] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
        "aspectmode": "data",
    }
    if reverse_axes:
        scene["xaxis"] = {"autorange": "reversed"}
        scene["yaxis"] = {"autorange": "reversed"}

    fig.update_layout(
        scene=scene,
        width=width,
        height=height,
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        template="plotly_white",
    )

    return fig


def visualize_detectors_plotly(
    detectors: Mapping[Coord3D, Iterable[NodeIndex]],
    *,
    canvas: Canvas | None = None,
    show_canvas_nodes: bool = True,
    show_canvas_edges: bool = True,
    show_node_indices_on_hover: bool = True,
    node_index_to_coord: Mapping[int, Coord3D] | None = None,
    detector_color: str = "red",
    detector_line_color: str = "darkred",
    detector_marker_size: int = 9,
    edge_width: float = 3.0,
    edge_color: str = "rgba(60, 60, 60, 0.6)",
    width: int = 900,
    height: int = 700,
    reverse_axes: bool = True,
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

    Returns
    -------
    go.Figure
        Plotly Figure object ready for `fig.show()` or `fig.write_html(...)`.
    """

    fig = go.Figure()

    # オプションでキャンバスのノード・エッジを背景に描画
    if canvas is not None and show_canvas_nodes:
        pauli_axes = canvas.pauli_axes
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

    # Detector を描画
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

    scene: dict[str, object] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
        "aspectmode": "data",
    }
    if reverse_axes:
        scene["xaxis"] = {"autorange": "reversed"}
        scene["yaxis"] = {"autorange": "reversed"}

    fig.update_layout(
        scene=scene,
        width=width,
        height=height,
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        template="plotly_white",
    )

    return fig
