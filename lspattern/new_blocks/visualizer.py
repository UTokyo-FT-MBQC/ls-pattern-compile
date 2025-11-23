from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import plotly.graph_objects as go

from lspattern.new_blocks.mytype import Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lspattern.new_blocks.canvas import Canvas


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
                text=[f"{spec['label']} ({c.x}, {c.y}, {c.z})" for c in coords],
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
