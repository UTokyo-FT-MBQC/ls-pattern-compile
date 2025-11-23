from __future__ import annotations

from collections.abc import Iterable, Mapping

import plotly.graph_objects as go

from lspattern.new_blocks.canvas import Canvas
from lspattern.new_blocks.mytype import Coord3D, NodeRole


_COLOR_MAP: dict[NodeRole, dict[str, object]] = {
    NodeRole.DATA: {"color": "white", "line_color": "black", "size": 8, "label": "Data"},
    NodeRole.ANCILLA_X: {"color": "green", "line_color": "darkgreen", "size": 7, "label": "X ancilla"},
    NodeRole.ANCILLA_Z: {"color": "blue", "line_color": "darkblue", "size": 7, "label": "Z ancilla"},
}


def _group_nodes(
    nodes: Iterable[Coord3D],
    coord2role: Mapping[Coord3D, NodeRole],
) -> dict[NodeRole, dict[str, list[object]]]:
    groups: dict[NodeRole, dict[str, list[object]]] = {
        role: {"x": [], "y": [], "z": [], "coords": []} for role in _COLOR_MAP
    }
    for coord in nodes:
        role = coord2role.get(coord, NodeRole.DATA)
        groups[role]["x"].append(coord.x)
        groups[role]["y"].append(coord.y)
        groups[role]["z"].append(coord.z)
        groups[role]["coords"].append(coord)
    return groups


def _edge_coordinates(edges: Iterable[tuple[Coord3D, Coord3D]]) -> tuple[list[float], list[float], list[float]]:
    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_z: list[float] = []
    for start, end in edges:
        edge_x.extend([float(start.x), float(end.x), float("nan")])
        edge_y.extend([float(start.y), float(end.y), float("nan")])
        edge_z.extend([float(start.z), float(end.z), float("nan")])
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
    """Plot a Canvas with Plotly using node coordinates."""

    nodes = canvas.nodes
    coord2role = canvas.coord2role
    groups = _group_nodes(nodes, coord2role)

    fig = go.Figure()

    for role, pts in groups.items():
        if not pts["x"]:
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
        edge_x, edge_y, edge_z = _edge_coordinates(edges)
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
