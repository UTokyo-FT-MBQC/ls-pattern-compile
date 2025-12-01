# ruff: noqa: PLR1702
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lspattern.canvas import CompiledRHGCanvas
    from lspattern.mytype import PhysCoordGlobal3D

# Minimum number of nodes required to visualize parity connections
_MIN_PARITY_NODES = 2


def _reverse_coord2node(
    coord2node: Mapping[PhysCoordGlobal3D, int],
) -> dict[int, tuple[int, int, int]]:
    node2coord: dict[int, tuple[int, int, int]] = {}
    for coord, nid in coord2node.items():
        node2coord[int(nid)] = (int(coord[0]), int(coord[1]), int(coord[2]))
    return node2coord


def visualize_compiled_canvas_plotly(  # noqa: C901
    cgraph: CompiledRHGCanvas,
    *,
    show_edges: bool = True,
    input_nodes: Iterable[int] | None = None,
    output_nodes: Iterable[int] | None = None,
    hilight_nodes: Iterable[int] | None = None,
    width: int = 800,
    height: int = 600,
    reverse_axes: bool = False,
    show_axes: bool = True,
    show_grid: bool = True,
    show_xparity: bool = True,
    zratio: float = 1.0,
) -> go.Figure:
    """CompiledRHGCanvas visualization (Plotly 3D).

    - Nodes are colored by z (z value is reflected in color).
    - Edges are drawn optionally.
    - Input/output nodes are highlighted with red diamonds.
    """

    node2coord = _reverse_coord2node(cgraph.coord2node or {})
    g = cgraph.global_graph

    # main scatter: color by z
    xs, ys, zs, texts = [], [], [], []
    for nid, (x, y, z) in node2coord.items():
        xs.append(x)
        ys.append(y)
        zs.append(z)
        texts.append(f"Node {nid}")

    fig = go.Figure()
    if xs:
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker={
                    "size": 4,
                    "color": zs,  # color by z
                    "colorscale": "Viridis",
                    "colorbar": {"title": "z"},
                    "line": {"color": "black", "width": 0.5},
                    "opacity": 0.95,
                },
                name="Nodes",
                text=texts,
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    # edges
    if show_edges and g is not None and hasattr(g, "physical_edges"):
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        for u, v in g.physical_edges:
            if u in node2coord and v in node2coord:
                x1, y1, z1 = node2coord[u]
                x2, y2, z2 = node2coord[v]
                edge_x.extend([float(x1), float(x2), None])
                edge_y.extend([float(y1), float(y2), None])
                edge_z.extend([float(z1), float(z2), None])
        if edge_x:
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line={"color": "black", "width": 1},
                    name="Edges",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    # inputs/outputs
    if input_nodes is None and g is not None and hasattr(g, "input_node_indices"):
        input_nodes = list(g.input_node_indices.keys())
    if output_nodes is None and g is not None and hasattr(g, "output_node_indices"):
        output_nodes = list(g.output_node_indices.keys())

    def _add_marker(nodes: Iterable[int], name: str, color: str) -> None:
        nodes = list(nodes or [])
        if not nodes:
            return
        xin = [node2coord[n][0] for n in nodes if n in node2coord]
        yin = [node2coord[n][1] for n in nodes if n in node2coord]
        zin = [node2coord[n][2] for n in nodes if n in node2coord]
        fig.add_trace(
            go.Scatter3d(
                x=xin,
                y=yin,
                z=zin,
                mode="markers",
                marker={
                    "size": 8,
                    "color": color,
                    "line": {"color": "darkred", "width": 2},
                    "symbol": "diamond",
                },
                name=name,
            )
        )

    _add_marker(input_nodes or [], "Input", "white")
    _add_marker(output_nodes or [], "Output", "red")

    if show_xparity and cgraph.parity is not None:
        checks = cgraph.parity.checks
        if checks:
            line_x: list[float | None] = []
            line_y: list[float | None] = []
            line_z: list[float | None] = []
            line_text: list[str | None] = []
            cone_x: list[float] = []
            cone_y: list[float] = []
            cone_z: list[float] = []
            cone_u: list[float] = []
            cone_v: list[float] = []
            cone_w: list[float] = []
            seen_pairs: set[tuple[int, int]] = set()

            for groups in checks.values():
                for nodes in groups.values():
                    ordered = [(node_id, node2coord[node_id]) for n in nodes if (node_id := int(n)) in node2coord]
                    if len(ordered) < _MIN_PARITY_NODES:
                        continue
                    ordered.sort(key=lambda item: (item[1][2], item[1][0], item[1][1], item[0]))
                    for idx in range(len(ordered) - 1):
                        start_id, start_coord = ordered[idx]
                        end_id, end_coord = ordered[idx + 1]
                        if start_id == end_id:
                            continue
                        pair = (start_id, end_id)
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        line_x.extend([float(start_coord[0]), float(end_coord[0]), None])
                        line_y.extend([float(start_coord[1]), float(end_coord[1]), None])
                        line_z.extend([float(start_coord[2]), float(end_coord[2]), None])
                        label = f"X parity: {start_id} → {end_id}"
                        line_text.extend([label, label, None])
                        cone_x.append(float(end_coord[0]))
                        cone_y.append(float(end_coord[1]))
                        cone_z.append(float(end_coord[2]))
                        cone_u.append(float(end_coord[0] - start_coord[0]))
                        cone_v.append(float(end_coord[1] - start_coord[1]))
                        cone_w.append(float(end_coord[2] - start_coord[2]))

            if line_x:
                fig.add_trace(
                    go.Scatter3d(
                        x=line_x,
                        y=line_y,
                        z=line_z,
                        mode="lines",
                        line={"color": "#e74c3c", "width": 1.5},
                        name="X parity",
                        hoverinfo="text",
                        text=line_text,
                        showlegend=True,
                    )
                )
            if cone_x:
                fig.add_trace(
                    go.Cone(
                        x=cone_x,
                        y=cone_y,
                        z=cone_z,
                        u=cone_u,
                        v=cone_v,
                        w=cone_w,
                        colorscale=[[0, "#e74c3c"], [1, "#e74c3c"]],
                        showscale=False,
                        sizemode="absolute",
                        sizeref=1.5,
                        anchor="tip",
                        name="X parity arrow",
                        hoverinfo="skip",
                    )
                )

    # highlighted nodes (same shape as base scatter, colored red)
    if hilight_nodes:
        highlight = [int(n) for n in hilight_nodes if int(n) in node2coord]
        if highlight:
            hx = [node2coord[n][0] for n in highlight]
            hy = [node2coord[n][1] for n in highlight]
            hz = [node2coord[n][2] for n in highlight]
            htext = [f"Node {n}" for n in highlight]
            fig.add_trace(
                go.Scatter3d(
                    x=hx,
                    y=hy,
                    z=hz,
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": "red",
                        "line": {"color": "black", "width": 0.5},
                        "opacity": 0.95,
                    },
                    name="Highlight",
                    hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                    text=htext,
                    showlegend=True,
                )
            )

    # layout
    scene: dict[str, Any] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": zratio},
        "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.2}},
    }
    if reverse_axes:
        scene["xaxis"] = {"autorange": "reversed"}
        scene["yaxis"] = {"autorange": "reversed"}

    def _axis_cfg(base: dict[str, Any] | None = None) -> dict[str, Any]:
        base = dict(base or {})
        if show_axes:
            base.update(
                {
                    "showgrid": bool(show_grid),
                    "zeroline": True,
                    "showline": True,
                    "mirror": True,
                    "ticks": "outside",
                }
            )
        else:
            base.update({"visible": False})
        return base

    scene["xaxis"] = _axis_cfg(scene.get("xaxis") if isinstance(scene.get("xaxis"), dict) else None)
    scene["yaxis"] = _axis_cfg(scene.get("yaxis") if isinstance(scene.get("yaxis"), dict) else None)
    scene["zaxis"] = _axis_cfg(scene.get("zaxis") if isinstance(scene.get("zaxis"), dict) else None)

    fig.update_layout(
        title=f"Compiled RHG Canvas (layers={len(getattr(cgraph, 'layers', []))})",
        scene=scene,
        width=width,
        height=height,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        legend={"itemsizing": "constant"},
    )

    return fig