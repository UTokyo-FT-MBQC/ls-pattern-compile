# ruff: noqa: PLR1702
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lspattern.canvas import CompiledRHGCanvas
    from lspattern.mytype import PhysCoordGlobal3D


def _reverse_coord2node(
    coord2node: Mapping[PhysCoordGlobal3D, int],
) -> dict[int, tuple[int, int, int]]:
    node2coord: dict[int, tuple[int, int, int]] = {}
    for coord, nid in coord2node.items():
        node2coord[int(nid)] = (int(coord[0]), int(coord[1]), int(coord[2]))
    return node2coord


def visualize_cgraph_xparity(  # noqa: C901
    cgraph: CompiledRHGCanvas,
    *,
    show_edges: bool = True,
    hilight_nodes: Iterable[int] | None = None,
    width: int = 800,
    height: int = 600,
    show_axes: bool = True,
    show_grid: bool = True,
    show_xparity: bool = True,
) -> go.Figure:
    """CompiledRHGCanvas 可視化(Plotly 3D)。"""

    raw_node2coord = cgraph.node2coord or {}
    node2coord: dict[int, tuple[int, int, int]] = {
        int(nid): (int(coord[0]), int(coord[1]), int(coord[2]))
        for nid, coord in raw_node2coord.items()
    }
    if not node2coord:
        node2coord = _reverse_coord2node(cgraph.coord2node or {})

    g = cgraph.global_graph

    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []
    texts: list[str] = []
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
                    "color": zs,
                    "colorscale": "Viridis",
                    "colorbar": {"title": "z"},
                    "line": {"color": "black", "width": 0.5},
                    "opacity": 0.9,
                },
                name="Nodes",
                text=texts,
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    if show_edges and g is not None and hasattr(g, "physical_edges"):
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        for u, v in g.physical_edges:
            if int(u) in node2coord and int(v) in node2coord:
                x1, y1, z1 = node2coord[int(u)]
                x2, y2, z2 = node2coord[int(v)]
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
                    ordered = [
                        (node_id, node2coord[node_id])
                        for n in nodes
                        if (node_id := int(n)) in node2coord
                    ]
                    if len(ordered) < 2:
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
                        "size": 5,
                        "color": "red",
                        "line": {"color": "black", "width": 1},
                        "opacity": 0.95,
                    },
                    name="Highlight",
                    hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                    text=htext,
                    showlegend=True,
                )
            )

    scene: dict[str, Any] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 1.0},
        "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.2}},
    }

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
        title=f"Compiled RHG Canvas (layers={len(cgraph.layers)})",
        scene=scene,
        width=width,
        height=height,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        legend={"itemsizing": "constant"},
    )

    return fig
