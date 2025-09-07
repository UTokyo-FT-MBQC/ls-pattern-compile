from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go


def _reverse_coord2node(coord2node: dict[tuple[int, int, int], int]) -> dict[int, tuple[int, int, int]]:
    node2coord: dict[int, tuple[int, int, int]] = {}
    for coord, nid in coord2node.items():
        node2coord[int(nid)] = (int(coord[0]), int(coord[1]), int(coord[2]))
    return node2coord


def visualize_compiled_canvas_plotly(
    cgraph,
    *,
    show_edges: bool = True,
    input_nodes: Iterable[int] | None = None,
    output_nodes: Iterable[int] | None = None,
    width: int = 800,
    height: int = 600,
    reverse_axes: bool = False,
    show_axes: bool = True,
    show_grid: bool = True,
):
    """CompiledRHGCanvas 可視化（Plotly 3D）。

    - ノードは z ごとに色分け（z 値をカラーに反映）。
    - エッジは任意で描画。
    - 入力/出力ノードは赤ダイヤで強調。
    """
    node2coord = _reverse_coord2node(cgraph.coord2node or {})
    g = cgraph.global_graph

    # main scatter: color by z
    xs, ys, zs, texts = [], [], [], []
    for nid, (x, y, z) in node2coord.items():
        xs.append(x); ys.append(y); zs.append(z)
        texts.append(f"Node {nid}")

    fig = go.Figure()
    if xs:
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=4,
                    color=zs,  # color by z
                    colorscale="Viridis",
                    colorbar=dict(title="z"),
                    line=dict(color="black", width=0.5),
                    opacity=0.95,
                ),
                name="Nodes",
                text=texts,
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    # edges
    if show_edges and g is not None and hasattr(g, "physical_edges"):
        edge_x: list[float] = []
        edge_y: list[float] = []
        edge_z: list[float] = []
        for u, v in g.physical_edges:
            if u in node2coord and v in node2coord:
                x1, y1, z1 = node2coord[u]
                x2, y2, z2 = node2coord[v]
                edge_x.extend([x1, x2, None])
                edge_y.extend([y1, y2, None])
                edge_z.extend([z1, z2, None])
        if edge_x:
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color="black", width=1),
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

    def _add_marker(nodes: Iterable[int], name: str, color: str):
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
                marker=dict(size=8, color=color, line=dict(color="darkred", width=2), symbol="diamond"),
                name=name,
            )
        )

    _add_marker(input_nodes, "Input", "white")
    _add_marker(output_nodes, "Output", "red")

    # layout
    scene: dict = dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="manual",
        aspectratio=dict(x=1.0, y=1.0, z=1.0),
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.2)),
    )
    if reverse_axes:
        scene["xaxis"] = dict(autorange="reversed")
        scene["yaxis"] = dict(autorange="reversed")

    def _axis_cfg(base: dict | None = None):
        base = dict(base or {})
        if show_axes:
            base.update(
                dict(
                    showgrid=bool(show_grid),
                    zeroline=True,
                    showline=True,
                    mirror=True,
                    ticks="outside",
                )
            )
        else:
            base.update(dict(visible=False))
        return base

    scene["xaxis"] = _axis_cfg(scene.get("xaxis"))
    scene["yaxis"] = _axis_cfg(scene.get("yaxis"))
    scene["zaxis"] = _axis_cfg(scene.get("zaxis"))

    fig.update_layout(
        title=f"Compiled RHG Canvas (layers={len(getattr(cgraph, 'layers', []))})",
        scene=scene,
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )

    return fig

