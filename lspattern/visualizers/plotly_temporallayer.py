from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from lspattern.consts import NodeRole, VisualizationKind
from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D
from lspattern.utils import infer_role

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lspattern.canvas import TemporalLayer


def visualize_temporal_layer_plotly(  # noqa: C901
    layer: TemporalLayer,
    *,
    node_roles: dict[NodeIdLocal, str] | None = None,
    ancilla_mode: VisualizationKind = VisualizationKind.BOTH,
    show_edges: bool = True,
    edge_width: float = 3.0,
    width: int = 800,
    height: int = 600,
    reverse_axes: bool = True,
    show_axes: bool = True,
    show_grid: bool = True,
    aspectmode: str = "cube",  # kept for backward-compat; ignored internally  # noqa: ARG001
    input_nodes: Iterable[int] | None = None,
    output_nodes: Iterable[int] | None = None,
) -> go.Figure:
    """Interactive 3D Plotly visualization for a TemporalLayer.

    Coloring/interaction is modeled after examples/visualize_initialize2.ipynb.

    - Groups nodes by role (data, ancilla_x, ancilla_z) if provided via
      `node_roles`. Otherwise, roles are inferred from parity.
    - Draws edges from `layer.local_graph.physical_edges` when available.
    - Highlights input/output nodes from GraphState registries if present.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive 3D Plotly figure for the temporal layer.

    Raises
    ------
    RuntimeError
        If plotly is not installed.
    """

    node2coord: dict[NodeIdLocal, PhysCoordGlobal3D] = layer.node2coord or {}
    g = layer.local_graph

    # Color mapping consistent with the notebook
    color_map = {
        NodeRole.DATA: {
            "color": "white",
            "line_color": "black",
            "size": 8,
            "name": "Data",
        },
        NodeRole.ANCILLA_X: {
            "color": "#2ecc71",
            "line_color": "#1e8449",
            "size": 7,
            "name": "X Ancilla",
        },
        NodeRole.ANCILLA_Z: {
            "color": "#3498db",
            "line_color": "#1f618d",
            "size": 7,
            "name": "Z Ancilla",
        },
    }

    # Build groups
    groups: dict[NodeRole, dict[str, list[int]]] = {k: {"x": [], "y": [], "z": [], "nodes": []} for k in color_map}

    # 役割は優先して TemporalLayer.node2role から取得(引数未指定時)
    if node_roles is None:
        node_roles = layer.node2role or None
    # Convert NodeIdLocal keys to int keys for compatibility
    node_roles_int: dict[int, str] | None = None
    if node_roles is not None:
        node_roles_int = {int(k): v for k, v in node_roles.items()}

    for n, coord in node2coord.items():
        role_str = node_roles_int.get(int(n)) if node_roles_int else None
        # Convert string role to NodeRole enum
        role: NodeRole
        if role_str in {NodeRole.DATA, NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}:
            role = NodeRole(role_str)
        else:
            role = infer_role(coord)
        if ancilla_mode == VisualizationKind.X and role == NodeRole.ANCILLA_Z:
            continue
        if ancilla_mode == VisualizationKind.Z and role == NodeRole.ANCILLA_X:
            continue
        groups[role]["x"].append(coord[0])
        groups[role]["y"].append(coord[1])
        groups[role]["z"].append(coord[2])
        groups[role]["nodes"].append(int(n))

    fig = go.Figure()

    # Add node traces
    for role, pts in groups.items():
        if not pts["x"]:
            continue
        spec = color_map[role]
        fig.add_trace(
            go.Scatter3d(
                x=pts["x"],
                y=pts["y"],
                z=pts["z"],
                mode="markers",
                marker={
                    "size": spec["size"],
                    "color": spec["color"],
                    "line": {"color": spec["line_color"], "width": 1},
                    "opacity": 0.9,
                },
                name=spec["name"],
                text=[f"Node {n}: {role}" for n in pts["nodes"]],
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    # Add edges
    if show_edges and g is not None and hasattr(g, "physical_edges"):
        edge_x: list[float] = []
        edge_y: list[float] = []
        edge_z: list[float] = []
        for u, v in g.physical_edges:
            if NodeIdLocal(u) in node2coord and NodeIdLocal(v) in node2coord:
                x1, y1, z1 = node2coord[NodeIdLocal(u)]
                x2, y2, z2 = node2coord[NodeIdLocal(v)]
                edge_x.extend([float(x1), float(x2), float("nan")])
                edge_y.extend([float(y1), float(y2), float("nan")])
                edge_z.extend([float(z1), float(z2), float("nan")])
        if edge_x:
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line={"color": "black", "width": edge_width},
                    name="Edges",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    # Highlight input/output nodes if present
    # Determine inputs/outputs: prefer explicit args, fall back to GraphState
    if input_nodes is None:
        input_nodes = list(g.input_node_indices.keys()) if g is not None else []
    if output_nodes is None:
        output_nodes = list(g.output_node_indices.keys()) if g is not None else []

    if input_nodes:
        xin = [node2coord[NodeIdLocal(n)][0] for n in input_nodes if NodeIdLocal(n) in node2coord]
        yin = [node2coord[NodeIdLocal(n)][1] for n in input_nodes if NodeIdLocal(n) in node2coord]
        zin = [node2coord[NodeIdLocal(n)][2] for n in input_nodes if NodeIdLocal(n) in node2coord]
        fig.add_trace(
            go.Scatter3d(
                x=xin,
                y=yin,
                z=zin,
                mode="markers",
                marker={
                    "size": 10,
                    "color": "white",
                    "line": {"color": "red", "width": 2},
                    "symbol": "diamond",
                },
                name="Input",
                text=[f"Input node {n}" for n in input_nodes],
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    if output_nodes:
        xout = [node2coord[NodeIdLocal(n)][0] for n in output_nodes if NodeIdLocal(n) in node2coord]
        yout = [node2coord[NodeIdLocal(n)][1] for n in output_nodes if NodeIdLocal(n) in node2coord]
        zout = [node2coord[NodeIdLocal(n)][2] for n in output_nodes if NodeIdLocal(n) in node2coord]
        fig.add_trace(
            go.Scatter3d(
                x=xout,
                y=yout,
                z=zout,
                mode="markers",
                marker={
                    "size": 10,
                    "color": "red",
                    "line": {"color": "darkred", "width": 2},
                    "symbol": "diamond",
                },
                name="Output",
                text=[f"Output node {n}" for n in output_nodes],
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    # Layout
    # 軸とレイアウト
    # Always fix aspect ratio to 1:1:1 regardless of cube/pipe/data ranges
    scene: dict[str, object] = {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "Z",
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 1.0},
        "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
    }
    if reverse_axes:
        scene["xaxis"] = {"autorange": "reversed"}
        scene["yaxis"] = {"autorange": "reversed"}

    # 軸の見た目制御
    def _axis_cfg(base: dict[str, object] | None = None) -> dict[str, object]:
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

    xaxis_obj = scene.get("xaxis")
    if isinstance(xaxis_obj, dict):
        scene["xaxis"] = _axis_cfg(xaxis_obj)
    else:
        scene["xaxis"] = _axis_cfg(None)

    yaxis_obj = scene.get("yaxis")
    if isinstance(yaxis_obj, dict):
        scene["yaxis"] = _axis_cfg(yaxis_obj)
    else:
        scene["yaxis"] = _axis_cfg(None)

    zaxis_obj = scene.get("zaxis")
    if isinstance(zaxis_obj, dict):
        scene["zaxis"] = _axis_cfg(zaxis_obj)
    else:
        scene["zaxis"] = _axis_cfg(None)

    fig.update_layout(
        title=f"Temporal Layer z={layer.z}",
        scene=scene,
        width=width,
        height=height,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    return fig
