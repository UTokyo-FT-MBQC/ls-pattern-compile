from __future__ import annotations

from collections.abc import Iterable


def visualize_temporal_layer_plotly(
    layer,
    *,
    node_roles: dict[int, str] | None = None,
    ancilla_mode: str = "both",  # 'both' | 'x' | 'z'
    show_edges: bool = True,
    edge_width: float = 3.0,
    width: int = 800,
    height: int = 600,
    reverse_axes: bool = True,
):
    """Interactive 3D Plotly visualization for a TemporalLayer.

    Coloring/interaction is modeled after examples/visualize_initialize2.ipynb.

    - Groups nodes by role (data, ancilla_x, ancilla_z) if provided via
      `node_roles`. Otherwise, roles are inferred from parity.
    - Draws edges from `layer.local_graph.physical_edges` when available.
    - Highlights input/output nodes from GraphState registries if present.
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "plotly is required for visualize_temporal_layer_plotly.\nInstall via `pip install plotly`."
        ) from e

    # Lazy import parity helpers
    from lspattern.geom.rhg_parity import is_ancilla_x, is_ancilla_z, is_data

    node2coord: dict[int, tuple[int, int, int]] = getattr(layer, "node2coord", {}) or {}
    g = getattr(layer, "local_graph", None)

    # Color mapping consistent with the notebook
    color_map = {
        "data": {"color": "white", "line_color": "black", "size": 8, "name": "Data"},
        "ancilla_x": {
            "color": "#2ecc71",
            "line_color": "#1e8449",
            "size": 7,
            "name": "X Ancilla",
        },
        "ancilla_z": {
            "color": "#3498db",
            "line_color": "#1f618d",
            "size": 7,
            "name": "Z Ancilla",
        },
    }

    # Build groups
    groups: dict[str, dict[str, list]] = {k: {"x": [], "y": [], "z": [], "nodes": []} for k in color_map}

    def infer_role(coord: tuple[int, int, int]) -> str:
        x, y, z = coord
        if is_data(x, y, z):
            return "data"
        if is_ancilla_x(x, y, z):
            return "ancilla_x"
        if is_ancilla_z(x, y, z):
            return "ancilla_z"
        return "data"

    # Roles are retrieved from TemporalLayer.node2role with priority (when arguments not specified)
    if node_roles is None:
        node_roles = getattr(layer, "node2role", {}) or None

    for n, coord in node2coord.items():
        role = node_roles.get(n) if node_roles else None
        if role not in ("data", "ancilla_x", "ancilla_z"):
            role = infer_role(coord)
        if ancilla_mode == "x" and role == "ancilla_z":
            continue
        if ancilla_mode == "z" and role == "ancilla_x":
            continue
        groups[role]["x"].append(coord[0])
        groups[role]["y"].append(coord[1])
        groups[role]["z"].append(coord[2])
        groups[role]["nodes"].append(n)

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
                marker=dict(
                    size=spec["size"],
                    color=spec["color"],
                    line=dict(color=spec["line_color"], width=1),
                    opacity=0.9,
                ),
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
                    line=dict(color="black", width=edge_width),
                    name="Edges",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    # Highlight input/output nodes if present
    try:
        in_nodes: Iterable[int] = list(getattr(g, "input_node_indices", {}).keys())
        out_nodes: Iterable[int] = list(getattr(g, "output_node_indices", {}).keys())
    except Exception:
        in_nodes, out_nodes = [], []

    if in_nodes:
        xin = [node2coord[n][0] for n in in_nodes if n in node2coord]
        yin = [node2coord[n][1] for n in in_nodes if n in node2coord]
        zin = [node2coord[n][2] for n in in_nodes if n in node2coord]
        fig.add_trace(
            go.Scatter3d(
                x=xin,
                y=yin,
                z=zin,
                mode="markers",
                marker=dict(size=10, color="red", symbol="diamond"),
                name="Input",
                text=[f"Input node {n}" for n in in_nodes],
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    if out_nodes:
        xout = [node2coord[n][0] for n in out_nodes if n in node2coord]
        yout = [node2coord[n][1] for n in out_nodes if n in node2coord]
        zout = [node2coord[n][2] for n in out_nodes if n in node2coord]
        fig.add_trace(
            go.Scatter3d(
                x=xout,
                y=yout,
                z=zout,
                mode="markers",
                marker=dict(size=10, color="darkred", symbol="diamond"),
                name="Output",
                text=[f"Output node {n}" for n in out_nodes],
                hovertemplate="<b>%{text}</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
            )
        )

    # Layout
    scene = dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    )
    if reverse_axes:
        scene["xaxis"] = dict(autorange="reversed")
        scene["yaxis"] = dict(autorange="reversed")

    fig.update_layout(
        title=f"Temporal Layer z={getattr(layer, 'z', '?')}",
        scene=scene,
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig
