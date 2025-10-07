from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lspattern.consts import VisualizationKind, VisualizationMode
from lspattern.mytype import NodeIdLocal
from lspattern.visualizers.plotly_temporallayer import visualize_temporal_layer_plotly
from lspattern.visualizers.temporallayer import visualize_temporal_layer

if TYPE_CHECKING:
    from lspattern.canvas import TemporalLayer


# Unified colors (match Plotly temporallayer visualizer palette)
COLOR_DATA = "white"
COLOR_X = "#2ecc71"  # ancilla X / X-related
COLOR_Z = "#3498db"  # ancilla Z / Z-related
COLOR_EDGE = "#555555"


# ----------------------
# Matplotlib visualizers
# ----------------------
def visualize_parity_mpl(  # noqa: C901
    layer: TemporalLayer,
    *,
    kind: VisualizationKind = VisualizationKind.BOTH,
    annotate: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    node2coord = layer.node2coord or {}
    par = layer.parity

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig_maybe_subfig = ax.get_figure()
        assert fig_maybe_subfig is not None  # noqa: S101
        # Get the root figure if this is a SubFigure
        fig = getattr(fig_maybe_subfig, "figure", None) or fig_maybe_subfig  # type: ignore[assignment]

    with contextlib.suppress(Exception):
        ax.set_box_aspect((1, 1, 1))  # type: ignore[arg-type]
    ax.grid(False)
    ax.set_axis_off()

    # Draw parity groups as nodes; optionally connect to emphasize grouping
    def draw_groups(groups: list[set[NodeIdLocal]], color: str, label: str) -> None:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for grp in groups:
            for n in grp:
                if n in node2coord:
                    x, y, z = node2coord[n]
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
        if xs:
            ax.scatter(xs, ys, zs, s=20, c=color, edgecolors="black", label=label, alpha=0.9)  # type: ignore[misc]

    if kind in {VisualizationKind.BOTH, VisualizationKind.X}:
        checks = [grp for coord_dict in par.checks.values() for grp in coord_dict.values()]
        draw_groups(checks, COLOR_X, "Parity X")

    if annotate:
        for nid, (x, y, z) in node2coord.items():
            ax.text(x, y, z, str(nid), color="black", fontsize=7)  # type: ignore[arg-type]

    ax.set_title(f"Parity (z={layer.z})")
    ax.legend()

    if save_path is not None:
        # Use root figure for saving if fig is a SubFigure
        root_fig = getattr(fig, "figure", fig)
        root_fig.savefig(save_path, bbox_inches="tight", dpi=120)  # pyright: ignore[reportAttributeAccessIssue]

    if show and created_fig:
        plt.show()

    return ax


def visualize_flow_mpl(  # noqa: C901
    layer: TemporalLayer,
    *,
    kind: VisualizationKind = VisualizationKind.BOTH,
    max_edges: int | None = None,
    save_path: str | None = None,
    show: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    node2coord = layer.node2coord or {}
    flow = layer.flow

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig_maybe_subfig = ax.get_figure()
        assert fig_maybe_subfig is not None  # noqa: S101
        # Get the root figure if this is a SubFigure
        fig = getattr(fig_maybe_subfig, "figure", None) or fig_maybe_subfig  # type: ignore[assignment]

    with contextlib.suppress(Exception):
        ax.set_box_aspect((1, 1, 1))  # type: ignore[arg-type]
    ax.grid(False)
    ax.set_axis_off()

    # Draw edges for flow relations
    def draw_edges(edges: dict[NodeIdLocal, set[NodeIdLocal]], color: str, _label: str) -> None:
        count = 0
        for u, vs in edges.items():
            for v in vs:
                if u in node2coord and v in node2coord:
                    x1, y1, z1 = node2coord[u]
                    x2, y2, z2 = node2coord[v]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], c=color, linewidth=2, alpha=0.8)
                    count += 1
                    if max_edges is not None and count >= max_edges:
                        return

    if kind in {VisualizationKind.BOTH, VisualizationKind.X} and flow.flow:
        draw_edges(flow.flow, COLOR_X, "X-flow")

    # Draw nodes lightly for context
    xs = [node2coord[n][0] for n in node2coord]
    ys = [node2coord[n][1] for n in node2coord]
    zs = [node2coord[n][2] for n in node2coord]
    if xs:
        ax.scatter(xs, ys, zs, s=10, c=COLOR_DATA, edgecolors="black", alpha=0.3, label="nodes")  # type: ignore[misc]

    ax.set_title(f"Flow (z={layer.z})")
    ax.legend()

    if save_path is not None:
        # Use root figure for saving if fig is a SubFigure
        root_fig = getattr(fig, "figure", fig)
        root_fig.savefig(save_path, bbox_inches="tight", dpi=120)  # pyright: ignore[reportAttributeAccessIssue]

    if show and created_fig:
        plt.show()

    return ax


def visualize_schedule_mpl(
    layer: TemporalLayer,
    *,
    mode: VisualizationMode = VisualizationMode.HIST,
    times: list[int] | None = None,
    save_path: str | None = None,
    show: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    sched = layer.schedule.schedule if getattr(layer, "schedule", None) else {}
    node2coord = layer.node2coord or {}

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 4), dpi=120)
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.get_figure()  # type: ignore[assignment]
        assert fig is not None  # noqa: S101

    if mode == VisualizationMode.HIST:
        ts = sorted(sched.keys())
        counts = [len(sched[t]) for t in ts]
        ax.bar(ts, counts, color="#888888")
        ax.set_xlabel("time (z)")
        ax.set_ylabel("#measured nodes")
        ax.set_title("Schedule (hist)")
    else:
        # slices: plot XY scatter for selected times
        if times is None:
            times = sorted(sched.keys())[:2]
        colors = [COLOR_X, COLOR_Z, "#aaaaaa", "#ff9900"]
        for i, t in enumerate(times):
            xs: list[float] = []
            ys: list[float] = []
            for n in sched.get(t, set()):
                n_local = NodeIdLocal(int(n))
                if n_local in node2coord:
                    x, y, _z = node2coord[n_local]
                    xs.append(x)
                    ys.append(y)
            if xs:
                ax.scatter(xs, ys, c=colors[i % len(colors)], s=20, edgecolors="black", label=f"t={t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Schedule (slices)")
        # Enforce equal XY aspect
        with contextlib.suppress(Exception):
            ax.set_aspect("equal", adjustable="box")
        ax.legend()

    if save_path is not None:
        # Use root figure for saving if fig is a SubFigure
        root_fig = getattr(fig, "figure", fig)
        root_fig.savefig(save_path, bbox_inches="tight", dpi=120)  # pyright: ignore[reportAttributeAccessIssue]
    if show and created_fig:
        plt.show()
    return ax


def visualize_temporal_layer_2x2_mpl(
    layer: TemporalLayer,
    *,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (12, 9),
    dpi: int = 120,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax11 = fig.add_subplot(221, projection="3d")
    ax12 = fig.add_subplot(222, projection="3d")
    ax21 = fig.add_subplot(223, projection="3d")
    ax22 = fig.add_subplot(224)

    visualize_temporal_layer(layer, show=False, ax=ax11)
    visualize_parity_mpl(layer, show=False, ax=ax12)
    visualize_flow_mpl(layer, show=False, ax=ax21)
    visualize_schedule_mpl(layer, mode=VisualizationMode.HIST, show=False, ax=ax22)

    fig.tight_layout()
    if save_path is not None:
        # Use root figure for saving if fig is a SubFigure
        root_fig = getattr(fig, "figure", fig)
        root_fig.savefig(save_path, bbox_inches="tight", dpi=dpi)  # pyright: ignore[reportAttributeAccessIssue]
    if show:
        plt.show()
    return fig


# -------------------
# Plotly visualizers
# -------------------
def visualize_parity_plotly(
    layer: TemporalLayer,
    *,
    kind: VisualizationKind = VisualizationKind.BOTH,
) -> go.Figure:
    node2coord = layer.node2coord or {}
    par = layer.parity

    fig = go.Figure()

    def add_group(groups: list[set[NodeIdLocal]], color: str, name: str) -> None:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for grp in groups:
            for n in grp:
                if n in node2coord:
                    x, y, z = node2coord[n]
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
        if xs:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker={"size": 5, "color": color, "line": {"color": "#000", "width": 1}},
                    name=name,
                )
            )

    if kind in {VisualizationKind.BOTH, VisualizationKind.X}:
        checks = [grp for coord_dict in par.checks.values() for grp in coord_dict.values()]
        add_group(checks, COLOR_X, "Parity X")

    fig.update_layout(
        title=f"Parity (z={layer.z})",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def visualize_flow_plotly(
    layer: TemporalLayer,
    *,
    kind: VisualizationKind = VisualizationKind.BOTH,
    max_edges: int | None = None,
) -> go.Figure:
    node2coord = layer.node2coord or {}
    flow = layer.flow

    fig = go.Figure()
    count = 0

    def add_edges(edges: dict[NodeIdLocal, set[NodeIdLocal]], color: str, name: str) -> None:
        nonlocal count
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        for u, vs in edges.items():
            for v in vs:
                if u in node2coord and v in node2coord:
                    x1, y1, z1 = node2coord[u]
                    x2, y2, z2 = node2coord[v]
                    edge_x.extend([x1, x2, None])
                    edge_y.extend([y1, y2, None])
                    edge_z.extend([z1, z2, None])
                    count += 1
                    if max_edges is not None and count >= max_edges:
                        break
        if edge_x:
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line={"color": color, "width": 4},
                    name=name,
                    hoverinfo="none",
                )
            )

    if kind in {VisualizationKind.BOTH, VisualizationKind.X} and flow.flow:
        add_edges(flow.flow, COLOR_X, "X-flow")

    fig.update_layout(
        title=f"Flow (z={layer.z})",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def visualize_schedule_plotly(
    layer: TemporalLayer,
    *,
    mode: VisualizationMode = VisualizationMode.HIST,
    times: list[int] | None = None,
) -> go.Figure:
    sched = layer.schedule.schedule if getattr(layer, "schedule", None) else {}
    node2coord = layer.node2coord or {}

    if mode == VisualizationMode.HIST:
        ts = sorted(sched.keys())
        counts = [len(sched[t]) for t in ts]
        return go.Figure(
            data=[go.Bar(x=ts, y=counts, marker={"color": "#888"})],
            layout=go.Layout(title="Schedule (hist)", xaxis_title="time (z)", yaxis_title="#measured"),
        )

    # slices mode
    fig = go.Figure()
    if times is None:
        times = sorted(sched.keys())[:2]
    colors = [COLOR_X, COLOR_Z, "#aaaaaa", "#ff9900"]
    for i, t in enumerate(times):
        xs: list[float] = []
        ys: list[float] = []
        for n in sched.get(t, set()):
            n_local = NodeIdLocal(int(n))
            if n_local in node2coord:
                x, y, _z = node2coord[n_local]
                xs.append(x)
                ys.append(y)
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers", marker={"color": colors[i % len(colors)], "size": 7}, name=f"t={t}"
                )
            )
    # Enforce equal XY aspect using scale anchors
    fig.update_layout(
        title="Schedule (slices)",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        yaxis={"constrain": "domain"},
    )
    return fig


def visualize_temporal_layer_2x2_plotly(layer: TemporalLayer) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Layer", "Parity", "Flow", "Schedule"),
    )

    # Layer
    fig_layer = visualize_temporal_layer_plotly(layer)
    for tr in fig_layer.data:
        fig.add_trace(tr, row=1, col=1)

    # Parity
    fig_par = visualize_parity_plotly(layer)
    for tr in fig_par.data:
        fig.add_trace(tr, row=1, col=2)

    # Flow
    fig_flow = visualize_flow_plotly(layer)
    for tr in fig_flow.data:
        fig.add_trace(tr, row=2, col=1)

    # Schedule (hist 2D)
    fig_sched = visualize_schedule_plotly(layer, mode=VisualizationMode.HIST)
    for tr in fig_sched.data:
        fig.add_trace(tr, row=2, col=2)

    # Set scene layout for three 3D panes
    fig.update_layout(
        scene={"aspectmode": "cube"},
        scene2={"aspectmode": "cube"},
        scene3={"aspectmode": "cube"},
        height=900,
        width=1200,
        title_text=f"TemporalLayer Overview z={layer.z}",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    return fig
