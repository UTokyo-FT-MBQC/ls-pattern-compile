from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    import matplotlib.axes
    import matplotlib.figure
    import plotly.graph_objects as go


def _ensure_mpl() -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:  # pragma: no cover
        msg = "matplotlib is required for accumulator visualizers.\nInstall via `pip install matplotlib`."
        raise RuntimeError(msg) from e


def _ensure_plotly() -> None:
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception as e:  # pragma: no cover
        msg = "plotly is required for accumulator visualizers.\nInstall via `pip install plotly`."
        raise RuntimeError(msg) from e


# Unified colors (match Plotly temporallayer visualizer palette)
COLOR_DATA = "white"
COLOR_X = "#2ecc71"  # ancilla X / X-related
COLOR_Z = "#3498db"  # ancilla Z / Z-related
COLOR_EDGE = "#555555"


# ----------------------
# Matplotlib visualizers
# ----------------------
def visualize_parity_mpl(
    layer,
    *,
    kind: Literal["both", "x", "z"] = "both",
    annotate: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}
    par = layer.parity

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_axis_off()

    # Draw parity groups as nodes; optionally connect to emphasize grouping
    def draw_groups(groups: list[set[int]], color: str, label: str) -> None:
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
            ax.scatter(xs, ys, zs, c=color, edgecolors="black", s=20, label=label, alpha=0.9)

    if kind in {"both", "x"}:
        draw_groups(par.x_checks, COLOR_X, "Parity X")
    if kind in {"both", "z"}:
        draw_groups(par.z_checks, COLOR_Z, "Parity Z")

    if annotate:
        for nid, (x, y, z) in node2coord.items():
            ax.text(x, y, z, str(nid), color="black", fontsize=7)

    ax.set_title(f"Parity (z={layer.z})")
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    if show and created_fig:
        plt.show()

    return ax


def visualize_flow_mpl(
    layer,
    *,
    kind: Literal["both", "x", "z"] = "both",
    max_edges: int | None = None,
    save_path: str | None = None,
    show: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}
    flow = layer.flow

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_axis_off()

    # Draw edges for flow relations
    def draw_edges(edges: dict[int, set[int]], color: str, label: str) -> None:
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

    if kind in {"both", "x"} and flow.xflow:
        draw_edges(flow.xflow, COLOR_X, "X-flow")
    if kind in {"both", "z"} and flow.zflow:
        draw_edges(flow.zflow, COLOR_Z, "Z-flow")

    # Draw nodes lightly for context
    xs = [node2coord[n][0] for n in node2coord]
    ys = [node2coord[n][1] for n in node2coord]
    zs = [node2coord[n][2] for n in node2coord]
    if xs:
        ax.scatter(xs, ys, zs, c=COLOR_DATA, edgecolors="black", s=10, alpha=0.3, label="nodes")

    ax.set_title(f"Flow (z={layer.z})")
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)

    if show and created_fig:
        plt.show()

    return ax


def visualize_schedule_mpl(
    layer,
    *,
    mode: Literal["hist", "slices"] = "hist",
    times: list[int] | None = None,
    save_path: str | None = None,
    show: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    sched = layer.schedule.schedule if getattr(layer, "schedule", None) else {}
    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 4), dpi=120)
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.get_figure()

    if mode == "hist":
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
                if n in node2coord:
                    x, y, _z = node2coord[n]
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
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    if show and created_fig:
        plt.show()
    return ax


def visualize_detectors_mpl(
    layer,
    *,
    detector=None,
    annotate: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # lazy import to avoid circular
    try:
        from lspattern.accumulator import DetectorAccumulator
    except Exception:  # pragma: no cover
        DetectorAccumulator = None  # type: ignore

    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()

    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_axis_off()

    # Build detector accumulator on the fly if not provided
    if detector is None and DetectorAccumulator is not None:
        det = DetectorAccumulator()
        ancillas = [n for n, r in layer.node2role.items() if str(r).startswith("ancilla")]
        for a in ancillas:
            det.update_at(a, layer)
    else:
        det = detector

    # draw all detectors as edges from anchor to data neighbors
    if det is not None:
        for a, group in det.detectors.items():
            if a not in node2coord:
                continue
            x1, y1, z1 = node2coord[a]
            for n in group:
                if n not in node2coord:
                    continue
                x2, y2, z2 = node2coord[n]
                ax.plot([x1, x2], [y1, y2], [z1, z2], c=COLOR_EDGE, linewidth=1.2, alpha=0.9)

    # also scatter anchors and data
    xs = [node2coord[n][0] for n in node2coord]
    ys = [node2coord[n][1] for n in node2coord]
    zs = [node2coord[n][2] for n in node2coord]
    ax.scatter(xs, ys, zs, c=COLOR_DATA, edgecolors="black", s=10, alpha=0.3, label="nodes")

    if annotate:
        for nid, (x, y, z) in node2coord.items():
            ax.text(x, y, z, str(nid), color="black", fontsize=7)

    ax.set_title(f"Detectors (z={layer.z})")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    if show and created_fig:
        plt.show()
    return ax


def visualize_temporal_layer_2x2_mpl(
    layer,
    *,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (12, 9),
    dpi: int = 120,
) -> matplotlib.figure.Figure:
    _ensure_mpl()
    import matplotlib.pyplot as plt

    from lspattern.visualizers.temporallayer import visualize_temporal_layer

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax11 = fig.add_subplot(221, projection="3d")
    ax12 = fig.add_subplot(222, projection="3d")
    ax21 = fig.add_subplot(223, projection="3d")
    ax22 = fig.add_subplot(224)

    visualize_temporal_layer(layer, show=False, ax=ax11)
    visualize_parity_mpl(layer, show=False, ax=ax12)
    visualize_flow_mpl(layer, show=False, ax=ax21)
    visualize_schedule_mpl(layer, mode="hist", show=False, ax=ax22)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    return fig


# -------------------
# Plotly visualizers
# -------------------
def visualize_parity_plotly(
    layer,
    *,
    kind: Literal["both", "x", "z"] = "both",
) -> go.Figure:
    _ensure_plotly()
    import plotly.graph_objects as go

    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}
    par = layer.parity

    fig = go.Figure()

    def add_group(groups: list[set[int]], color: str, name: str) -> None:
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

    if kind in {"both", "x"}:
        add_group(par.x_checks, COLOR_X, "Parity X")
    if kind in {"both", "z"}:
        add_group(par.z_checks, COLOR_Z, "Parity Z")

    fig.update_layout(
        title=f"Parity (z={layer.z})",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def visualize_flow_plotly(
    layer,
    *,
    kind: Literal["both", "x", "z"] = "both",
    max_edges: int | None = None,
) -> go.Figure:
    _ensure_plotly()
    import plotly.graph_objects as go

    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}
    flow = layer.flow

    fig = go.Figure()
    count = 0

    def add_edges(edges: dict[int, set[int]], color: str, name: str) -> None:
        nonlocal count
        edge_x: list[float] = []
        edge_y: list[float] = []
        edge_z: list[float] = []
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

    if kind in {"both", "x"} and flow.xflow:
        add_edges(flow.xflow, COLOR_X, "X-flow")
    if kind in {"both", "z"} and flow.zflow:
        add_edges(flow.zflow, COLOR_Z, "Z-flow")

    fig.update_layout(
        title=f"Flow (z={layer.z})",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def visualize_schedule_plotly(
    layer,
    *,
    mode: Literal["hist", "slices"] = "hist",
    times: list[int] | None = None,
) -> go.Figure:
    _ensure_plotly()
    import plotly.graph_objects as go

    sched = layer.schedule.schedule if getattr(layer, "schedule", None) else {}
    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}

    if mode == "hist":
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
            if n in node2coord:
                x, y, _z = node2coord[n]
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


def visualize_detectors_plotly(layer, *, detector=None) -> go.Figure:
    _ensure_plotly()
    import plotly.graph_objects as go

    try:
        from lspattern.accumulator import DetectorAccumulator
    except Exception:  # pragma: no cover
        DetectorAccumulator = None  # type: ignore

    node2coord: dict[int, Sequence[int]] = layer.node2coord or {}

    if detector is None and DetectorAccumulator is not None:
        det = DetectorAccumulator()
        ancillas = [n for n, r in layer.node2role.items() if str(r).startswith("ancilla")]
        for a in ancillas:
            det.update_at(a, layer)
    else:
        det = detector

    fig = go.Figure()
    # Edges
    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_z: list[float] = []
    if det is not None:
        for a, group in det.detectors.items():
            if a not in node2coord:
                continue
            x1, y1, z1 = node2coord[a]
            for n in group:
                if n not in node2coord:
                    continue
                x2, y2, z2 = node2coord[n]
                edge_x.extend([x1, x2, None])
                edge_y.extend([y1, y2, None])
                edge_z.extend([z1, z2, None])
    if edge_x:
        fig.add_trace(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z, mode="lines", line={"color": COLOR_EDGE, "width": 3}, name="detectors"
            )
        )

    # Nodes (context)
    xs = [node2coord[n][0] for n in node2coord]
    ys = [node2coord[n][1] for n in node2coord]
    zs = [node2coord[n][2] for n in node2coord]
    if xs:
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker={"size": 4, "color": COLOR_DATA, "line": {"color": "#000", "width": 1}},
                name="nodes",
                opacity=0.5,
            )
        )

    fig.update_layout(
        title=f"Detectors (z={layer.z})",
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def visualize_temporal_layer_2x2_plotly(layer) -> go.Figure:
    _ensure_plotly()
    from plotly.subplots import make_subplots

    from lspattern.visualizers.plotly_temporallayer import visualize_temporal_layer_plotly

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
    fig_sched = visualize_schedule_plotly(layer, mode="hist")
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
