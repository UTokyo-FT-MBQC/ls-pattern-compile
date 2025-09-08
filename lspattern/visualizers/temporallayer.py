from __future__ import annotations

import os
import pathlib

import matplotlib.pyplot as plt

from lspattern.geom.rhg_parity import is_ancilla_x, is_ancilla_z


def visualize_temporal_layer(
    layer,
    *,
    indicated_nodes: set[int] | None = None,
    input_nodes: set[int] | None = None,
    output_nodes: set[int] | None = None,
    annotate: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax=None,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 120,
    show_axes: bool = True,
    show_grid: bool = True,
):
    """Visualize a single TemporalLayer in 3D with parity-based coloring.

    Parameters
    ----------
    layer : TemporalLayer
        The temporal layer with `node2coord` mapping and optional `local_graph`.
    indicated_nodes : set[int] | None
        Optional set of node ids to highlight in black.
    annotate : bool
        If True, annotate nodes with their ids.
    save_path : str | None
        If provided, save the figure to this path (directories created as needed).
    show : bool
        If True, display the plot; otherwise, close the figure after saving.
    figsize : tuple[int, int]
        Matplotlib figure size.
    dpi : int
        Matplotlib figure DPI.
    """
    node2coord: dict[int, tuple[int, int, int]] = layer.node2coord or {}

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()
    ax.set_box_aspect((1, 1, 1))
    # 軸とグリッドの表示制御(デフォルトON)
    if show_axes:
        ax.set_axis_on()
    else:
        ax.set_axis_off()
    ax.grid(bool(show_grid))

    # 役割ベースでグルーピングして凡例を表示(z 偶奇による分岐は行わない)
    roles: dict[int, str] = layer.node2role or {}
    groups: dict[str, dict[str, list]] = {
        "data": {"x": [], "y": [], "z": []},
        "ancilla_x": {"x": [], "y": [], "z": []},
        "ancilla_z": {"x": [], "y": [], "z": []},
    }
    for nid, (x, y, z) in node2coord.items():
        role = roles.get(nid)
        if role == "ancilla_x":
            g = groups["ancilla_x"]
        elif role == "ancilla_z":
            g = groups["ancilla_z"]
        # 役割がない場合はパリティから推定(それでも ancilla 判定されなければ data 扱い)
        elif role is None:
            if is_ancilla_x(x, y, z):
                g = groups["ancilla_x"]
            elif is_ancilla_z(x, y, z):
                g = groups["ancilla_z"]
            else:
                g = groups["data"]
        else:
            g = groups["data"]
        g["x"].append(x)
        g["y"].append(y)
        g["z"].append(z)

    def scat(gkey: str, color: str, label: str | None):
        pts = groups[gkey]
        if pts["x"]:
            ax.scatter(
                pts["x"], pts["y"], pts["z"],
                c=color,
                edgecolors="black",
                s=50,
                depthshade=True,
                label=label,
            )

    scat("data", "white", "data")
    # unify palette with Plotly temporallayer: X=green, Z=blue
    scat("ancilla_x", "#2ecc71", "ancilla X")
    scat("ancilla_z", "#3498db", "ancilla Z")

    # Draw edges if we have a local graph
    local_graph = layer.local_graph
    if local_graph is not None and hasattr(local_graph, "physical_edges"):
        for u, v in local_graph.physical_edges:
            if u in node2coord and v in node2coord:
                x1, y1, z1 = node2coord[u]
                x2, y2, z2 = node2coord[v]
                ax.plot([x1, x2], [y1, y2], [z1, z2], c="gray", linewidth=1, alpha=0.5)

    # Highlight indicated nodes
    if indicated_nodes:
        for nid in indicated_nodes:
            if nid in node2coord:
                x, y, z = node2coord[nid]
                ax.scatter(x, y, z, c="black", edgecolors="black", s=55)

    # Highlight input/output nodes with legend
    if input_nodes:
        xin = [node2coord[n][0] for n in input_nodes if n in node2coord]
        yin = [node2coord[n][1] for n in input_nodes if n in node2coord]
        zin = [node2coord[n][2] for n in input_nodes if n in node2coord]
        if xin:
            ax.scatter(
                xin,
                yin,
                zin,
                s=70,
                facecolors="white",
                edgecolors="#e74c3c",  # softer red
                linewidths=1.8,
                marker="D",
                label="Input",
            )
    if output_nodes:
        xout = [node2coord[n][0] for n in output_nodes if n in node2coord]
        yout = [node2coord[n][1] for n in output_nodes if n in node2coord]
        zout = [node2coord[n][2] for n in output_nodes if n in node2coord]
        if xout:
            ax.scatter(
                xout,
                yout,
                zout,
                s=70,
                c="#e74c3c",          # softer red fill
                edgecolors="#c0392b",  # darker red edge
                linewidths=1.8,
                marker="D",
                label="Output",
            )

    # Optional annotations
    if annotate:
        for nid, (x, y, z) in node2coord.items():
            ax.text(x, y, z, str(nid), color="black", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.tight_layout()

    # Save figure if requested
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not pathlib.Path(save_dir).exists():
            pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to: {save_path}")

    if show and created_fig:
        # In Jupyter show() is fine; in headless skip
        try:
            get_ipython()  # type: ignore[name-defined]
            plt.show()
        except NameError:
            if os.environ.get("DISPLAY") or os.name == "nt":
                plt.show()
            else:
                print("Display not available; use save_path to save the figure.")
    elif created_fig is False:
        # When embedding into external figure, do not manage closing.
        pass
    else:
        plt.close(fig)

    return fig, ax
