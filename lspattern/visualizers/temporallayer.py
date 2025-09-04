from __future__ import annotations

from typing import Optional, Tuple, Dict

import os
import matplotlib.pyplot as plt

from lspattern.geom.rhg_parity import is_data, is_ancilla_x, is_ancilla_z


def _color_for_coord(x: int, y: int, z: int) -> str:
    """Return node color based on RHG parity, matching existing visualizers.

    - DATA: white (even z), red (odd z)
    - ancilla-X: blue
    - ancilla-Z: green
    """
    if is_data(x, y, z):
        return "white" if (z % 2 == 0) else "red"
    if is_ancilla_x(x, y, z):
        return "blue"
    if is_ancilla_z(x, y, z):
        return "green"
    return "gray"


def visualize_temporal_layer(
    layer,
    *,
    indicated_nodes: set[int] | None = None,
    annotate: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 120,
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

    node2coord: Dict[int, Tuple[int, int, int]] = getattr(layer, "node2coord", {}) or {}

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_axis_off()

    # Scatter nodes
    xs, ys, zs, colors = [], [], [], []
    for nid, (x, y, z) in node2coord.items():
        xs.append(x)
        ys.append(y)
        zs.append(z)
        colors.append(_color_for_coord(x, y, z))

    if xs:
        ax.scatter(
            xs,
            ys,
            zs,
            c=colors,
            edgecolors="black",
            s=50,
            depthshade=True,
            label="nodes",
        )

    # Draw edges if we have a local graph
    local_graph = getattr(layer, "local_graph", None)
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
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to: {save_path}")

    if show:
        # In Jupyter show() is fine; in headless skip
        try:
            get_ipython()  # type: ignore[name-defined]
            plt.show()
        except NameError:
            if os.environ.get("DISPLAY") or os.name == "nt":
                plt.show()
            else:
                print("Display not available; use save_path to save the figure.")
    else:
        plt.close(fig)

    return fig, ax
