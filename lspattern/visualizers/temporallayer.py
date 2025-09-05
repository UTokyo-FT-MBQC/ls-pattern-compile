from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from lspattern.geom.rhg_parity import is_ancilla_x, is_ancilla_z


def visualize_temporal_layer(  # noqa: C901, PLR0912, PLR0913
    layer: Any,
    *,
    indicated_nodes: set[int] | None = None,
    annotate: bool = False,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 120,
) -> None:
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
    node2coord: dict[int, tuple[int, int, int]] = getattr(layer, "node2coord", {}) or {}

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_axis_off()

    # Group by role and display legend (no branching by z even/odd)
    roles: dict[int, str] = getattr(layer, "node2role", {}) or {}
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
        # If no role, estimate from parity (if still not ancilla, treat as data)
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

    def scat(gkey: str, color: str, label: str | None) -> None:
        pts = groups[gkey]
        if pts["x"]:
            ax.scatter(
                pts["x"],
                pts["y"],
                pts["z"],
                c=color,
                edgecolors="black",
                s=50,
                depthshade=True,
                label=label,
            )

    scat("data", "white", "data")
    scat("ancilla_x", "blue", "ancilla X")
    scat("ancilla_z", "green", "ancilla Z")

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
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
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
