from __future__ import annotations

import os
import pathlib
from typing import Any

import matplotlib.pyplot as plt

from lspattern.geom.rhg_parity import is_ancilla_x, is_ancilla_z, is_data


def _node_to_coord(canvas: Any) -> dict[int, tuple[int, int, int]]:
    """Invert canvas.coord_to_node -> node -> (x,y,z)."""
    return {nid: coord for coord, nid in canvas.coord_to_node.items()}


def visualize_canvas(
    canvas: Any,
    *,
    indicated_nodes: set[int] | None = None,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 120,
) -> None:
    """Visualizes the Raussendorf lattice with nodes colored based on their parity.
    Nodes with allowed parities are colored white, others are red.
    Physical edges are drawn in gray.

    Parameters
    ----------
    canvas : RHGCanvas
        The canvas describing MBQC on the Raussendorf lattice.
    save_path : Optional[str], optional
        Path to save the figure. If None, the figure is not saved.
        Directory will be created if it doesn't exist, by default None
    show : bool, optional
        Whether to display the figure. Set to False for non-interactive environments
        or when only saving is desired, by default True
    figsize : tuple[int, int], optional
        Figure size in inches (width, height), by default (6, 6)
    dpi : int, optional
        Figure resolution in dots per inch, by default 120

    """
    node2coord = _node_to_coord(canvas)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))  # Set aspect ratio to be equal
    ax.grid(False)
    ax.set_axis_off()

    xs, ys, zs = [], [], []
    colors = []
    for x, y, z in node2coord.values():
        xs.append(x)
        ys.append(y)
        zs.append(z)

        if is_data(x, y, z) and z % 2 == 0:
            colors.append("white")
        elif is_data(x, y, z) and z % 2 == 1:
            colors.append("red")
        elif is_ancilla_x(x, y, z):
            colors.append("blue")
        elif is_ancilla_z(x, y, z):
            colors.append("green")

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

    for u, v in canvas.graph.physical_edges:
        # Extract coordinates from coord2node
        x1, y1, z1 = node2coord[u]
        x2, y2, z2 = node2coord[v]
        ax.plot([x1, x2], [y1, y2], [z1, z2], c="gray", linewidth=1, alpha=0.5)
    if indicated_nodes is not None:
        for node in indicated_nodes:
            x, y, z = node2coord[node]
            ax.scatter(
                x,
                y,
                z,
                c="black",
                edgecolors="black",
                s=50,
                # depthshade=True,
            )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.tight_layout()

    # Save figure if path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_dir = pathlib.Path(save_path).parent
        if str(save_dir) and not save_dir.exists():
            save_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to: {save_path}")

    # Show figure if requested and in interactive mode
    if show:
        # Check if we're in a Jupyter notebook
        try:
            get_ipython()  # type: ignore[name-defined]
            # In Jupyter, just display the plot
            plt.show()
        except NameError:
            # Not in Jupyter, check if display is available
            if os.environ.get("DISPLAY") or os.name == "nt":
                plt.show()
            else:
                print("Display not available. Use save_path parameter to save the figure.")
    else:
        plt.close(fig)
