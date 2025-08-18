
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import os

from lspattern.geom.rhg_parity import is_data, is_ancilla_x, is_ancilla_z

def _node_to_coord(canvas) -> Dict[int, Tuple[int,int,int]]:
    """Invert canvas.coord_to_node -> node -> (x,y,z)."""
    return { nid: coord for coord, nid in canvas.coord_to_node.items() }

def visualize_canvas(
    canvas,
    *,
    indicated_nodes: set[int] | None = None,
    annotate: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 120,
):
    """Quick visualization of an RHGCanvas with parity-aware node types.

    - Data nodes and ancilla (X/Z) nodes are drawn with different markers.
    - Colors are *not* specified (uses matplotlib defaults).
    """

    node2coord = _node_to_coord(canvas)
    nodes = list(node2coord.keys())
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))  # Set aspect ratio to be equal
    ax.grid(False)
    ax.set_axis_off()
    
    xs, ys, zs = [], [], []
    colors = []
    for _, (x, y, z) in node2coord.items():
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
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Figure saved to: {save_path}")

    # Show figure if requested and in interactive mode
    if show:
        # Check if we're in a Jupyter notebook
        try:
            get_ipython()  # type: ignore
            # In Jupyter, just display the plot
            plt.show()
        except NameError:
            # Not in Jupyter, check if display is available
            if os.environ.get("DISPLAY") or os.name == "nt":
                plt.show()
            else:
                print(
                    "Display not available. Use save_path parameter to save the figure."
                )
    else:
        plt.close(fig)
