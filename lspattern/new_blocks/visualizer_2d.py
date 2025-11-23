from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt

from lspattern.new_blocks.mytype import Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from matplotlib.figure import Figure

    from lspattern.new_blocks.canvas import Canvas


class NodeStyleSpec(TypedDict):
    """Style specification for node visualization.

    Attributes
    ----------
    color : str
        Fill color of the node marker.
    line_color : str
        Border color of the node marker.
    size : int
        Size of the node marker in pixels.
    label : str
        Display label for the node type in the legend.
    """

    color: str
    line_color: str
    size: int
    label: str


class NodeGroup2D(TypedDict):
    """Grouped node data for 2D visualization.

    Attributes
    ----------
    x : list[int]
        List of x coordinates.
    y : list[int]
        List of y coordinates.
    coords : list[Coord3D]
        List of Coord3D objects.
    """

    x: list[int]
    y: list[int]
    coords: list[Coord3D]


_COLOR_MAP: dict[NodeRole, NodeStyleSpec] = {
    NodeRole.DATA: {"color": "white", "line_color": "black", "size": 100, "label": "Data"},
    NodeRole.ANCILLA_X: {"color": "green", "line_color": "darkgreen", "size": 80, "label": "X ancilla"},
    NodeRole.ANCILLA_Z: {"color": "blue", "line_color": "darkblue", "size": 80, "label": "Z ancilla"},
}


def _group_nodes_2d(
    nodes: Iterable[Coord3D],
    coord2role: Mapping[Coord3D, NodeRole],
) -> dict[NodeRole, NodeGroup2D]:
    """Group nodes by their role for 2D visualization traces.

    Parameters
    ----------
    nodes : Iterable[Coord3D]
        Iterable of 3D coordinates representing nodes in the canvas.
    coord2role : Mapping[Coord3D, NodeRole]
        Mapping from coordinates to their roles. Nodes not in the mapping
        default to NodeRole.DATA.

    Returns
    -------
    dict[NodeRole, NodeGroup2D]
        Dictionary mapping each node role to its grouped 2D coordinate data.
        Each NodeGroup2D contains separate lists for x, y coordinates
        and the original Coord3D objects.
    """
    groups: dict[NodeRole, NodeGroup2D] = {role: {"x": [], "y": [], "coords": []} for role in _COLOR_MAP}
    for coord in nodes:
        role = coord2role.get(coord, NodeRole.DATA)
        groups[role]["x"].append(coord.x)
        groups[role]["y"].append(coord.y)
        groups[role]["coords"].append(coord)
    return groups


def visualize_canvas_matplotlib_2d(
    canvas: Canvas,
    target_z: int,
    *,
    show_edges: bool = True,
    edge_color: str = "gray",
    edge_width: float = 0.5,
    edge_alpha: float = 0.5,
    figsize: tuple[int, int] = (8, 8),
    show_grid: bool = True,
    reverse_axes: bool = False,
) -> Figure:
    """Create a 2D visualization of a Canvas at a specific Z-slice using Matplotlib.

    This function renders nodes at a specific Z coordinate, colored by their role
    (data, X ancilla, Z ancilla) and optionally displays edges connecting them.
    The visualization uses 2D scatter plots with customizable styling.

    Parameters
    ----------
    canvas : Canvas
        The Canvas object to visualize, containing nodes, edges, and role information.
    target_z : int
        The Z coordinate of the slice to visualize.
    show_edges : bool, optional
        Whether to display edges between nodes, by default True.
    edge_color : str, optional
        Color string for edges, by default "gray".
    edge_width : float, optional
        Width of edge lines, by default 0.5.
    edge_alpha : float, optional
        Transparency of edge lines (0.0 to 1.0), by default 0.5.
    figsize : tuple[int, int], optional
        Figure size in inches (width, height), by default (8, 8).
    show_grid : bool, optional
        Whether to show grid lines, by default True.
    reverse_axes : bool, optional
        Reverse X and Y axes to match quantum circuit layout convention,
        by default False.

    Returns
    -------
    Figure
        Matplotlib Figure object ready for display or further customization.

    Examples
    --------
    >>> fig = visualize_canvas_matplotlib_2d(canvas, target_z=0)
    >>> plt.show()
    """
    # Filter nodes at target Z coordinate
    nodes_at_z = {node for node in canvas.nodes if node.z == target_z}

    # Group nodes by role
    coord2role = canvas.coord2role
    groups = _group_nodes_2d(nodes_at_z, coord2role)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot nodes grouped by role
    for role, pts in groups.items():
        if not pts["coords"]:  # Skip empty groups
            continue
        spec = _COLOR_MAP[role]
        ax.scatter(
            pts["x"],
            pts["y"],
            c=spec["color"],
            edgecolors=spec["line_color"],
            s=spec["size"],
            label=spec["label"],
            alpha=0.9,
            linewidths=1.5,
            zorder=2,  # Draw nodes on top of edges
        )

    # Draw edges if requested
    if show_edges:
        edges = canvas.edges
        for start, end in edges:
            # Only draw edges where both endpoints are in the current Z slice
            if start not in nodes_at_z or end not in nodes_at_z:
                continue
            ax.plot(
                [start.x, end.x],
                [start.y, end.y],
                color=edge_color,
                linewidth=edge_width,
                alpha=edge_alpha,
                zorder=1,  # Draw edges behind nodes
            )

    # Configure axes
    ax.set_aspect("equal", "box")
    ax.grid(show_grid, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Canvas 2D Slice at Z={target_z}")
    ax.legend(loc="best")

    # Reverse axes if requested (quantum circuit convention)
    if reverse_axes:
        # Reverse setting: invert X axis only
        ax.invert_xaxis()
    else:
        # Default: invert Y axis only
        ax.invert_yaxis()

    return fig
