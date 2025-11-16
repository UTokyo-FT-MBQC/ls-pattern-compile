from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from lspattern.canvas import CompiledRHGCanvas
    from lspattern.mytype import PhysCoordGlobal3D


def _reverse_coord2node(coord2node: Mapping[PhysCoordGlobal3D, int]) -> dict[int, tuple[int, int, int]]:
    """Return node->coord map from coord->node (CompiledRHGCanvas format)."""
    node2coord: dict[int, tuple[int, int, int]] = {}
    for coord, nid in coord2node.items():
        node2coord[int(nid)] = (int(coord[0]), int(coord[1]), int(coord[2]))
    return node2coord


def visualize_compiled_canvas(  # noqa: C901
    cgraph: CompiledRHGCanvas,
    *,
    annotate: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (7, 5),
    dpi: int = 120,
    show_axes: bool = True,
    show_grid: bool = True,
    show_edges: bool = True,
    color_by_z: bool = True,
    input_nodes: Iterable[int] | None = None,
    output_nodes: Iterable[int] | None = None,
) -> Figure:
    """CompiledRHGCanvas visualization (Matplotlib 3D).

    - Displays nodes as scatter plot using CompiledRHGCanvas's `coord2node`.
    - Draws edges if `global_graph.physical_edges` is available.
    - Input/output nodes are highlighted with red diamonds (uses GraphState properties if not specified).
    - Role information is not stored globally, so color is represented by z-based coloring (color_by_z=True).
    """
    node2coord = _reverse_coord2node(cgraph.coord2node or {})

    created_fig = False
    fig: Figure
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        temp_fig = ax.get_figure()
        # mypy doesn't understand matplotlib's Figure hierarchy properly
        fig = temp_fig  # type: ignore[assignment]
    if ax is None:
        msg = "ax should not be None here"
        raise ValueError(msg)

    # Axes and grid
    ax.set_box_aspect([1, 1, 1])  # type: ignore[arg-type]  # matplotlib 3D axis expects list
    ax.grid(bool(show_grid))
    if show_axes:
        ax.set_axis_on()
    else:
        ax.set_axis_off()

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    by_z: dict[int, dict[str, list[int]]] = {}
    for nid, (x, y, z) in node2coord.items():
        g = by_z.setdefault(int(z), {"x": [], "y": [], "z": [], "n": []})
        g["x"].append(int(x))
        g["y"].append(int(y))
        g["z"].append(int(z))
        g["n"].append(int(nid))

    for i, (z, pts) in enumerate(sorted(by_z.items())):
        color = palette[i % len(palette)] if color_by_z else "#ffffff"
        if pts["x"]:
            ax.scatter(
                pts["x"],
                pts["y"],
                zs=pts["z"],  # pyright: ignore[reportArgumentType]
                c=color,
                edgecolors="black",
                s=40,
                label=f"z={z}",
            )

    # Draw edges
    graph = cgraph.global_graph
    if show_edges and graph is not None and hasattr(graph, "physical_edges"):
        for u, v in graph.physical_edges:
            if u in node2coord and v in node2coord:
                x1, y1, z1 = node2coord[u]
                x2, y2, z2 = node2coord[v]
                ax.plot([x1, x2], [y1, y2], [z1, z2], c="gray", linewidth=1, alpha=0.5)

    # Highlight input/output nodes (red diamonds)
    if input_nodes is None and graph is not None and hasattr(graph, "input_node_indices"):
        try:
            input_nodes = list(graph.input_node_indices.keys())
        except AttributeError:
            input_nodes = []
    if output_nodes is None and graph is not None and hasattr(graph, "output_node_indices"):
        try:
            output_nodes = list(graph.output_node_indices.keys())
        except AttributeError:
            output_nodes = []

    def _scatter_marker(nodes: Iterable[int], face: str, color: str) -> None:
        nodes = list(nodes or [])
        if not nodes:
            return
        xs = [node2coord[n][0] for n in nodes if n in node2coord]
        ys = [node2coord[n][1] for n in nodes if n in node2coord]
        zs = [node2coord[n][2] for n in nodes if n in node2coord]
        if xs:
            ax.scatter(
                xs,
                ys,
                zs=zs,  # pyright: ignore[reportArgumentType]
                c=color,
                edgecolors="darkred",
                s=70,
                marker="D",
                label=face,
            )

    _scatter_marker(input_nodes or [], "Input", "white")
    _scatter_marker(output_nodes or [], "Output", "red")

    # Annotations
    if annotate:
        for nid, (x, y, z) in node2coord.items():
            ax.text(x, y, z, str(nid), fontsize=6)  # type: ignore[arg-type]

    ax.legend(loc="best")

    # Save/display
    if save_path:
        p = pathlib.Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p))
    if created_fig and show:
        plt.show()

    return ax.get_figure()  # type: ignore[return-value]
