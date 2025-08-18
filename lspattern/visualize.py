
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

from .geom.rhg_parity import is_data, is_ancilla_x, is_ancilla_z

def _node_to_coord(canvas) -> Dict[int, Tuple[int,int,int]]:
    """Invert canvas.coord_to_node -> node -> (x,y,z)."""
    return { nid: coord for coord, nid in canvas.coord_to_node.items() }

def visualize_canvas(
    canvas,
    *,
    projection: str = "xz",   # "xz", "xy", "yz", or "3d"
    show_nodes: bool = True,
    show_edges: bool = True,
    annotate: bool = False,
    figsize: Tuple[int,int] = (9, 6),
    ax = None,
):
    """Quick visualization of an RHGCanvas with parity-aware node types.

    - Data nodes and ancilla (X/Z) nodes are drawn with different markers.
    - Colors are *not* specified (uses matplotlib defaults).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for visualization") from e

    n2c = _node_to_coord(canvas)
    nodes = list(n2c.keys())

    # Choose dimensionality
    is3d = (projection == "3d")
    if ax is None:
        if is3d:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Helpers to project coordinates
    def proj(coord):
        x, y, z = coord
        if projection == "xz":
            return (x, z)
        if projection == "xy":
            return (x, y)
        if projection == "yz":
            return (y, z)
        return coord  # 3d

    # Draw nodes (split by parity type, use markers; no colors specified)
    if show_nodes:
        data_nodes: List[int] = []
        ancx_nodes: List[int] = []
        ancz_nodes: List[int] = []
        for n in nodes:
            x,y,z = n2c[n]
            if is_data(x,y,z):
                data_nodes.append(n)
            elif is_ancilla_x(x,y,z):
                ancx_nodes.append(n)
            elif is_ancilla_z(x,y,z):
                ancz_nodes.append(n)

        def scatter_nodes(ns: List[int], marker: str):
            if not ns:
                return
            if is3d:
                xs = [n2c[n][0] for n in ns]
                ys = [n2c[n][1] for n in ns]
                zs = [n2c[n][2] for n in ns]
                ax.scatter(xs, ys, zs, marker=marker)
            else:
                xs = []
                ys = []
                for n in ns:
                    a, b = proj(n2c[n])
                    xs.append(a); ys.append(b)
                ax.scatter(xs, ys, marker=marker, s=50)

        scatter_nodes(data_nodes, 'o')   # data
        scatter_nodes(ancx_nodes, '^')   # ancilla-X
        scatter_nodes(ancz_nodes, 's')   # ancilla-Z
        
        # Draw edges
    if show_edges and getattr(canvas.graph, "physical_edges", None) is not None:
        for u, v in canvas.graph.physical_edges:
            if u in n2c and v in n2c:
                cu = n2c[u]; cv = n2c[v]
                if is3d:
                    xs, ys, zs = zip(cu, cv)
                    ax.plot(xs, ys, zs, linestyle='-', color='black')
                else:
                    xu, yu = proj(cu)
                    xv, yv = proj(cv)
                    ax.plot([xu, xv], [yu, yv], linestyle='-', color='black')

    # Annotate (node ids and logical boundaries)
    if annotate:
        if is3d:
            for n in nodes:
                x, y, z = n2c[n]
                ax.text(x, y, z, str(n))
        else:
            for n in nodes:
                a, b = proj(n2c[n])
                ax.text(a, b, str(n))

        # Mark logical boundaries with their logical index near a representative node
        for lidx, qmap in canvas.logical_registry.boundary_qidx.items():
            node_set = set(qmap.keys())
            if not node_set:
                continue
            rep = min(node_set)
            if rep in n2c:
                coord = n2c[rep]
                if is3d:
                    ax.text(coord[0], coord[1], coord[2], f"L{lidx}")
                else:
                    a, b = proj(coord)
                    ax.text(a, b, f"L{lidx}")

    # Labels
    if projection == "xz":
        ax.set_xlabel("x"); ax.set_ylabel("z")
    elif projection == "xy":
        ax.set_xlabel("x"); ax.set_ylabel("y")
    elif projection == "yz":
        ax.set_xlabel("y"); ax.set_ylabel("z")
    else:
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if is3d:
            ax.set_zlabel("z")

    ax.set_title("RHGCanvas view ({})".format(projection))
    fig.tight_layout()
    return ax
