from __future__ import annotations

import os
import pathlib
from typing import Iterable

import matplotlib.pyplot as plt


def _reverse_coord2node(coord2node: dict[tuple[int, int, int], int]) -> dict[int, tuple[int, int, int]]:
    """Return node->coord map from coord->node (CompiledRHGCanvas形式)。"""
    node2coord: dict[int, tuple[int, int, int]] = {}
    for coord, nid in coord2node.items():
        node2coord[int(nid)] = (int(coord[0]), int(coord[1]), int(coord[2]))
    return node2coord


def visualize_compiled_canvas(
    cgraph,
    *,
    annotate: bool = False,
    save_path: str | None = None,
    show: bool = True,
    ax=None,
    figsize: tuple[int, int] = (7, 5),
    dpi: int = 120,
    show_axes: bool = True,
    show_grid: bool = True,
    show_edges: bool = True,
    color_by_z: bool = True,
    input_nodes: Iterable[int] | None = None,
    output_nodes: Iterable[int] | None = None,
):
    """CompiledRHGCanvas 可視化（Matplotlib 3D）。

    - CompiledRHGCanvas の `coord2node` を用いてノードを散布表示。
    - `global_graph.physical_edges` があればエッジも描画。
    - 入力/出力ノードは赤ダイヤで強調（指定が無い場合は GraphState の property を使用）。
    - 役割情報は global には保持しないため、色は z ごとの色分け（color_by_z=True）で表現。
    """
    node2coord = _reverse_coord2node(cgraph.coord2node or {})

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()

    # 軸とグリッド
    ax.set_box_aspect((1, 1, 1))
    ax.grid(bool(show_grid))
    if show_axes:
        ax.set_axis_on()
    else:
        ax.set_axis_off()

    # zごとに色分け
    import itertools

    palette = (
        "#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd #8c564b #e377c2 #7f7f7f #bcbd22 #17becf"
    ).split()
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
                pts["x"], pts["y"], pts["z"],
                c=color,
                edgecolors="black",
                s=40,
                depthshade=True,
                label=f"z={z}",
            )

    # エッジ描画
    g = cgraph.global_graph
    if show_edges and g is not None and hasattr(g, "physical_edges"):
        for u, v in g.physical_edges:
            if u in node2coord and v in node2coord:
                x1, y1, z1 = node2coord[u]
                x2, y2, z2 = node2coord[v]
                ax.plot([x1, x2], [y1, y2], [z1, z2], c="gray", linewidth=1, alpha=0.5)

    # 入力/出力ノードを強調（赤ダイヤ）
    if input_nodes is None and g is not None and hasattr(g, "input_node_indices"):
        try:
            input_nodes = list(g.input_node_indices.keys())
        except Exception:
            input_nodes = []
    if output_nodes is None and g is not None and hasattr(g, "output_node_indices"):
        try:
            output_nodes = list(g.output_node_indices.keys())
        except Exception:
            output_nodes = []

    def _scatter_marker(nodes: Iterable[int], face: str, color: str):
        nodes = list(nodes or [])
        if not nodes:
            return
        xs = [node2coord[n][0] for n in nodes if n in node2coord]
        ys = [node2coord[n][1] for n in nodes if n in node2coord]
        zs = [node2coord[n][2] for n in nodes if n in node2coord]
        if xs:
            ax.scatter(
                xs, ys, zs,
                c=color,
                edgecolors="darkred",
                s=70,
                marker="D",
                label=face,
            )

    _scatter_marker(input_nodes, "Input", "white")
    _scatter_marker(output_nodes, "Output", "red")

    # 注釈
    if annotate:
        for nid, (x, y, z) in node2coord.items():
            ax.text(x, y, z, str(nid), fontsize=6)

    ax.legend(loc="best")

    # 保存/表示
    if save_path:
        p = pathlib.Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p))
    if created_fig and show:
        plt.show()

    return ax.get_figure()

