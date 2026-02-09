"""Debug utilities for flow cycle detection at coordinate level."""

from __future__ import annotations

from functools import reduce
from operator import xor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lspattern.mytype import Coord3D


def check_coord_flow_cycle(
    xflow: dict[Coord3D, set[Coord3D]],
    edges: set[tuple[Coord3D, Coord3D]],
) -> None:
    """Check for cycles in the flow graph at coordinate level.

    This function performs the same DAG construction as graphqomb.qompile() but
    at the coordinate level, providing more informative error messages when
    cycles are detected.

    Parameters
    ----------
    xflow : dict[Coord3D, set[Coord3D]]
        X correction flow at coordinate level (from CoordFlowAccumulator.flow).
    edges : set[tuple[Coord3D, Coord3D]]
        Graph edges at coordinate level (from Canvas.edges).

    Raises
    ------
    ValueError
        If a cycle is detected, with coordinate information in the message.
    """
    # Build adjacency once
    adjacency: dict[Coord3D, set[Coord3D]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    # Compute zflow from xflow: zflow[node] = XOR of neighbors of xflow targets
    zflow: dict[Coord3D, set[Coord3D]] = {}
    for node, targets in xflow.items():
        neighbor_sets = [adjacency.get(t, set()) for t in targets]
        zflow[node] = reduce(xor, neighbor_sets, set())

    # Build DAG: (xflow | zflow) - self-loops
    dag: dict[Coord3D, set[Coord3D]] = {}
    for node in xflow:
        x_targets = xflow.get(node, set())
        z_targets = zflow.get(node, set())
        dag[node] = (x_targets | z_targets) - {node}  # Remove self-loops

    # Check for direct cycles (A -> B and B -> A)
    for node, children in dag.items():
        for child in children:
            if child in dag and node in dag[child]:
                msg = f"Cycle detected in flow graph at coordinate level:\n  {node} -> {child}\n  {child} -> {node}"
                raise ValueError(msg)
