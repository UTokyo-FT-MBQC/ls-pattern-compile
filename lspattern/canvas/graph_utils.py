"""Graph remapping utilities for RHG canvas compilation.

This module provides utility functions for remapping graph nodes, edges,
and measurement bases when composing temporal layers.
"""

from __future__ import annotations

from graphix_zx.graphstate import BaseGraphState, GraphState

from lspattern.mytype import NodeIdLocal


def remap_graph_nodes(gsrc: BaseGraphState, nmap: dict[NodeIdLocal, NodeIdLocal]) -> tuple[dict[int, int], GraphState]:
    """Create new nodes in destination graph.

    Parameters
    ----------
    gsrc : BaseGraphState
        Source graph state to remap.
    nmap : dict[NodeIdLocal, NodeIdLocal]
        Node mapping from old to new IDs.

    Returns
    -------
    tuple[dict[int, int], GraphState]
        Created node mapping and new graph state.
    """
    gdst = GraphState()
    created: dict[int, int] = {}
    for old in gsrc.physical_nodes:
        new_id = nmap.get(NodeIdLocal(old), NodeIdLocal(old))
        if int(new_id) in created:
            continue
        created[int(new_id)] = gdst.add_physical_node()
    return created, gdst


def remap_measurement_bases(
    gsrc: BaseGraphState,
    gdst: BaseGraphState,
    nmap: dict[NodeIdLocal, NodeIdLocal],
    created: dict[int, int],
) -> None:
    """Remap measurement bases from source to destination graph.

    Parameters
    ----------
    gsrc : BaseGraphState
        Source graph state.
    gdst : BaseGraphState
        Destination graph state.
    nmap : dict[NodeIdLocal, NodeIdLocal]
        Node mapping from old to new IDs.
    created : dict[int, int]
        Mapping of created nodes.
    """
    for old, new_id in nmap.items():
        mb = gsrc.meas_bases.get(int(old))
        if mb is not None:
            gdst.assign_meas_basis(created.get(int(new_id), int(new_id)), mb)


def remap_graph_edges(
    gsrc: BaseGraphState,
    gdst: BaseGraphState,
    nmap: dict[NodeIdLocal, NodeIdLocal],
    created: dict[int, int],
) -> None:
    """Remap graph edges from source to destination graph.

    Parameters
    ----------
    gsrc : BaseGraphState
        Source graph state.
    gdst : BaseGraphState
        Destination graph state.
    nmap : dict[NodeIdLocal, NodeIdLocal]
        Node mapping from old to new IDs.
    created : dict[int, int]
        Mapping of created nodes.
    """
    for u, v in gsrc.physical_edges:
        nu = nmap.get(NodeIdLocal(u), NodeIdLocal(u))
        nv = nmap.get(NodeIdLocal(v), NodeIdLocal(v))
        gdst.add_physical_edge(created.get(int(nu), int(nu)), created.get(int(nv), int(nv)))


def create_remapped_graphstate(gsrc: BaseGraphState | None, nmap: dict[NodeIdLocal, NodeIdLocal]) -> GraphState | None:
    """Create a remapped GraphState.

    Parameters
    ----------
    gsrc : BaseGraphState | None
        Source graph state to remap, or None.
    nmap : dict[NodeIdLocal, NodeIdLocal]
        Node mapping from old to new IDs.

    Returns
    -------
    GraphState | None
        Remapped graph state, or None if source was None.
    """
    if gsrc is None:
        return None
    created, gdst = remap_graph_nodes(gsrc, nmap)
    remap_measurement_bases(gsrc, gdst, nmap, created)
    remap_graph_edges(gsrc, gdst, nmap, created)
    return gdst
