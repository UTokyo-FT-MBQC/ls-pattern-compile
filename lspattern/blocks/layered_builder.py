"""Common builder logic for layered RHG blocks.

This module provides shared functionality for building 3D graph structures
from sequences of UnitLayer objects, used by both LayeredRHGCube and LayeredRHGPipe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.accumulator import FlowAccumulator, ParityAccumulator, ScheduleAccumulator
from lspattern.blocks.unit_layer import UnitLayer
from lspattern.consts import NodeRole, TemporalBoundarySpecValue
from lspattern.mytype import NodeIdGlobal, NodeIdLocal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix_zx.graphstate import GraphState

    from lspattern.tiling.template import ScalableTemplate


def build_layered_graph(  # noqa: C901
    unit_layers: Sequence[UnitLayer],
    d: int,
    source: tuple[int, int, int],
    template: ScalableTemplate,
    final_layer: TemporalBoundarySpecValue,
    graph: GraphState,
) -> tuple[
    GraphState,
    dict[int, tuple[int, int, int]],
    dict[tuple[int, int, int], int],
    dict[int, NodeRole],
    ScheduleAccumulator,
    FlowAccumulator,
    ParityAccumulator,
]:
    """Build 3D RHG graph structure layer-by-layer.

    This function constructs a graph by iteratively building each UnitLayer and
    merging the results. It handles empty layers and temporal edge connections.

    Parameters
    ----------
    unit_layers : list[UnitLayer]
        Sequence of unit layers to compose.
    d : int
        Code distance.
    source : tuple[int, int, int]
        Source coordinate for z-offset calculation.
    template : ScalableTemplate
        Template providing data/ancilla coordinates.
    final_layer : TemporalBoundarySpecValue
        Time boundary specification for final layer.
    graph : GraphState
        Graph state to add nodes and edges to (modified in place).

    Returns
    -------
    tuple
        (graph, node2coord, coord2node, node2role, schedule, flow, parity) for the complete block.
    """
    # Initialize accumulators
    schedule_accumulator = ScheduleAccumulator()
    flow_accumulator = FlowAccumulator()
    parity_accumulator = ParityAccumulator()
    z0 = int(source[2]) * (2 * d)

    # Accumulate all layer data
    all_nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}
    all_node2coord: dict[int, tuple[int, int, int]] = {}
    all_coord2node: dict[tuple[int, int, int], int] = {}
    all_node2role: dict[int, NodeRole] = {}

    for i, unit_layer in enumerate(unit_layers):  # noqa: PLR1702
        layer_z = z0 + 2 * i
        layer_data = unit_layer.build_layer(graph, layer_z, template)

        # Check if this layer is empty (no nodes)
        is_empty = not layer_data.nodes_by_z

        if not is_empty:
            # Merge layer data
            all_nodes_by_z.update(layer_data.nodes_by_z)
            all_node2coord.update(layer_data.node2coord)
            all_coord2node.update(layer_data.coord2node)
            all_node2role.update(layer_data.node2role)

            # Merge accumulators (accumulator methods return new objects)
            schedule_accumulator = schedule_accumulator.compose_sequential(layer_data.schedule)
            flow_accumulator = flow_accumulator.merge_with(layer_data.flow)
            parity_accumulator = parity_accumulator.merge_with(layer_data.parity)

            # Add temporal edges to the immediately previous z-layer if it exists
            current_layer_zs = sorted(layer_data.nodes_by_z.keys())
            if current_layer_zs:
                first_z = current_layer_zs[0]
                current_layer_nodes = layer_data.nodes_by_z[first_z]

                # Look for nodes at z-1 (immediately previous layer)
                prev_z = first_z - 1
                prev_layer_nodes = all_nodes_by_z.get(prev_z, {})

                # Only connect if previous layer has nodes (empty layers will be skipped)
                if prev_layer_nodes:
                    # Build temporal flow for this connection
                    temp_flow: dict[NodeIdLocal, set[NodeIdLocal]] = {}
                    for xy, u in current_layer_nodes.items():
                        v = prev_layer_nodes.get(xy)
                        if v is not None:
                            graph.add_physical_edge(u, v)
                            temp_flow.setdefault(NodeIdLocal(v), set()).add(NodeIdLocal(u))
                    # Merge temporal flow into accumulator
                    if temp_flow:
                        flow_accumulator = flow_accumulator.merge_with(FlowAccumulator(flow=temp_flow))

    # Add final data layer if final_layer is 'O' (open)
    if final_layer == TemporalBoundarySpecValue.O:
        data2d = list(template.data_coords or [])
        final_z = z0 + 2 * len(unit_layers)
        final_layer_dict: dict[tuple[int, int], int] = {}

        for x, y in data2d:
            n = graph.add_physical_node()
            all_node2coord[n] = (int(x), int(y), int(final_z))
            all_coord2node[int(x), int(y), int(final_z)] = n
            all_node2role[n] = NodeRole.DATA
            final_layer_dict[int(x), int(y)] = n

        all_nodes_by_z[final_z] = final_layer_dict

        # Add spatial edges for final layer
        UnitLayer.add_spatial_edges(graph, final_layer_dict)

        # Add temporal edges connecting to previous layer
        if all_nodes_by_z:
            prev_z = final_z - 1
            if prev_z in all_nodes_by_z:
                prev_layer = all_nodes_by_z[prev_z]
                final_flow: dict[NodeIdLocal, set[NodeIdLocal]] = {}
                for xy, u in final_layer_dict.items():
                    v = prev_layer.get(xy)
                    if v is not None:
                        graph.add_physical_edge(u, v)
                        final_flow.setdefault(NodeIdLocal(v), set()).add(NodeIdLocal(u))
                # Merge temporal flow into accumulator
                if final_flow:
                    flow_accumulator = flow_accumulator.merge_with(FlowAccumulator(flow=final_flow))

        # Add final layer to schedule
        final_data_nodes = {NodeIdGlobal(n) for n in final_layer_dict.values()}
        if final_data_nodes:
            # Create new schedule with final layer
            final_schedule = ScheduleAccumulator(schedule={2 * final_z + 1: final_data_nodes})
            schedule_accumulator = schedule_accumulator.compose_parallel(final_schedule)

    return (
        graph,
        all_node2coord,
        all_coord2node,
        all_node2role,
        schedule_accumulator,
        flow_accumulator,
        parity_accumulator,
    )
