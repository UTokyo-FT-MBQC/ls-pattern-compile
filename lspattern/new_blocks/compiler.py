"""Compile metadata into measurement pattern and stim circuit"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

from lspattern.new_blocks.detector import construct_detector, remove_non_deterministic_det

if TYPE_CHECKING:
    from lspattern.new_blocks.canvas import Canvas
    from lspattern.new_blocks.mytype import Coord3D


def compile_canvas_to_stim(
    canvas: Canvas,
    logical_observables: dict[int, set[Coord3D]],
    p_depol_after_clifford: float,
    p_before_meas_flip: float,
) -> str:
    graph, node_map = GraphState.from_graph(
        nodes=canvas.nodes,
        edges=canvas.edges,
        meas_bases={node: AxisMeasBasis(canvas.pauli_axes[node], Sign.PLUS) for node in canvas.nodes},
    )

    flow = canvas.flow.to_node_flow(node_map)

    # construct scheduler
    # Convert time->nodes to node->time mappings
    time_to_nodes = canvas.scheduler.to_node_schedule(node_map)
    prep_time: dict[int, int] = {}
    meas_time: dict[int, int] = {}
    for time, nodes in time_to_nodes.items():
        for node in nodes:
            prep_time[node] = time
            meas_time[node] = time + 1  # measure one step after prepare
    scheduler = Scheduler(graph, flow)
    scheduler.manual_schedule(prep_time, meas_time)

    # construct detectors
    deterministic_accumulator = remove_non_deterministic_det(canvas)
    coord2detectors = construct_detector(deterministic_accumulator)
    detectors: list[frozenset[int]] = []
    for det_coords in coord2detectors.values():
        det_nodes = {node_map[coord] for coord in det_coords if coord in node_map}
        if det_nodes:
            detectors.append(frozenset(det_nodes))
    pattern = qompile(graph, flow, parity_check_group=detectors, scheduler=scheduler)

    # extract logical observables
    logical_observables_nodes: dict[int, set[int]] = {}
    for key, coord_set in logical_observables.items():
        nodes = set()
        for coord in coord_set:
            nodes |= canvas.couts[coord]
        logical_observables_nodes[key] = {node_map[coord] for coord in nodes}

    return stim_compile(
        pattern,
        logical_observables_nodes,
        p_depol_after_clifford=p_depol_after_clifford,
        p_before_meas_flip=p_before_meas_flip,
    )
