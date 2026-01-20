"""Compile metadata into measurement pattern and stim circuit"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

from lspattern.detector import construct_detector, remove_non_deterministic_det

if TYPE_CHECKING:
    from lspattern.canvas import Canvas
    from lspattern.canvas_loader import CompositeLogicalObservableSpec
    from lspattern.mytype import Coord3D


def _collect_logical_observable_nodes(
    canvas: Canvas,
    composite_logical_obs: CompositeLogicalObservableSpec,
) -> set[Coord3D]:
    """Collect all node coordinates for a composite logical observable.

    Parameters
    ----------
    canvas : Canvas
        The canvas containing couts data.
    composite_logical_obs : CompositeLogicalObservableSpec
        The composite observable specifying cubes and pipes.

    Returns
    -------
    set[Coord3D]
        All node coordinates contributing to this logical observable.
    """
    nodes: set[Coord3D] = set()

    for cube in composite_logical_obs.cubes:
        cube_couts = canvas.couts.get(cube, {})
        if composite_logical_obs.label is not None:
            if composite_logical_obs.label not in cube_couts:
                msg = (
                    f"Label '{composite_logical_obs.label}' not found in cube {cube}. "
                    f"Available labels: {sorted(cube_couts.keys())}"
                )
                raise KeyError(msg)
            nodes |= cube_couts[composite_logical_obs.label]
        else:
            for cout_set in cube_couts.values():
                nodes |= cout_set

    for pipe in composite_logical_obs.pipes:
        pipe_couts = canvas.pipe_couts.get(pipe, {})
        if composite_logical_obs.label is not None:
            if composite_logical_obs.label not in pipe_couts:
                msg = (
                    f"Label '{composite_logical_obs.label}' not found in pipe {pipe}. "
                    f"Available labels: {sorted(pipe_couts.keys())}"
                )
                raise KeyError(msg)
            nodes |= pipe_couts[composite_logical_obs.label]
        else:
            for cout_set in pipe_couts.values():
                nodes |= cout_set

    return nodes


def compile_canvas_to_stim(
    canvas: Canvas,
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
    prep_time, meas_time, entangle_time = canvas.scheduler.to_node_schedule(node_map)
    scheduler = Scheduler(graph, flow)
    scheduler.manual_schedule(prep_time, meas_time, entangle_time)

    # construct detectors
    deterministic_accumulator = remove_non_deterministic_det(canvas)
    coord2detectors = construct_detector(deterministic_accumulator)
    detectors: list[frozenset[int]] = []
    for det_coords in coord2detectors.values():
        det_nodes = {node_map[coord] for coord in det_coords if coord in node_map}
        if det_nodes:
            detectors.append(frozenset(det_nodes))
    pattern = qompile(graph, flow, parity_check_group=detectors, scheduler=scheduler)

    # extract logical observables from canvas
    logical_observables_nodes: dict[int, set[int]] = {}
    for key, composite_logical_obs in enumerate(canvas.logical_observables):
        nodes = _collect_logical_observable_nodes(canvas, composite_logical_obs)
        logical_observables_nodes[key] = {node_map[coord] for coord in nodes}

    result: str = stim_compile(
        pattern,
        logical_observables_nodes,
        p_depol_after_clifford=p_depol_after_clifford,
        p_before_meas_flip=p_before_meas_flip,
    )
    return result
