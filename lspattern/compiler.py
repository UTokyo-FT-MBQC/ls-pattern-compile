"""Compile metadata into measurement pattern and stim circuit"""

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

from graphqomb.common import AxisMeasBasis, Sign
from graphqomb.graphstate import GraphState
from graphqomb.noise_model import DepolarizingNoiseModel, MeasurementFlipNoiseModel
from graphqomb.ptn_format import dump as dump_ptn
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

from lspattern.detector import construct_detector

if TYPE_CHECKING:
    from pathlib import Path

    from graphqomb.pattern import Pattern

    from lspattern.canvas import Canvas
    from lspattern.canvas_loader import CompositeLogicalObservableSpec
    from lspattern.mytype import Coord3D


def _collect_logical_observable_nodes(  # noqa: C901
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

    for cube_ref in composite_logical_obs.cubes:
        if cube_ref.position not in canvas.couts:
            msg = f"Cube {cube_ref.position} not found in canvas.couts. Available cubes: {sorted(canvas.couts.keys())}"
            raise KeyError(msg)
        cube_couts = canvas.couts[cube_ref.position]
        if cube_ref.label is not None:
            if cube_ref.label not in cube_couts:
                msg = (
                    f"Label '{cube_ref.label}' not found in cube {cube_ref.position}. "
                    f"Available labels: {sorted(cube_couts.keys())}"
                )
                raise KeyError(msg)
            nodes |= cube_couts[cube_ref.label]
        else:
            for cout_set in cube_couts.values():
                nodes |= cout_set

    for pipe_ref in composite_logical_obs.pipes:
        pipe_key = (pipe_ref.start, pipe_ref.end)
        if pipe_key not in canvas.pipe_couts:
            msg = f"Pipe {pipe_key} not found in canvas.pipe_couts. Available pipes: {sorted(canvas.pipe_couts.keys())}"
            raise KeyError(msg)
        pipe_couts = canvas.pipe_couts[pipe_key]
        if pipe_ref.label is not None:
            if pipe_ref.label not in pipe_couts:
                msg = (
                    f"Label '{pipe_ref.label}' not found in pipe {pipe_key}. "
                    f"Available labels: {sorted(pipe_couts.keys())}"
                )
                raise KeyError(msg)
            nodes |= pipe_couts[pipe_ref.label]
        else:
            for cout_set in pipe_couts.values():
                nodes |= cout_set

    return nodes


def _delay_measurements_for_flow(scheduler: Scheduler) -> None:
    """Delay equal-time measurements until the manual schedule satisfies flow causality."""
    predecessors: dict[int, set[int]] = {node: set() for node in scheduler.measure_time}
    for source, successors in scheduler.dag.items():
        if source not in scheduler.measure_time:
            continue
        for successor in successors:
            if successor in scheduler.measure_time:
                predecessors[successor].add(source)

    for node in TopologicalSorter(predecessors).static_order():
        measure_time = scheduler.measure_time[node]
        if measure_time is None:
            continue

        min_measure_time = measure_time
        for predecessor in predecessors[node]:
            predecessor_time = scheduler.measure_time[predecessor]
            if predecessor_time is not None and predecessor_time >= min_measure_time:
                min_measure_time = predecessor_time + 1
        scheduler.measure_time[node] = min_measure_time


def compile_canvas_to_stim(
    canvas: Canvas,
    p_depol_after_clifford: float,
    p_before_meas_flip: float,
) -> str:
    pattern = compile_canvas_to_pattern(canvas)
    result: str = stim_compile(
        pattern,
        emit_qubit_coords=False,
        noise_models=[
            DepolarizingNoiseModel(p1=p_depol_after_clifford),
            MeasurementFlipNoiseModel(p=p_before_meas_flip),
        ],
    )
    return result


def compile_canvas_to_pattern(canvas: Canvas) -> Pattern:
    """Compile a canvas into a GraphQOMB measurement pattern.

    Parameters
    ----------
    canvas : Canvas
        The canvas to compile.

    Returns
    -------
    Pattern
        The compiled GraphQOMB measurement pattern.
    """
    nodes = sorted(canvas.nodes)
    graph, node_map = GraphState.from_graph(
        nodes=nodes,
        edges=sorted(canvas.edges),
        meas_bases={node: AxisMeasBasis(canvas.pauli_axes[node], Sign.PLUS) for node in nodes},
        coordinates={node: tuple(float(value) for value in node) for node in nodes},
    )

    flow = canvas.flow.to_node_flow(node_map)

    # construct scheduler
    # Convert time->nodes to node->time mappings
    prep_time, meas_time, entangle_time = canvas.scheduler.to_node_schedule(node_map)
    scheduler = Scheduler(graph, flow)
    scheduler.manual_schedule(prep_time, meas_time, entangle_time)
    _delay_measurements_for_flow(scheduler)
    scheduler.validate_schedule()

    # construct detectors
    coord2detectors = construct_detector(canvas.parity_accumulator)
    detectors: list[frozenset[int]] = []
    for _, det_coords in sorted(coord2detectors.items()):
        det_nodes = {node_map[coord] for coord in det_coords if coord in node_map}
        if det_nodes:
            detectors.append(frozenset(det_nodes))

    # extract logical observables from canvas
    logical_observables_nodes: dict[int, set[int]] = {}
    for key, composite_logical_obs in enumerate(canvas.logical_observables):
        nodes = _collect_logical_observable_nodes(canvas, composite_logical_obs)
        logical_observables_nodes[key] = {node_map[coord] for coord in nodes}

    return qompile(
        graph,
        flow,
        parity_check_group=detectors,
        logical_observables=logical_observables_nodes,
        scheduler=scheduler,
    )


def export_canvas_to_ptn(canvas: Canvas, output_path: str | Path) -> None:
    """Export a canvas to GraphQOMB's human-readable .ptn format.

    Parameters
    ----------
    canvas : Canvas
        The canvas to compile and export.
    output_path : str | Path
        Path to write the .ptn file.
    """
    dump_ptn(compile_canvas_to_pattern(canvas), output_path)
