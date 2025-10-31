from __future__ import annotations

from typing import TYPE_CHECKING, cast

import stim
from graphqomb.qompiler import qompile
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState
    from graphqomb.pattern import Pattern

    from lspattern.canvas.compiled import CompiledRHGCanvas
    from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D


def compile_canvas(
    graph: BaseGraphState,
    flow: Mapping[int, AbstractSet[int]],
    parity: Sequence[AbstractSet[int]] | None = None,
    scheduler: Scheduler | None = None,
) -> Pattern:
    """
    Thin wrapper around `graphqomb.qompile` for an RHG canvas.

    Parameters
    ----------
    graph : BaseGraphState
        A `GraphState` (or compatible) instance produced by the canvas.
    flow : collections.abc.Mapping[int, collections.abc.Set[int]] | None
        flow mapping (node -> correction target nodes).
    parity : collections.abc.Sequence[collections.abc.Set[int]] | None
        Optional list of parity check groups (GLOBAL node-id sets).
    scheduler : Scheduler | None
        Optional measurement scheduler. If `None`, backend default is used.

    Returns
    -------
    Pattern
        Whatever `graphqomb.qompile` returns (pattern, circuit, etc.).
    """
    return qompile(
        graph=graph,
        xflow=flow,
        parity_check_group=parity,
        scheduler=scheduler,
    )


def compile_to_stim(  # noqa: C901
    compiled_canvas: CompiledRHGCanvas,
    logical_observable_coords: Mapping[int, Sequence[PatchCoordGlobal3D | PipeCoordGlobal3D]],
    *,
    p_depol_after_clifford: float = 0.0,
    p_before_meas_flip: float = 0.0,
) -> stim.Circuit:
    """
    Compile a CompiledRHGCanvas to a stim.Circuit.

    This is a unified API that streamlines the conversion from a compiled canvas
    to a stim circuit by internally handling scheduler setup, parity extraction,
    pattern compilation, and logical observable resolution.

    Parameters
    ----------
    compiled_canvas : CompiledRHGCanvas
        The compiled canvas to convert.
    logical_observable_coords : Mapping[str | int, Sequence[PatchCoordGlobal3D | PipeCoordGlobal3D]]
        Mapping from logical observable keys to sequences of patch or pipe coordinates.
        Each coordinate can be either:
        - PatchCoordGlobal3D: for cube-based observables
        - PipeCoordGlobal3D: for pipe-based observables
    p_depol_after_clifford : float
        Depolarization noise rate after Clifford gates (default: 0.0).
    p_before_meas_flip : float
        Measurement bit-flip error rate (default: 0.0).

    Returns
    -------
    stim.Circuit
        The compiled quantum circuit with detectors and observables.

    Raises
    ------
    ValueError
        If compiled_canvas.global_graph is None.
    KeyError
        If any coordinate in logical_observable_coords is not found in cout ports.

    Examples
    --------
    >>> circuit = compile_to_stim(
    ...     compiled_canvas,
    ...     logical_observable_coords={0: [PatchCoordGlobal3D((0, 0, 2))]},
    ...     p_before_meas_flip=0.001,
    ... )
    """
    # 1. Validate global_graph
    if compiled_canvas.global_graph is None:
        msg = "Global graph is None"
        raise ValueError(msg)

    # 2. Extract xflow (NodeIdLocal → int)
    xflow: dict[int, set[int]] = {}
    for src, dsts in compiled_canvas.flow.flow.items():
        xflow[int(src)] = {int(dst) for dst in dsts}

    # 3. Setup scheduler
    scheduler = Scheduler(compiled_canvas.global_graph, xflow=xflow)
    compact_schedule = compiled_canvas.schedule.compact()

    prep_time: dict[int, int] = {}
    meas_time: dict[int, int] = {}

    input_nodes = set(compiled_canvas.global_graph.input_node_indices.keys())
    output_nodes = set(compiled_canvas.global_graph.output_node_indices.keys())

    for node in compiled_canvas.global_graph.physical_nodes:
        if node not in input_nodes:
            prep_time[node] = 0

        if node not in output_nodes:
            meas_time[node] = 1
            for time_slot, scheduled_nodes in compact_schedule.schedule.items():
                if node in scheduled_nodes:
                    meas_time[node] = time_slot + 1
                    break

    scheduler.manual_schedule(prepare_time=prep_time, measure_time=meas_time)

    # 4. Extract parity
    parity: list[set[int]] = []
    for group_dict in compiled_canvas.parity.checks.values():
        parity.extend({int(node) for node in group} for group in group_dict.values())

    # 5. Compile pattern
    pattern = compile_canvas(
        compiled_canvas.global_graph,
        flow=xflow,
        parity=parity,
        scheduler=scheduler,
    )

    # 6. Resolve logical observables
    cout_portmap_cube = compiled_canvas.cout_portset_cube
    cout_portmap_pipe = compiled_canvas.cout_portset_pipe

    logical_observables: dict[int, set[int]] = {}
    for key, coords in logical_observable_coords.items():
        observable_nodes: set[int] = set()
        for coord in coords:
            # Type check to determine if pipe or cube coordinate
            # PipeCoordGlobal3D is a tuple of two PatchCoordGlobal3D
            # PatchCoordGlobal3D is a tuple of three ints
            # Check if coord is a pipe coordinate (nested tuple structure)
            if isinstance(coord, tuple) and len(coord) == 2 and isinstance(coord[0], tuple):
                # It's a PipeCoordGlobal3D
                pipe_coord = cast("PipeCoordGlobal3D", coord)  # type: ignore[redundant-cast]
                if pipe_coord in cout_portmap_pipe:
                    observable_nodes.update(int(n) for n in cout_portmap_pipe[pipe_coord])
                else:
                    msg = f"Pipe coordinate {coord} not found in cout ports"
                    raise KeyError(msg)
            else:
                # It's a PatchCoordGlobal3D
                patch_coord = cast("PatchCoordGlobal3D", coord)  # type: ignore[redundant-cast]
                if patch_coord in cout_portmap_cube:
                    observable_nodes.update(int(n) for n in cout_portmap_cube[patch_coord])
                else:
                    msg = f"Patch coordinate {coord} not found in cout ports"
                    raise KeyError(msg)
        logical_observables[key] = observable_nodes

    # 7. Generate stim circuit
    # Note: stim_compile expects int keys, but we allow str | int for flexibility
    stim_str = stim_compile(
        pattern,
        logical_observables=logical_observables,
        p_depol_after_clifford=p_depol_after_clifford,
        p_before_meas_flip=p_before_meas_flip,
    )

    return stim.Circuit(stim_str)
