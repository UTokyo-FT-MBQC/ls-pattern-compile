"""Merge and Split error rate simulation with noise probability sweep."""

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import sinter
import stim
from graphqomb.scheduler import Scheduler
from graphqomb.stim_compiler import stim_compile

if TYPE_CHECKING:
    from lspattern.canvas import CompiledRHGCanvas

from lspattern.blocks.cubes.initialize import InitZeroCubeThinLayerSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D


def _create_merge_split_skeleton(d: int) -> RHGCanvasSkeleton:
    """Create RHG canvas skeleton for merge and split operation."""
    canvass = RHGCanvasSkeleton("Merge and Split")

    edgespec: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec1: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec_trimmed: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec_measure_trimmed: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.O}

    blocks = [
        (
            PatchCoordGlobal3D((0, 0, 0)),
            InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 0, 0)),
            InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 0, 1)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 0, 1)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 0, 2)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec1),
        ),
        (
            PatchCoordGlobal3D((1, 0, 2)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec2),
        ),
        (
            PatchCoordGlobal3D((0, 0, 3)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 0, 3)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 0, 4)),
            MeasureZSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 0, 4)),
            MeasureZSkeleton(d=d, edgespec=edgespec),
        ),
    ]
    pipes = [
        (
            PatchCoordGlobal3D((0, 0, 2)),
            PatchCoordGlobal3D((1, 0, 2)),
            InitPlusPipeSkeleton(d=d, edgespec=edgespec_trimmed),
        ),
        (
            PatchCoordGlobal3D((0, 0, 3)),
            PatchCoordGlobal3D((1, 0, 3)),
            MeasureXPipeSkeleton(d=d, edgespec=edgespec_measure_trimmed),
        ),
    ]

    for block in blocks:
        canvass.add_cube(*block)
    for pipe in pipes:
        canvass.add_pipe(*pipe)

    return canvass


def _setup_scheduler(
    compiled_canvas: "CompiledRHGCanvas",
) -> tuple[Scheduler, dict[int, set[int]]]:
    """Set up scheduler with timing information."""
    if compiled_canvas.global_graph is None:
        raise ValueError("Global graph is None")

    xflow = {}
    for src, dsts in compiled_canvas.flow.flow.items():
        xflow[int(src)] = {int(dst) for dst in dsts}

    scheduler = Scheduler(compiled_canvas.global_graph, xflow=xflow)
    compact_schedule = compiled_canvas.schedule.compact()

    prep_time = {}
    meas_time = {}

    if compiled_canvas.global_graph is not None:
        input_nodes = set(compiled_canvas.global_graph.input_node_indices.keys())
        for node in compiled_canvas.global_graph.physical_nodes:
            if node not in input_nodes:
                prep_time[node] = 0

        output_indices = compiled_canvas.global_graph.output_node_indices or {}
        output_nodes = set(output_indices.keys())
        for node in compiled_canvas.global_graph.physical_nodes:
            if node not in output_nodes:
                meas_time[node] = 1
                for time_slot, nodes in compact_schedule.schedule.items():
                    if node in nodes:
                        meas_time[node] = time_slot + 1
                        break

    scheduler.manual_schedule(prepare_time=prep_time, measure_time=meas_time)
    return scheduler, xflow


def create_circuit(d: int, noise: float) -> stim.Circuit:
    """Create merge and split circuit with specified parameters.

    Parameters
    ----------
    d : int
        Distance parameter
    noise : float
        Noise probability

    Returns
    -------
    stim.Circuit
        The compiled merge and split circuit.
    """
    skeleton = _create_merge_split_skeleton(d)
    canvas = skeleton.to_canvas()
    compiled_canvas = canvas.compile()

    scheduler, xflow = _setup_scheduler(compiled_canvas)

    if compiled_canvas.global_graph is None:
        raise ValueError("Global graph is None")

    x_parity: list[set[int]] = []
    for group_dict in compiled_canvas.parity.checks.values():
        for group in group_dict.values():
            x_parity.append({int(node) for node in group})

    pattern = compile_canvas(
        compiled_canvas.global_graph,
        xflow=xflow,
        x_parity=x_parity,
        z_parity=[],
        scheduler=scheduler,
    )

    # Set logical observables - use the first output patch only
    cout_portmap = compiled_canvas.cout_portset
    coord2logical_group = {
        0: {PatchCoordGlobal3D((0, 0, 4))},  # First output patch
    }
    logical_observables = {}
    for i, group in coord2logical_group.items():
        nodes = []
        for coord in group:
            if coord in cout_portmap:
                nodes.extend(cout_portmap[coord])
        logical_observables[i] = set(nodes)

    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=0,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)


if __name__ == "__main__":
    # Create tasks for different distances and noise levels
    merge_split_tasks = [
        sinter.Task(
            circuit=create_circuit(d, noise),
            json_metadata={"d": d, "r": 1, "p": noise, "circuit_type": "merge_split"},
        )
        for d in [3, 5, 7]
        for noise in [1e-2, 5e-2, 2e-2, 1e-1]
    ]

    # Collect statistics
    collected_stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count() or 1,
        tasks=merge_split_tasks,
        decoders=["pymatching"],
        max_shots=1_000_000,
        max_errors=5_000,
        print_progress=True,
    )

    # Plot the results
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: f"d={stat.json_metadata['d']}",
        failure_units_per_shot_func=lambda stat: stat.json_metadata["r"],
    )

    ax.loglog()
    ax.set_title("Merge and Split Error Rates per Round under Circuit Noise")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)
    fig.savefig("figures/merge_split_error_sim.png", bbox_inches="tight")
    plt.show()
