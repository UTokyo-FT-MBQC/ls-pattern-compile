"""Merge and Split error rate simulation with noise probability sweep."""

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
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
from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D


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

    parity: list[set[int]] = []
    for group_dict in compiled_canvas.parity.checks.values():
        for group in group_dict.values():
            parity.append({int(node) for node in group})

    pattern = compile_canvas(
        compiled_canvas.global_graph,
        xflow=xflow,
        parity=parity,
        scheduler=scheduler,
    )

    # Set logical observables
    cout_portmap = compiled_canvas.cout_portset_cube
    cout_portmap_pipe = compiled_canvas.cout_portset_pipe
    coord2logical_group = {
        0: {PatchCoordGlobal3D((0, 0, 4))},  # First output patch
        1: {PatchCoordGlobal3D((1, 0, 4))},  # Second output patch
        2: {PipeCoordGlobal3D((PatchCoordGlobal3D((0, 0, 2)), PatchCoordGlobal3D((1, 0, 2))))},  # InitPlus pipe
    }
    logical_observables = {}
    for i, group in coord2logical_group.items():
        nodes = []
        for coord in group:
            # PipeCoordGlobal3D is a 2-tuple of PatchCoordGlobal3D (nested tuples)
            # PatchCoordGlobal3D is a 3-tuple of ints
            if isinstance(coord, tuple) and len(coord) == 2 and all(isinstance(c, tuple) for c in coord):
                # This is a PipeCoordGlobal3D
                if coord in cout_portmap_pipe:
                    nodes.extend(cout_portmap_pipe[coord])
            elif coord in cout_portmap:
                # This is a PatchCoordGlobal3D
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
        count_observable_error_combos=True,
        print_progress=True,
    )

    # Extract per-observable error rates with uncertainty
    def extract_per_observable_error_rates(
        stats: sinter.TaskStats, n_obs: int = 3
    ) -> list[tuple[float, float]]:
        """Extract per-observable error rates and standard errors from custom_counts.

        Returns
        -------
        list[tuple[float, float]]
            List of (error_rate, std_error) tuples for each observable.
        """
        err_counts = [0] * n_obs
        for key, count in stats.custom_counts.items():
            if not key.startswith("obs_mistake_mask="):
                continue
            mask = key.split("=", 1)[1]
            for i, ch in enumerate(mask[:n_obs]):
                if ch == "E":
                    err_counts[i] += count

        results = []
        for err_count in err_counts:
            if stats.shots > 0:
                error_rate = err_count / stats.shots
                # Standard error using binomial distribution
                std_error = np.sqrt(error_rate * (1 - error_rate) / stats.shots)
                results.append((error_rate, std_error))
            else:
                results.append((0.0, 0.0))
        return results

    # Create per-observable TaskStats for sinter.plot_error_rate
    from dataclasses import replace

    stats_per_obs: dict[int, list[sinter.TaskStats]] = {0: [], 1: [], 2: []}

    for stat in collected_stats:
        per_obs_data = extract_per_observable_error_rates(stat)
        for obs_id, (error_rate, std_error) in enumerate(per_obs_data):
            # Calculate error count for this observable
            err_count = int(error_rate * stat.shots)
            # Create new TaskStats with observable-specific error count
            new_stat = replace(stat, errors=err_count)
            stats_per_obs[obs_id].append(new_stat)

    # Plot the results with 3 subplots (one per observable)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    observable_names = [
        "Output patch 0 (0,0,4)",
        "Output patch 1 (1,0,4)",
        "InitPlus pipe (0,0,2)-(1,0,2)",
    ]

    for obs_id, (ax, obs_name) in enumerate(zip(axes, observable_names)):
        sinter.plot_error_rate(
            ax=ax,
            stats=stats_per_obs[obs_id],
            x_func=lambda stat: stat.json_metadata["p"],
            group_func=lambda stat: f"d={stat.json_metadata['d']}",
            failure_units_per_shot_func=lambda stat: stat.json_metadata["r"],
        )
        ax.set_title(f"Observable {obs_id}: {obs_name}")
        ax.loglog()
        ax.grid(which="major")
        ax.grid(which="minor")
        ax.legend()

    fig.suptitle("Merge and Split: Per-Observable Error Rates under Circuit Noise", fontsize=14)
    fig.tight_layout()
    fig.set_dpi(120)
    fig.savefig("figures/merge_split_error_sim.png", bbox_inches="tight")
    plt.show()
