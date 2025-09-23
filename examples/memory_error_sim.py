"""RHG memory simulation with noise probability sweep."""

import os
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import sinter
import stim
from graphix_zx.scheduler import Scheduler
from graphix_zx.stim_compiler import stim_compile

if TYPE_CHECKING:
    from lspattern.canvas import CompiledRHGCanvas

from lspattern.blocks.cubes.initialize import InitPlusCubeThinLayerSkeleton, InitZeroCubeThinLayerSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton, MeasureZSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.mytype import PatchCoordGlobal3D


def _create_skeleton(d: int, init_type: Literal["plus", "zero"]) -> RHGCanvasSkeleton:
    """Create RHG canvas skeleton with specified parameters."""
    skeleton = RHGCanvasSkeleton(name=f"RHG Memory Circuit d={d}, init={init_type}")
    edgespec: dict[str, Literal["X", "Z", "O"]] = {"TOP": "X", "BOTTOM": "X", "LEFT": "Z", "RIGHT": "Z"}

    # Add initialization cube at the beginning based on init_type
    if init_type == "plus":
        init_skeleton = InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)
        measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
    elif init_type == "zero":
        init_skeleton = InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec)
        measure_skeleton = MeasureZSkeleton(d=d, edgespec=edgespec)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

    # Add memory cube in the middle
    memory_skeleton = MemoryCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), memory_skeleton)

    # Add measurement cube at the end
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 2)), measure_skeleton)

    return skeleton


def _setup_scheduler(compiled_canvas: "CompiledRHGCanvas") -> tuple[Scheduler, dict[int, set[int]]]:
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

        output_nodes = set(compiled_canvas.global_graph.output_node_indices.keys())
        for node in compiled_canvas.global_graph.physical_nodes:
            if node not in output_nodes:
                meas_time[node] = 1
                for time_slot, nodes in compact_schedule.schedule.items():
                    if node in nodes:
                        meas_time[node] = time_slot + 1
                        break

    scheduler.manual_schedule(prepare_time=prep_time, measure_time=meas_time)
    return scheduler, xflow


def create_circuit(d: int, noise: float, init_type: Literal["plus", "zero"]) -> stim.Circuit:
    """Create RHG memory circuit with specified parameters.

    Parameters
    ----------
    d : int
        Distance parameter
    noise : float
        Noise probability
    init_type : Literal["plus", "zero"]
        Initialization type

    Returns
    -------
    stim.Circuit
        The compiled memory circuit.
    """
    skeleton = _create_skeleton(d, init_type)
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
    # Set logical observables using coordinate-based approach (similar to plus_initialization.py)
    cout_portmap = compiled_canvas.cout_portset
    coord2logical_group = {0: PatchCoordGlobal3D((0, 0, 2))}  # Measurement cube is at position (0, 0, 2)
    logical_observables = {i: cout_portmap[coord] for i, coord in coord2logical_group.items()}
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=0,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)


if __name__ == "__main__":
    # Select initialization type: "plus", "zero", or "both"
    init_mode = "both"  # Change this to "zero" or "both" for different evaluations

    if init_mode == "both":
        init_types: list[Literal["plus", "zero"]] = ["plus", "zero"]
    else:
        init_types = [init_mode]  # type: ignore[list-item]

    # Create tasks for different distances and noise levels
    rhg_code_tasks = [
        sinter.Task(
            circuit=create_circuit(d, noise, init_type),
            json_metadata={"d": d, "r": 1, "p": noise, "init_type": init_type},
        )
        for init_type in init_types
        for d in [3, 5, 7]
        for noise in [1e-2, 5e-2, 2e-2, 1e-1]
    ]

    # Collect statistics
    collected_rhg_code_stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count() or 1,
        tasks=rhg_code_tasks,
        decoders=["pymatching"],
        max_shots=1_000_000,
        max_errors=5_000,
        print_progress=True,
    )

    # Plot the results
    fig, ax = plt.subplots(1, 1)
    if init_mode == "both":
        # Group by both distance and initialization type
        sinter.plot_error_rate(
            ax=ax,
            stats=collected_rhg_code_stats,
            x_func=lambda stat: stat.json_metadata["p"],
            group_func=lambda stat: f"d={stat.json_metadata['d']}, {stat.json_metadata['init_type']}",
            failure_units_per_shot_func=lambda stat: stat.json_metadata["r"],
        )
        title = "RHG Code Error Rates per Round under Circuit Noise (Plus vs Zero)"
        filename = "figures/memory_sim_comparison.png"
    else:
        # Group by distance only
        sinter.plot_error_rate(
            ax=ax,
            stats=collected_rhg_code_stats,
            x_func=lambda stat: stat.json_metadata["p"],
            group_func=lambda stat: stat.json_metadata["d"],
            failure_units_per_shot_func=lambda stat: stat.json_metadata["r"],
        )
        title = f"RHG Code Error Rates per Round under Circuit Noise ({init_mode} initialization)"
        filename = f"figures/memory_sim_{init_mode}.png"

    # ax.set_ylim(5e-3, 5e-2)
    # ax.set_xlim(0.008, 0.012)
    ax.loglog()
    ax.set_title(title)
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)
    fig.savefig(filename, bbox_inches="tight")
    plt.show()
