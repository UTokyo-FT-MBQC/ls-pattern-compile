"""Merge and Split error rate simulation with noise probability sweep."""

import os

import matplotlib.pyplot as plt
import sinter
import stim

from lspattern.blocks.cubes.initialize import InitPlusCubeThinLayerSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_to_stim
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.utils import to_edgespec
from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D


def _create_merge_split_skeleton(d: int) -> RHGCanvasSkeleton:
    """Create RHG canvas skeleton for merge and split operation."""
    canvass = RHGCanvasSkeleton("Merge and Split")

    edgespec: dict[BoundarySide, EdgeSpecValue] = to_edgespec("XXZZ")
    edgespec1: dict[BoundarySide, EdgeSpecValue] = to_edgespec("XOZZ")
    edgespec2: dict[BoundarySide, EdgeSpecValue] = to_edgespec("OXZZ")
    edgespec_trimmed: dict[BoundarySide, EdgeSpecValue] = to_edgespec("OOZZ")
    edgespec_measure_trimmed: dict[BoundarySide, EdgeSpecValue] = to_edgespec("OOOO")

    blocks = [
        (
            PatchCoordGlobal3D((0, 0, 0)),
            InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 0, 0)),
            InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
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
            MeasureXSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 0, 4)),
            MeasureXSkeleton(d=d, edgespec=edgespec),
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

    return compile_to_stim(
        compiled_canvas,
        logical_observable_coords={
            0: [
                PatchCoordGlobal3D((0, 0, 4)),
                PatchCoordGlobal3D((1, 0, 4)),
                PipeCoordGlobal3D((PatchCoordGlobal3D((0, 0, 3)), PatchCoordGlobal3D((1, 0, 3)))),
            ]
        },
        p_depol_after_clifford=noise,
        p_before_meas_flip=noise,
    )


if __name__ == "__main__":
    # Create tasks for different distances and noise levels
    merge_split_tasks = [
        sinter.Task(
            circuit=create_circuit(d, noise),
            json_metadata={"d": d, "r": 1, "p": noise, "circuit_type": "merge_split"},
        )
        for d in [3, 5, 7]
        for noise in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
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
    ax.set_title("Merge and Split Error Rates per Shot under Circuit Noise")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)
    fig.savefig("figures/merge_split_error_sim_xx_init.png", bbox_inches="tight")
    plt.show()
