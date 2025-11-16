"""CNOT gate error rate simulation with noise probability sweep."""

import os

import matplotlib.pyplot as plt
import sinter
import stim

from lspattern.blocks.cubes.initialize import (
    InitPlusCubeThinLayerSkeleton,
    InitZeroCubeThinLayerSkeleton,
)
from lspattern.blocks.cubes.measure import MeasureXSkeleton, MeasureZSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton, InitZeroPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton, MeasureZPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_to_stim
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.utils import to_edgespec


def _create_cnot_skeleton(d: int) -> RHGCanvasSkeleton:
    """Create RHG canvas skeleton for CNOT gate.

    The CNOT circuit implements a controlled-NOT gate using the following sequence:
    1. Initialize control qubit (0,0,0) in |0⟩, target qubit (1,1,0) in |+⟩, ancilla (0,1,0) in |+⟩
    2. Memory operations to preserve state (clock 1)
    3. ZZ-basis merge between control and target using InitPlus pipe (clock 2-3)
    4. XX-basis merge between target and ancilla using InitZero pipe (clock 4-5)
    5. Final measurements: control in Z basis, target and ancilla in X basis (clock 6)
    """
    canvas = RHGCanvasSkeleton("CNOT")

    edgespec: dict[BoundarySide, EdgeSpecValue] = to_edgespec("ZZXX")

    blocks = [
        # Clock 0 (initialization)
        (
            PatchCoordGlobal3D((0, 0, 0)),
            InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 1, 0)),
            InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 1, 0)),
            InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
        ),
        # Clock 1 (memory)
        (
            PatchCoordGlobal3D((0, 0, 1)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 1, 1)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 1, 1)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        # Clock 2 (merge ZZ preparation)
        (
            PatchCoordGlobal3D((0, 0, 2)),
            MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZOX")),
        ),
        (
            PatchCoordGlobal3D((0, 1, 2)),
            MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZXO")),
        ),
        (
            PatchCoordGlobal3D((1, 1, 2)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        # Clock 3 (split XX for ZZ merge)
        (
            PatchCoordGlobal3D((0, 0, 3)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 1, 3)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 1, 3)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        # Clock 4 (merge XX preparation)
        (
            PatchCoordGlobal3D((0, 0, 4)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 1, 4)),
            MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZOXX")),
        ),
        (
            PatchCoordGlobal3D((1, 1, 4)),
            MemoryCubeSkeleton(d=d, edgespec=to_edgespec("OZXX")),
        ),
        # Clock 5 (memory)
        (
            PatchCoordGlobal3D((0, 0, 5)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 1, 5)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 1, 5)),
            MemoryCubeSkeleton(d=d, edgespec=edgespec),
        ),
        # Clock 6 (measurements)
        (
            PatchCoordGlobal3D((0, 0, 6)),
            MeasureZSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((0, 1, 6)),
            MeasureXSkeleton(d=d, edgespec=edgespec),
        ),
        (
            PatchCoordGlobal3D((1, 1, 6)),
            MeasureXSkeleton(d=d, edgespec=edgespec),
        ),
    ]

    pipes = [
        # Clock 2-3 (ZZ merge via InitPlus pipe)
        (
            PatchCoordGlobal3D((0, 0, 2)),
            PatchCoordGlobal3D((0, 1, 2)),
            InitPlusPipeSkeleton(d=d, edgespec=to_edgespec("ZZOO")),
        ),
        # Clock 3 (XX split via MeasureX pipe)
        (
            PatchCoordGlobal3D((0, 0, 3)),
            PatchCoordGlobal3D((0, 1, 3)),
            MeasureXPipeSkeleton(d=d, edgespec=to_edgespec("OOOO")),
        ),
        # Clock 4-5 (XX merge via InitZero pipe)
        (
            PatchCoordGlobal3D((0, 1, 4)),
            PatchCoordGlobal3D((1, 1, 4)),
            InitZeroPipeSkeleton(d=d, edgespec=to_edgespec("OOXX")),
        ),
        # Clock 5 (ZZ split via MeasureZ pipe)
        (
            PatchCoordGlobal3D((0, 1, 5)),
            PatchCoordGlobal3D((1, 1, 5)),
            MeasureZPipeSkeleton(d=d, edgespec=to_edgespec("OOOO")),
        ),
    ]

    for block in blocks:
        canvas.add_cube(*block)
    for pipe in pipes:
        canvas.add_pipe(*pipe)

    return canvas


def create_circuit(d: int, noise: float) -> stim.Circuit:
    """Create CNOT circuit with specified parameters.

    Parameters
    ----------
    d : int
        Distance parameter
    noise : float
        Noise probability

    Returns
    -------
    stim.Circuit
        The compiled CNOT circuit.
    """
    skeleton = _create_cnot_skeleton(d)
    canvas = skeleton.to_canvas()
    compiled_canvas = canvas.compile()

    # Compile to stim circuit using the new unified API
    # Logical observables: control qubit output (0,0,6) and target qubit output (1,1,6)
    return compile_to_stim(
        compiled_canvas,
        logical_observable_coords={
            0: [PatchCoordGlobal3D((0, 0, 6))],  # Control qubit output
            1: [PatchCoordGlobal3D((1, 1, 6))],  # Target qubit output
        },
        p_before_meas_flip=noise,
    )


if __name__ == "__main__":
    # Create tasks for different distances and noise levels
    cnot_tasks = [
        sinter.Task(
            circuit=create_circuit(d, noise),
            json_metadata={"d": d, "r": 1, "p": noise, "circuit_type": "cnot"},
        )
        for d in [3, 5, 7]
        # Finer sampling is needed to see threshold behavior
        for noise in [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-1]
    ]

    # Collect statistics
    collected_stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count() or 1,
        tasks=cnot_tasks,
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
    ax.set_title("CNOT Gate Error Rates per Round under Circuit Noise")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)
    fig.savefig("figures/cnot_error_sim.png", bbox_inches="tight")
    print("\nSaved error rate plot to figures/cnot_error_sim.png")
    plt.show()
