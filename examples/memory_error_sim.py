"""RHG memory simulation with noise probability sweep."""

import os

import matplotlib.pyplot as plt
import sinter
import stim
from graphix_zx.stim_compiler import stim_compile
from lspattern.ops import memory


def create_circuit(d: int, rounds: int, noise: float) -> stim.Circuit:
    """Create RHG memory circuit with specified parameters."""
    pattern = memory(d, rounds)
    length = 2 * d - 1
    logical_observables = {0: {length * i for i in range(d)}}
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)


if __name__ == "__main__":
    # Create tasks for different distances and noise levels
    rhg_code_tasks = [
        sinter.Task(
            circuit=create_circuit(d, d * 3, noise),
            json_metadata={"d": d, "r": d * 3, "p": noise},
        )
        for d in [3, 7, 11]
        for noise in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    ]

    # Collect statistics
    collected_rhg_code_stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=rhg_code_tasks,
        decoders=["pymatching"],
        max_shots=1_000_000,
        max_errors=5_000,
        print_progress=True,
    )

    # Plot the results
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=collected_rhg_code_stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: stat.json_metadata["d"],
        failure_units_per_shot_func=lambda stat: stat.json_metadata["r"],
    )
    # ax.set_ylim(5e-3, 5e-2)
    # ax.set_xlim(0.008, 0.012)
    ax.loglog()
    ax.set_title("RHG Code Error Rates per Round under Circuit Noise")
    ax.set_xlabel("Physical Error Rate")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)
    fig.savefig("figures/memory_sim.png", bbox_inches="tight")
    plt.show()
