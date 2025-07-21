"""RHG memory simulation example."""

# %%

import numpy as np
import scipy.stats
import scipy
import matplotlib.pyplot as plt
import stim
import sinter
from graphix_zx.pattern import print_pattern
from graphix_zx.stim_compiler import stim_compile

from lspattern.rhg import create_rhg, visualize_rhg
from lspattern.ops import memory

# %%
d = 3
r = d
rhg_lattice, coord2node, x, z = create_rhg(d, r)
visualize_rhg(rhg_lattice, coord2node)

length = 2 * d - 1
logical = set(range(d**2 + (d - 1) ** 2))
print(f"logical Z: {logical}")
logical_observables = {0: logical}

# %%
pattern = memory(d, r)
print_pattern(pattern)

# %%
# compile to stim
stim_str = stim_compile(
    pattern,
    logical_observables,
    after_clifford_depolarization=0.001,
    before_measure_flip_probability=0.01,
)
print(stim_str)

# %%

noise = 0.001


def create_circuit(d: int, rounds: int, noise: float) -> stim.Circuit:
    pattern = memory(d, rounds)
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)


if __name__ == "__main__":
    rhg_code_tasks = [
        sinter.Task(
            circuit=create_circuit(d, d, noise),
            json_metadata={"d": d, "r": d, "noise": noise},
        )
        for d in [3, 5, 7, 9]
    ]

    collected_rhg_code_stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=1,
        tasks=rhg_code_tasks,
        decoders=["pymatching"],
        max_shots=5_000_000,
        max_errors=100,
        print_progress=True,
    )

    # %%
    # Compute the line fit.
    xs = []
    ys = []
    log_ys = []
    for stats in collected_rhg_code_stats:
        d = stats.json_metadata["d"]
        if not stats.errors:
            print(f"Didn't see any errors for d={d}")
            continue
        per_shot = stats.errors / stats.shots
        per_round = sinter.shot_error_rate_to_piece_error_rate(
            per_shot, pieces=stats.json_metadata["r"]
        )
        xs.append(d)
        ys.append(per_round)
        log_ys.append(np.log(per_round))
    fit = scipy.stats.linregress(xs, log_ys)
    print(fit)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, ys, label=f"sampled logical error rate at p={noise}")
    ax.plot(
        [0, 25],
        [np.exp(fit.intercept), np.exp(fit.intercept + fit.slope * 25)],
        linestyle="--",
        label="least squares line fit",
    )
    ax.set_ylim(1e-12, 1e-0)
    ax.set_xlim(0, 25)
    ax.semilogy()
    ax.set_title("Projecting distance needed to survive a trillion rounds")
    ax.set_xlabel("Code Distance")
    ax.set_ylabel("Logical Error Rate per Round")
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.legend()
    fig.set_dpi(120)  # Show it bigger
