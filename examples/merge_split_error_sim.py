"""Merge and Split error rate simulation with noise probability sweep."""

import os

import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim

from lspattern.blocks.cubes.initialize import InitZeroCubeThinLayerSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_to_stim
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D


def _create_merge_split_skeleton(d: int) -> RHGCanvasSkeleton:
    """Create RHG canvas skeleton for merge and split operation."""
    canvass = RHGCanvasSkeleton("Merge and Split")

    edgespec: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec1: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.O,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec2: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec_trimmed: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.O,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec_measure_trimmed: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.O,
        BoundarySide.TOP: EdgeSpecValue.O,
        BoundarySide.BOTTOM: EdgeSpecValue.O,
    }

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

    # Compile to stim circuit using the new unified API
    # Note: Converting sets to lists for the API
    return compile_to_stim(
        compiled_canvas,
        logical_observable_coords={
            0: [PatchCoordGlobal3D((0, 0, 4))],  # First output patch
            1: [PatchCoordGlobal3D((1, 0, 4))],  # Second output patch
            2: [PipeCoordGlobal3D((PatchCoordGlobal3D((0, 0, 2)), PatchCoordGlobal3D((1, 0, 2))))],  # InitPlus pipe
        },
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
    def extract_per_observable_error_rates(stats: sinter.TaskStats, n_obs: int = 3) -> list[tuple[float, float]]:
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

    def analyze_correlations(stats: sinter.TaskStats, n_obs: int = 3) -> dict[str, Any]:
        """Analyze correlations between observables.

        Returns
        -------
        dict
            Dictionary containing correlation analysis results.
        """
        # Extract error patterns
        error_patterns: dict[str, int] = {}
        for key, count in stats.custom_counts.items():
            if not key.startswith("obs_mistake_mask="):
                continue
            mask = key.split("=", 1)[1][:n_obs]
            error_patterns[mask] = count

        # Calculate correlation coefficients
        total_shots = stats.shots
        if total_shots == 0:
            return {"error": "No shots"}

        # Build error matrices for each observable pair
        corr_matrix = np.zeros((n_obs, n_obs))
        joint_probs: dict[tuple[int, int], dict[str, float]] = {}

        for i in range(n_obs):
            for j in range(n_obs):
                # Count shots where both i and j have errors
                both_error = 0
                i_error = 0
                j_error = 0
                neither_error = 0

                for pattern, count in error_patterns.items():
                    i_has_error = pattern[i] == "E" if i < len(pattern) else False
                    j_has_error = pattern[j] == "E" if j < len(pattern) else False

                    if i_has_error and j_has_error:
                        both_error += count
                    if i_has_error:
                        i_error += count
                    if j_has_error:
                        j_error += count
                    if not i_has_error and not j_has_error:
                        neither_error += count

                # Calculate correlation coefficient (Pearson)
                p_i = i_error / total_shots
                p_j = j_error / total_shots
                p_both = both_error / total_shots

                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Correlation: (E[XY] - E[X]E[Y]) / sqrt(Var[X]Var[Y])
                    numerator = p_both - p_i * p_j
                    denominator = np.sqrt(p_i * (1 - p_i) * p_j * (1 - p_j))
                    if denominator > 0:
                        corr_matrix[i, j] = numerator / denominator
                    else:
                        corr_matrix[i, j] = 0.0

                joint_probs[(i, j)] = {
                    "both_error": both_error / total_shots,
                    "i_error": p_i,
                    "j_error": p_j,
                    "conditional_j_given_i": both_error / i_error if i_error > 0 else 0.0,
                }

        return {
            "correlation_matrix": corr_matrix,
            "joint_probabilities": joint_probs,
            "error_patterns": error_patterns,
            "total_shots": total_shots,
        }

    # Analyze error patterns and correlations for all cases
    print("\n" + "=" * 80)
    print("ERROR PATTERN STATISTICS AND CORRELATION ANALYSIS")
    print("=" * 80)

    for stat in collected_stats:
        d_val = stat.json_metadata["d"]
        p_val = stat.json_metadata["p"]
        corr_data = analyze_correlations(stat)

        print(f"\n{'─' * 80}")
        print(f"Distance d={d_val}, Physical Error Rate p={p_val:.4f}")
        print(f"Total shots: {corr_data['total_shots']:,}")
        print(f"{'─' * 80}")

        # Individual observable error rates
        print("\nIndividual Observable Error Rates:")
        for i in range(3):
            p_i = corr_data["joint_probabilities"][(i, i)]["i_error"]
            print(f"  Observable {i}: {p_i:.6f} ({100 * p_i:.3f}%)")

        # Correlation matrix
        print("\nCorrelation Matrix (Pearson):")
        print("        Obs0    Obs1    Obs2")
        for i in range(3):
            row = f"  Obs{i}"
            for j in range(3):
                row += f" {corr_data['correlation_matrix'][i, j]:7.3f}"
            print(row)

        # Independence test
        print("\nIndependence Test (ratio should be ~1.0 if independent):")
        for i in range(3):
            for j in range(i + 1, 3):
                jp = corr_data["joint_probabilities"][(i, j)]
                expected_both = jp["i_error"] * jp["j_error"]
                actual_both = jp["both_error"]
                ratio = actual_both / expected_both if expected_both > 0 else 0
                print(f"  P(Obs{i} ∩ Obs{j}) / [P(Obs{i}) × P(Obs{j})] = {ratio:.3f}")

        # Top error patterns
        print("\nTop Error Patterns (format: Obs0 Obs1 Obs2, E=Error, _=Correct):")
        sorted_patterns = sorted(corr_data["error_patterns"].items(), key=lambda x: x[1], reverse=True)
        for idx, (pattern, count) in enumerate(sorted_patterns[:15], 1):
            percentage = 100 * count / corr_data["total_shots"]
            num_errors = pattern.count("E")
            print(
                f"  {idx:2d}. {pattern:5s}: {count:7,} shots ({percentage:5.2f}%) - {num_errors} observable(s) with errors"
            )

    print("\n" + "=" * 80 + "\n")

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
    print(f"\nSaved per-observable error rate plot to figures/merge_split_error_sim.png")

    # Create error pattern distribution visualization
    # Select a reference noise level for pattern visualization
    ref_noise = 0.02
    available_stats = [s for s in collected_stats if abs(s.json_metadata["p"] - ref_noise) < 1e-6]

    if available_stats:
        n_plots = len(available_stats)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes2 = [axes2]
        else:
            axes2 = axes2.flatten() if n_plots > 1 else [axes2]

        for idx, stat in enumerate(available_stats):
            if idx >= len(axes2):
                break

            d_val = stat.json_metadata["d"]
            p_val = stat.json_metadata["p"]
            corr_data = analyze_correlations(stat)
            ax = axes2[idx]

            # Get top error patterns
            sorted_patterns = sorted(corr_data["error_patterns"].items(), key=lambda x: x[1], reverse=True)
            top_patterns = sorted_patterns[:10]  # Show top 10 patterns

            pattern_labels = [p[0] for p in top_patterns]
            pattern_counts = [p[1] for p in top_patterns]
            pattern_percentages = [100 * c / corr_data["total_shots"] for c in pattern_counts]

            # Create bar plot
            bars = ax.bar(range(len(pattern_labels)), pattern_percentages)
            ax.set_xticks(range(len(pattern_labels)))
            ax.set_xticklabels(pattern_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Percentage of Shots (%)", fontsize=10)
            ax.set_title(f"d={d_val}, p={p_val:.3f}\n(Total shots: {corr_data['total_shots']:,})", fontsize=11)
            ax.grid(axis="y", alpha=0.3)

            # Color bars based on number of errors
            colors = ["green", "yellow", "orange", "red"]
            for i, (bar, pattern) in enumerate(zip(bars, pattern_labels)):
                num_errors = pattern.count("E")
                bar.set_color(colors[min(num_errors, 3)])

        # Hide unused subplots
        for idx in range(len(available_stats), len(axes2)):
            axes2[idx].axis("off")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", label="No errors (_____)"),
            Patch(facecolor="yellow", label="1 error"),
            Patch(facecolor="orange", label="2 errors"),
            Patch(facecolor="red", label="3+ errors"),
        ]
        if len(available_stats) < len(axes2):
            axes2[-1].legend(handles=legend_elements, loc="center", fontsize=12)
        else:
            fig2.legend(handles=legend_elements, loc="upper right", fontsize=10)

        fig2.suptitle(f"Error Pattern Distribution (p={ref_noise})", fontsize=14, y=0.995)
        fig2.tight_layout()
        fig2.set_dpi(120)
        fig2.savefig("figures/merge_split_error_patterns.png", bbox_inches="tight")
        print(f"Saved error pattern visualization to figures/merge_split_error_patterns.png")

    plt.show()
