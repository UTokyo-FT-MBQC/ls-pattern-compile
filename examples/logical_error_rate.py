"""Logical error rate evaluation example for canvas specifications.

This script demonstrates how to use the simulator utilities to evaluate
logical error rates for canvas specifications across multiple code distances
and noise rates.

Usage:
    python -m lspattern.new_blocks.examples.logical_error_rate

To use a different canvas specification, modify the SPEC_NAME variable below.
Available specs:
    - "memory_canvas.yml" (default)
    - "design/cnot.yml"
    - "design/merge_split.yml"
    - etc.
"""

# %%
from __future__ import annotations

import pathlib
from functools import partial

import matplotlib.pyplot as plt
import stim

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.compiler import compile_canvas_to_stim
from lspattern.new_blocks.plot_error_rate import (
    ObservablePlotConfig,
    PlotConfig,
    create_error_rate_figure,
    create_multi_observable_figure,
    print_fitting_summary,
    save_figure,
)
from lspattern.new_blocks.simulator import (
    SimulationConfig,
    fit_logical_error_rate,
    simulate_logical_error_rate,
)

# Canvas specification to use for simulation
# Change this to use different design specs, e.g., "design/cnot.yml"
SPEC_NAME = "memory_canvas.yml"


def infer_circuit_type(spec_name: str) -> str:
    """Infer circuit type from spec name.

    Parameters
    ----------
    spec_name : str
        Canvas specification file name.

    Returns
    -------
    str
        Inferred circuit type for metadata.

    Examples
    --------
    >>> infer_circuit_type("memory_canvas.yml")
    'memory'
    >>> infer_circuit_type("design/cnot.yml")
    'cnot'
    """
    stem = pathlib.Path(spec_name).stem
    return stem.removesuffix("_canvas")


def create_circuit(code_distance: int, noise_rate: float, spec_name: str) -> stim.Circuit:
    """Create a stim circuit from canvas specification with given code distance and noise rate.

    Parameters
    ----------
    code_distance : int
        Code distance for the surface code.
    noise_rate : float
        Physical error rate for depolarizing noise and measurement flip.
    spec_name : str
        Canvas specification file name (e.g., "memory_canvas.yml", "design/cnot.yml").

    Returns
    -------
    stim.Circuit
        Compiled stim circuit ready for simulation.
    """
    canvas, _ = load_canvas(spec_name, code_distance=code_distance)

    circuit_str = compile_canvas_to_stim(
        canvas,
        p_depol_after_clifford=0,
        p_before_meas_flip=noise_rate,
    )
    return stim.Circuit(circuit_str)


def main() -> None:
    """Run logical error rate simulation and generate plots."""
    # Infer circuit type from spec name
    circuit_type = infer_circuit_type(SPEC_NAME)

    # Configuration for simulation
    # You can adjust these parameters based on your needs
    config = SimulationConfig(
        code_distances=[3, 5, 7],
        noise_rates=[5e-3, 1e-2, 2e-2, 3e-2, 5e-2],
        max_shots=100_000,  # Reduce for faster testing
        max_errors=1_000,
        decoders=["pymatching"],
    )

    print("=" * 60)
    print(f"Logical Error Rate Simulation for {SPEC_NAME}")
    print("=" * 60)
    print(f"Spec: {SPEC_NAME}")
    print(f"Circuit type: {circuit_type}")
    print(f"Code distances: {config.code_distances}")
    print(f"Noise rates: {config.noise_rates}")
    print(f"Max shots per task: {config.max_shots}")
    print(f"Max errors per task: {config.max_errors}")
    print()

    # Create circuit factory with spec_name bound
    circuit_factory = partial(create_circuit, spec_name=SPEC_NAME)

    # Run simulation
    print("Running simulation...")
    stats = simulate_logical_error_rate(
        circuit_factory=circuit_factory,
        config=config,
        circuit_type=circuit_type,
        print_progress=True,
    )
    print(f"\nCollected statistics for {len(stats)} tasks")

    # Print raw statistics
    print("\nRaw Statistics:")
    print("-" * 60)
    for stat in stats:
        d = stat.json_metadata["d"]
        p = stat.json_metadata["p"]
        error_rate = stat.errors / stat.shots if stat.shots > 0 else 0
        print(f"d={d}, p={p:.4f}: errors={stat.errors}/{stat.shots} = {error_rate:.4e}")

    # Perform fitting
    print("\nFitting to model: p_L = A * p^(B*x) * exp(C*d)")
    print("  where x = (d+1)/2 for odd d, d/2 for even d")
    print("-" * 60)
    try:
        fitting_result = fit_logical_error_rate(stats)
        print_fitting_summary([fitting_result])
    except ValueError as e:
        print(f"Fitting failed: {e}")
        fitting_result = None

    # Create plot configuration
    # Users can customize titles and labels here
    plot_config = PlotConfig(
        title=f"{circuit_type.replace('_', ' ').title()} Logical Error Rate",
        xlabel="Physical Error Rate (p)",
        ylabel="Logical Error Rate (p_L)",
        xscale="log",
        yscale="log",
        figsize=(10, 8),
        show_fitting_curve=fitting_result is not None,
        show_fitting_params=True,
    )

    # Create and save figure
    print("\nGenerating plot...")
    fig, _ax, _ = create_error_rate_figure(
        stats,
        config=plot_config,
    )

    # Ensure output directory exists
    output_dir = pathlib.Path("figures")
    output_dir.mkdir(exist_ok=True)

    saved_files = save_figure(fig, str(output_dir / f"{circuit_type}_logical_error_rate"), formats=["png"])
    print(f"Saved figures: {saved_files}")

    # Show interactive plot
    plt.show()


# %%
# Example of customizing for multiple observables
def example_multi_observable() -> None:
    """Example of handling multiple logical observables.

    This function shows how to customize plots when your circuit has
    multiple logical observables (e.g., X and Z logical operators).
    """
    # This is a placeholder showing how to customize for multiple observables
    # In a real scenario, you would have separate statistics for each observable

    # Example observable configuration
    obs_config = ObservablePlotConfig(
        observable_titles={
            0: "Logical Z Error Rate",
            1: "Logical X Error Rate",
        },
        observable_labels={
            0: "Z Observable",
            1: "X Observable",
        },
    )

    print("Observable configuration created:")
    print(f"  Title for obs 0: {obs_config.get_title(0)}")
    print(f"  Title for obs 1: {obs_config.get_title(1)}")

    # When you have stats for multiple observables:
    # stats_by_obs = {0: stats_z, 1: stats_x}
    # fig, axes, fittings = create_multi_observable_figure(
    #     stats_by_obs,
    #     observable_config=obs_config,
    #     show_fitting=True,
    #     layout="vertical",
    # )

    # Suppress unused import warning by referencing the function
    _ = create_multi_observable_figure


# %%
if __name__ == "__main__":
    main()
