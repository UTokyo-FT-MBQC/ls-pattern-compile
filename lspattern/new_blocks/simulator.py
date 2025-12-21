"""Sinter-based logical error rate simulation utilities.

This module provides utilities for evaluating logical error rates of
compiled MBQC circuits using sinter (stim's batch sampling framework).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sinter

if TYPE_CHECKING:
    from collections.abc import Callable

    import stim

# Minimum number of data points required for curve fitting (3 parameters: A, B, C)
MIN_FITTING_POINTS = 3


@dataclass
class SimulationConfig:
    """Configuration for logical error rate simulation.

    Attributes
    ----------
    code_distances : list[int]
        List of code distances to simulate.
    noise_rates : list[float]
        List of physical error rates to simulate.
    max_shots : int
        Maximum number of shots per task.
    max_errors : int
        Maximum number of errors to collect per task.
    num_workers : int | None
        Number of parallel workers. If None, uses all available CPUs.
    decoders : list[str]
        List of decoder names to use (e.g., ["pymatching"]).
    """

    code_distances: list[int]
    noise_rates: list[float]
    max_shots: int = 1_000_000
    max_errors: int = 5_000
    num_workers: int | None = None
    decoders: list[str] | None = None

    def __post_init__(self) -> None:
        if self.decoders is None:
            self.decoders = ["pymatching"]
        if self.num_workers is None:
            self.num_workers = os.cpu_count() or 1


def compute_x_from_d(d: int) -> float:
    """Compute x value from code distance d for fitting model.

    The scaling factor x is:
    - x = (d + 1) / 2 for odd d
    - x = d / 2 for even d

    This is equivalent to ceil(d / 2).

    Parameters
    ----------
    d : int
        Code distance.

    Returns
    -------
    float
        Scaling factor x for the fitting model.
    """
    if d % 2 == 1:  # odd
        return (d + 1) / 2
    return d / 2


@dataclass
class FittingResult:
    """Result of logical error rate fitting.

    The fitting model is: p_L = A * p^(B * x) * exp(C * d)
    where p_L is logical error rate, p is physical error rate, d is code distance,
    and x = (d+1)/2 for odd d, x = d/2 for even d.

    In log-space: log(p_L) = log(A) + B * x * log(p) + C * d

    Attributes
    ----------
    A : float
        Fitting coefficient A.
    B : float
        Fitting exponent coefficient B (typically positive).
    C : float
        Exponential decay/growth coefficient in exp(C * d).
    A_err : float
        Standard error of A.
    B_err : float
        Standard error of B.
    C_err : float
        Standard error of C.
    observable_index : int | None
        Index of the logical observable this fitting applies to.
    """

    A: float
    B: float
    C: float
    A_err: float
    B_err: float
    C_err: float
    observable_index: int | None = None


def create_sinter_tasks(
    circuit_factory: Callable[[int, float], stim.Circuit],
    config: SimulationConfig,
    circuit_type: str = "default",
) -> list[sinter.Task]:
    """Create sinter tasks for logical error rate simulation.

    Parameters
    ----------
    circuit_factory : Callable[[int, float], stim.Circuit]
        A function that takes (code_distance, noise_rate) and returns a stim.Circuit.
    config : SimulationConfig
        Simulation configuration.
    circuit_type : str
        Label for the circuit type in metadata.

    Returns
    -------
    list[sinter.Task]
        List of sinter tasks ready for simulation.

    Examples
    --------
    >>> def my_circuit_factory(d: int, p: float) -> stim.Circuit:
    ...     # Build circuit with distance d and noise rate p
    ...     return circuit
    >>> config = SimulationConfig(
    ...     code_distances=[3, 5, 7],
    ...     noise_rates=[0.001, 0.005, 0.01],
    ... )
    >>> tasks = create_sinter_tasks(my_circuit_factory, config)
    """
    tasks: list[sinter.Task] = []
    for d in config.code_distances:
        for noise in config.noise_rates:
            circuit = circuit_factory(d, noise)
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        "d": d,
                        "p": noise,
                        "circuit_type": circuit_type,
                    },
                )
            )
    return tasks


def run_simulation(
    tasks: list[sinter.Task],
    config: SimulationConfig,
    print_progress: bool = True,
) -> list[sinter.TaskStats]:
    """Run sinter simulation and collect statistics.

    Parameters
    ----------
    tasks : list[sinter.Task]
        List of sinter tasks to simulate.
    config : SimulationConfig
        Simulation configuration.
    print_progress : bool
        Whether to print progress during simulation.

    Returns
    -------
    list[sinter.TaskStats]
        Collected statistics for each task.

    Examples
    --------
    >>> stats = run_simulation(tasks, config)
    >>> for stat in stats:
    ...     print(f"d={stat.json_metadata['d']}, p={stat.json_metadata['p']}")
    ...     print(f"  errors/shots = {stat.errors}/{stat.shots}")
    """
    if config.decoders is None:
        msg = "decoders must be set in config"
        raise ValueError(msg)
    collected_stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=config.num_workers or (os.cpu_count() or 1),
        tasks=tasks,
        decoders=config.decoders,
        max_shots=config.max_shots,
        max_errors=config.max_errors,
        print_progress=print_progress,
    )
    return collected_stats


def simulate_logical_error_rate(
    circuit_factory: Callable[[int, float], stim.Circuit],
    config: SimulationConfig,
    circuit_type: str = "default",
    print_progress: bool = True,
) -> list[sinter.TaskStats]:
    """Convenience function to create tasks and run simulation in one step.

    Parameters
    ----------
    circuit_factory : Callable[[int, float], stim.Circuit]
        A function that takes (code_distance, noise_rate) and returns a stim.Circuit.
    config : SimulationConfig
        Simulation configuration.
    circuit_type : str
        Label for the circuit type in metadata.
    print_progress : bool
        Whether to print progress during simulation.

    Returns
    -------
    list[sinter.TaskStats]
        Collected statistics for each task.
    """
    tasks = create_sinter_tasks(circuit_factory, config, circuit_type)
    return run_simulation(tasks, config, print_progress)


class InsufficientDataError(ValueError):
    """Raised when there are insufficient data points for fitting."""

    def __init__(self, n_points: int, min_required: int = MIN_FITTING_POINTS) -> None:
        super().__init__(f"Insufficient data points for fitting (got {n_points}, need at least {min_required})")
        self.n_points = n_points
        self.min_required = min_required


class FittingError(ValueError):
    """Raised when curve fitting fails."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(f"Fitting failed: {cause}")
        self.cause = cause


def fit_logical_error_rate(
    stats: list[sinter.TaskStats],
    observable_index: int | None = None,
) -> FittingResult:
    """Fit logical error rate data to model: p_L = A * p^(B * x) * exp(C * d).

    Where x = (d+1)/2 for odd d, x = d/2 for even d (equivalent to ceil(d/2)).

    Fitting is performed in log-space for numerical stability:
    log(p_L) = log(A) + B * x * log(p) + C * d

    Parameters
    ----------
    stats : list[sinter.TaskStats]
        Collected statistics from simulation.
    observable_index : int | None
        Index of the logical observable to fit. If None, uses the first observable (index 0).

    Returns
    -------
    FittingResult
        Fitting parameters A, B, and C with their uncertainties.

    Raises
    ------
    InsufficientDataError
        If insufficient data points for fitting.
    FittingError
        If curve fitting fails.

    Examples
    --------
    >>> result = fit_logical_error_rate(stats)
    >>> print(f"A = {result.A:.4e} ± {result.A_err:.4e}")
    >>> print(f"B = {result.B:.4f} ± {result.B_err:.4f}")
    >>> print(f"C = {result.C:.4e} ± {result.C_err:.4e}")
    """
    if observable_index is None:
        observable_index = 0

    # Extract data points
    p_values: list[float] = []
    d_values: list[float] = []
    error_rates: list[float] = []

    for stat in stats:
        if stat.shots == 0:
            continue
        # Get error rate for specific observable
        error_rate = stat.errors / stat.shots
        if error_rate <= 0:
            continue

        p_values.append(stat.json_metadata["p"])
        d_values.append(stat.json_metadata["d"])
        error_rates.append(error_rate)

    if len(p_values) < MIN_FITTING_POINTS:
        raise InsufficientDataError(len(p_values))

    # Convert to numpy arrays
    p_arr = np.array(p_values)
    d_arr = np.array(d_values)
    error_arr = np.array(error_rates)

    # Compute x values: x = (d+1)/2 for odd d, x = d/2 for even d
    x_arr = np.array([compute_x_from_d(int(d)) for d in d_arr])

    # Fit in log-space: log(p_L) = log(A) + B * x * log(p) + C * d
    # This is a linear regression: y = c0 + c1 * x1 + c2 * x2
    # where y = log(p_L), x1 = x * log(p), x2 = d
    # c0 = log(A), c1 = B, c2 = C
    log_p = np.log(p_arr)
    log_error = np.log(error_arr)
    x1 = x_arr * log_p  # x * log(p)
    x2 = d_arr  # d

    # Design matrix: [1, x1, x2] for fitting y = c0 + c1*x1 + c2*x2
    design_matrix = np.column_stack([np.ones_like(x1), x1, x2])

    try:
        # Solve least squares: (X^T X)^-1 X^T y
        coeffs, _residuals, _rank, _s = np.linalg.lstsq(design_matrix, log_error, rcond=None)
        log_a, b, c = coeffs[0], coeffs[1], coeffs[2]
        a = np.exp(log_a)

        # Estimate standard errors
        n = len(log_error)
        n_params = MIN_FITTING_POINTS  # Number of parameters (A, B, C)
        if n > n_params:
            y_pred = design_matrix @ coeffs
            mse = np.sum((log_error - y_pred) ** 2) / (n - n_params)
            var_coeffs = mse * np.linalg.inv(design_matrix.T @ design_matrix)
            log_a_err = np.sqrt(var_coeffs[0, 0])
            b_err = np.sqrt(var_coeffs[1, 1])
            c_err = np.sqrt(var_coeffs[2, 2])
            # Error propagation for A = exp(log_A): dA = A * d(log_A)
            a_err = a * log_a_err
        else:
            a_err = 0.0
            b_err = 0.0
            c_err = 0.0

        return FittingResult(
            A=float(a),
            B=float(b),
            C=float(c),
            A_err=float(a_err),
            B_err=float(b_err),
            C_err=float(c_err),
            observable_index=observable_index,
        )
    except np.linalg.LinAlgError as e:
        raise FittingError(e) from e


def stats_to_dataframe(stats: list[sinter.TaskStats]) -> dict[str, list[int | float | str]]:
    """Convert sinter statistics to a dictionary suitable for DataFrame creation.

    Parameters
    ----------
    stats : list[sinter.TaskStats]
        Collected statistics from simulation.

    Returns
    -------
    dict[str, list[int | float | str]]
        Dictionary with keys: 'd', 'p', 'shots', 'errors', 'error_rate', 'circuit_type'.

    Examples
    --------
    >>> data = stats_to_dataframe(stats)
    >>> import pandas as pd
    >>> df = pd.DataFrame(data)
    """
    data: dict[str, list[int | float | str]] = {
        "d": [],
        "p": [],
        "shots": [],
        "errors": [],
        "error_rate": [],
        "circuit_type": [],
    }

    for stat in stats:
        data["d"].append(stat.json_metadata["d"])
        data["p"].append(stat.json_metadata["p"])
        data["shots"].append(stat.shots)
        data["errors"].append(stat.errors)
        data["error_rate"].append(stat.errors / stat.shots if stat.shots > 0 else 0)
        data["circuit_type"].append(stat.json_metadata.get("circuit_type", "default"))

    return data
