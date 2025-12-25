"""Visualization utilities for logical error rate analysis.

This module provides plotting functions for visualizing logical error rates
with support for log-log and semi-log scales, multiple observables, and
customizable titles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import sinter

from lspattern.new_blocks.simulator import (
    FittingResult,
    compute_x_from_d,
    fit_logical_error_rate,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _default_x_func(stat: sinter.TaskStats) -> float:
    """Default x-value extractor for plotting."""
    return stat.json_metadata["p"]  # type: ignore[no-any-return]


def _default_group_func(stat: sinter.TaskStats) -> str:
    """Default group label generator for plotting."""
    return f"d={stat.json_metadata['d']}"


@dataclass
class PlotConfig:
    """Configuration for error rate plots.

    Attributes
    ----------
    title : str | None
        Plot title. If None, a default title is generated.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    xscale : str
        X-axis scale ('log' or 'linear').
    yscale : str
        Y-axis scale ('log' or 'linear').
    figsize : tuple[float, float]
        Figure size in inches.
    legend_loc : str
        Legend location.
    show_fitting_curve : bool
        Whether to show fitting curves.
    show_fitting_params : bool
        Whether to show fitting parameters in legend.
    color_by_distance : bool
        If True, color by code distance. If False, color by observable.
    """

    title: str | None = None
    xlabel: str = "Physical Error Rate"
    ylabel: str = "Logical Error Rate"
    xscale: str = "log"
    yscale: str = "log"
    figsize: tuple[float, float] = (10, 8)
    legend_loc: str = "best"
    show_fitting_curve: bool = False
    show_fitting_params: bool = True
    color_by_distance: bool = True


@dataclass
class ObservablePlotConfig:
    """Configuration for per-observable plot customization.

    Allows users to customize plot settings for each logical observable.

    Attributes
    ----------
    observable_titles : dict[int, str]
        Custom titles for each observable index.
    observable_labels : dict[int, str]
        Custom labels for each observable in the legend.
    """

    observable_titles: dict[int, str] = field(default_factory=dict)
    observable_labels: dict[int, str] = field(default_factory=dict)

    def get_title(self, obs_index: int, default: str | None = None) -> str:
        """Get the title for a specific observable."""
        if obs_index in self.observable_titles:
            return self.observable_titles[obs_index]
        return default or f"Observable {obs_index}"

    def get_label(self, obs_index: int, default: str | None = None) -> str:
        """Get the legend label for a specific observable."""
        if obs_index in self.observable_labels:
            return self.observable_labels[obs_index]
        return default or f"Obs {obs_index}"


def plot_error_rate(
    ax: Axes,
    stats: list[sinter.TaskStats],
    config: PlotConfig | None = None,
    x_func: Callable[[sinter.TaskStats], float] | None = None,
    group_func: Callable[[sinter.TaskStats], str] | None = None,
) -> Axes:
    """Plot logical error rate using sinter's built-in plotting.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    stats : list[sinter.TaskStats]
        Collected statistics from simulation.
    config : PlotConfig | None
        Plot configuration. If None, uses defaults.
    x_func : Callable[[sinter.TaskStats], float] | None
        Function to extract x-value from stats. Default extracts 'p' from metadata.
    group_func : Callable[[sinter.TaskStats], str] | None
        Function to generate group label. Default groups by code distance.

    Returns
    -------
    Axes
        The axes with the plot.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_error_rate(ax, stats)
    >>> plt.show()
    """
    if config is None:
        config = PlotConfig()

    if x_func is None:
        x_func = _default_x_func

    if group_func is None:
        group_func = _default_group_func

    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=x_func,
        group_func=group_func,
    )

    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_xscale(config.xscale)
    ax.set_yscale(config.yscale)
    if config.title:
        ax.set_title(config.title)
    ax.legend(loc=config.legend_loc)
    ax.grid(True, alpha=0.3)

    return ax


def plot_error_rate_with_fitting(
    ax: Axes,
    stats: list[sinter.TaskStats],
    fitting_result: FittingResult | None = None,
    config: PlotConfig | None = None,
) -> tuple[Axes, FittingResult | None]:
    """Plot logical error rate with optional fitting curve.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    stats : list[sinter.TaskStats]
        Collected statistics from simulation.
    fitting_result : FittingResult | None
        Pre-computed fitting result. If None and config.show_fitting_curve is True,
        fitting is performed automatically.
    config : PlotConfig | None
        Plot configuration. If None, uses defaults.

    Returns
    -------
    tuple[Axes, FittingResult | None]
        The axes with the plot and the fitting result (if computed).
    """
    if config is None:
        config = PlotConfig()

    # First plot the data points
    plot_error_rate(ax, stats, config)

    # Compute fitting if requested and not provided
    if config.show_fitting_curve and fitting_result is None:
        try:
            fitting_result = fit_logical_error_rate(stats)
        except ValueError:
            # Not enough data for fitting
            fitting_result = None

    # Add fitting curve if available
    if fitting_result is not None and config.show_fitting_curve:
        _add_fitting_curve(ax, stats, fitting_result, config)

    return ax, fitting_result


def _add_fitting_curve(
    ax: Axes,
    stats: list[sinter.TaskStats],
    fitting_result: FittingResult,
    config: PlotConfig,
) -> None:
    """Add fitting curve to the plot."""
    # Get unique code distances and p range
    distances = sorted({stat.json_metadata["d"] for stat in stats})
    p_values = sorted({stat.json_metadata["p"] for stat in stats})
    p_min, p_max = min(p_values), max(p_values)

    # Generate smooth p values for curve
    p_smooth = np.logspace(np.log10(p_min), np.log10(p_max), 100)

    # Use the same color cycle as matplotlib default (which sinter uses)
    # Get colors from the axes' existing lines to match sinter's colors
    existing_lines = ax.get_lines()
    color_map: dict[int, object] = {}
    for line in existing_lines:
        # Parse the label to extract distance
        label = line.get_label()
        if isinstance(label, str) and label.startswith("d="):
            try:
                d_val = int(label.split("=")[1])
                color_map[d_val] = line.get_color()
            except (ValueError, IndexError):
                pass

    # Plot fitting curve for each distance using matched colors
    # Model: p_L = A * p^(B*x) * exp(C*d) where x = (d+1)/2 for odd d, d/2 for even d
    for i, d in enumerate(distances):
        x = compute_x_from_d(d)
        y_fit = fitting_result.A * np.power(p_smooth, fitting_result.B * x) * np.exp(fitting_result.C * d)
        color = color_map.get(d, f"C{i}")  # Fallback to default color cycle
        label = None
        if config.show_fitting_params and i == 0:
            label = f"Fit: A={fitting_result.A:.3e}, B={fitting_result.B:.3f}, C={fitting_result.C:.3e}"
        ax.plot(p_smooth, y_fit, "--", color=color, alpha=0.7, label=label)

    if config.show_fitting_params:
        ax.legend(loc=config.legend_loc)


def create_error_rate_figure(
    stats: list[sinter.TaskStats],
    config: PlotConfig | None = None,
    observable_config: ObservablePlotConfig | None = None,
) -> tuple[Figure, Axes | list[Axes], list[FittingResult]]:
    """Create a complete error rate figure.

    This is a convenience function that creates a figure with appropriate
    axes and plots the error rate data.

    Parameters
    ----------
    stats : list[sinter.TaskStats]
        Collected statistics from simulation.
    config : PlotConfig | None
        Plot configuration. If None, uses defaults.
        Use config.show_fitting_curve to control fitting curve display.
    observable_config : ObservablePlotConfig | None
        Per-observable customization. If None, uses defaults.

    Returns
    -------
    tuple[Figure, Axes | list[Axes], list[FittingResult]]
        The figure, axes (single or list for multiple observables), and fitting results.

    Examples
    --------
    >>> config = PlotConfig(show_fitting_curve=True)
    >>> fig, ax, fittings = create_error_rate_figure(stats, config=config)
    >>> for fit in fittings:
    ...     print(f"A={fit.A:.3e}, B={fit.B:.3f}")
    >>> plt.show()
    """
    if config is None:
        config = PlotConfig()

    if observable_config is None:
        observable_config = ObservablePlotConfig()

    fig, ax = plt.subplots(1, 1, figsize=config.figsize)

    fitting_results: list[FittingResult] = []

    if config.show_fitting_curve:
        _, fitting_result = plot_error_rate_with_fitting(ax, stats, config=config)
        if fitting_result is not None:
            fitting_results.append(fitting_result)
    else:
        plot_error_rate(ax, stats, config)

    return fig, ax, fitting_results


def create_multi_observable_figure(
    stats_per_observable: dict[int, list[sinter.TaskStats]],
    config: PlotConfig | None = None,
    observable_config: ObservablePlotConfig | None = None,
    layout: str = "vertical",
) -> tuple[Figure, list[Axes], dict[int, FittingResult]]:
    """Create a figure with subplots for multiple logical observables.

    Parameters
    ----------
    stats_per_observable : dict[int, list[sinter.TaskStats]]
        Dictionary mapping observable index to its statistics.
    config : PlotConfig | None
        Base plot configuration. If None, uses defaults.
        Use config.show_fitting_curve to control fitting curve display.
    observable_config : ObservablePlotConfig | None
        Per-observable customization for titles and labels.
    layout : str
        Layout of subplots: 'vertical', 'horizontal', or 'grid'.

    Returns
    -------
    tuple[Figure, list[Axes], dict[int, FittingResult]]
        The figure, list of axes, and fitting results per observable.

    Examples
    --------
    >>> # For circuits with multiple logical observables
    >>> stats_by_obs = {0: stats_x, 1: stats_z}
    >>> obs_config = ObservablePlotConfig(observable_titles={0: "X Observable", 1: "Z Observable"})
    >>> plot_config = PlotConfig(show_fitting_curve=True)
    >>> fig, axes, fittings = create_multi_observable_figure(
    ...     stats_by_obs, config=plot_config, observable_config=obs_config
    ... )
    >>> plt.show()
    """
    if config is None:
        config = PlotConfig()

    if observable_config is None:
        observable_config = ObservablePlotConfig()

    n_obs = len(stats_per_observable)

    # Determine subplot layout
    if layout == "vertical":
        nrows, ncols = n_obs, 1
        fig_height = config.figsize[1] * n_obs * 0.6
        figsize = (config.figsize[0], fig_height)
    elif layout == "horizontal":
        nrows, ncols = 1, n_obs
        fig_width = config.figsize[0] * n_obs * 0.6
        figsize = (fig_width, config.figsize[1])
    else:  # grid
        ncols = int(np.ceil(np.sqrt(n_obs)))
        nrows = int(np.ceil(n_obs / ncols))
        figsize = (config.figsize[0] * ncols * 0.5, config.figsize[1] * nrows * 0.5)

    fig, axes_array = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes_array.flatten().tolist()

    fitting_results: dict[int, FittingResult] = {}

    for i, (obs_idx, stats) in enumerate(sorted(stats_per_observable.items())):
        ax = axes_flat[i]
        title = observable_config.get_title(obs_idx, f"Logical Observable {obs_idx}")

        subplot_config = PlotConfig(
            title=title,
            xlabel=config.xlabel,
            ylabel=config.ylabel,
            xscale=config.xscale,
            yscale=config.yscale,
            figsize=config.figsize,
            legend_loc=config.legend_loc,
            show_fitting_curve=config.show_fitting_curve,
            show_fitting_params=config.show_fitting_params,
            color_by_distance=config.color_by_distance,
        )

        if config.show_fitting_curve:
            _, fitting_result = plot_error_rate_with_fitting(ax, stats, config=subplot_config)
            if fitting_result is not None:
                fitting_result.observable_index = obs_idx
                fitting_results[obs_idx] = fitting_result
        else:
            plot_error_rate(ax, stats, subplot_config)

    # Hide unused axes
    for i in range(len(stats_per_observable), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout()
    return fig, axes_flat[: len(stats_per_observable)], fitting_results


def save_figure(
    fig: Figure,
    filename: str,
    dpi: int = 150,
    formats: list[str] | None = None,
) -> list[str]:
    """Save figure in multiple formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    filename : str
        Base filename (without extension).
    dpi : int
        Resolution in dots per inch.
    formats : list[str] | None
        List of formats to save. Default is ['png', 'pdf'].

    Returns
    -------
    list[str]
        List of saved file paths.

    Examples
    --------
    >>> saved_files = save_figure(fig, "memory_error_rate", formats=["png", "pdf", "svg"])
    """
    if formats is None:
        formats = ["png", "pdf"]

    saved_files: list[str] = []
    for fmt in formats:
        filepath = f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        saved_files.append(filepath)

    return saved_files


def print_fitting_summary(fitting_results: list[FittingResult] | dict[int, FittingResult]) -> None:
    """Print a summary of fitting results.

    Parameters
    ----------
    fitting_results : list[FittingResult] | dict[int, FittingResult]
        Fitting results from simulation.

    Examples
    --------
    >>> print_fitting_summary(fitting_results)
    Fitting Summary
    ===============
    Observable 0:
      A = 1.234e-01 ± 5.678e-03
      B = 0.567 ± 0.012
    """
    results = list(fitting_results.values()) if isinstance(fitting_results, dict) else fitting_results

    print("Fitting Summary")
    print("=" * 40)

    for result in results:
        obs_label = f"Observable {result.observable_index}" if result.observable_index is not None else "All"
        print(f"\n{obs_label}:")
        print(f"  A = {result.A:.4e} ± {result.A_err:.4e}")
        print(f"  B = {result.B:.4f} ± {result.B_err:.4f}")
        print(f"  C = {result.C:.4e} ± {result.C_err:.4e}")
        print("  Model: p_L = A * p^(B * x) * exp(C * d)")
        print("  where x = (d+1)/2 for odd d, d/2 for even d")
