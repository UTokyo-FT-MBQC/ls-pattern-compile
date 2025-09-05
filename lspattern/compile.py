from __future__ import annotations

from typing import Any

from graphix_zx.qompiler import qompile


def compile_canvas(
    *,
    graph: Any,
    xflow: dict[int, set[int]] | None = None,
    x_parity: list[set[int]] | None = None,
    z_parity: list[set[int]] | None = None,
    scheduler: Any | None = None,
) -> Any:
    """
    Thin wrapper around `graphix_zx.qompile` for an RHG canvas.

    Parameters
    ----------
    graph : Any
        A `GraphState` (or compatible) instance produced by the canvas.
    xflow : dict[int, set[int]] | None
        Optional X-flow mapping (node -> correction target nodes).
        Pass `None` to let the backend derive it if supported.
    x_parity : list[set[int]] | None
        Optional list of X-parity check groups (GLOBAL node-id sets).
    z_parity : list[set[int]] | None
        Optional list of Z-parity check groups (GLOBAL node-id sets).
    scheduler : Any | None
        Optional measurement scheduler. If `None`, backend default is used.

    Returns
    -------
    Any
        Whatever `graphix_zx.qompile` returns (pattern, circuit, etc.).
    """
    return qompile(
        graph=graph,
        xflow=xflow,
        x_parity_check_group=x_parity,
        z_parity_check_group=z_parity,
        scheduler=scheduler,
    )
