from __future__ import annotations

from typing import TYPE_CHECKING

from graphqomb.qompiler import qompile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from graphqomb.graphstate import BaseGraphState
    from graphqomb.pattern import Pattern
    from graphqomb.scheduler import Scheduler


def compile_canvas(
    graph: BaseGraphState,
    xflow: Mapping[int, AbstractSet[int]],
    parity: Sequence[AbstractSet[int]] | None = None,
    scheduler: Scheduler | None = None,
) -> Pattern:
    """
    Thin wrapper around `graphqomb.qompile` for an RHG canvas.

    Parameters
    ----------
    graph : Any
        A `GraphState` (or compatible) instance produced by the canvas.
    xflow : dict[int, set[int]] | None
        Optional X-flow mapping (node -> correction target nodes).
        Pass `None` to let the backend derive it if supported.
    parity : list[set[int]] | None
        Optional list of parity check groups (GLOBAL node-id sets).
    scheduler : Any | None
        Optional measurement scheduler. If `None`, backend default is used.

    Returns
    -------
    Pattern
        Whatever `graphqomb.qompile` returns (pattern, circuit, etc.).
    """
    return qompile(
        graph=graph,
        xflow=xflow,
        parity_check_group=parity,
        scheduler=scheduler,
    )
