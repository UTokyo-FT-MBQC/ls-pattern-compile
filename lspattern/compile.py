
from __future__ import annotations

from typing import Any, Optional

from graphix_zx.qompiler import qompile

import stim  # optional; only for exporting circuits if available


def compile_canvas(*, graph, xflow=None, x_parity=None, z_parity=None,
                   scheduler=None):
    """Thin wrapper around graphix_zx.qompile for our canvas.

    Parameters
    ----------
    graph : GraphState
    xflow, zflow : dict | None
        Flow mappings; pass None to let the backend derive missing parts.
    x_parity, z_parity : list[set[int]] | None
        Parity check groups for X/Z stabilizers.
    scheduler : Any | None
        Measurement scheduler; pass None for default.
    correct_output : bool
        Whether to apply Pauli frame corrections on outputs.
    """
    return qompile(
        graph=graph,
        xflow=xflow,
        x_parity_check_group=x_parity,
        z_parity_check_group=z_parity,
        scheduler=scheduler,
    )
