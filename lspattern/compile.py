
from __future__ import annotations

from typing import Any, Optional

try:
    from graphix_zx.qompiler import qompile
except Exception:  # pragma: no cover
    def qompile(*args, **kwargs):  # type: ignore
        raise RuntimeError("qompile requires graphix_zx. Please install graphix_zx.")

try:
    import stim  # optional; only for exporting circuits if available
except Exception:  # pragma: no cover
    stim = None  # type: ignore


def compile_canvas(*, graph, xflow=None, zflow=None, x_parity=None, z_parity=None,
                   scheduler=None, correct_output: bool = True):
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
        zflow=zflow,
        x_parity_check_group=x_parity,
        z_parity_check_group=z_parity,
        scheduler=scheduler,
        correct_output=correct_output,
    )


def pattern_to_stim(pattern) -> "stim.Circuit":
    """Convert a compiled Pattern to a stim.Circuit if `stim` is available.

    This is a small convenience; if `stim` isn't installed, it raises.
    """
    if stim is None:
        raise RuntimeError("`stim` is not available. Install `stim` to export circuits.")
    # Many Pattern implementations already expose .to_stim(); otherwise adjust here.
    if hasattr(pattern, "to_stim"):
        return pattern.to_stim()
    raise AttributeError("Pattern object has no `to_stim()` method.")
