
#!/usr/bin/env python3
"""New-API demo: RHG memory experiment (InitPlus -> Memory -> MeasureX).

This example builds a single-logical memory line by stacking blocks on a growing canvas.

If graphix_zx (and optionally stim) are installed in your environment, this script will:
  - compile the canvas into a Pattern via graphix_zx.qompile
  - (optionally) export to a stim.Circuit if `stim` is available
Otherwise, it will run in DRY mode and just print the build plan.

Usage:
  python examples/rhg_memory.py
"""

from __future__ import annotations

import sys

try:
    import graphix_zx  # noqa: F401
    GRAPHIX_AVAILABLE = True
except Exception:
    GRAPHIX_AVAILABLE = False

try:
    import stim  # noqa: F401
    STIM_AVAILABLE = True
except Exception:
    STIM_AVAILABLE = False


def main():
    # Lazy imports from our package
    from lspattern.canvas import RHGCanvas
    from lspattern.blocks import InitPlus, Memory, MeasureX
    from lspattern.compile import compile_canvas, pattern_to_stim

    logical = 0
    dx, dy = 5, 5
    rounds = 8

    canvas = RHGCanvas()

    # Stack the blocks
    canvas.append(InitPlus(logical=logical, dx=dx, dy=dy))
    canvas.append(Memory(logical=logical, rounds=rounds))
    canvas.append(MeasureX(logical=logical))

    if not GRAPHIX_AVAILABLE:
        print("[DRY] graphix_zx is not available. Built canvas with:")
        print(f"  - graph: {type(canvas.graph).__name__ if canvas.graph is not None else None}")
        print(f"  - logical boundaries: {canvas.logical_registry.boundary}")
        print(f"  - #X checks: {len(canvas.parity_accum.x_groups)}, #Z checks: {len(canvas.parity_accum.z_groups)}")
        print(f"  - xflow entries: {len(canvas.flow_accum.xflow)}")
        return 0

    # Compile into a Pattern
    pattern = canvas.compile()
    print("[OK] Compiled Pattern:", type(pattern).__name__)

    if STIM_AVAILABLE:
        try:
            circ = pattern_to_stim(pattern)
            print("[OK] Converted to stim.Circuit with", len(list(circ)), "operations")
        except Exception as e:
            print("[WARN] Failed to convert to stim:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
