#!/usr/bin/env python3
"""T30: PipeTemplate の DIRECTION ごとの tiling と offset 可視化（d=3,5）

各 DIRECTION（RIGHT/LEFT/TOP/BOTTOM）について、
- テンプレートそのまま（raw）の2D
- 代表シームへのオフセット適用後（offset）の2D
を並べて可視化します。数値検証は行わず、図を保存して目視確認します。
"""

from __future__ import annotations

import pathlib
import sys
from typing import Tuple


# repo paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GRAPHIX = SRC / "graphix_zx"
for p in (ROOT, SRC, SRC_GRAPHIX):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def edge_spec_for(direction: str) -> dict[str, str]:
    d = direction.upper()
    if d in ("RIGHT", "LEFT"):
        # Horizontal pipe: TOP/BOTTOM open
        return {"TOP": "O", "BOTTOM": "O", "LEFT": "X", "RIGHT": "Z"}
    if d in ("TOP", "BOTTOM"):
        # Vertical pipe: LEFT/RIGHT open
        return {"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"}
    raise ValueError(f"invalid direction {direction}")


def sample_seam(direction: str) -> Tuple[tuple[int, int, int], tuple[int, int, int]]:
    d = direction.upper()
    if d == "RIGHT":
        return (0, 0, 0), (1, 0, 0)
    if d == "LEFT":
        return (1, 0, 0), (0, 0, 0)
    if d == "TOP":
        return (0, 0, 0), (0, 1, 0)
    if d == "BOTTOM":
        return (0, 1, 0), (0, 0, 0)
    raise ValueError(direction)


def plot_pipe(d: int, direction: str, save_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    from lspattern.tiling.template import (
        RotatedPlanarPipetemplate,
        pipe_offset_xy,
        offset_tiling,
    )
    from lspattern.consts.consts import PIPEDIRECTION

    spec = edge_spec_for(direction)
    tmpl = RotatedPlanarPipetemplate(d=d, edgespec=spec)
    tmpl.to_tiling()

    # Compute offset for a sample seam
    src, snk = sample_seam(direction)
    dir_enum = getattr(PIPEDIRECTION, direction.upper())
    dx, dy = pipe_offset_xy(d, src, snk, dir_enum)
    off = offset_tiling(tmpl, dx, dy)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def scatter(ax, t, title: str):
        data = list(t.data_coords or [])
        xs = list(t.x_coords or [])
        zs = list(t.z_coords or [])
        if data:
            ax.scatter([x for x, _ in data], [y for _, y in data], s=50, facecolors="white", edgecolors="black", label="data")
        if xs:
            ax.scatter([x for x, _ in xs], [y for _, y in xs], s=40, c="#2ecc71", edgecolors="#1e8449", label="X")
        if zs:
            ax.scatter([x for x, _ in zs], [y for _, y in zs], s=40, c="#3498db", edgecolors="#1f618d", label="Z")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(title)
        ax.legend(loc="best")

    scatter(axes[0], tmpl, f"raw (d={d}, dir={direction})")
    scatter(axes[1], off, f"offset dx={dx}, dy={dy}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"Saved: {save_path}")


def main() -> None:
    out_dir = ROOT / "examples"
    for d in (3, 5):
        for direction in ("RIGHT", "LEFT", "TOP", "BOTTOM"):
            out_png = out_dir / f"fig_T30_pipe_d{d}_{direction}.png"
            plot_pipe(d, direction, out_png)


if __name__ == "__main__":
    main()
