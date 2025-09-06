from __future__ import annotations

# Minimal sanity check for T19: coord2id-based connection restriction

from lspattern.tiling.base import ConnectedTiling
from lspattern.tiling.template import (
    RotatedPlanarBlockTemplate,
    RotatedPlanarPipetemplate,
    offset_tiling,
    cube_offset_xy,
    pipe_offset_xy,
)
from lspattern.consts.consts import PIPEDIRECTION


def build_two_block_tilings(d: int = 3):
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    t1 = RotatedPlanarBlockTemplate(d=d, edgespec=edgespec)
    t2 = RotatedPlanarBlockTemplate(d=d, edgespec=edgespec)
    t1.to_tiling()
    t2.to_tiling()
    # positions
    pos_a = (0, 0, 0)
    pos_b = (1, 0, 0)
    dx1, dy1 = cube_offset_xy(d, pos_a, anchor="inner")
    dx2, dy2 = cube_offset_xy(d, pos_b, anchor="inner")
    return offset_tiling(t1, dx1, dy1), offset_tiling(t2, dx2, dy2), pos_a, pos_b


def run_no_pipe_then_with_pipe() -> None:
    d = 3
    t1, t2, pos_a, pos_b = build_two_block_tilings(d)

    # Case 1: no pipe
    ct1 = ConnectedTiling([t1, t2])  # legacy API
    gids1 = set((ct1.coord2id or {}).values())
    print("no-pipe: group_count=", len(gids1) if gids1 else 0)

    # Case 2: with a simple pipe bridging A->B
    ptemp = RotatedPlanarPipetemplate(d=d, edgespec={"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"})
    ptemp.to_tiling()
    dxp, dyp = pipe_offset_xy(d, pos_a, pos_b, PIPEDIRECTION.RIGHT)
    p = offset_tiling(ptemp, dxp, dyp)
    ct2 = ConnectedTiling([t1, t2], [p])
    gids2 = set((ct2.coord2id or {}).values())
    print("with-pipe: group_count=", len(gids2) if gids2 else 0)

    # Expected: without pipe => group_count==0 (legacy), with pipe => >=1 (new)
    assert len(gids2) >= 1, "expected grouped coord2id with pipe"
    print("T19 debug check: OK")


if __name__ == "__main__":
    run_no_pipe_then_with_pipe()
