from __future__ import annotations

"""
Smoke tests for T9 placement helpers:
- Blocks placed at patch (0,0,0) and (1,0,0) do not collide after offset.
- Vertical adjacency also works.
- Without offset, collision is detected.
Run: python examples/test_t9.py
"""

from lspattern.consts.consts import PIPEDIRECTION
from lspattern.tiling.base import ConnectedTiling
from lspattern.tiling.template import (
    RotatedPlanarPipetemplate,
    RotatedPlanarTemplate,
    block_offset_xy,
    offset_tiling,
    pipe_offset_xy,
)


def mk_block_template(d: int) -> RotatedPlanarTemplate:
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    t = RotatedPlanarTemplate(d=d, edgespec=edgespec)
    t.to_tiling()
    return t


def mk_pipe_template_horiz(d: int) -> RotatedPlanarPipetemplate:
    # Horizontal along X: TOP/BOTTOM must be 'O'
    spec = {"TOP": "O", "BOTTOM": "O", "LEFT": "X", "RIGHT": "Z"}
    p = RotatedPlanarPipetemplate(d=d, edgespec=spec)
    p.to_tiling()
    return p


def test_horizontal_ok() -> None:
    d = 3
    b0 = mk_block_template(d)
    b1 = mk_block_template(d)

    # Place at (0,0,0) and (1,0,0) using inner anchor
    dx0, dy0 = block_offset_xy(d, (0, 0, 0), anchor="inner")
    dx1, dy1 = block_offset_xy(d, (1, 0, 0), anchor="inner")
    t0 = offset_tiling(b0, dx0, dy0)
    t1 = offset_tiling(b1, dx1, dy1)

    # Pipe RIGHT from (0,0,0) -> (1,0,0)
    ptemp = mk_pipe_template_horiz(d)
    px, py = pipe_offset_xy(d, (0, 0, 0), (1, 0, 0), PIPEDIRECTION.RIGHT)
    tp = offset_tiling(ptemp, px, py)

    _ = ConnectedTiling([t0, t1, tp], check_collisions=True)


def test_vertical_ok() -> None:
    d = 3
    b0 = mk_block_template(d)
    b1 = mk_block_template(d)
    dx0, dy0 = block_offset_xy(d, (0, 0, 0), anchor="inner")
    dx1, dy1 = block_offset_xy(d, (0, 1, 0), anchor="inner")
    t0 = offset_tiling(b0, dx0, dy0)
    t1 = offset_tiling(b1, dx1, dy1)

    # Vertical pipe TOP: use vertical pipe template (LEFT/RIGHT 'O')
    spec = {"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"}
    ptemp = RotatedPlanarPipetemplate(d=d, edgespec=spec)
    ptemp.to_tiling()
    px, py = pipe_offset_xy(d, (0, 0, 0), (0, 1, 0), PIPEDIRECTION.TOP)
    tp = offset_tiling(ptemp, px, py)

    _ = ConnectedTiling([t0, t1, tp], check_collisions=True)


def test_collision_without_offset() -> None:
    d = 3
    b0 = mk_block_template(d)
    b1 = mk_block_template(d)
    try:
        _ = ConnectedTiling([b0, b1], check_collisions=True)
        raise AssertionError("expected collision was not raised")
    except ValueError:
        pass


if __name__ == "__main__":
    test_horizontal_ok()
    test_vertical_ok()
    test_collision_without_offset()
    print("test_t9: OK")
