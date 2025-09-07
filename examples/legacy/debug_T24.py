#!/usr/bin/env python3
"""T24 debug: materialize RHG blocks (cube/pipe) and verify coords/shift.

This script constructs a couple of skeletons, converts them to blocks via
to_block(), calls block.materialize(), and then checks that the underlying
template has populated data/X/Z coordinates. It also exercises shift_coords
to ensure the coordinates move consistently afterwards.
"""

from __future__ import annotations

import pathlib
import sys
from pprint import pprint


def _ensure_paths() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    gzx = src / "graphix_zx"
    for p in (src, gzx):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_ensure_paths()

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.mytype import PatchCoordGlobal3D


def summarize_block(block, label: str) -> dict:
    t = block.template
    return {
        "label": label,
        "d": int(block.d),
        "data": len(t.data_coords or []),
        "X": len(t.x_coords or []),
        "Z": len(t.z_coords or []),
        "sample": {
            "data": (t.data_coords or [None])[:5],
            "X": (t.x_coords or [None])[:5],
            "Z": (t.z_coords or [None])[:5],
        },
    }


def main() -> None:
    d = 3
    # Edge specs for cubes
    edges_a = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edges_b = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}

    # Build cube skeletons -> blocks -> materialize
    s1 = InitPlusCubeSkeleton(d=d, edgespec=edges_a)
    s2 = InitPlusCubeSkeleton(d=d, edgespec=edges_b)

    b1 = s1.to_block().materialize()
    b2 = s2.to_block().materialize()

    # Pipe skeleton between them (spatial X+)
    psk = InitPlusPipeSkeleton(d=d, edgespec=None)
    src = PatchCoordGlobal3D((0, 0, 0))
    snk = PatchCoordGlobal3D((1, 0, 0))
    pipe = psk.to_block(src, snk).materialize()

    print("[T24] After materialize() — coordinate counts/samples")
    pprint(summarize_block(b1, "cube-1"))
    pprint(summarize_block(b2, "cube-2"))
    pprint(summarize_block(pipe, "pipe"))

    assert b1.template.data_coords, "cube-1 has no data coords"
    assert pipe.template.x_coords or pipe.template.z_coords, "pipe has no ancilla coords"

    # Exercise shift_coords
    off = (2 * d, 0, 0)
    pre_first = b1.template.data_coords[0]
    b1.shift_coords(off)
    post_first = b1.template.data_coords[0]
    dx = post_first[0] - pre_first[0]
    dy = post_first[1] - pre_first[1]
    assert dx == off[0] and dy == off[1], "cube shift_coords did not apply expected XY offset"

    # Pipe shift by one patch to the right
    pipe_pre = (pipe.template.data_coords or pipe.template.x_coords or pipe.template.z_coords)[0]
    pipe.shift_coords((1, 0, 0))
    pipe_post = (pipe.template.data_coords or pipe.template.x_coords or pipe.template.z_coords)[0]
    assert pipe_post != pipe_pre, "pipe coords did not change after shift"

    print("[T24] shift_coords OK (cube+pipe)")


if __name__ == "__main__":
    main()

