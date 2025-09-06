#!/usr/bin/env python3
"""T32: RHGBlock.get_boundary_nodes refactor test

Checks grouping and face/depth selection using the shared utility.
"""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GRAPHIX = SRC / "graphix_zx"
for p in (ROOT, SRC, SRC_GRAPHIX):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def build_block_fixture():
    from lspattern.blocks.base import RHGBlock

    b = RHGBlock(d=3)
    # 2x2x2 cube of coordinates: x,y,z in {0,1}
    coords = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    b.coord2node = {c: i for i, c in enumerate(coords)}
    b.node2coord = {i: c for i, c in enumerate(coords)}
    # Assign roles on z=1 layer: alternate ancilla_x/ancilla_z
    role_map = {}
    for i, c in enumerate(coords):
        if c[2] == 1:
            role_map[i] = "ancilla_x" if (c[0] + c[1]) % 2 == 0 else "ancilla_z"
    b.node2role = role_map
    return b


def main() -> None:
    b = build_block_fixture()

    # z- at depth 0 should pick z == 0 and have only 'data' (no roles for z=0)
    r0 = b.get_boundary_nodes(face="z-", depth=[0])
    assert len(r0["data"]) == 4 and not r0["xcheck"] and not r0["zcheck"], r0

    # z+ at depth 0 should pick z == 1 and split by roles
    r1 = b.get_boundary_nodes(face="z+", depth=[0])
    assert len(r1["xcheck"]) + len(r1["zcheck"]) == 4 and not r1["data"], r1

    # x- across both depths (0 and 1) should cover 8 total points when unioned via API
    r2 = b.get_boundary_nodes(face="x-", depth=[0, 1])
    assert len(r2["data"]) + len(r2["xcheck"]) + len(r2["zcheck"]) == 8, r2

    # y+ at depth 0 should pick y == 1 with 4 points total
    r3 = b.get_boundary_nodes(face="y+", depth=[0])
    assert (
        len(r3["data"]) + len(r3["xcheck"]) + len(r3["zcheck"]) == 4
    ), r3

    print(
        "[T32] OK:",
        {
            "z-": r0,
            "z+": {k: len(v) for k, v in r1.items()},
            "x-": {k: len(v) for k, v in r2.items()},
            "y+": {k: len(v) for k, v in r3.items()},
        },
    )


if __name__ == "__main__":
    main()

