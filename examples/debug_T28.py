#!/usr/bin/env python3
"""T28: X/Z 重複（交互残存）の検出テストと回避検証。

2キューブ + 1パイプ（水平/垂直）のレイアウトを2Dで再構成し、
全体の X/Z 集合の交差が空であることを検証する。
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


def collect_xy(cubes, pipes):
    from lspattern.tiling.template import cube_offset_xy, pipe_offset_xy, offset_tiling
    from lspattern.utils import get_direction

    union_x: set[tuple[int, int]] = set()
    union_z: set[tuple[int, int]] = set()

    for pos, b in cubes.items():
        dx, dy = cube_offset_xy(b.d, pos)
        t = offset_tiling(b.template, dx, dy)
        union_x |= set((int(x), int(y)) for (x, y) in t.x_coords)
        union_z |= set((int(x), int(y)) for (x, y) in t.z_coords)

    for (u, v), p in pipes.items():
        direction = get_direction(u, v)
        dx, dy = pipe_offset_xy(p.d, u, v, direction)
        t = offset_tiling(p.template, dx, dy)
        union_x |= set((int(x), int(y)) for (x, y) in t.x_coords)
        union_z |= set((int(x), int(y)) for (x, y) in t.z_coords)

    return union_x, union_z


def scenario_horizontal():
    from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.mytype import PatchCoordGlobal3D

    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_pipe_h = {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"}
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    cubes = {
        a: InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube).to_block(),
        b: InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube).to_block(),
    }
    pipes = {
        (a, b): InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_h).to_block(a, b)
    }
    return cubes, pipes


def scenario_vertical():
    from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.mytype import PatchCoordGlobal3D

    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_pipe_v = {"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"}
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 1, 0))
    cubes = {
        a: InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube).to_block(),
        b: InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube).to_block(),
    }
    pipes = {
        (a, b): InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_v).to_block(a, b)
    }
    return cubes, pipes


def main() -> None:
    for name, builder in [("horizontal", scenario_horizontal), ("vertical", scenario_vertical)]:
        cubes, pipes = builder()
        ux, uz = collect_xy(cubes, pipes)
        overlap = ux & uz
        print({name: {"x": len(ux), "z": len(uz), "overlap": len(overlap), "sample": sorted(overlap)[:8]}})
        assert not overlap, f"X/Z overlap detected in scenario={name}: {sorted(overlap)[:8]}"
    print("[T28] OK: no X/Z overlaps across cubes+pipes in both scenarios")


if __name__ == "__main__":
    main()

