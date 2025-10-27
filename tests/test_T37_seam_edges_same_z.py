from __future__ import annotations

from typing import Any

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D


def _cross_region_edge_count(layer: Any) -> int:
    # Count edges across cube↔pipe XY regions using layer's compiled artifacts
    cube_xy: set[tuple[int, int]] = set()
    pipe_xy: set[tuple[int, int]] = set()
    for blk in layer.cubes_.values():
        t = blk.template
        for L in (t.data_coords, t.x_coords, t.z_coords):
            for x, y in L or []:
                cube_xy.add((int(x), int(y)))
    for pipe in layer.pipes_.values():
        t = pipe.template
        for L in (t.data_coords, t.x_coords, t.z_coords):
            for x, y in L or []:
                pipe_xy.add((int(x), int(y)))

    g = layer.local_graph
    n2c = layer.node2coord
    cnt = 0
    for u, v in getattr(g, "physical_edges", []) or []:
        cu = n2c.get(u)
        cv = n2c.get(v)
        if not cu or not cv:
            continue
        xu, yu = int(cu[0]), int(cu[1])
        xv, yv = int(cv[0]), int(cv[1])
        if ((xu, yu) in cube_xy and (xv, yv) in pipe_xy) or (
            (xv, yv) in cube_xy and (xu, yu) in pipe_xy
        ):
            cnt += 1
    return cnt


def test_T37_horizontal_seam_edges_present() -> None:
    d = 3
    edgespec_cube: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec_pipe_h: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.Z,
        BoundarySide.TOP: EdgeSpecValue.O,
        BoundarySide.BOTTOM: EdgeSpecValue.O,
    }

    sk = RHGCanvasSkeleton("T37 horiz")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_h))
    layer = sk.to_canvas().to_temporal_layers()[0]
    assert _cross_region_edge_count(layer) > 0


def test_T37_vertical_seam_edges_present() -> None:
    d = 3
    edgespec_cube: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec_pipe_v: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.TOP: EdgeSpecValue.X,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.O,
    }

    sk = RHGCanvasSkeleton("T37 vert")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 1, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_v))
    layer = sk.to_canvas().to_temporal_layers()[0]
    assert _cross_region_edge_count(layer) > 0
