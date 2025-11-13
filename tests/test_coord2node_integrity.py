"""Tests for ensuring compiled coord2node mappings are consistent."""

from lspattern.blocks.cubes.initialize import (
    InitPlusCubeThinLayerSkeleton,
    InitZeroCubeThinLayerSkeleton,
)
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.utils import to_edgespec


def _build_cnot_canvas() -> RHGCanvasSkeleton:
    d = 3
    edgespec = to_edgespec("ZZXX")

    canvas = RHGCanvasSkeleton("CNOT")
    blocks = [
        (PatchCoordGlobal3D((0, 0, 0)), InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 1, 0)), InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 1, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZOX"))),
        (PatchCoordGlobal3D((0, 1, 2)), MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZXO"))),
        (PatchCoordGlobal3D((0, 0, 3)), MeasureXSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 1, 3)), MeasureXSkeleton(d=d, edgespec=edgespec)),
    ]

    pipes = [
        (
            PatchCoordGlobal3D((0, 0, 2)),
            PatchCoordGlobal3D((0, 1, 2)),
            InitPlusPipeSkeleton(d=d, edgespec=to_edgespec("ZZOO")),
        ),
        (
            PatchCoordGlobal3D((0, 0, 3)),
            PatchCoordGlobal3D((0, 1, 3)),
            MeasureXPipeSkeleton(d=d, edgespec=to_edgespec("OOOO")),
        ),
    ]

    for block in blocks:
        canvas.add_cube(*block)
    for pipe in pipes:
        canvas.add_pipe(*pipe)

    return canvas


def test_compiled_canvas_coord2node_has_unique_incremental_ids() -> None:
    canvas = _build_cnot_canvas().to_canvas()
    compiled = canvas.compile()

    coord_values = [int(node) for node in compiled.coord2node.values()]

    assert len(coord_values) == len(set(coord_values))
    assert compiled.global_graph is not None
    assert set(coord_values) == {int(n) for n in compiled.global_graph.physical_nodes}
