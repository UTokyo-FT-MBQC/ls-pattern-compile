from __future__ import annotations

from lspattern.blocks.cubes.initialize import InitPlusCubeThinLayerSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitZeroPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureZPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.canvas._canvas_impl import CompiledRHGCanvas
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D, PhysCoordLocal2D


def _build_compiled_canvas() -> CompiledRHGCanvas:
    """Construct the XX merge/split canvas used by the example script."""
    d = 3
    canvas = RHGCanvasSkeleton("merge-split-xx-test")

    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.Z,
        BoundarySide.RIGHT: EdgeSpecValue.Z,
        BoundarySide.TOP: EdgeSpecValue.X,
        BoundarySide.BOTTOM: EdgeSpecValue.X,
    }
    edgespec_left_open = edgespec | {BoundarySide.RIGHT: EdgeSpecValue.O}
    edgespec_right_open = edgespec | {BoundarySide.LEFT: EdgeSpecValue.O}
    edgespec_trimmed = {
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.O,
        BoundarySide.TOP: EdgeSpecValue.X,
        BoundarySide.BOTTOM: EdgeSpecValue.X,
    }
    edgespec_measure_trimmed = {
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.O,
        BoundarySide.TOP: EdgeSpecValue.O,
        BoundarySide.BOTTOM: EdgeSpecValue.O,
    }

    block_specs = {
        PatchCoordGlobal3D((0, 0, 0)): InitPlusCubeThinLayerSkeleton(
            d=d, edgespec=edgespec
        ),
        PatchCoordGlobal3D((1, 0, 0)): InitPlusCubeThinLayerSkeleton(
            d=d, edgespec=edgespec
        ),
        PatchCoordGlobal3D((0, 0, 1)): MemoryCubeSkeleton(d=d, edgespec=edgespec),
        PatchCoordGlobal3D((1, 0, 1)): MemoryCubeSkeleton(d=d, edgespec=edgespec),
        PatchCoordGlobal3D((0, 0, 2)): MemoryCubeSkeleton(
            d=d, edgespec=edgespec_left_open
        ),
        PatchCoordGlobal3D((1, 0, 2)): MemoryCubeSkeleton(
            d=d, edgespec=edgespec_right_open
        ),
        PatchCoordGlobal3D((0, 0, 3)): MemoryCubeSkeleton(d=d, edgespec=edgespec),
        PatchCoordGlobal3D((1, 0, 3)): MemoryCubeSkeleton(d=d, edgespec=edgespec),
        PatchCoordGlobal3D((0, 0, 4)): MeasureXSkeleton(d=d, edgespec=edgespec),
        PatchCoordGlobal3D((1, 0, 4)): MeasureXSkeleton(d=d, edgespec=edgespec),
    }
    pipe_specs = {
        (
            PatchCoordGlobal3D((0, 0, 2)),
            PatchCoordGlobal3D((1, 0, 2)),
        ): InitZeroPipeSkeleton(d=d, edgespec=edgespec_trimmed),
        (
            PatchCoordGlobal3D((0, 0, 3)),
            PatchCoordGlobal3D((1, 0, 3)),
        ): MeasureZPipeSkeleton(d=d, edgespec=edgespec_measure_trimmed),
    }

    for coord, block in block_specs.items():
        canvas.add_cube(coord, block)
    for (src, dst), pipe in pipe_specs.items():
        canvas.add_pipe(src, dst, pipe)

    return canvas.to_canvas().compile()


def test_init_zero_pipe_first_x_seam_is_paired() -> None:
    compiled = _build_compiled_canvas()
    seam_coord = PhysCoordLocal2D((5, 3))

    seam_groups = [
        {int(node) for node in group}
        for group in compiled.parity.checks[seam_coord].values()
    ]

    # The first detector should now be the paired {363, 375}.
    assert {363, 375} in seam_groups
    assert {363} not in seam_groups


def test_z_side_seam_remains_multi_layer_chain() -> None:
    compiled = _build_compiled_canvas()
    z_coord = PhysCoordLocal2D((5, 1))

    seam_groups = [
        {int(node) for node in group}
        for group in compiled.parity.checks[z_coord].values()
    ]

    # Ensure Z side still reports its chained detectors involving node 368.
    assert any(368 in group for group in seam_groups)
