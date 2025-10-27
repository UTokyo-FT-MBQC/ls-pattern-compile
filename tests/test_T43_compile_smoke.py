from __future__ import annotations

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D


def test_T43_compile_two_layers_smoke() -> None:
    edgespec: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    d = 3
    sk = RHGCanvasSkeleton("T43 compiled viz")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 0, 1))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec))
    sk.add_cube(b, MemoryCubeSkeleton(d=d, edgespec=edgespec))
    sk.add_pipe(a, b, MemoryPipeSkeleton(d=d, edgespec=edgespec))

    cgraph = sk.to_canvas().compile()
    global_nodes = len(getattr(cgraph.global_graph, "physical_nodes", []) or [])
    assert global_nodes > 0
    assert list(getattr(cgraph, "zlist", [])) == [0, 1]
