from __future__ import annotations

from typing import Literal

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D


def test_T43_compile_two_layers_smoke() -> None:
    edgespec: dict[str, Literal["X", "Z", "O"]] = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
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
