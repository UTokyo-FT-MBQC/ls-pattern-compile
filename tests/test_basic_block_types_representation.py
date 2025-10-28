from __future__ import annotations

from typing import Any, Literal

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.mytype import PatchCoordGlobal3D


def _summarize_block(block: Any) -> tuple[int, int, int, int]:
    b = block.materialize()
    minus = b.get_boundary_nodes(face="z-")
    plus = b.get_boundary_nodes(face="z+")
    return len(b.in_ports), len(b.out_ports), len(minus["data"]), len(plus["data"])


def test_representative_blocks_counts() -> None:
    # InitPlusCube d=3 spec=A
    cube_spec_a: dict[str, Literal["X", "Z", "O"]] = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    init_cube = InitPlusCubeSkeleton(d=3, edgespec=cube_spec_a).to_block()
    _, i_out, i_zm, i_zp = _summarize_block(init_cube)
    # For Init-type blocks, out contains all data indices, in depends on template (empty allowed)
    assert i_out == 9
    assert i_zm == 9
    assert i_zp == 9

    # MemoryCube d=3 spec=B
    cube_spec_b: dict[str, Literal["X", "Z", "O"]] = {"LEFT": "Z", "RIGHT": "Z", "TOP": "X", "BOTTOM": "X"}
    mem_cube = MemoryCubeSkeleton(d=3, edgespec=cube_spec_b).to_block()
    m_in, m_out, m_zm, m_zp = _summarize_block(mem_cube)
    assert m_in == 9
    assert m_out == 9
    assert m_zm == 9
    assert m_zp == 9

    # InitPlusPipe d=3 spec=H1 RIGHT
    pipe_spec_h1: dict[str, Literal["X", "Z", "O"]] = {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"}
    init_pipe_h = InitPlusPipeSkeleton(d=3, edgespec=pipe_spec_h1).to_block(
        source=PatchCoordGlobal3D((0, 0, 0)), sink=PatchCoordGlobal3D((1, 0, 0))
    )
    _, pi_out, pi_zm, pi_zp = _summarize_block(init_pipe_h)
    assert pi_out == 3
    assert pi_zm == 3
    assert pi_zp == 3

    # MemoryPipe d=3 spec=V2 TOP
    pipe_spec_b2: dict[str, Literal["X", "Z", "O"]] = {"LEFT": "O", "RIGHT": "O", "TOP": "Z", "BOTTOM": "X"}
    mem_pipe_v = MemoryPipeSkeleton(d=3, edgespec=pipe_spec_b2).to_block(
        source=PatchCoordGlobal3D((0, 0, 0)), sink=PatchCoordGlobal3D((0, 1, 0))
    )
    pm_in, pm_out, pm_zm, pm_zp = _summarize_block(mem_pipe_v)
    assert pm_in == 3
    assert pm_out == 3
    assert pm_zm == 3
    assert pm_zp == 3
