from __future__ import annotations

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton


def _summarize_block(block) -> tuple[int, int, int, int]:
    b = block.materialize()
    minus = b.get_boundary_nodes(face="z-")
    plus = b.get_boundary_nodes(face="z+")
    return len(b.in_ports), len(b.out_ports), len(minus["data"]), len(plus["data"])


def test_T41_representative_blocks_counts():
    # InitPlusCube d=3 spec=A
    cube_spec_A = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    init_cube = InitPlusCubeSkeleton(d=3, edgespec=cube_spec_A).to_block()
    i_in, i_out, i_zm, i_zp = _summarize_block(init_cube)
    # Init系は out に data の全インデックス、inはテンプレート依存（空許容）
    assert i_out == 9 and i_zm == 9 and i_zp == 9

    # MemoryCube d=3 spec=B
    cube_spec_B = {"LEFT": "Z", "RIGHT": "Z", "TOP": "X", "BOTTOM": "X"}
    mem_cube = MemoryCubeSkeleton(d=3, edgespec=cube_spec_B).to_block()
    m_in, m_out, m_zm, m_zp = _summarize_block(mem_cube)
    assert m_in == 9 and m_out == 9 and m_zm == 9 and m_zp == 9

    # InitPlusPipe d=3 spec=H1 RIGHT
    pipe_spec_H1 = {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"}
    init_pipe_h = InitPlusPipeSkeleton(d=3, edgespec=pipe_spec_H1).to_block(
        source=(0, 0, 0), sink=(1, 0, 0)
    )
    pi_in, pi_out, pi_zm, pi_zp = _summarize_block(init_pipe_h)
    assert pi_out == 3 and pi_zm == 3 and pi_zp == 3

    # MemoryPipe d=3 spec=V2 TOP
    pipe_spec_V2 = {"LEFT": "O", "RIGHT": "O", "TOP": "Z", "BOTTOM": "X"}
    mem_pipe_v = MemoryPipeSkeleton(d=3, edgespec=pipe_spec_V2).to_block(
        source=(0, 0, 0), sink=(0, 1, 0)
    )
    pm_in, pm_out, pm_zm, pm_zp = _summarize_block(mem_pipe_v)
    assert pm_in == 3 and pm_out == 3 and pm_zm == 3 and pm_zp == 3

