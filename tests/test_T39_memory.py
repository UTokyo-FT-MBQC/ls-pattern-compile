import os
import sys
from pathlib import Path

# Ensure vendored graphix_zx (in src/) is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SRC_GZX = SRC / "graphix_zx"
for p in (SRC, SRC_GZX):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton


def test_memory_cube_ports_and_boundaries():
    spec = {"TOP": "Z", "BOTTOM": "Z", "LEFT": "X", "RIGHT": "X"}
    block = MemoryCubeSkeleton(d=3, edgespec=spec).to_block().materialize()

    # in/out are non-empty and equal to number of data in template (d*d)
    assert len(block.in_ports) == 9
    assert len(block.out_ports) == 9

    minus = block.get_boundary_nodes(face="z-")
    plus = block.get_boundary_nodes(face="z+")
    assert len(minus["data"]) == 9
    assert len(plus["data"]) == 9


def test_memory_pipe_ports_and_boundaries():
    skel = MemoryPipeSkeleton(d=5, edgespec={"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "Z"})
    block = skel.to_block(source=(0, 0, 0), sink=(1, 0, 0)).materialize()

    # in/out are non-empty and equal to number of data in pipe template (d)
    assert len(block.in_ports) == 5
    assert len(block.out_ports) == 5

    minus = block.get_boundary_nodes(face="z-")
    plus = block.get_boundary_nodes(face="z+")
    assert len(minus["data"]) == 5
    assert len(plus["data"]) == 5
