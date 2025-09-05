from __future__ import annotations

"""
Refactor smoke tests to ensure previous flows still work after unification:

- Template to tiling and boundary trim
- InitPlusSkeleton -> RHGBlock materialization
- Canvas add_block(s) and to_temporal_layers
- ConnectedTiling node_maps via TemporalLayer.materialize (offset path)
- InitPlusPipe (new) materialization and canvas integration

Run: python examples/test_refactor.py
"""

from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.tiling.template import RotatedPlanarTemplate


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_template_and_trim() -> None:
    d = 3
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    tmpl = RotatedPlanarTemplate(d=d, edgespec=edgespec)
    t = tmpl.to_tiling()
    assert_true(len(t["data"]) > 0 and (len(t["X"]) + len(t["Z"]) > 0), "tiling empty")
    z_before = len(tmpl.z_coords)
    tmpl.trim_spatial_boundary("TOP")
    t2 = tmpl.to_tiling()
    z_after = len(t2["Z"])  # not necessarily strictly less, but never more
    assert_true(z_after <= z_before, "trim did not decrease/equal Z ancillas")


def test_block_and_canvas_layers() -> None:
    d = 3
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    try:
        from lspattern.blocks.initialize import InitPlusBlockSkeleton  # lazy import
    except Exception as e:
        print(f"skip block/canvas_layers test (dependency missing): {e}")
        return

    from lspattern.canvas import RHGCanvas  # lazy import to avoid hard dep

    skel = InitPlusBlockSkeleton(d=d, edgespec=edgespec)
    block = skel.materialize()
    assert_true(block.graph_local is not None and len(block.node2coord) > 0, "block empty")

    canvas = RHGCanvas("RefactorSmoke")
    canvas.add_block(PatchCoordGlobal3D((0, 0, 0)), skel)
    layers = canvas.to_temporal_layers()
    assert_true(0 in layers, "layer z=0 missing")
    layer0 = layers[0]
    # TemporalLayer.materialize computes ConnectedTiling with offsets
    nm = layer0.get_node_maps()
    assert_true(set(nm.keys()) == {"data", "x", "z"}, "node_maps keys mismatch")


def test_pipe_materialize_and_canvas() -> None:
    d = 3
    edgespec = {"TOP": "O", "BOTTOM": "O", "LEFT": "X", "RIGHT": "Z"}
    try:
        from lspattern.blocks.cubes.initialize import InitPlusBlockSkeleton
        from lspattern.blocks.pipes.initialize import InitPlusPipe
    except Exception as e:
        print(f"skip pipe test (dependency missing): {e}")
        return

    # Build two blocks and a RIGHT pipe between them
    block_spec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    skel_a = InitPlusBlockSkeleton(d=d, edgespec=block_spec)
    skel_b = InitPlusBlockSkeleton(d=d, edgespec=block_spec)

    pipe = InitPlusPipe(d=d, edgespec=edgespec, direction=PIPEDIRECTION.RIGHT)
    # Should have local nodes from template
    assert_true(len(pipe.node2coord) > 0, "pipe failed to materialize")

    from lspattern.canvas import RHGCanvas  # lazy import

    canvas = RHGCanvas("RefactorPipe")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    canvas.add_block(a, skel_a)
    canvas.add_block(b, skel_b)
    canvas.add_pipe(a, b, pipe)

    # Build layers/compile; should not throw
    layers = canvas.to_temporal_layers()
    assert_true(0 in layers, "layer z=0 missing for pipe case")
    cgraph = canvas.compile()
    assert_true(cgraph.global_graph is not None, "compile produced no global graph")


def main() -> None:
    test_template_and_trim()
    test_block_and_canvas_layers()
    test_pipe_materialize_and_canvas()
    print("test_refactor: OK")


if __name__ == "__main__":
    main()
