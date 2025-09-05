from __future__ import annotations

"""
Smoke tests for T10:

- Build two blocks + one pipe via skeletons, materialize into a layer,
  and run plot_layer_tiling(show=False) without exceptions.
- Provide mixed d to confirm ValueError from connected-tiling reconstruction.

Run: python examples/test_t10.py
"""


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_layer_viz_no_error() -> None:
    from lspattern.mytype import PatchCoordGlobal3D

    try:
        from lspattern.blocks.cubes.initialize import InitPlusBlockSkeleton as _BlockSkel
    except Exception:
        from lspattern.blocks.cubes.initialize import InitPlusBlockSkeleton as _BlockSkel
    from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
    from lspattern.canvas import RHGCanvasSkeleton
    from lspattern.tiling.visualize import plot_layer_tiling

    d = 3
    block_spec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((1, 0, 0))
    skel_a = _BlockSkel(d=d, edgespec=block_spec)
    skel_b = _BlockSkel(d=d, edgespec=block_spec)
    p_skel = InitPlusPipeSkeleton(logical=0, d=d)

    canvas = RHGCanvasSkeleton("T10Smoke")
    canvas.add_block(a, skel_a)
    canvas.add_block(b, skel_b)
    canvas.add_pipe(a, b, p_skel)

    canvas2 = canvas.to_canvas()
    layers = canvas2.to_temporal_layers()
    assert_true(0 in layers, "layer z=0 missing")
    layer0 = layers[0]
    # ensure it plots without showing
    _ = plot_layer_tiling(layer0, anchor="inner", show=False)


def test_mixed_d_raises() -> None:
    from lspattern.mytype import PatchCoordGlobal3D

    try:
        from lspattern.blocks.cubes.initialize import InitPlusBlockSkeleton as _BlockSkel2
    except Exception:
        from lspattern.blocks.cubes.initialize import InitPlusBlockSkeleton as _BlockSkel2
    from lspattern.canvas import TemporalLayer

    d1 = 3
    d2 = 5
    spec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    skel_a = _BlockSkel2(d=d1, edgespec=spec)
    skel_b = _BlockSkel2(d=d2, edgespec=spec)

    layer = TemporalLayer(0)
    layer.add_block(PatchCoordGlobal3D((0, 0, 0)), skel_a)
    layer.add_block(PatchCoordGlobal3D((1, 0, 0)), skel_b)

    try:
        _ = layer.get_connected_tiling(anchor="inner")
        raise AssertionError("expected ValueError for mixed d not raised")
    except ValueError:
        pass


if __name__ == "__main__":
    test_layer_viz_no_error()
    test_mixed_d_raises()
    print("test_t10: OK")
