# %%
"""
T26: パイプのinnerアンカー適用後の可視化

水平/垂直パイプの layer0 を Matplotlib/Plotly で可視化し、負座標が出ていないことを確認します。
"""

# %%
import pathlib

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers.plotly_temporallayer import visualize_temporal_layer_plotly
from lspattern.visualizers.temporallayer import visualize_temporal_layer

MeasureXCubeSkeleton = InitPlusCubeSkeleton


def build_horizontal():  # type: ignore[no-untyped-def]
    """Build horizontal pipe configuration."""
    d = 3
    edgespec_cube1: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec_cube2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec_pipe_h: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    sk = RHGCanvasSkeleton("T26 horiz")
    a = PatchCoordGlobal3D((0, 0, 1))
    b = PatchCoordGlobal3D((1, 0, 1))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube1))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube2))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_h))
    # sk.add_cube(a, MeasureXCubeSkeleton(d=d, edgespec=edgespec_cube1))
    # sk.add_cube(a, MeasureZCubeSkeleton(d=d, edgespec=edgespec_cube1))
    canvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    return layers[0]


def build_vertical():  # type: ignore[no-untyped-def]
    """Build vertical pipe configuration."""
    d = 3
    edgespec_cube1: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec_cube2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.O}
    edgespec_pipe_v: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.O}
    sk = RHGCanvasSkeleton("T26 vert")
    a = PatchCoordGlobal3D((0, 0, 0))
    b = PatchCoordGlobal3D((0, 1, 0))
    sk.add_cube(a, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube1))
    sk.add_cube(b, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube2))
    sk.add_pipe(a, b, InitPlusPipeSkeleton(d=d, edgespec=edgespec_pipe_v))
    canvas = sk.to_canvas()
    layers = canvas.to_temporal_layers()
    return layers[0]


# %%
# Matplotlib 可視化(水平)
layer_h = build_horizontal()  # type: ignore[no-untyped-call]
print(len(layer_h.cubes_), "cubes")
print(len(layer_h.node2coord))
print(layer_h.node2coord)
# assert no duplicate values in the layer_h.node2coord
assert len(layer_h.node2coord) == len(set(layer_h.node2coord.values()))
out_png = pathlib.Path("./").resolve().with_name("fig_T26_horiz.png")
visualize_temporal_layer(layer_h, save_path=str(out_png), show=False, show_axes=True, show_grid=True)
print("Saved:", out_png)

# Plotly 可視化(水平)
fig1 = visualize_temporal_layer_plotly(layer_h, aspectmode="data", reverse_axes=True, show_axes=True, show_grid=True)
fig1.show()

# %%
# Matplotlib 可視化(垂直)
layer_v = build_vertical()  # type: ignore[no-untyped-call]
out_png2 = pathlib.Path(".").resolve().with_name("fig_T26_vert.png")
visualize_temporal_layer(layer_v, save_path=str(out_png2), show=False, show_axes=True, show_grid=True)
print("Saved:", out_png2)

# Plotly 可視化(垂直)
fig2 = visualize_temporal_layer_plotly(layer_v, aspectmode="data", reverse_axes=False, show_axes=True, show_grid=True)
fig2.show()

# %%
