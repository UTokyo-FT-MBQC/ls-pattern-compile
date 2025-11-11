# Visualizer for tiling

"""
Tiling visualization for different edge specifications.

This script demonstrates various tiling patterns for rotated planar cubes
and pipes with different boundary conditions.
"""

import matplotlib.pyplot as plt
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.tiling.template import RotatedPlanarCubeTemplate, RotatedPlanarPipetemplate

# %%
# Case: Rotated Planar Tiling (XXZZ)
print("=" * 60)
print("Case: Rotated Planar Tiling (XXZZ)")
print("=" * 60)

d = 3
edgespec_xxzz: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.X,
    BoundarySide.RIGHT: EdgeSpecValue.X,
    BoundarySide.TOP: EdgeSpecValue.Z,
    BoundarySide.BOTTOM: EdgeSpecValue.Z,
}

tiling_xxzz = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec_xxzz)
tiling_xxzz.to_tiling()

print(f"\nData qubits (n={len(tiling_xxzz.data_coords)}):")
print(tiling_xxzz.data_coords)
print(f"\nX ancilla qubits (n={len(tiling_xxzz.x_coords)}):")
print(tiling_xxzz.x_coords)
print(f"\nZ ancilla qubits (n={len(tiling_xxzz.z_coords)}):")
print(tiling_xxzz.z_coords)

fig, ax = plt.subplots(figsize=(8, 8))
tiling_xxzz.visualize_tiling(ax=ax, show=False, title_suffix="XXZZ")
plt.tight_layout()
plt.savefig("tiling_xxzz.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: tiling_xxzz.png")
plt.show()

# %%
# Case: Rotated Planar Tiling (ZZXX)
print("\n" + "=" * 60)
print("Case: Rotated Planar Tiling (ZZXX)")
print("=" * 60)

edgespec_zzxx: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}

tiling_zzxx = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec_zzxx)
tiling_zzxx.to_tiling()

print(f"\nData qubits (n={len(tiling_zzxx.data_coords)}):")
print(tiling_zzxx.data_coords)
print(f"\nX ancilla qubits (n={len(tiling_zzxx.x_coords)}):")
print(tiling_zzxx.x_coords)
print(f"\nZ ancilla qubits (n={len(tiling_zzxx.z_coords)}):")
print(tiling_zzxx.z_coords)

fig, ax = plt.subplots(figsize=(8, 8))
tiling_zzxx.visualize_tiling(ax=ax, show=False, title_suffix="ZZXX")
plt.tight_layout()
plt.savefig("tiling_zzxx.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: tiling_zzxx.png")
plt.show()

# %%
# Case: Rotated Planar Tiling (OOOO)
print("\n" + "=" * 60)
print("Case: Rotated Planar Tiling (OOOO)")
print("=" * 60)

edgespec_oooo: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.O,
    BoundarySide.BOTTOM: EdgeSpecValue.O,
}

tiling_oooo = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec_oooo)
tiling_oooo.to_tiling()

print(f"\nData qubits (n={len(tiling_oooo.data_coords)}):")
print(tiling_oooo.data_coords)
print(f"\nX ancilla qubits (n={len(tiling_oooo.x_coords)}):")
print(tiling_oooo.x_coords)
print(f"\nZ ancilla qubits (n={len(tiling_oooo.z_coords)}):")
print(tiling_oooo.z_coords)

fig, ax = plt.subplots(figsize=(8, 8))
tiling_oooo.visualize_tiling(ax=ax, show=False, title_suffix="OOOO")
plt.tight_layout()
plt.savefig("tiling_oooo.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: tiling_oooo.png")
plt.show()

# %%
# Case: Rotated Planar Tiling (XXXX) (<- see if the corners are trimmed)
print("\n" + "=" * 60)
print("Case: Rotated Planar Tiling (ZXZX) - Check corner trimming")
print("=" * 60)

edgespec_zxzx: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.X,
    BoundarySide.TOP: EdgeSpecValue.Z,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}

tiling_zxzx = RotatedPlanarCubeTemplate(d=d, edgespec=edgespec_zxzx)
tiling_zxzx.to_tiling()

print(f"\nData qubits (n={len(tiling_zxzx.data_coords)}):")
print(tiling_zxzx.data_coords)
print(f"Expected: {d*d - 4} (corners should be trimmed)")
print(f"\nX ancilla qubits (n={len(tiling_zxzx.x_coords)}):")
print(tiling_zxzx.x_coords)
print(f"\nZ ancilla qubits (n={len(tiling_zxzx.z_coords)}):")
print(tiling_zxzx.z_coords)

fig, ax = plt.subplots(figsize=(8, 8))
tiling_zxzx.visualize_tiling(ax=ax, show=False, title_suffix="ZXZX (corners trimmed)")
plt.tight_layout()
plt.savefig("tiling_zxzx.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: tiling_zxzx.png")
plt.show()

# %%
# Case: Pipe tiling (OOXX) (Vertical)
print("\n" + "=" * 60)
print("Case: Pipe tiling (OOXX) - Vertical")
print("=" * 60)

edgespec_ooxx: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}

tiling_pipe_ooxx = RotatedPlanarPipetemplate(d=d, edgespec=edgespec_ooxx)
tiling_pipe_ooxx.to_tiling()

print(f"\nData qubits (n={len(tiling_pipe_ooxx.data_coords)}):")
print(tiling_pipe_ooxx.data_coords)
print(f"\nX ancilla qubits (n={len(tiling_pipe_ooxx.x_coords)}):")
print(tiling_pipe_ooxx.x_coords)
print(f"\nZ ancilla qubits (n={len(tiling_pipe_ooxx.z_coords)}):")
print(tiling_pipe_ooxx.z_coords)

fig, ax = plt.subplots(figsize=(8, 8))
tiling_pipe_ooxx.visualize_tiling(ax=ax, show=False, title_suffix="Pipe OOXX (Vertical)")
plt.tight_layout()
plt.savefig("tiling_pipe_ooxx_vertical.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: tiling_pipe_ooxx_vertical.png")
plt.show()

# %%
# Case: Pipe tiling (ZZOO) (Horizontal)
print("\n" + "=" * 60)
print("Case: Pipe tiling (ZZOO) - Horizontal")
print("=" * 60)

edgespec_zzoo: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
    BoundarySide.TOP: EdgeSpecValue.O,
    BoundarySide.BOTTOM: EdgeSpecValue.O,
}

tiling_pipe_zzoo = RotatedPlanarPipetemplate(d=d, edgespec=edgespec_zzoo)
tiling_pipe_zzoo.to_tiling()

print(f"\nData qubits (n={len(tiling_pipe_zzoo.data_coords)}):")
print(tiling_pipe_zzoo.data_coords)
print(f"\nX ancilla qubits (n={len(tiling_pipe_zzoo.x_coords)}):")
print(tiling_pipe_zzoo.x_coords)
print(f"\nZ ancilla qubits (n={len(tiling_pipe_zzoo.z_coords)}):")
print(tiling_pipe_zzoo.z_coords)

fig, ax = plt.subplots(figsize=(8, 8))
tiling_pipe_zzoo.visualize_tiling(ax=ax, show=False, title_suffix="Pipe ZZOO (Horizontal)")
plt.tight_layout()
plt.savefig("tiling_pipe_zzoo_horizontal.png", dpi=150, bbox_inches="tight")
print("\nFigure saved: tiling_pipe_zzoo_horizontal.png")
plt.show()
