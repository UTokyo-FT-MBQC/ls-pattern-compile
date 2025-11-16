from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from lspattern.consts import BoundarySide, CoordinateSystem, EdgeSpecValue
from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import (
    QubitIndexLocal,
    SpatialEdgeSpec,
    TilingCoord2D,
)
from lspattern.tiling.base import Tiling
from lspattern.utils import sort_xy

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def calculate_qindex_base_cube(patch_coord: tuple[int, int], d: int) -> int:
    """Calculate the starting q_index for data qubits in a cube patch.

    Each patch at coordinate (px, py) gets a unique range of q_indices.
    This ensures that patches at the same coordinate always get the same q_indices,
    enabling consistent mapping across different temporal layers.

    Parameters
    ----------
    patch_coord : tuple[int, int]
        The (px, py) coordinate of the patch in the global tiling
    d : int
        The distance parameter, determines number of data qubits per patch

    Returns
    -------
    int
        The starting q_index for this patch's data qubits
    """
    px, py = patch_coord

    # Use a large enough stride to ensure no overlap between different grid positions
    # Each grid position can have:
    # - 1 cube with d*d data qubits
    # - up to 2 pipes (horizontal + vertical) with d data qubits each
    max_qubits_per_grid_position = d * d + 2 * d

    # Create a unique base index for this grid coordinate
    # Use a grid layout with sufficient spacing
    patch_index = py * 1000 + px  # 1000 should be enough for reasonable grid sizes
    return patch_index * max_qubits_per_grid_position


def calculate_qindex_base_pipe(patch_coord: tuple[int, int], sink_patch: tuple[int, int], d: int) -> int:
    """Calculate the starting q_index for data qubits in a pipe patch.

    Pipes use boundary-based indexing to ensure that pipes sharing the same
    physical data qubits get the same qindex.

    Parameters
    ----------
    patch_coord : tuple[int, int]
        The (px, py) coordinate of the source patch in the global tiling
    sink_patch : tuple[int, int]
        The (px, py) coordinate of the sink patch (required for pipes)
    d : int
        The distance parameter, determines number of data qubits per patch

    Returns
    -------
    int
        The starting q_index for this pipe's data qubits
    """
    return _calculate_pipe_boundary_qindex(patch_coord, sink_patch, d)


def _calculate_pipe_boundary_qindex(source_patch: tuple[int, int], sink_patch: tuple[int, int], d: int) -> int:
    """Calculate qindex for a pipe based on the boundary between source and sink patches.

    This ensures that pipes sharing the same physical data qubits get the same qindex.

    Parameters
    ----------
    source_patch : tuple[int, int]
        Source patch coordinate (px, py)
    sink_patch : tuple[int, int]
        Sink patch coordinate (px, py)
    d : int
        Distance parameter

    Returns
    -------
    int
        The base qindex for the pipe's data qubits
    """
    spx, spy = source_patch
    dpx, dpy = sink_patch

    # Determine pipe direction and boundary position
    dx = dpx - spx
    dy = dpy - spy

    # Validate that patches are adjacent
    if abs(dx) + abs(dy) != 1:
        msg = f"Source {source_patch} and sink {sink_patch} must be adjacent (Manhattan distance 1)"
        raise ValueError(msg)

    # Calculate boundary position - use the lower coordinate value for consistency
    if dx != 0:  # Horizontal pipe
        boundary_x = min(spx, dpx)
        boundary_y = spy  # Same Y coordinate
        boundary_pos = (boundary_x, boundary_y)
        direction = "horizontal"
    else:  # Vertical pipe (dy != 0)
        boundary_x = spx  # Same X coordinate
        boundary_y = min(spy, dpy)
        boundary_pos = (boundary_x, boundary_y)
        direction = "vertical"

    # Calculate qindex based on boundary position
    px, py = boundary_pos
    max_qubits_per_grid_position = d * d + 2 * d
    patch_index = py * 1000 + px
    base_index = patch_index * max_qubits_per_grid_position

    # Add offset for pipe type
    cube_range = d * d
    if direction == "horizontal":
        return base_index + cube_range
    # vertical
    return base_index + cube_range + d


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(kw_only=True)
class ScalableTemplate(Tiling):
    d: int
    edgespec: SpatialEdgeSpec  # e.g., {"top":"X","bottom":"Z",...}

    data_coords: list[TilingCoord2D] = field(default_factory=list)
    data_indices: list[QubitIndexLocal] = field(default_factory=list)
    x_coords: list[TilingCoord2D] = field(default_factory=list)
    z_coords: list[TilingCoord2D] = field(default_factory=list)

    trimmed: bool = False

    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:
        raise NotImplementedError

    def _spec(self, side: BoundarySide) -> EdgeSpecValue:
        """Return standardized spec value (EdgeSpecValue enum).

        Accepts BoundarySide enum. Defaults to EdgeSpecValue.O.
        """
        v = None
        if isinstance(self.edgespec, dict):
            # Try enum key first
            v = self.edgespec.get(side)
            # Fall back to string key for backward compatibility
            if v is None:
                v = self.edgespec.get(side.value)  # type: ignore[call-overload]
        if v is None:
            v = EdgeSpecValue.O
        # Convert string to enum if needed
        elif isinstance(v, str):
            v = EdgeSpecValue(v.upper())
        return v

    def get_data_indices_cube(
        self,
        patch_coord: tuple[int, int] | None = None,
    ) -> dict[TilingCoord2D, QubitIndexLocal]:
        """Get data qubit indices for a cube patch.

        Parameters
        ----------
        patch_coord : tuple[int, int] | None, default None
            The (px, py) coordinate of the patch in the global tiling.
            If None, generates default indices starting from 0.

        Returns
        -------
        dict[TilingCoord2D, QubitIndexLocal]
            Mapping from 2D tiling coordinates to local qubit indices
        """
        coord_set = {(coord[0], coord[1]) for coord in self.data_coords}
        sorted_coords = sort_xy(coord_set)

        # If data_indices have been explicitly set, use them (legacy compatibility)
        if self.data_indices and len(self.data_indices) == len(sorted_coords):
            return {TilingCoord2D(coor): self.data_indices[i] for i, coor in enumerate(sorted_coords)}

        # If patch coordinate is provided, use it to calculate consistent q_indices
        if patch_coord is not None:
            base_qindex = calculate_qindex_base_cube(patch_coord, self.d)
            return {TilingCoord2D(coor): QubitIndexLocal(base_qindex + i) for i, coor in enumerate(sorted_coords)}

        # Otherwise, generate default indices starting from 0 (fallback for backward compatibility)
        return {TilingCoord2D(coor): QubitIndexLocal(i) for i, coor in enumerate(sorted_coords)}

    def get_data_indices_pipe(
        self,
        patch_coord: tuple[int, int],
        sink_patch: tuple[int, int],
    ) -> dict[TilingCoord2D, QubitIndexLocal]:
        """Get data qubit indices for a pipe patch.

        Parameters
        ----------
        patch_coord : tuple[int, int]
            The (px, py) coordinate of the source patch in the global tiling
        sink_patch : tuple[int, int]
            The (px, py) coordinate of the sink patch (required for pipes)

        Returns
        -------
        dict[TilingCoord2D, QubitIndexLocal]
            Mapping from 2D tiling coordinates to local qubit indices
        """
        coord_set = {(coord[0], coord[1]) for coord in self.data_coords}
        sorted_coords = sort_xy(coord_set)

        # If data_indices have been explicitly set, use them (legacy compatibility)
        if self.data_indices and len(self.data_indices) == len(sorted_coords):
            return {TilingCoord2D(coor): self.data_indices[i] for i, coor in enumerate(sorted_coords)}

        # Calculate base qindex for pipes using boundary-based indexing
        base_qindex = calculate_qindex_base_pipe(patch_coord, sink_patch, self.d)
        return {TilingCoord2D(coor): QubitIndexLocal(base_qindex + i) for i, coor in enumerate(sorted_coords)}

    # ---- Coordinate and index shifting APIs ---------------------------------
    def _shift_lists_inplace(self, dx: int, dy: int) -> None:
        if self.data_coords:
            self.data_coords = [TilingCoord2D((x + dx, y + dy)) for (x, y) in self.data_coords]
        if self.x_coords:
            self.x_coords = [TilingCoord2D((x + dx, y + dy)) for (x, y) in self.x_coords]
        if self.z_coords:
            self.z_coords = [TilingCoord2D((x + dx, y + dy)) for (x, y) in self.z_coords]

    def shift_coords(
        self,
        by: tuple[int, int] | tuple[int, int, int],
        *,
        coordinate: CoordinateSystem = CoordinateSystem.TILING_2D,
        inplace: bool = True,
    ) -> ScalableTemplate:
        """Shift template 2D coords based on the given coordinate system.

        - tiling2d: by=(dx,dy) used directly
        - phys3d: by=(x,y,z), uses (x,y)
        - patch3d: by=(px,py,pz), converts to (dx,dy) via block offset rule (INNER)

        Note: Pipe-specific patch3d handling is defined in subclass override.
        """
        if not (self.data_coords or self.x_coords or self.z_coords):
            self.to_tiling()

        dx: int
        dy: int
        if coordinate == CoordinateSystem.TILING_2D:
            bx, by_ = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == CoordinateSystem.PHYS_3D:
            bx, by_, _ = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == CoordinateSystem.PATCH_3D:
            # Default block-style behavior (INNER offset)
            px, py, pz = by  # type: ignore[misc]
            dx, dy = cube_offset_xy(self.d, (int(px), int(py), int(pz)))
        else:
            msg = "coordinate must be one of: tiling2d, phys3d, patch3d"
            raise ValueError(msg)

        if inplace:
            self._shift_lists_inplace(dx, dy)
            return self
        # create a shallow Tiling copy with shifted coordinates
        t = offset_tiling(self, dx, dy)
        # rebuild a new instance of the same class, carrying d/edgespec
        new = type(self)(d=self.d, edgespec=self.edgespec)
        new.data_coords = t.data_coords
        new.x_coords = t.x_coords
        new.z_coords = t.z_coords
        return new

    def trim_spatial_boundary(self, direction: BoundarySide) -> None:
        """Remove ancilla/two-body checks on a given boundary in 2D tiling.

        Only X/Z ancilla on the target boundary line are removed. Data qubits
        remain intact.

        Parameters
        ----------
        direction : BoundarySide
            Boundary side to trim (LEFT, RIGHT, TOP, or BOTTOM).
            UP and DOWN are temporal boundaries and not supported for spatial trimming.
        """
        if not (self.data_coords or self.x_coords or self.z_coords):
            self.to_tiling()

        axis: int
        target: int

        match direction:
            case BoundarySide.TOP:
                axis = 1
                target = 2 * self.d - 1
            case BoundarySide.BOTTOM:
                axis = 1
                target = -1
            case BoundarySide.LEFT:
                axis = 0
                target = -1
            case BoundarySide.RIGHT:
                axis = 0
                target = 2 * self.d - 1
            case _:
                msg = (
                    f"Invalid direction for spatial boundary: {direction}. "
                    "Only TOP, BOTTOM, LEFT, RIGHT are supported (UP/DOWN are temporal boundaries)."
                )
                raise ValueError(msg)

        self.x_coords = [p for p in (self.x_coords or []) if p[axis] != target]
        self.z_coords = [p for p in (self.z_coords or []) if p[axis] != target]

    def visualize_tiling(  # noqa: C901
        self, ax: Axes | None = None, show: bool = True, title_suffix: str | None = None
    ) -> None:
        """Visualize the tiling using matplotlib.

        - data qubits: white-filled circles with black edge
        - X faces: green circles
        - Z faces: blue circles
        """

        data = [(coord[0], coord[1]) for coord in (self.data_coords or [])]
        xs = [(coord[0], coord[1]) for coord in (self.x_coords or [])]
        zs = [(coord[0], coord[1]) for coord in (self.z_coords or [])]

        created_fig: Figure | None = None
        if ax is None:
            created_fig, ax = plt.subplots(figsize=(6, 6))
        if ax is None:
            msg = "ax should not be None here"
            raise ValueError(msg)

        def unpack(coords: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
            if not coords:
                return [], []
            x_vals, y_vals = zip(*coords, strict=False)
            return list(x_vals), list(y_vals)

        dx, dy = unpack(data)
        xx, xy = unpack(xs)
        zx, zy = unpack(zs)

        if dx:
            ax.scatter(
                dx,
                dy,
                s=120,
                facecolors="white",
                edgecolors="black",
                linewidths=1.8,
                label="data",
            )
        if xx:
            ax.scatter(
                xx,
                xy,
                s=90,
                color="#2ecc71",
                edgecolors="#1e8449",
                linewidths=1.0,
                label="X",
            )
        if zx:
            ax.scatter(
                zx,
                zy,
                s=90,
                color="#3498db",
                edgecolors="#1f618d",
                linewidths=1.0,
                label="Z",
            )

        all_x = (dx or []) + (xx or []) + (zx or [])
        all_y = (dy or []) + (xy or []) + (zy or [])
        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            pad = 1
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_aspect("equal")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        title_core = f"d={self.d}"
        if title_suffix:
            title_core += f" | {title_suffix}"
        ax.set_title(title_core)

        if any((dx, xx, zx)):
            ax.legend(loc="upper right", frameon=True)

        if created_fig is not None:
            created_fig.tight_layout()
        if show and created_fig is not None:
            plt.show()


class RotatedPlanarCubeTemplate(ScalableTemplate):
    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:  # noqa: C901
        d = self.d
        data_coords: set[tuple[int, int]] = set()
        x_coords: set[tuple[int, int]] = set()
        z_coords: set[tuple[int, int]] = set()

        # Data qubits at even-even coordinates in [0, 2d-2]
        data_coords = {(2 * i, 2 * j) for i in range(d) for j in range(d)}

        # Bulk checks (odd-odd), two interleaving lattices per type
        for x0, y0 in ((1, 3), (3, 1)):
            for x in range(x0, 2 * d - 1, 4):
                x_coords.update((x, y) for y in range(y0, 2 * d - 1, 4))
        for x0, y0 in ((1, 1), (3, 3)):
            for x in range(x0, 2 * d - 1, 4):
                z_coords.update((x, y) for y in range(y0, 2 * d - 1, 4))

        # Boundaries
        match self._spec(BoundarySide.LEFT):
            case EdgeSpecValue.X:
                x_coords.update((-1, y) for y in range(1, 2 * d - 1, 4))
            case EdgeSpecValue.Z:
                z_coords.update((-1, y) for y in range(2 * d - 3, -1, -4))
            case _:
                pass
        match self._spec(BoundarySide.RIGHT):
            case EdgeSpecValue.X:
                x_coords.update((2 * d - 1, y) for y in range(2 * d - 3, -1, -4))
            case EdgeSpecValue.Z:
                z_coords.update((2 * d - 1, y) for y in range(1, 2 * d - 1, 4))
            case _:
                pass
        match self._spec(BoundarySide.BOTTOM):
            case EdgeSpecValue.X:
                x_coords.update((x, -1) for x in range(1, 2 * d - 1, 4))
            case EdgeSpecValue.Z:
                z_coords.update((x, -1) for x in range(2 * d - 3, -1, -4))
            case _:
                pass
        match self._spec(BoundarySide.TOP):
            case EdgeSpecValue.X:
                x_coords.update((x, 2 * d - 1) for x in range(2 * d - 3, -1, -4))
            case EdgeSpecValue.Z:
                z_coords.update((x, 2 * d - 1) for x in range(1, 2 * d - 1, 4))
            case _:
                pass

        result = {
            "data": sort_xy(data_coords),
            "X": sort_xy(x_coords),
            "Z": sort_xy(z_coords),
        }
        # Sanity: ensure no X/Z overlap within this template
        if x_coords & z_coords:
            overlap = sorted(x_coords & z_coords)[:10]
            msg = f"RotatedPlanarCubeTemplate X/Z overlap: sample={overlap}"
            raise ValueError(msg)
        self.data_coords = [TilingCoord2D(coord) for coord in result["data"]]
        self.x_coords = [TilingCoord2D(coord) for coord in result["X"]]
        self.z_coords = [TilingCoord2D(coord) for coord in result["Z"]]
        return result


# --- Spatial merge helper APIs (Trim -> Merge -> Unify) ---------------------


def _offset_coords(coords: Sequence[TilingCoord2D] | None, dx: int, dy: int) -> list[TilingCoord2D]:
    if not coords:
        return []
    return [TilingCoord2D((x + dx, y + dy)) for (x, y) in coords]


def _copy_with_offset(t: Tiling, dx: int, dy: int) -> Tiling:
    """Create a shallow Tiling copy with all 2D coords offset by (dx, dy).

    Does not mutate the input instance.
    """
    return Tiling(
        data_coords=_offset_coords(t.data_coords, dx, dy),
        x_coords=_offset_coords(t.x_coords, dx, dy),
        z_coords=_offset_coords(t.z_coords, dx, dy),
    )


def offset_tiling(t: Tiling, dx: int, dy: int) -> Tiling:
    """Public wrapper to create a shifted Tiling copy without mutating input."""
    return _copy_with_offset(t, dx, dy)


def cube_offset_xy(
    d: int,
    patch: tuple[int, int, int],
) -> tuple[int, int]:
    px, py, _ = patch
    base_x = 2 * (d + 1) * int(px)
    base_y = 2 * (d + 1) * int(py)
    # INNER anchor only (global policy)
    return base_x, base_y


def merge_pair_spatial(
    a: ScalableTemplate,
    b: ScalableTemplate,
    direction: str,
    *,
    check_collisions: bool = True,
) -> Tiling:
    """Trim the facing boundaries of `a` and `b`, then merge their tilings.

    - direction: one of "X+", "X-", "Y+", "Y-" (case-insensitive)
    - Trims ancilla/two-body checks on the seam (RIGHT/LEFT for X, TOP/BOTTOM for Y)
    - Offsets `b` so it sits adjacent to `a` (grid step = 2*d)
    - Returns a ConnectedTiling which stably de-duplicates within-type coords
      and optionally checks for across-type overlaps.
    """
    d_a = a.d
    d_b = b.d
    if not isinstance(d_a, int) or not isinstance(d_b, int):
        msg = "Both templates must have integer distance 'd'."
        raise TypeError(msg)

    # Ensure coordinates are populated
    if not (a.data_coords or a.x_coords or a.z_coords):
        a.to_tiling()
    if not (b.data_coords or b.x_coords or b.z_coords):
        b.to_tiling()

    diru = direction.upper()
    if diru not in {"X+", "X-", "Y+", "Y-"}:
        msg = "direction must be one of: X+, X-, Y+, Y-"
        raise ValueError(msg)

    # 1) Trim the seam boundaries
    if diru == "X+":
        a.trim_spatial_boundary(BoundarySide.RIGHT)
        b.trim_spatial_boundary(BoundarySide.LEFT)
        off_b = (2 * d_a, 0)
    elif diru == "X-":
        a.trim_spatial_boundary(BoundarySide.LEFT)
        b.trim_spatial_boundary(BoundarySide.RIGHT)
        off_b = (-2 * d_b, 0)
    elif diru == "Y+":
        a.trim_spatial_boundary(BoundarySide.TOP)
        b.trim_spatial_boundary(BoundarySide.BOTTOM)
        off_b = (0, 2 * d_a)
    else:  # "Y-"
        a.trim_spatial_boundary(BoundarySide.BOTTOM)
        b.trim_spatial_boundary(BoundarySide.TOP)
        off_b = (0, -2 * d_b)

    # 2) Build offset copies and merge
    a_copy = _copy_with_offset(a, 0, 0)
    b_copy = _copy_with_offset(b, *off_b)
    # Minimal merged tiling without requiring ConnectedTiling class
    data_list = (a_copy.data_coords or []) + (b_copy.data_coords or [])
    xs_list = (a_copy.x_coords or []) + (b_copy.x_coords or [])
    zs_list = (a_copy.z_coords or []) + (b_copy.z_coords or [])

    data = list(dict.fromkeys(data_list))
    xs = list(dict.fromkeys(xs_list))
    zs = list(dict.fromkeys(zs_list))

    if check_collisions:
        overlap_set = set(xs) & set(zs)
        if overlap_set:
            overlap = sorted(overlap_set)[:10]
            msg = f"merge_pair_spatial X/Z overlap: sample={overlap}"
            raise ValueError(msg)

    return Tiling(
        data_coords=[TilingCoord2D(coord) for coord in sort_xy(set(data))],
        x_coords=[TilingCoord2D(coord) for coord in sort_xy(set(xs))],
        z_coords=[TilingCoord2D(coord) for coord in sort_xy(set(zs))],
    )


@dataclass(kw_only=True)
class RotatedPlanarPipetemplate(ScalableTemplate):
    direction: PIPEDIRECTION

    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:  # noqa: C901
        d = self.d
        data_coords: set[tuple[int, int]] = set()
        x_coords: set[tuple[int, int]] = set()
        z_coords: set[tuple[int, int]] = set()
        if self.direction in {PIPEDIRECTION.RIGHT, PIPEDIRECTION.LEFT}:
            # Pipe along Y (vertical), x fixed at 0
            data_coords.update((0, y) for y in range(0, 2 * d, 2))
            for n in range(d - 1):
                y = 2 * n + 1
                x_coords.add((((-1) ** n), y))
                z_coords.add((-((-1) ** n), y))

            match self._spec(BoundarySide.TOP):
                case EdgeSpecValue.X:
                    x_coords.add((1, 2 * d - 1))
                case EdgeSpecValue.Z:
                    z_coords.add((-1, 2 * d - 1))
                case _:
                    pass
            match self._spec(BoundarySide.BOTTOM):
                case EdgeSpecValue.X:
                    x_coords.add((-1, -1))
                case EdgeSpecValue.Z:
                    z_coords.add((1, -1))
                case _:
                    pass

        elif self.direction in {PIPEDIRECTION.TOP, PIPEDIRECTION.BOTTOM}:
            # Pipe along X (horizontal), y fixed at 0
            data_coords.update((x, 0) for x in range(0, 2 * d, 2))
            for n in range(d - 1):
                x = 2 * n + 1
                x_coords.add((x, (-1) ** n))
                z_coords.add((x, -((-1) ** n)))

            match self._spec(BoundarySide.LEFT):
                case EdgeSpecValue.X:
                    x_coords.add((-1, -1))
                case EdgeSpecValue.Z:
                    z_coords.add((-1, 1))
                case _:
                    pass
            match self._spec(BoundarySide.RIGHT):
                case EdgeSpecValue.X:
                    x_coords.add((2 * d - 1, 1))
                case EdgeSpecValue.Z:
                    z_coords.add((2 * d - 1, -1))
                case _:
                    pass
        else:
            msg = f"Unknown pipe direction. Got: {self.direction}"
            raise ValueError(msg)

        result = {
            "data": sort_xy(data_coords),
            "X": sort_xy(x_coords),
            "Z": sort_xy(z_coords),
        }
        # Sanity: ensure no X/Z overlap within pipe template
        if x_coords & z_coords:
            overlap = sorted(x_coords & z_coords)[:10]
            msg = f"RotatedPlanarPipetemplate X/Z overlap: sample={overlap}"
            raise ValueError(msg)
        self.data_coords = [TilingCoord2D(coord) for coord in result["data"]]
        self.x_coords = [TilingCoord2D(coord) for coord in result["X"]]
        self.z_coords = [TilingCoord2D(coord) for coord in result["Z"]]
        return result

    # Pipe-specific shift including patch3d to (dx,dy) conversion.
    def shift_coords(
        self,
        by: tuple[int, int] | tuple[int, int, int],
        *,
        coordinate: CoordinateSystem = CoordinateSystem.TILING_2D,
        direction: PIPEDIRECTION | None = None,
        inplace: bool = True,
    ) -> RotatedPlanarPipetemplate:
        if not (self.data_coords or self.x_coords or self.z_coords):
            self.to_tiling()

        dx: int
        dy: int
        if coordinate == CoordinateSystem.TILING_2D:
            bx, by_ = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == CoordinateSystem.PHYS_3D:
            bx, by_, _ = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == CoordinateSystem.PATCH_3D:
            if direction is None:
                msg = "direction is required for patch3d pipe shift"
                raise ValueError(msg)
            px, py, pz = by  # type: ignore[misc]
            source_sink = (int(px), int(py), int(pz))
            dx, dy = pipe_offset_xy(self.d, source_sink, source_sink, direction)
        else:
            msg = "coordinate must be one of: tiling2d, phys3d, patch3d"
            raise ValueError(msg)

        if inplace:
            if self.data_coords:
                self.data_coords = [TilingCoord2D((x + dx, y + dy)) for (x, y) in self.data_coords]
            if self.x_coords:
                self.x_coords = [TilingCoord2D((x + dx, y + dy)) for (x, y) in self.x_coords]
            if self.z_coords:
                self.z_coords = [TilingCoord2D((x + dx, y + dy)) for (x, y) in self.z_coords]
            return self
        t = offset_tiling(self, dx, dy)
        new = RotatedPlanarPipetemplate(d=self.d, edgespec=self.edgespec, direction=self.direction)
        new.data_coords = t.data_coords
        new.x_coords = t.x_coords
        new.z_coords = t.z_coords
        return new


def pipe_offset_xy(
    d: int,
    source: tuple[int, int, int],
    sink: tuple[int, int, int],
    direction: PIPEDIRECTION,
) -> tuple[int, int]:
    """Convert a pipe defined by (source, sink, direction) in patch3d to (dx, dy).
    Note: source and sink are patch3d coordinates (px,py,pz).
    - direction: one of PIPEDIRECTION.LEFT/RIGHT/TOP/BOTTOM/UP/DOWN
    - For LEFT/RIGHT, the pipe runs along Y (vertical), x fixed at seam
    - For TOP/BOTTOM, the pipe runs along X (horizontal), y fixed at seam
    - For UP/DOWN, NotImplementedError (temporal pipe not supported yet)
    - The source and sink must be axis neighbors (Manhattan distance 1)
    - The source and sink must share the same z for spatial pipes
    - The returned (dx, dy) is the offset to apply to the pipe template's
        internal (0,0) anchor to place it correctly in the global tiling.
    """
    if direction in {PIPEDIRECTION.UP, PIPEDIRECTION.DOWN}:
        msg = "Temporal pipe (UP/DOWN) not supported for 2D tiling placement"
        raise NotImplementedError(msg)
    if source[2] != sink[2]:
        msg = "source and sink must share the same z for spatial pipe"
        raise ValueError(msg)
    if abs(sink[0] - source[0]) + abs(sink[1] - source[1]) != 1:
        msg = "source and sink must be axis neighbors (Manhattan distance 1)"
        raise ValueError(msg)

    if direction in {PIPEDIRECTION.LEFT, PIPEDIRECTION.BOTTOM}:
        source, sink = sink, source

    # Aligned to RIGHT direction
    if direction in {PIPEDIRECTION.RIGHT, PIPEDIRECTION.LEFT}:
        tx, ty = sink[0], sink[1]
        base_x = (2 * d + 2) * tx - 2  # RIGHT direction -> tx > sx
        base_y = (2 * d + 2) * ty  # RIGHT direction -> ty == sy
        return base_x, base_y

    if direction in {PIPEDIRECTION.TOP, PIPEDIRECTION.BOTTOM}:
        # Place pipe along the seam center between patches in Y at an even row
        # to preserve even parity for data coords after offset.
        tx, ty = sink[0], sink[1]
        base_x = (2 * d + 2) * tx  # TOP direction -> sx == tx
        base_y = (2 * d + 2) * ty - 2  # TOP direction -> sy < ty
        return base_x, base_y

    msg = f"Invalid direction for pipe offset: {direction}"
    raise ValueError(msg)
