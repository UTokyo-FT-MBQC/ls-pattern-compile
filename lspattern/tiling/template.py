from __future__ import annotations

# ruff: noqa: I001  # import layout acceptable; avoid heavy reordering for clarity

from dataclasses import dataclass, field
from typing import Literal

from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import (
    SpatialEdgeSpec,
    QubitIndexLocal,
    TilingCoord2D,
)
from lspattern.tiling.base import Tiling
from lspattern.utils import sort_xy


@dataclass(kw_only=True)
class ScalableTemplate(Tiling):
    d: int
    edgespec: SpatialEdgeSpec  # e.g., {"top":"X","bottom":"Z",...}

    data_coords: list[tuple[int, int]] = field(default_factory=list)
    data_indices: list[int] = field(default_factory=list)
    x_coords: list[tuple[int, int]] = field(default_factory=list)
    z_coords: list[tuple[int, int]] = field(default_factory=list)

    trimmed: bool = False

    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:
        raise NotImplementedError

    def _spec(self, side: str) -> str:
        """Return standardized spec value ("X"/"Z"/"O").

        Accepts side in any case (e.g., "left"/"LEFT"). Defaults to "O".
        """
        v = None
        if isinstance(self.edgespec, dict):
            v = self.edgespec.get(side.upper())
            if v is None:
                v = self.edgespec.get(side.lower())
        if v is None:
            v = "O"
        return str(v).upper()

    def get_data_indices(self) -> dict[TilingCoord2D, QubitIndexLocal]:
        data_index = {coor: i for i, coor in enumerate(sort_xy(self.data_coords))}
        return data_index

    # ---- Coordinate and index shifting APIs ---------------------------------
    def _shift_lists_inplace(self, dx: int, dy: int) -> None:
        if self.data_coords:
            self.data_coords = [(x + dx, y + dy) for (x, y) in self.data_coords]
        if self.x_coords:
            self.x_coords = [(x + dx, y + dy) for (x, y) in self.x_coords]
        if self.z_coords:
            self.z_coords = [(x + dx, y + dy) for (x, y) in self.z_coords]

    def shift_coords(
        self,
        by: tuple[int, int] | tuple[int, int, int],
        *,
        coordinate: Literal["tiling2d", "phys3d", "patch3d"] = "tiling2d",
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
        if coordinate == "tiling2d":
            bx, by_ = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == "phys3d":
            bx, by_, _bz = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == "patch3d":
            # Default block-style behavior (INNER offset)
            px, py, pz = by  # type: ignore[misc]
            dx, dy = cube_offset_xy(self.d, (int(px), int(py), int(pz)))
        else:
            raise ValueError("coordinate must be one of: tiling2d, phys3d, patch3d")

        if inplace:
            self._shift_lists_inplace(dx, dy)
            return self
        # create a shallow Tiling copy with shifted coordinates
        t = offset_tiling(self, dx, dy)
        # rebuild a new instance of the same class, carrying d/edgespec
        new = type(self)(d=self.d, edgespec=self.edgespec)  # type: ignore[call-arg]
        new.data_coords = t.data_coords
        new.x_coords = t.x_coords
        new.z_coords = t.z_coords
        return new

    def shift_qindex(self, by: int, *, inplace: bool = True) -> ScalableTemplate:
        """Shift local qubit indices, if present in this template instance.

        This is optional; ConnectedTiling rebuilds indexes, but callers may
        use this for standalone composition.
        """
        if self.data_indices:
            shifted = [int(i) + int(by) for i in self.data_indices]
            if inplace:
                self.data_indices = shifted
            else:
                new = type(self)(d=self.d, edgespec=self.edgespec)  # type: ignore[call-arg]
                new.data_coords = list(self.data_coords)
                new.x_coords = list(self.x_coords)
                new.z_coords = list(self.z_coords)
                new.data_indices = shifted
                return new
        return self

    def trim_spatial_boundary(self, direction: str) -> None:
        """Remove ancilla/two-body checks on a given boundary in 2D tiling.

        Only X/Z ancilla on the target boundary line are removed. Data qubits
        remain intact. Supported directions: LEFT/RIGHT/TOP/BOTTOM or X±/Y±.
        """
        if not (self.data_coords or self.x_coords or self.z_coords):
            self.to_tiling()

        axis: int
        target: int

        match direction.upper():
            case "TOP" | "Y+":
                axis = 1
                target = 2 * self.d - 1
            case "BOTTOM" | "Y-":
                axis = 1
                target = -1
            case "LEFT" | "X-":
                axis = 0
                target = -1
            case "RIGHT" | "X+":
                axis = 0
                target = 2 * self.d - 1
            case _:
                raise ValueError("Invalid direction for trim_spatial_boundary")

        self.x_coords = [p for p in (self.x_coords or []) if p[axis] != target]
        self.z_coords = [p for p in (self.z_coords or []) if p[axis] != target]

    def visualize_tiling(self, ax=None, show: bool = True, title_suffix: str | None = None) -> None:
        """Visualize the tiling using matplotlib.

        - data qubits: white-filled circles with black edge
        - X faces: green circles
        - Z faces: blue circles
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        data = list(self.data_coords or [])
        xs = list(self.x_coords or [])
        zs = list(self.z_coords or [])

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots(figsize=(6, 6))

        def unpack(coords: list[tuple[int, int]]):
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
            import matplotlib.pyplot as plt  # noqa: PLC0415

            plt.show()


class RotatedPlanarCubeTemplate(ScalableTemplate):
    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:
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
        match self._spec("LEFT"):
            case "X":
                x_coords.update((-1, y) for y in range(1, 2 * d - 1, 4))
            case "Z":
                z_coords.update((-1, y) for y in range(2 * d - 3, -1, -4))
            case _:
                pass
        match self._spec("RIGHT"):
            case "X":
                x_coords.update((2 * d - 1, y) for y in range(2 * d - 3, -1, -4))
            case "Z":
                z_coords.update((2 * d - 1, y) for y in range(1, 2 * d - 1, 4))
            case _:
                pass
        match self._spec("BOTTOM"):
            case "X":
                x_coords.update((x, -1) for x in range(1, 2 * d - 1, 4))
            case "Z":
                z_coords.update((x, -1) for x in range(2 * d - 3, -1, -4))
            case _:
                pass
        match self._spec("TOP"):
            case "X":
                x_coords.update((x, 2 * d - 1) for x in range(2 * d - 3, -1, -4))
            case "Z":
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
            raise ValueError(f"RotatedPlanarCubeTemplate X/Z overlap: sample={overlap}")
        self.data_coords = result["data"]
        self.x_coords = result["X"]
        self.z_coords = result["Z"]
        return result


# --- Spatial merge helper APIs (Trim -> Merge -> Unify) ---------------------


def _offset_coords(coords: list[tuple[int, int]] | None, dx: int, dy: int) -> list[tuple[int, int]]:
    if not coords:
        return []
    return [(x + dx, y + dy) for (x, y) in coords]


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
    px, py, _pz = patch
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
        raise ValueError("Both templates must have integer distance 'd'.")

    # Ensure coordinates are populated
    if not (a.data_coords or a.x_coords or a.z_coords):
        a.to_tiling()
    if not (b.data_coords or b.x_coords or b.z_coords):
        b.to_tiling()

    diru = direction.upper()
    if diru not in {"X+", "X-", "Y+", "Y-"}:
        raise ValueError("direction must be one of: X+, X-, Y+, Y-")

    # 1) Trim the seam boundaries
    if diru == "X+":
        a.trim_spatial_boundary("RIGHT")
        b.trim_spatial_boundary("LEFT")
        off_b = (2 * d_a, 0)
    elif diru == "X-":
        a.trim_spatial_boundary("LEFT")
        b.trim_spatial_boundary("RIGHT")
        off_b = (-2 * d_b, 0)
    elif diru == "Y+":
        a.trim_spatial_boundary("TOP")
        b.trim_spatial_boundary("BOTTOM")
        off_b = (0, 2 * d_a)
    else:  # "Y-"
        a.trim_spatial_boundary("BOTTOM")
        b.trim_spatial_boundary("TOP")
        off_b = (0, -2 * d_b)

    # 2) Build offset copies and merge
    a_copy = _copy_with_offset(a, 0, 0)
    b_copy = _copy_with_offset(b, *off_b)
    # Minimal merged tiling without requiring ConnectedTiling class
    data = list(dict.fromkeys((a_copy.data_coords or []) + (b_copy.data_coords or [])))
    xs = list(dict.fromkeys((a_copy.x_coords or []) + (b_copy.x_coords or [])))
    zs = list(dict.fromkeys((a_copy.z_coords or []) + (b_copy.z_coords or [])))
    if check_collisions:
        overlap_set = set(xs) & set(zs)
        if overlap_set:
            overlap = sorted(overlap_set)[:10]
            raise ValueError(f"merge_pair_spatial X/Z overlap: sample={overlap}")
    return Tiling(data_coords=sort_xy(data), x_coords=sort_xy(xs), z_coords=sort_xy(zs))


class RotatedPlanarPipetemplate(ScalableTemplate):
    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:
        d = self.d
        data_coords: set[tuple[int, int]] = set()
        x_coords: set[tuple[int, int]] = set()
        z_coords: set[tuple[int, int]] = set()

        is_x_dir = self._spec("LEFT") == "O" and self._spec("RIGHT") == "O"
        is_y_dir = self._spec("TOP") == "O" and self._spec("BOTTOM") == "O"

        if is_x_dir:
            # Pipe along Y (vertical), x fixed at 0
            data_coords.update((0, y) for y in range(0, 2 * d, 2))
            for n in range(d - 1):
                y = 2 * n + 1
                x_coords.add((((-1) ** n), y))
                z_coords.add((-((-1) ** n), y))

            match self._spec("TOP"):
                case "X":
                    x_coords.add((1, 2 * d - 1))
                case "Z":
                    z_coords.add((-1, 2 * d - 1))
                case _:
                    pass
            match self._spec("BOTTOM"):
                case "X":
                    x_coords.add((-1, -1))
                case "Z":
                    z_coords.add((1, -1))
                case _:
                    pass

        elif is_y_dir:
            # Pipe along X (horizontal), y fixed at 0
            data_coords.update((x, 0) for x in range(0, 2 * d, 2))
            for n in range(d - 1):
                x = 2 * n + 1
                x_coords.add((x, (-1) ** n))
                z_coords.add((x, -((-1) ** n)))

            match self._spec("LEFT"):
                case "X":
                    x_coords.add((-1, -1))
                case "Z":
                    z_coords.add((-1, 1))
                case _:
                    pass
            match self._spec("RIGHT"):
                case "X":
                    x_coords.add((2 * d - 1, 1))
                case "Z":
                    z_coords.add((2 * d - 1, -1))
                case _:
                    pass

        elif self._spec("UP") == "O" or self._spec("DOWN") == "O":
            raise NotImplementedError("Temporal pipe not supported yet")
        else:
            raise ValueError("This pipe has no connection boundary (EdgeSpec)")

        result = {
            "data": sort_xy(data_coords),
            "X": sort_xy(x_coords),
            "Z": sort_xy(z_coords),
        }
        # Sanity: ensure no X/Z overlap within pipe template
        if x_coords & z_coords:
            overlap = sorted(x_coords & z_coords)[:10]
            raise ValueError(f"RotatedPlanarPipetemplate X/Z overlap: sample={overlap}")
        self.data_coords = result["data"]
        self.x_coords = result["X"]
        self.z_coords = result["Z"]
        return result

    # Pipe-specific shift including patch3d to (dx,dy) conversion.
    def shift_coords(
        self,
        by: tuple[int, int] | tuple[int, int, int],
        *,
        coordinate: Literal["tiling2d", "phys3d", "patch3d"] = "tiling2d",
        direction: PIPEDIRECTION | None = None,
        inplace: bool = True,
    ) -> RotatedPlanarPipetemplate:
        if not (self.data_coords or self.x_coords or self.z_coords):
            self.to_tiling()

        dx: int
        dy: int
        if coordinate == "tiling2d":
            bx, by_ = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == "phys3d":
            bx, by_, _bz = by  # type: ignore[misc]
            dx, dy = int(bx), int(by_)
        elif coordinate == "patch3d":
            if direction is None:
                raise ValueError("direction is required for patch3d pipe shift")
            px, py, pz = by  # type: ignore[misc]
            dx, dy = pipe_offset_xy(self.d, (int(px), int(py), int(pz)), None, direction)
        else:
            raise ValueError("coordinate must be one of: tiling2d, phys3d, patch3d")

        if inplace:
            if self.data_coords:
                self.data_coords = [(x + dx, y + dy) for (x, y) in self.data_coords]
            if self.x_coords:
                self.x_coords = [(x + dx, y + dy) for (x, y) in self.x_coords]
            if self.z_coords:
                self.z_coords = [(x + dx, y + dy) for (x, y) in self.z_coords]
            return self
        t = offset_tiling(self, dx, dy)
        new = RotatedPlanarPipetemplate(d=self.d, edgespec=self.edgespec)
        new.data_coords = t.data_coords
        new.x_coords = t.x_coords
        new.z_coords = t.z_coords
        return new

    def shift_qindex(self, by: int, *, inplace: bool = True) -> RotatedPlanarPipetemplate:
        return super().shift_qindex(by, inplace=inplace)  # type: ignore[return-value]


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
        raise NotImplementedError("Temporal pipe (UP/DOWN) not supported for 2D tiling placement")
    if source[2] != sink[2]:
        raise ValueError("source and sink must share the same z for spatial pipe")
    if abs(sink[0] - source[0]) + abs(sink[1] - source[1]) != 1:
        raise ValueError("source and sink must be axis neighbors (Manhattan distance 1)")

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

    raise ValueError(f"Invalid direction for pipe offset: {direction}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from lspattern.mytype import EdgeSpec

    def set_edgespec(**kw):
        EdgeSpec.update(
            {
                "TOP": "O",
                "BOTTOM": "O",
                "LEFT": "O",
                "RIGHT": "O",
                "UP": "O",
                "DOWN": "O",
            }
        )
        EdgeSpec.update({k.upper(): v for k, v in kw.items()})

    SHOW_BLOCK = False
    SHOW_PIPE = True

    if SHOW_BLOCK:
        d = 3
        configs = [
            ("L/R=X, T/B=Z", {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}),
            ("L/R=Z, T/B=X", {"LEFT": "Z", "RIGHT": "Z", "TOP": "X", "BOTTOM": "X"}),
            ("All X", {"LEFT": "X", "RIGHT": "X", "TOP": "X", "BOTTOM": "X"}),
            ("All Z", {"LEFT": "Z", "RIGHT": "Z", "TOP": "Z", "BOTTOM": "Z"}),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for (label, spec), ax in zip(configs, axes.ravel(), strict=False):
            set_edgespec(**spec)
            template = RotatedPlanarCubeTemplate(d=d, edgespec=spec)
            template.to_tiling()
            template.visualize_tiling(ax=ax, show=False, title_suffix=label)

        fig.suptitle(f"Rotated Planar (EdgeSpec-driven) d={d}")
        fig.tight_layout()
        plt.show()

    if SHOW_PIPE:
        d = 7
        pipe_cfgs = [
            (
                "Pipe X: TOP=X, BOTTOM=Z",
                {"TOP": "X", "BOTTOM": "Z", "LEFT": "O", "RIGHT": "O"},
            ),
            (
                "Pipe X: TOP=Z, BOTTOM=X",
                {"TOP": "Z", "BOTTOM": "X", "LEFT": "O", "RIGHT": "O"},
            ),
            (
                "Pipe Y: LEFT=X, RIGHT=Z",
                {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"},
            ),
            (
                "Pipe Y: LEFT=Z, RIGHT=X",
                {"LEFT": "Z", "RIGHT": "X", "TOP": "O", "BOTTOM": "O"},
            ),
        ]

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
        for (label, spec), ax in zip(pipe_cfgs, axes2.ravel(), strict=False):
            set_edgespec(**spec)
            ptemp = RotatedPlanarPipetemplate(d=d, edgespec=spec)
            ptemp.to_tiling()
            ptemp.visualize_tiling(ax=ax, show=False, title_suffix=label)

        fig2.suptitle(f"Rotated Planar Pipes (EdgeSpec) d={d}")
        fig2.tight_layout()
        plt.show()
