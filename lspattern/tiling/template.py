"""Template classes for scalable tiling patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from lspattern.consts.consts import PIPEDIRECTION
from lspattern.mytype import (
    SpatialEdgeSpec,
    TilingConsistentQubitId,
    TilingCoord2D,
)
from lspattern.tiling.base import ConnectedTiling, Tiling
from lspattern.utils import sort_xy


@dataclass(kw_only=True)
class ScalableTemplate(Tiling):
    """Base class for scalable tiling templates with configurable edge specifications."""

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

        Accepts side in any case (e.g., "left"/"LEFT"). Falls back to "O".
        """
        v = None
        if isinstance(self.edgespec, dict):
            v = self.edgespec.get(side.lower())
            if v is None:
                v = self.edgespec.get(side.upper())
        if v is None:
            try:
                v = getattr(self.edgespec, side.upper())  # type: ignore[attr-defined]
            except Exception:
                v = "O"
        try:
            return str(v).upper()
        except Exception:
            return "O"

    def get_data_indices(self) -> dict[TilingCoord2D, TilingConsistentQubitId]:
        return {coor: i for i, coor in enumerate(sort_xy(self.data_coords))}

    # ---- Coordinate and index shifting APIs ---------------------------------
    def _shift_lists_inplace(self, dx: int, dy: int) -> None:
        if getattr(self, "data_coords", None):
            self.data_coords = [(x + dx, y + dy) for (x, y) in self.data_coords]
        if getattr(self, "x_coords", None):
            self.x_coords = [(x + dx, y + dy) for (x, y) in self.x_coords]
        if getattr(self, "z_coords", None):
            self.z_coords = [(x + dx, y + dy) for (x, y) in self.z_coords]

    def shift_coords(
        self,
        by: tuple[int, int] | tuple[int, int, int],
        *,
        coordinate: Literal["tiling2d", "phys3d", "patch3d"] = "tiling2d",
        anchor: Literal["seam", "inner"] = "seam",
        inplace: bool = True,
    ) -> ScalableTemplate:
        """Shift template 2D coords based on the given coordinate system.

        - tiling2d: by=(dx,dy) used directly
        - phys3d: by=(x,y,z), uses (x,y)
        - patch3d: by=(px,py,pz), converts to (dx,dy) via block offset rule

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
            # Default block-style behavior: use block offset
            px, py, pz = by  # type: ignore[misc]
            dx, dy = block_offset_xy(self.d, (int(px), int(py), int(pz)), anchor=anchor)
        else:
            msg = "coordinate must be one of: tiling2d, phys3d, patch3d"
            raise ValueError(msg)

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
        data_indices = getattr(self, "data_indices", None)
        if data_indices is not None:
            shifted = [int(i) + int(by) for i in data_indices]
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
                msg = "Invalid direction for trim_spatial_boundary"
                raise ValueError(msg)

        self.x_coords = [p for p in (self.x_coords or []) if p[axis] != target]
        self.z_coords = [p for p in (self.z_coords or []) if p[axis] != target]

    def visualize_tiling(self, ax: Any = None, show: bool = True, title_suffix: str | None = None) -> None:  # noqa: C901, PLR0914
        """Visualize the tiling using matplotlib.

        - data qubits: white-filled circles with black edge
        - X faces: green circles
        - Z faces: blue circles
        """
        import matplotlib.pyplot as plt

        data = list(getattr(self, "data_coords", []) or [])
        xs = list(getattr(self, "x_coords", []) or [])
        zs = list(getattr(self, "z_coords", []) or [])

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
            ax.scatter(dx, dy, s=120, facecolors="white", edgecolors="black", linewidths=1.8, label="data")
        if xx:
            ax.scatter(xx, xy, s=90, color="#2ecc71", edgecolors="#1e8449", linewidths=1.0, label="X")
        if zx:
            ax.scatter(zx, zy, s=90, color="#3498db", edgecolors="#1f618d", linewidths=1.0, label="Z")

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

        title_core = f"d={getattr(self, 'd', '?')}"
        if title_suffix:
            title_core += f" | {title_suffix}"
        ax.set_title(title_core)

        if any((dx, xx, zx)):
            ax.legend(loc="upper right", frameon=True)

        if created_fig is not None:
            created_fig.tight_layout()
        if show and created_fig is not None:
            import matplotlib.pyplot as plt  # local import to avoid confusion

            plt.show()


class RotatedPlanarTemplate(ScalableTemplate):
    """Rotated planar template for RHG lattice patterns."""

    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:  # noqa: C901, PLR0912
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

        result = {"data": sort_xy(data_coords), "X": sort_xy(x_coords), "Z": sort_xy(z_coords)}
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
        data_coords=_offset_coords(getattr(t, "data_coords", []), dx, dy),
        x_coords=_offset_coords(getattr(t, "x_coords", []), dx, dy),
        z_coords=_offset_coords(getattr(t, "z_coords", []), dx, dy),
    )


def offset_tiling(t: Tiling, dx: int, dy: int) -> Tiling:
    """Public wrapper to create a shifted Tiling copy without mutating input."""
    return _copy_with_offset(t, dx, dy)


def block_offset_xy(
    d: int,
    patch: tuple[int, int, int],
    *,
    anchor: Literal["seam", "inner"] = "seam",
) -> tuple[int, int]:
    px, py, _pz = patch
    base_x = 2 * d * int(px)
    base_y = 2 * d * int(py)
    if anchor == "inner":
        return base_x + 2, base_y + 2
    return base_x, base_y


def merge_pair_spatial(
    a: ScalableTemplate,
    b: ScalableTemplate,
    direction: str,
    *,
    check_collisions: bool = True,
) -> ConnectedTiling:
    """Trim the facing boundaries of `a` and `b`, then merge their tilings.

    - direction: one of "X+", "X-", "Y+", "Y-" (case-insensitive)
    - Trims ancilla/two-body checks on the seam (RIGHT/LEFT for X, TOP/BOTTOM for Y)
    - Offsets `b` so it sits adjacent to `a` (grid step = 2*d)
    - Returns a ConnectedTiling which stably de-duplicates within-type coords
      and optionally checks for across-type overlaps.
    """
    d_a = getattr(a, "d", None)
    d_b = getattr(b, "d", None)
    if not isinstance(d_a, int) or not isinstance(d_b, int):
        msg = "Both templates must have integer distance 'd'."
        raise ValueError(msg)

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
    return ConnectedTiling([a_copy, b_copy], check_collisions=check_collisions)


class RotatedPlanarPipetemplate(ScalableTemplate):
    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:  # noqa: C901
        d = self.d
        data_coords: set[tuple[int, int]] = set()
        x_coords: set[tuple[int, int]] = set()
        z_coords: set[tuple[int, int]] = set()

        is_x_dir = self._spec("LEFT") == "O" and self._spec("RIGHT") == "O"
        is_y_dir = self._spec("TOP") == "O" and self._spec("BOTTOM") == "O"

        if is_x_dir:
            # Pipe along Y (vertical), x fixed at 0
            data_coords.update((0, y) for y in range(0, 2 * d, 2))
            for n in range(d - 2):
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
            for n in range(d - 2):
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
            msg = "Temporal pipe not supported yet"
            raise NotImplementedError(msg)
        else:
            msg = "This pipe has no connection boundary (EdgeSpec)"
            raise ValueError(msg)

        result = {"data": sort_xy(data_coords), "X": sort_xy(x_coords), "Z": sort_xy(z_coords)}
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
                msg = "direction is required for patch3d pipe shift"
                raise ValueError(msg)
            px, py, pz = by  # type: ignore[misc]
            dx, dy = pipe_offset_xy(self.d, (int(px), int(py), int(pz)), None, direction)
        else:
            msg = "coordinate must be one of: tiling2d, phys3d, patch3d"
            raise ValueError(msg)

        if inplace:
            if getattr(self, "data_coords", None):
                self.data_coords = [(x + dx, y + dy) for (x, y) in self.data_coords]
            if getattr(self, "x_coords", None):
                self.x_coords = [(x + dx, y + dy) for (x, y) in self.x_coords]
            if getattr(self, "z_coords", None):
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
    sink: tuple[int, int, int] | None,
    direction: PIPEDIRECTION,
) -> tuple[int, int]:
    sx, sy, sz = source
    if direction in {PIPEDIRECTION.UP, PIPEDIRECTION.DOWN}:
        msg = "Temporal pipe (UP/DOWN) not supported for 2D tiling placement"
        raise NotImplementedError(msg)

    if sink is not None:
        tx, ty, tz = sink
        if sz != tz:
            msg = "source and sink must share the same z for spatial pipe"
            raise ValueError(msg)
        if abs(tx - sx) + abs(ty - sy) != 1:
            msg = "source and sink must be axis neighbors (Manhattan distance 1)"
            raise ValueError(msg)

    if direction in {PIPEDIRECTION.RIGHT, PIPEDIRECTION.LEFT}:
        base_x = 2 * d * min(sx, (sink[0] if sink else sx))
        base_y = 2 * d * sy
        return base_x, base_y
    if direction in {PIPEDIRECTION.TOP, PIPEDIRECTION.BOTTOM}:
        base_x = 2 * d * sx
        base_y = 2 * d * min(sy, (sink[1] if sink else sy))
        return base_x, base_y
    msg = "Invalid direction for pipe offset"
    raise ValueError(msg)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from lspattern.mytype import EdgeSpec

    def set_edgespec(**kw) -> None:
        EdgeSpec.update({"TOP": "O", "BOTTOM": "O", "LEFT": "O", "RIGHT": "O", "UP": "O", "DOWN": "O"})
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
            template = RotatedPlanarTemplate(d=d, edgespec=spec)
            template.to_tiling()
            template.visualize_tiling(ax=ax, show=False, title_suffix=label)

        fig.suptitle(f"Rotated Planar (EdgeSpec-driven) d={d}")
        fig.tight_layout()
        plt.show()

    if SHOW_PIPE:
        d = 7
        pipe_cfgs = [
            ("Pipe X: TOP=X, BOTTOM=Z", {"TOP": "X", "BOTTOM": "Z", "LEFT": "O", "RIGHT": "O"}),
            ("Pipe X: TOP=Z, BOTTOM=X", {"TOP": "Z", "BOTTOM": "X", "LEFT": "O", "RIGHT": "O"}),
            ("Pipe Y: LEFT=X, RIGHT=Z", {"LEFT": "X", "RIGHT": "Z", "TOP": "O", "BOTTOM": "O"}),
            ("Pipe Y: LEFT=Z, RIGHT=X", {"LEFT": "Z", "RIGHT": "X", "TOP": "O", "BOTTOM": "O"}),
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
