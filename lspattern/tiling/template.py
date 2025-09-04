from dataclasses import dataclass, field

from mytype import EdgeSpec

from lspattern.mytype import TilingConsistentQubitId, TilingCoord2D
from lspattern.tiling.base import Tiling
from lspattern.utils import sort_xy


# TODO: remove kind. Use edgespec
@dataclass
class ScalableTemplate(Tiling):
    d: int
    kind: tuple[str, str, str]  # (X, Y, Z) faces 3-4 chars
    edgespec: EdgeSpec

    data_coords: list[tuple[int, int]] = field(default_factory=list)
    data_indices: list[int] = field(default_factory=list)
    x_coords: list[tuple[int, int]] = field(default_factory=list)
    z_coords: list[tuple[int, int]] = field(default_factory=list)

    def to_tiling(self) -> dict[str, list[tuple[int, int]]]: ...

    def get_data_indices(self) -> dict[TilingCoord2D, TilingConsistentQubitId]:
        data_index = {coor: i for i, coor in enumerate(sort_xy(self.data_coords))}
        return data_index

    # ---- T3 additions: EdgeSpec 利用と Trim API ----
    def _resolve_edge(self, side: str) -> str:
        s = side.upper()
        # Prefer per-instance edgespec if provided; fallback to 'O'
        try:
            val = getattr(self.edgespec, s)
        except Exception:
            val = "O"
        if val not in ("X", "Z", "O"):
            return "O"
        return val

    def _dir_to_side(self, direction: str) -> str:
        d = direction.upper()
        # 追加指示: LEFT/RIGHT/TOP/BOTTOM 以外は来ない前提で簡略化
        if d in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
            return d
        return d

    def _ensure_coords_populated(self) -> None:
        if not (self.data_coords or self.x_coords or self.z_coords):
            try:
                self.to_tiling()
            except Exception:
                pass

    def trim_spatial_boundary(self, direction: str) -> None:
        """Remove ancilla/two-body checks on a given boundary in 2D tiling.

        Only X/Z ancilla on the target boundary line are removed. Data qubits
        remain intact. Supported directions: LEFT/RIGHT/TOP/BOTTOM or X±/Y±.
        """
        self._ensure_coords_populated()
        side = self._dir_to_side(direction)
        d = getattr(self, "d", None)
        if not isinstance(d, int):
            return

        def on_side(pt: tuple[int, int]) -> bool:
            x, y = pt
            if side == "LEFT":
                return x == -1
            if side == "RIGHT":
                return x == 2 * d - 1
            if side == "BOTTOM":
                return y == -1
            if side == "TOP":
                return y == 2 * d - 1
            return False

        self.x_coords = [p for p in (self.x_coords or []) if not on_side(p)]
        self.z_coords = [p for p in (self.z_coords or []) if not on_side(p)]

    def visualize_tiling(
        self, ax=None, show: bool = True, title_suffix: str | None = None
    ) -> None:
        """Visualize the tiling using matplotlib.

        - data qubits: white-filled circles with black edge
        - X faces: green circles
        - Z faces: blue circles

        Adds x/y ticks and grid; title shows d and kind.
        """
        import matplotlib.pyplot as plt

        # Prepare coordinate arrays (robust to None/empties)
        data = list(getattr(self, "data_coords", []) or [])
        xs = list(getattr(self, "x_coords", []) or [])
        zs = list(getattr(self, "z_coords", []) or [])

        # Build figure/axes
        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots(figsize=(6, 6))

        # Helper to unpack list[(x,y)] -> two lists
        def unpack(coords: list[tuple[int, int]]):
            if not coords:
                return [], []
            x_vals, y_vals = zip(*coords)
            return list(x_vals), list(y_vals)

        # Plot points
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

        # Axis limits: pad by 1 around all shown points
        all_x = (dx or []) + (xx or []) + (zx or [])
        all_y = (dy or []) + (xy or []) + (zy or [])
        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            pad = 1
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

            # Ticks at integer grid points within limits
            xticks = list(range(int(xmin) - pad, int(xmax) + pad + 1))
            yticks = list(range(int(ymin) - pad, int(ymax) + pad + 1))
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

        # Grid and aspect
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")

        # Axes labels and title
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        title_core = (
            f"Tiling d={getattr(self, 'd', '?')} kind={getattr(self, 'kind', '?')}"
        )
        if title_suffix:
            title_core += f" ({title_suffix})"
        ax.set_title(title_core)

        # Legend only if something is plotted
        if any((dx, xx, zx)):
            ax.legend(loc="upper right", frameon=True)

        if created_fig is not None:
            created_fig.tight_layout()
        if show and created_fig is not None:
            import matplotlib.pyplot as plt  # local import to avoid confusion

            plt.show()


class RotatedPlanarTemplate(ScalableTemplate):
    # Denote = as X boundary and - as Z boundary
    def to_tiling(self) -> dict[str, list[tuple[int, int]]]:
        d = self.d
        # Local containers (avoid relying on dataclass defaults)
        data_coords: set[tuple[int, int]] = set()
        x_coords: set[tuple[int, int]] = set()
        z_coords: set[tuple[int, int]] = set()

        # 1) Data qubits at even-even coordinates within [0, 2d-2]
        data_coords = {(2 * i, 2 * j) for i in range(d) for j in range(d)}

        # 2) Bulk checks (odd-odd), two interleaving lattices per type
        # X bulk seeds: (1,3) and (3,1)
        for x0, y0 in ((1, 3), (3, 1)):
            for x in range(x0, 2 * d - 1, 4):
                for y in range(y0, 2 * d - 1, 4):
                    x_coords.add((x, y))
        # Z bulk seeds: (1,1) and (3,3)
        for x0, y0 in ((1, 1), (3, 3)):
            for x in range(x0, 2 * d - 1, 4):
                for y in range(y0, 2 * d - 1, 4):
                    z_coords.add((x, y))

        # 3) X faces (left/right boundaries along x)
        # kind[0] chooses which type lives on vertical boundaries
        # TODO: refactor this part
        # このあたりkind -> edgespecに置き換えたい。変更が大きくなるので注意して取り組む
        # kind ベースの境界ロジックは廃止（EdgeSpec で決定）
        # 3) 垂直境界（LEFT/RIGHT）は EdgeSpec で決定
        left_spec = self._resolve_edge("LEFT")
        right_spec = self._resolve_edge("RIGHT")
        if left_spec == "X":
            for y in range(1, 2 * d - 1, 4):
                x_coords.add((-1, y))
        elif left_spec == "Z":
            for y in range(2 * d - 3, -1, -4):
                z_coords.add((-1, y))

        if right_spec == "X":
            for y in range(2 * d - 3, -1, -4):
                x_coords.add((2 * d - 1, y))
        elif right_spec == "Z":
            for y in range(1, 2 * d - 1, 4):
                z_coords.add((2 * d - 1, y))

        # 4) 水平境界（BOTTOM/TOP）は EdgeSpec で決定
        bottom_spec = self._resolve_edge("BOTTOM")
        top_spec = self._resolve_edge("TOP")
        if bottom_spec == "X":
            for x in range(1, 2 * d - 1, 4):
                x_coords.add((x, -1))
        elif bottom_spec == "Z":
            for x in range(2 * d - 3, -1, -4):
                z_coords.add((x, -1))

        if top_spec == "X":
            for x in range(2 * d - 3, -1, -4):
                x_coords.add((x, 2 * d - 1))
        elif top_spec == "Z":
            for x in range(1, 2 * d - 1, 4):
                z_coords.add((x, 2 * d - 1))

        # 5) Reconcile per-side EdgeSpec on boundaries (EdgeSpec overrides kind)
        def on_side(pt: tuple[int, int], side: str) -> bool:
            x, y = pt
            if side == "LEFT":
                return x == -1
            if side == "RIGHT":
                return x == 2 * d - 1
            if side == "BOTTOM":
                return y == -1
            if side == "TOP":
                return y == 2 * d - 1
            return False

        for side in ("LEFT", "RIGHT", "BOTTOM", "TOP"):
            spec = self._resolve_edge(side)
            if spec == "X":
                # only X allowed on this side
                z_coords = {p for p in z_coords if not on_side(p, side)}
            elif spec == "Z":
                # only Z allowed on this side
                x_coords = {p for p in x_coords if not on_side(p, side)}
            elif spec == "O":
                # trimmed: remove both X and Z on this side
                x_coords = {p for p in x_coords if not on_side(p, side)}
                z_coords = {p for p in z_coords if not on_side(p, side)}

        result = {
            "data": sort_xy(data_coords),
            "X": sort_xy(x_coords),
            "Z": sort_xy(z_coords),
        }

        # Also populate instance attrs if present (optional for visualization)
        try:
            self.data_coords = result["data"]  # type: ignore[attr-defined]
            self.x_coords = result["X"]  # type: ignore[attr-defined]
            self.z_coords = result["Z"]  # type: ignore[attr-defined]
        except Exception:
            pass

        return result


class RotatedPlanarPipetemplate(ScalableTemplate):
    def to_tiling(self):
        d = self.d
        # Local containers (avoid relying on dataclass defaults)
        data_coords: set[tuple[int, int]] = set()
        x_coords: set[tuple[int, int]] = set()
        z_coords: set[tuple[int, int]] = set()
        # kind廃止の際はself.directionがXpm,Ypm, Zpmで取得できるようにする
        if self.kind[0] == "O":
            """
            if self.kind[0] == "O", which means this pipe is piping along X direction

            The schematic. = shows X boundary and - shows Z boundary
            ===== KKK =====
            |   | x x |   |
            |   | x x |   |
            ===== KKK =====

            So the tiling has the footprint of (1, d), excluding the ancilla qubits. If we include them it will be (3, d+1)
            Here K means the boundary designated by kind[1]
            """
            # data qubits
            for y in range(0, 2 * d, 2):
                data_coords.add((0, y))
            # x ancillas
            for x0, y0 in ((-1, 3), (1, 1)):
                for iy in range(d // 2):
                    x_coords.add((x0, y0 + 4 * iy))
            # z ancillas
            for x0, y0 in ((-1, 1), (1, 3)):
                for iy in range(d // 2):
                    z_coords.add((x0, y0 + 4 * iy))

            if self.kind[1] == "X":
                # insert two-body X stabilizer at (-1, -1) and (1, 2d-1)
                x_coords.add((-1, -1))
                x_coords.add((1, 2 * d - 1))
            elif self.kind[1] == "Z":
                # insert two-body Z stabilizer at (1, -1) and (-1, 2d-1)
                z_coords.add((1, -1))
                z_coords.add((-1, 2 * d - 1))
        elif self.kind[1] == "O":
            """
            If `self.kind[1] == "O"`, the pipe runs along the Y direction
            (horizontal faces are open/trimmed). This is the 90-degree-rotated
            counterpart of the X-direction pipe above.

            Schematic (now piping along Y). "=" shows X boundary, "-" shows Z boundary.
            
            =====
            |   |
            |   |
            =====
            KxxxK
            K   K
            KxxxK    
            =====
            |   |
            |   |
            =====

            Footprint (excluding ancilla qubits): (d, 1). Including ancilla:
            approximately (d+1, 3). Here K means the boundary designated by kind[0]
            (left/right faces), which decides whether two-body X or Z stabilizers
            are placed at the pipe ends.
            """
            # data qubits along x (y fixed at 0): (0,0), (2,0), .., (2d-2, 0)
            for x in range(0, 2 * d, 2):
                data_coords.add((x, 0))

            # X ancillas (rotated from the X-direction case): seed at (3,0) and (1,1)
            for x0, y0 in ((3, -1), (1, 1)):
                for ix in range(d // 2):
                    x_coords.add((x0 + 4 * ix, y0))

            # Z ancillas (rotated from the X-direction case): seed at (1,-1) and (3,1)
            for x0, y0 in ((1, -1), (3, 1)):
                for ix in range(d // 2):
                    z_coords.add((x0 + 4 * ix, y0))

            # Two-body stabilizers at left/right ends decided by kind[0]
            if self.kind[0] == "X":
                # Insert two-body X stabilizers near x=-1 and x=2d-1
                x_coords.add((-1, -1))
                x_coords.add((2 * d - 1, 1))
            elif self.kind[0] == "Z":
                # Insert two-body Z stabilizers near x=-1 and x=2d-1
                z_coords.add((-1, 1))
                z_coords.add((2 * d - 1, -1))
        elif self.kind[2] == "O":
            raise NotImplementedError("Temporal pipe not supported yet")
        else:
            raise ValueError("This pipe has no connection boundary")

        # Build deterministic result and also populate instance attributes (optional)
        result = {
            "data": sort_xy(data_coords),
            "X": sort_xy(x_coords),
            "Z": sort_xy(z_coords),
        }

        try:
            self.data_coords = result["data"]  # type: ignore[attr-defined]
            self.x_coords = result["X"]  # type: ignore[attr-defined]
            self.z_coords = result["Z"]  # type: ignore[attr-defined]
        except Exception:
            pass

        return result


# simple testing (manual)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SHOW_BLOCK = False
    SHOW_PIPE = True
    if SHOW_BLOCK:
        d = 3
        kinds = [("X", "X", "*"), ("X", "Z", "*"), ("Z", "X", "*"), ("Z", "Z", "*")]
        labels = ["XX", "XZ", "ZX", "ZZ"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for (kx, ky, kz), label, ax in zip(kinds, labels, axes.ravel()):
            template = RotatedPlanarTemplate(d=d, kind=(kx, ky, kz))
            tiling = template.to_tiling()
            print(label, {k: len(v) for k, v in tiling.items()})
            template.visualize_tiling(ax=ax, show=False, title_suffix=label)

        fig.suptitle(f"Rotated Planar XY Faces (d={d})")
        fig.tight_layout()
        plt.show()

    if SHOW_PIPE:
        d = 7
        pipe_kinds = [
            ("O", "X", "*"),  # pipe along X, horizontal faces X
            ("O", "Z", "*"),  # pipe along X, horizontal faces Z
            ("X", "O", "*"),  # pipe along Y, vertical faces X
            ("Z", "O", "*"),  # pipe along Y, vertical faces Z
        ]
        labels = ["Pipe OX", "Pipe OZ", "Pipe XO", "Pipe ZO"]

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
        for kind, label, ax in zip(pipe_kinds, labels, axes2.ravel()):
            ptemp = RotatedPlanarPipetemplate(d=d, kind=kind)
            tiling = ptemp.to_tiling()
            print(label, {k: len(v) for k, v in tiling.items()})
            ptemp.visualize_tiling(ax=ax, show=False, title_suffix=label)

        fig2.suptitle(f"Rotated Planar Pipes (d={d})")
        fig2.tight_layout()
        plt.show()
