from dataclasses import dataclass, field

from lspattern.mytype import EdgeSpec, TilingConsistentQubitId, TilingCoord2D
from lspattern.tiling.base import Tiling
from lspattern.utils import sort_xy


# TODO: remove kind. Use edgespec
@dataclass
class ScalableTemplate(Tiling):
    d: int
    edgespec: EdgeSpec  # XXZX

    data_coords: list[tuple[int, int]] = field(default_factory=list)
    data_indices: list[int] = field(default_factory=list)
    x_coords: list[tuple[int, int]] = field(default_factory=list)
    z_coords: list[tuple[int, int]] = field(default_factory=list)

    def to_tiling(self) -> dict[str, list[tuple[int, int]]]: ...

    def get_data_indices(self) -> dict[TilingCoord2D, TilingConsistentQubitId]:
        data_index = {coor: i for i, coor in enumerate(sort_xy(self.data_coords))}
        return data_index

    def trim_spatial_boundary(self, direction: str) -> None:
        """Remove ancilla/two-body checks on a given boundary in 2D tiling.

        Only X/Z ancilla on the target boundary line are removed. Data qubits
        remain intact. Supported directions: LEFT/RIGHT/TOP/BOTTOM or X±/Y±.
        """
        if not (self.data_coords or self.x_coords or self.z_coords):
            self.to_tiling()

        axis: int = -1
        target: int = -1

        # check TOP
        match direction:
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
        match self.edgespec.LEFT:
            case "X":
                for y in range(1, 2 * d - 1, 4):
                    x_coords.add((-1, y))
            case "Z":
                for y in range(2 * d - 3, -1, -4):
                    z_coords.add((-1, y))
            case "O":
                # nothing
                pass
        match self.edgespec.RIGHT:
            case "X":
                for y in range(2 * d - 3, -1, -4):
                    x_coords.add((2 * d - 1, y))
            case "Z":
                for y in range(1, 2 * d - 1, 4):
                    z_coords.add((2 * d - 1, y))
            case _:
                pass
        match self.edgespec.BOTTOM:
            case "X":
                for x in range(1, 2 * d - 1, 4):
                    x_coords.add((x, -1))
            case "Z":
                for x in range(2 * d - 3, -1, -4):
                    z_coords.add((x, -1))
            case _:
                pass
        match self.edgespec.TOP:
            case "X":
                for x in range(2 * d - 3, -1, -4):
                    x_coords.add((x, 2 * d - 1))
            case "Z":
                for x in range(1, 2 * d - 1, 4):
                    z_coords.add((x, 2 * d - 1))
            case _:
                pass

        result = {
            "data": sort_xy(data_coords),
            "X": sort_xy(x_coords),
            "Z": sort_xy(z_coords),
        }

        self.data_coords = result["data"]
        self.x_coords = result["X"]
        self.z_coords = result["Z"]

        return result


class RotatedPlanarPipetemplate(ScalableTemplate):
    def to_tiling(self):
        d = self.d
        # Local containers (avoid relying on dataclass defaults)
        data_coords: set[tuple[int, int]] = set()
        x_coords: set[tuple[int, int]] = set()
        z_coords: set[tuple[int, int]] = set()
        # kind廃止の際はself.directionがXpm,Ypm, Zpmで取得できるようにする
        if self.edgespec.LEFT == self.edgespec.RIGHT == "O":
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
            # x ancillas (bulk)
            for n in range(d - 2):
                x = (-1) ** n
                y = 2 * n + 1
                x_coords.add((x, y))
            # z ancillas (bulk)
            for n in range(d - 2):
                x = -((-1) ** n)
                y = 2 * n + 1
                z_coords.add((x, y))

            match self.edgespec.TOP:
                case "X":
                    x_coords.add((1, 2 * d + 1))
                case "Z":
                    z_coords.add((-1, 2 * d + 1))
                case "O":
                    pass

            match self.edgespec.BOTTOM:
                case "X":
                    x_coords.add((-1, -1))
                case "Z":
                    z_coords.add((1, -1))
                case "O":
                    pass

        elif self.edgespec.TOP == self.edgespec.BOTTOM == "O":
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

            # x ancillas (bulk)
            for n in range(d - 2):
                x = 2 * n + 1
                y = (-1) ** n
                x_coords.add((x, y))
            # z ancillas (bulk)
            for n in range(d - 2):
                x = 2 * n + 1
                y = -((-1) ** n)
                z_coords.add((x, y))

            # Two-body stabilizers at left/right ends decided by EdgeSpec
            match self.edgespec.LEFT:
                case "X":
                    x_coords.add((-1, -1))
                case "Z":
                    z_coords.add((-1, 1))
                case "O":
                    pass

            match self.edgespec.RIGHT:
                case "X":
                    x_coords.add((2 * d - 1, 1))
                case "Z":
                    z_coords.add((2 * d - 1, -1))
                case "O":
                    pass
        elif self.edgespec.UP == "O" or self.edgespec.DOWN == "O":
            raise NotImplementedError("Temporal pipe not supported yet")
        else:
            raise ValueError("This pipe has no connection boundary")

        # Build deterministic result and also populate instance attributes (optional)
        result = {
            "data": sort_xy(data_coords),
            "X": sort_xy(x_coords),
            "Z": sort_xy(z_coords),
        }

        self.data_coords = result["data"]  # type: ignore[attr-defined]
        self.x_coords = result["X"]  # type: ignore[attr-defined]
        self.z_coords = result["Z"]  # type: ignore[attr-defined]

        return result


# simple testing (manual)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mytype import EdgeSpec  # use global class-level spec

    def set_edgespec(**kw):
        # Reset to open by default, then apply overrides
        EdgeSpec.update({"TOP": "O", "BOTTOM": "O", "LEFT": "O", "RIGHT": "O"})
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
        for (label, spec), ax in zip(configs, axes.ravel()):
            set_edgespec(**spec)
            template = RotatedPlanarTemplate(d=d, kind=("X", "X", "*"), edgespec=EdgeSpec)
            tiling = template.to_tiling()
            print(label, {k: len(v) for k, v in tiling.items()})
            template.visualize_tiling(ax=ax, show=False, title_suffix=label)

        fig.suptitle(f"Rotated Planar (EdgeSpec-driven) d={d}")
        fig.tight_layout()
        plt.show()

    if SHOW_PIPE:
        d = 7
        pipe_cfgs = [
            ("Pipe X: TOP=X, BOTTOM=Z", ("O", "X", "*"), {"TOP": "X", "BOTTOM": "Z"}),
            ("Pipe X: TOP=Z, BOTTOM=X", ("O", "Z", "*"), {"TOP": "Z", "BOTTOM": "X"}),
            ("Pipe Y: LEFT=X, RIGHT=Z", ("X", "O", "*"), {"LEFT": "X", "RIGHT": "Z"}),
            ("Pipe Y: LEFT=Z, RIGHT=X", ("Z", "O", "*"), {"LEFT": "Z", "RIGHT": "X"}),
        ]

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
        for (label, kind, spec), ax in zip(pipe_cfgs, axes2.ravel()):
            set_edgespec(**spec)
            ptemp = RotatedPlanarPipetemplate(d=d, kind=kind, edgespec=EdgeSpec)
            tiling = ptemp.to_tiling()
            print(label, {k: len(v) for k, v in tiling.items()})
            ptemp.visualize_tiling(ax=ax, show=False, title_suffix=label)

        fig2.suptitle(f"Rotated Planar Pipes (EdgeSpec) d={d}")
        fig2.tight_layout()
        plt.show()
