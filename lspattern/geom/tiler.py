from __future__ import annotations

from dataclasses import dataclass, field


# ----------------------------
# Geometry primitives
# ----------------------------
@dataclass(frozen=True)
class Rect:
    """Closed-open rectangle [x0, x1) x [y0, y1) on an integer grid."""

    x0: int
    y0: int
    dx: int
    dy: int

    @property
    def x1(self) -> int:
        """Right edge (exclusive)."""
        return self.x0 + self.dx

    @property
    def y1(self) -> int:
        """Top edge (exclusive)."""
        return self.y0 + self.dy

    def moved(self, x: int, y: int) -> Rect:
        """Return a copy translated so its anchor is (x, y)."""
        return Rect(x, y, self.dx, self.dy)


def _overlap_1d(a0: int, a1: int, b0: int, b1: int, margin: int) -> bool:
    """Return True iff [a0, a1) and [b0, b1) overlap when expanded by `margin` on both sides."""
    return not (a1 + margin <= b0 - margin or b1 + margin <= a0 - margin)


def _rects_collide(a: Rect, b: Rect, margin_x: int, margin_y: int) -> bool:
    """Return True if two rectangles collide under axis-wise margins."""
    return _overlap_1d(a.x0, a.x1, b.x0, b.x1, margin_x) and _overlap_1d(a.y0, a.y1, b.y0, b.y1, margin_y)


# ----------------------------
# Patch tiler
# ----------------------------
@dataclass
class PatchTiler:
    """Assign non-overlapping (x0, y0) anchors for rectangular patches.

    Coordinates are integer grid units (user-defined). This tiler scans a grid
    (x increases fastest, then y), enforcing margins between patches. You may
    also reserve explicit positions.

    Parameters
    ----------
    pitch_x : int, default=16
        Horizontal spacing between potential patch positions in grid units.
    pitch_y : int, default=16
        Vertical spacing between potential patch positions in grid units.
    margin_x : int, default=0
        Minimum horizontal margin required between patches in grid units.
    margin_y : int, default=0
        Minimum vertical margin required between patches in grid units.

    Attributes
    ----------
    pitch_x : int
        Horizontal spacing between potential patch positions.
    pitch_y : int
        Vertical spacing between potential patch positions.
    margin_x : int
        Minimum horizontal margin required between patches.
    margin_y : int
        Minimum vertical margin required between patches.

    Example
    -------
    >>> tiler = PatchTiler(pitch_x=16, pitch_y=16, margin_x=2, margin_y=2)
    >>> x0, y0 = tiler.alloc(logical=0, dx=7, dy=7)
    >>> tiler.reserve(1, x0=40, y0=0, dx=7, dy=7)
    """

    pitch_x: int = 16
    pitch_y: int = 16
    margin_x: int = 0
    margin_y: int = 0
    _occupied: dict[int, Rect] = field(default_factory=dict)
    _scan_limit: int = 10_000  # safety cap on search iterations

    # -------------------
    # Public API
    # -------------------
    def alloc(self, logical: int, dx: int, dy: int, *, prefer_row: int = 0) -> tuple[int, int]:
        """Find a free anchor (x0, y0) to place a dx-by-dy patch for `logical`.

        The anchor is the lower-left corner. Raises ValueError if no spot is found
        within the scan limit.

        Parameters
        ----------
        logical : int
            Logical index identifier for the patch.
        dx : int
            Width of the patch in grid units.
        dy : int
            Height of the patch in grid units.
        prefer_row : int, optional
            Starting row preference (y = prefer_row * pitch_y), by default 0.

        Returns
        -------
        tuple[int, int]
            Anchor position (x0, y0) where the patch was placed.

        Raises
        ------
        ValueError
            If the logical index is already occupied or no suitable position
            is found within the scan limit.
        """
        if logical in self._occupied:
            msg = f"logical {logical} is already placed at {self._occupied[logical]}"
            raise ValueError(msg)

        rect = Rect(0, 0, dx, dy)
        x = 0
        y = prefer_row * self.pitch_y

        steps = 0
        while steps < self._scan_limit:
            steps += 1
            cand = rect.moved(x, y)
            if self._fits(cand):
                self._occupied[logical] = cand
                return cand.x0, cand.y0

            # advance scan
            x += self.pitch_x
            # wrap row if x passes a heuristic frontier based on current max x
            if x > self._max_x() + self.pitch_x * 4:
                x = 0
                y += self.pitch_y

        msg = "PatchTiler.alloc: failed to find space (increase scan_limit or adjust pitch/margins)."
        raise ValueError(msg)

    def reserve(self, logical: int, *, x0: int, y0: int, dx: int, dy: int) -> None:
        """Reserve an explicit rectangle for a logical index (raises if it collides).

        Parameters
        ----------
        logical : int
            Logical index identifier for the patch.
        x0 : int
            Left edge (anchor x-coordinate) of the rectangle.
        y0 : int
            Bottom edge (anchor y-coordinate) of the rectangle.
        dx : int
            Width of the rectangle in grid units.
        dy : int
            Height of the rectangle in grid units.

        Raises
        ------
        ValueError
            If the requested rectangle collides with existing patches or
            if the logical index is already occupied.
        """
        rect = Rect(x0, y0, dx, dy)
        if not self._fits(rect):
            msg = f"Requested reservation collides with existing patches: {rect}"
            raise ValueError(msg)
        if logical in self._occupied:
            msg = f"logical {logical} is already placed at {self._occupied[logical]}"
            raise ValueError(msg)
        self._occupied[logical] = rect

    def get(self, logical: int) -> Rect:
        """Return the reserved/allocated rectangle for `logical`.

        Parameters
        ----------
        logical : int
            Logical index identifier for the patch.

        Returns
        -------
        Rect
            Rectangle object representing the occupied patch.

        Raises
        ------
        KeyError
            If the logical index is not found in occupied patches.
        """
        return self._occupied[logical]

    def release(self, logical: int) -> None:
        """Release a previously reserved/allocated rectangle.

        Parameters
        ----------
        logical : int
            Logical index identifier for the patch to release.

        Notes
        -----
        This method silently does nothing if the logical index is not found.
        """
        self._occupied.pop(logical, None)

    def list_occupied(self) -> list[tuple[int, Rect]]:
        """List occupied patches as (logical, Rect), ordered by (y0, x0).

        Returns
        -------
        list[tuple[int, Rect]]
            List of tuples containing logical index and rectangle pairs,
            sorted first by y-coordinate, then by x-coordinate.
        """
        return sorted(self._occupied.items(), key=lambda kv: (kv[1].y0, kv[1].x0))

    def bbox(self) -> Rect | None:
        """Return the bounding box covering all occupied patches (or None if empty).

        Returns
        -------
        Rect or None
            Rectangle representing the minimal bounding box that covers all
            occupied patches. Returns None if no patches are occupied.
        """
        if not self._occupied:
            return None
        xs0 = min(r.x0 for r in self._occupied.values())
        ys0 = min(r.y0 for r in self._occupied.values())
        xs1 = max(r.x1 for r in self._occupied.values())
        ys1 = max(r.y1 for r in self._occupied.values())
        return Rect(xs0, ys0, xs1 - xs0, ys1 - ys0)

    # -------------------
    # Internals
    # -------------------
    def _fits(self, cand: Rect) -> bool:
        """Return True if `cand` does not collide with any occupied rectangle."""
        return all(not _rects_collide(cand, r, self.margin_x, self.margin_y) for r in self._occupied.values())

    def _max_x(self) -> int:
        """Return the current maximum x1 among occupied rectangles (0 if none)."""
        return max((r.x1 for r in self._occupied.values()), default=0)
