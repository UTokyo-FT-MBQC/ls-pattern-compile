
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Optional, List


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    dx: int
    dy: int

    @property
    def x1(self) -> int:
        return self.x0 + self.dx

    @property
    def y1(self) -> int:
        return self.y0 + self.dy

    def moved(self, x: int, y: int) -> "Rect":
        return Rect(x, y, self.dx, self.dy)


def _overlap_1d(a0: int, a1: int, b0: int, b1: int, margin: int) -> bool:
    # intervals [a0-margin, a1+margin] and [b0-margin, b1+margin] overlap?
    return not (a1 + margin <= b0 - margin or b1 + margin <= a0 - margin)


def _rects_collide(a: Rect, b: Rect, margin_x: int, margin_y: int) -> bool:
    return _overlap_1d(a.x0, a.x1, b.x0, b.x1, margin_x) and _overlap_1d(a.y0, a.y1, b.y0, b.y1, margin_y)


@dataclass
class PatchTiler:
    """Assigns non-overlapping (x0, y0) anchor positions for rectangular patches.

    Coordinates are integer grid units (user-defined). This tiler is simple but robust:
      - Places patches on a rectilinear scan (x increases fastest, then y).
      - Enforces margins between patches (margin_x, margin_y).
      - Allows explicit reservation and release.

    Typical usage:
        tiler = PatchTiler(pitch_x=16, pitch_y=16, margin_x=2, margin_y=2)
        x0, y0 = tiler.alloc(logical=0, dx=7, dy=7)  # returns a free anchor
        tiler.reserve(1, x0=40, y0=0, dx=7, dy=7)    # explicit placement

    """
    pitch_x: int = 16
    pitch_y: int = 16
    margin_x: int = 0
    margin_y: int = 0
    _occupied: Dict[int, Rect] = field(default_factory=dict)
    _scan_limit: int = 10_000  # safety cap on search iterations

    # -------------------
    # public API
    # -------------------
    def alloc(self, logical: int, dx: int, dy: int, *, prefer_row: int = 0) -> Tuple[int, int]:
        """Find a free anchor (x0, y0) to place a dx-by-dy patch for `logical`.

        The anchor is the lower-left corner of the rectangle (x0, y0).
        Raises ValueError if cannot find a spot within scan limit.
        """
        if logical in self._occupied:
            raise ValueError(f"logical {logical} is already placed at {self._occupied[logical]}")

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
            # wrap row if too crowded: heuristic based on current max x
            if x > self._max_x() + self.pitch_x * 4:
                x = 0
                y += self.pitch_y

        raise ValueError("PatchTiler.alloc: failed to find space (increase scan_limit or adjust pitch/margins).")

    def reserve(self, logical: int, *, x0: int, y0: int, dx: int, dy: int) -> None:
        """Reserve an explicit rectangle for a logical index (raises if collides)."""
        rect = Rect(x0, y0, dx, dy)
        if not self._fits(rect):
            raise ValueError(f"Requested reservation collides with existing patches: {rect}")
        if logical in self._occupied:
            raise ValueError(f"logical {logical} is already placed at {self._occupied[logical]}")
        self._occupied[logical] = rect

    def get(self, logical: int) -> Rect:
        return self._occupied[logical]

    def release(self, logical: int) -> None:
        self._occupied.pop(logical, None)

    def list_occupied(self) -> List[Tuple[int, Rect]]:
        return sorted(self._occupied.items(), key=lambda kv: (kv[1].y0, kv[1].x0))

    def bbox(self) -> Optional[Rect]:
        """Return the bounding box covering all occupied patches (or None if empty)."""
        if not self._occupied:
            return None
        xs0 = min(r.x0 for r in self._occupied.values())
        ys0 = min(r.y0 for r in self._occupied.values())
        xs1 = max(r.x1 for r in self._occupied.values())
        ys1 = max(r.y1 for r in self._occupied.values())
        return Rect(xs0, ys0, xs1 - xs0, ys1 - ys0)

    # -------------------
    # internals
    # -------------------
    def _fits(self, cand: Rect) -> bool:
        for r in self._occupied.values():
            if _rects_collide(cand, r, self.margin_x, self.margin_y):
                return False
        return True

    def _max_x(self) -> int:
        if not self._occupied:
            return 0
        return max(r.x1 for r in self._occupied.values())
