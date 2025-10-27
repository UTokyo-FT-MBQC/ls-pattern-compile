from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING

from lspattern.consts.consts import (
    ANCILLA_X_PARITY,
    ANCILLA_Z_PARITY,
    DATA_PARITIES,
    PIPEDIRECTION,
    NodeRole,
    EdgeSpecValue,
    BoundarySide,
)
from lspattern.mytype import (
    PatchCoordGlobal3D,
    QubitGroupIdLocal,
    TilingId,
)

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet


def to_edgespec(espec_str: str) -> dict[BoundarySide, EdgeSpecValue]:
    """Decode a four-character edge specification into boundary assignments.

    The string is interpreted in left, right, top, bottom order and accepts
    the characters ``O``, ``X``, or ``Z`` in any case. Each character maps to
    the corresponding ``EdgeSpecValue`` enumerator.

    Args:
        espec_str: Four-character boundary description string.

    Returns:
        Dictionary mapping each ``BoundarySide`` to its ``EdgeSpecValue``.

    Raises:
        AssertionError: If ``espec_str`` is not exactly four characters long.
        ValueError: If an unsupported character is provided.
    """
    assert len(espec_str) == 4, "Edge spec string must have length 4"

    espec_values: list[EdgeSpecValue] = []

    for char in espec_str:
        char = char.upper()
        match char:
            case "O":
                espec_values.append(EdgeSpecValue.O)
            case "X":
                espec_values.append(EdgeSpecValue.X)
            case "Z":
                espec_values.append(EdgeSpecValue.Z)
            case _:
                msg = f"Invalid edge spec character: {char}"
                raise ValueError(msg)

    # left, right, top, bottom
    ret = {
        BoundarySide.LEFT: espec_values[0],
        BoundarySide.RIGHT: espec_values[1],
        BoundarySide.TOP: espec_values[2],
        BoundarySide.BOTTOM: espec_values[3],
    }

    return ret

def get_direction(
    source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D
) -> PIPEDIRECTION:
    dx = sink[0] - source[0]
    dy = sink[1] - source[1]
    dz = sink[2] - source[2]

    match (dx, dy, dz):
        case (1, 0, 0):
            return PIPEDIRECTION.RIGHT
        case (-1, 0, 0):
            return PIPEDIRECTION.LEFT
        case (0, 1, 0):
            return PIPEDIRECTION.TOP
        case (0, -1, 0):
            return PIPEDIRECTION.BOTTOM
        case (0, 0, 1):
            return PIPEDIRECTION.UP
        case (0, 0, -1):
            return PIPEDIRECTION.DOWN
        case _:
            msg = "Invalid direction"
            raise ValueError(msg)


# Prepare outputs as sorted lists for determinism
def sort_xy(points: AbstractSet[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(points, key=itemgetter(1, 0))


def is_allowed_pair(
    u: QubitGroupIdLocal | TilingId | None,
    v: QubitGroupIdLocal | TilingId | None,
    allowed_pairs: (
        AbstractSet[tuple[QubitGroupIdLocal, QubitGroupIdLocal]]
        | AbstractSet[tuple[TilingId, TilingId]]
    ),
) -> bool:
    """Return True if an (unordered) pair is allowed.

    - None-safe: if either ``u`` or ``v`` is None, returns False.
    - Normalizes to ``int`` for comparison to be robust against NewType wrappers.
    """
    if u is None or v is None:
        return False
    try:
        uu, vv = int(u), int(v)
    except (ValueError, TypeError):
        return False
    # empty set => nothing allowed
    if not allowed_pairs:
        return False
    m, max_val = (uu, vv) if uu <= vv else (vv, uu)
    return (m, max_val) in allowed_pairs


class UnionFind:
    """Simple Union-Find (Disjoint Set Union) with path compression.

    Unions always attach the larger representative to the smaller one to keep
    group ids deterministic (min representative), matching prior behavior.
    """

    parent: dict[int, int]

    def __init__(self) -> None:
        self.parent = {}

    def add(self, a: int) -> None:
        a = int(a)
        if a not in self.parent:
            self.parent[a] = a

    def find(self, a: int) -> int:
        a = int(a)
        self.add(a)
        p = self.parent[a]
        if p != a:
            self.parent[a] = self.find(p)
        return self.parent[a]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        m = min(ra, rb)
        self.parent[ra] = m
        self.parent[rb] = m


def infer_role(coord: tuple[int, int, int]) -> NodeRole:
    parity = (coord[0] & 1, coord[1] & 1, coord[2] & 1)
    fg_data = parity in DATA_PARITIES
    fg_ancz = parity in ANCILLA_Z_PARITY
    fg_ancx = parity in ANCILLA_X_PARITY

    match (fg_data, fg_ancx, fg_ancz):
        case (True, False, False):
            return NodeRole.DATA
        case (False, True, False):
            return NodeRole.ANCILLA_X
        case (False, False, True):
            return NodeRole.ANCILLA_Z
        case _:
            msg = f"Cannot infer role from coord {coord}"
            raise ValueError(msg)


if __name__ == "__main__":
    # Simple self-test for utils
    print("[utils] Running self-test...")

    # Test UnionFind determinism and connectivity
    uf = UnionFind()
    for i in range(1, 6):
        uf.add(i)
    uf.union(1, 3)
    uf.union(3, 5)
    uf.union(2, 4)
    # Representatives should be minimal in each set
    r135 = {uf.find(1), uf.find(3), uf.find(5)}
    r24 = {uf.find(2), uf.find(4)}
    if not (len(r135) == 1 and min(r135) == 1):
        msg = f"UF group {r135} should be rep=1"
        raise AssertionError(msg)
    EXPECTED_REP2 = 2
    if not (len(r24) == 1 and min(r24) == EXPECTED_REP2):
        msg = f"UF group {r24} should be rep=2"
        raise AssertionError(msg)

    # Test get_direction
    p0: PatchCoordGlobal3D = PatchCoordGlobal3D((0, 0, 0))
    px: PatchCoordGlobal3D = PatchCoordGlobal3D((1, 0, 0))
    py: PatchCoordGlobal3D = PatchCoordGlobal3D((0, 1, 0))
    pm: PatchCoordGlobal3D = PatchCoordGlobal3D((1, 1, -1))
    p11: PatchCoordGlobal3D = PatchCoordGlobal3D((1, 1, 0))
    if get_direction(p0, px).name != "RIGHT":
        msg = "direction RIGHT failed"
        raise AssertionError(msg)
    if get_direction(p0, py).name != "TOP":
        msg = "direction TOP failed"
        raise AssertionError(msg)
    if get_direction(p11, pm).name != "DOWN":
        msg = "direction DOWN failed"
        raise AssertionError(msg)

    # Test is_allowed_pair
    allow: set[tuple[QubitGroupIdLocal, QubitGroupIdLocal]] = {
        (QubitGroupIdLocal(1), QubitGroupIdLocal(2)),
        (QubitGroupIdLocal(3), QubitGroupIdLocal(3)),
    }
    if not is_allowed_pair(QubitGroupIdLocal(1), QubitGroupIdLocal(2), allow):
        msg = "pair (1,2) should be allowed"
        raise AssertionError(msg)
    if not is_allowed_pair(QubitGroupIdLocal(2), QubitGroupIdLocal(1), allow):
        msg = "pair (2,1) should be allowed"
        raise AssertionError(msg)
    if not is_allowed_pair(QubitGroupIdLocal(3), QubitGroupIdLocal(3), allow):
        msg = "pair (3,3) should be allowed"
        raise AssertionError(msg)
    if is_allowed_pair(QubitGroupIdLocal(1), QubitGroupIdLocal(3), allow):
        msg = "pair (1,3) should not be allowed"
        raise AssertionError(msg)

    print("[utils] All tests passed.")
