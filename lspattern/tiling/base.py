from dataclasses import dataclass, field

from lspattern.mytype import QubitIndex, TilingCoord2D


@dataclass
class Tiling:
    """
    Base class for physical qubit tiling patterns.
    """

    data_coords: list[TilingCoord2D] = field(default_factory=list)
    # TODO: do it
    # Tilingを継承しているすべてのclassでqubit indexをcoord2qubitindexに置き換える
    coord2qubitindex: dict[TilingCoord2D, QubitIndex] = field(default_factory=dict)

    x_coords: list[TilingCoord2D] = field(default_factory=list)
    z_coords: list[TilingCoord2D] = field(default_factory=list)

    def shift_qubit_indices(self, by: int):
        """Shift assigned qubit indices by `by` in-place.

        Operates on `coord2qubitindex` which maps tiling coordinates to
        contiguous qubit indices. If no indices are assigned yet, this is a no-op.
        """
        if not self.coord2qubitindex:
            return
        delta = int(by)
        self.coord2qubitindex = {c: QubitIndex(int(qi) + delta) for c, qi in self.coord2qubitindex.items()}


@dataclass(init=False)
class ConnectedTiling(Tiling):
    """
    Combine multiple tilings into a single connected tiling.

    - Concatenates data/X/Z coordinates across parts.
    - Optionally shifts qubit indices to avoid collisions by making them
      contiguous across parts in the given order.
    - Optionally checks for coordinate collisions (within-type duplicates and
      across-type overlaps) and raises ValueError if found.

    Usage:
        tilings = [tile1, tile2, tile3]
        ct = ConnectedTiling(tilings, shift_qubits=True, check_collisions=True)
    """

    parts: list[Tiling]

    def __init__(
        self,
        tilings: list[Tiling] | tuple[Tiling, ...],
        *,
        check_collisions: bool = True,
    ) -> None:
        # Keep references to parts; do not mutate them
        self.parts = list(tilings)
        self.node_maps: dict[str, dict[TilingCoord2D, QubitIndex]] = {}
        self.coord2qubitindex: dict[TilingCoord2D, QubitIndex] = {}

        # Gather coordinates
        data_list: list[TilingCoord2D] = []
        x_list: list[TilingCoord2D] = []
        z_list: list[TilingCoord2D] = []

        data_set: set[TilingCoord2D] = set()
        x_set: set[TilingCoord2D] = set()
        z_set: set[TilingCoord2D] = set()

        for t in self.parts:
            if t.data_coords:
                data_list.extend(t.data_coords)
                data_set.update(t.data_coords)
            if t.x_coords:
                x_list.extend(t.x_coords)
                x_set.update(t.x_coords)
            if t.z_coords:
                z_list.extend(t.z_coords)
                z_set.update(t.z_coords)

        # TODO: 関数の外に出してコードをきれいにする
        if check_collisions:
            _check_collisions_and_raise(
                data_list,
                x_list,
                z_list,
                data_set,
                x_set,
                z_set,
            )

        # Stable de-duplication while preserving part order
        self.data_coords = list(dict.fromkeys(data_list))
        self.x_coords = list(dict.fromkeys(x_list))
        self.z_coords = list(dict.fromkeys(z_list))

        # Precompute index maps (O(n)) to avoid repeated list.index (O(n^2))
        data_idx = {c: i for i, c in enumerate(self.data_coords)}
        x_idx = {c: i for i, c in enumerate(self.x_coords)}
        z_idx = {c: i for i, c in enumerate(self.z_coords)}

        # Build flat coord -> contiguous qubit index map
        base_x = len(self.data_coords)
        base_z = base_x + len(self.x_coords)
        self.coord2qubitindex.update({c: QubitIndex(i) for c, i in data_idx.items()})
        self.coord2qubitindex.update({c: QubitIndex(base_x + i) for c, i in x_idx.items()})
        self.coord2qubitindex.update({c: QubitIndex(base_z + i) for c, i in z_idx.items()})

        # Fast node maps using the precomputed index maps
        self.node_maps = {
            "data": {c: data_idx[c] for t in self.parts for c in t.data_coords},
            "x": {c: x_idx[c] for t in self.parts for c in t.x_coords},
            "z": {c: z_idx[c] for t in self.parts for c in t.z_coords},
        }


def _find_duplicates(seq: list[TilingCoord2D]) -> set[TilingCoord2D]:
    seen: set[TilingCoord2D] = set()
    dups: set[TilingCoord2D] = set()
    for item in seq:
        if item in seen:
            dups.add(item)
        else:
            seen.add(item)
    return dups


def _check_collisions_and_raise(
    data_list: list[TilingCoord2D],
    x_list: list[TilingCoord2D],
    z_list: list[TilingCoord2D],
    data_set: set[TilingCoord2D],
    x_set: set[TilingCoord2D],
    z_set: set[TilingCoord2D],
) -> None:
    dup_data = _find_duplicates(data_list)
    dup_x = _find_duplicates(x_list)
    dup_z = _find_duplicates(z_list)

    overlap_dx = data_set & x_set
    overlap_dz = data_set & z_set
    overlap_xz = x_set & z_set

    if dup_data or dup_x or dup_z or overlap_dx or overlap_dz or overlap_xz:
        problems: list[str] = []
        if dup_data:
            problems.append(f"duplicate data coords: {sorted(dup_data)}")
        if dup_x:
            problems.append(f"duplicate X coords: {sorted(dup_x)}")
        if dup_z:
            problems.append(f"duplicate Z coords: {sorted(dup_z)}")
        if overlap_dx:
            problems.append(f"data/X overlap: {sorted(overlap_dx)}")
        if overlap_dz:
            problems.append(f"data/Z overlap: {sorted(overlap_dz)}")
        if overlap_xz:
            problems.append(f"X/Z overlap: {sorted(overlap_xz)}")
        raise ValueError("ConnectedTiling coordinate collisions: " + "; ".join(problems))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
