from dataclasses import dataclass, field
from itertools import count

from lspattern.mytype import PatchCoordGlobal3D, QubitIndex, TilingCoord2D, TilingId
from lspattern.consts.consts import DIRECTIONS2D


def _next_tiling_id() -> int:
    """Return the next globally unique Tiling id.

    Increments by 1 for every Tiling (and subclass) instantiation.
    Starts from 1.
    """
    return next(_TILING_ID_COUNTER)


def reset_tiling_id_counter(start_at: int = 1) -> None:
    """Reset the global Tiling id counter (primarily for tests)."""
    global _TILING_ID_COUNTER
    _TILING_ID_COUNTER = count(int(start_at))


# Global counter shared by Tiling and all subclasses
_TILING_ID_COUNTER = count(1)


@dataclass
class Tiling:
    """
    Base class for physical qubit tiling patterns.
    """

    id_: TilingId = field(init=False)  # unique identifier, auto-assigned on init
    data_coords: list[TilingCoord2D] = field(default_factory=list)
    coord2qubitindex: dict[TilingCoord2D, QubitIndex] = field(default_factory=dict)
    coord2id: dict[TilingCoord2D, int] = field(default_factory=dict)

    x_coords: list[TilingCoord2D] = field(default_factory=list)
    z_coords: list[TilingCoord2D] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Assign a globally unique, incrementing id
        self.id_ = _next_tiling_id()

    def set_ids(self, id_: int) -> None:
        """
        Set the same group id for all coordinates in coord2id.

        Parameters
        ----------
        id_ : int
            The group id to assign to all coordinates.
        """
        self.coord2id = dict.fromkeys(self.coord2id, id_)

    def shift_qubit_indices(self, by: int) -> None:
        """Shift assigned qubit indices by `by` in-place.

        Operates on `coord2qubitindex` which maps tiling coordinates to
        contiguous qubit indices. If no indices are assigned yet, this is a no-op.
        """
        if not self.coord2qubitindex:
            return
        delta = int(by)
        self.coord2qubitindex = {
            c: QubitIndex(int(qi) + delta) for c, qi in self.coord2qubitindex.items()
        }


# @dataclass(init=False)
# class ConnectedTiling(Tiling):
#     """
#     Combine multiple tilings into a single connected tiling.

#     - Concatenates data/X/Z coordinates across parts.
#     - Optionally shifts qubit indices to avoid collisions by making them
#       contiguous across parts in the given order.
#     - Optionally checks for coordinate collisions (within-type duplicates and
#       across-type overlaps) and raises ValueError if found.

#     Usage:
#         tilings = [tile1, tile2, tile3]
#         ct = ConnectedTiling(tilings, shift_qubits=True, check_collisions=True)
#     """

#     cube_parts: dict[PatchCoordGlobal3D, Tiling] = field(default_factory=dict)
#     pipe_parts: dict[tuple[PatchCoordGlobal3D, PatchCoordGlobal3D], Tiling] = field(default_factory=dict)
#     node_maps: dict[PatchCoordGlobal3D, dict[TilingCoord2D, QubitIndex], int] = field(default_factory=dict)

#     def __init__(
#         self,
#         cube_tilings: dict[PatchCoordGlobal3D, Tiling],
#         pipe_tilings: dict[tuple[PatchCoordGlobal3D, PatchCoordGlobal3D], Tiling],
#         *,
#         check_collisions: bool = True,
#     ) -> None:
#         # Assign unique id for this ConnectedTiling instance as well
#         self.id_ = _next_tiling_id()

#         # Keep references to parts; do not mutate them
#         self.cube_parts = cube_tilings
#         self.pipe_parts = pipe_tilings

#         self.node_maps = {}
#         self.coord2qubitindex = {}
#         self.coord2id = {}

#         # Gather coordinates
#         data_list: list[TilingCoord2D] = []
#         x_list: list[TilingCoord2D] = []
#         z_list: list[TilingCoord2D] = []

#         data_set: set[TilingCoord2D] = set()
#         x_set: set[TilingCoord2D] = set()
#         z_set: set[TilingCoord2D] = set()

#         for t in [*cube_tilings.values(), *pipe_tilings.values()]:
#             if t.data_coords:
#                 data_list.extend(t.data_coords)
#                 data_set.update(t.data_coords)
#             if t.x_coords:
#                 x_list.extend(t.x_coords)
#                 x_set.update(t.x_coords)
#             if t.z_coords:
#                 z_list.extend(t.z_coords)
#                 z_set.update(t.z_coords)

#         if check_collisions:
#             _check_collisions_and_raise(
#                 data_list,
#                 x_list,
#                 z_list,
#                 data_set,
#                 x_set,
#                 z_set,
#             )

#         # Stable de-duplication while preserving part order
#         self.data_coords = list(dict.fromkeys(data_list))
#         self.x_coords = list(dict.fromkeys(x_list))
#         self.z_coords = list(dict.fromkeys(z_list))

#         # Precompute index maps (O(n)) to avoid repeated list.index (O(n^2))
#         data_idx = {c: i for i, c in enumerate(self.data_coords)}
#         x_idx = {c: i for i, c in enumerate(self.x_coords)}
#         z_idx = {c: i for i, c in enumerate(self.z_coords)}

#         # Build flat coord -> contiguous qubit index map
#         base_x = len(self.data_coords)
#         base_z = base_x + len(self.x_coords)
#         self.coord2qubitindex.update({c: QubitIndex(i) for c, i in data_idx.items()})
#         self.coord2qubitindex.update({c: QubitIndex(base_x + i) for c, i in x_idx.items()})
#         self.coord2qubitindex.update({c: QubitIndex(base_z + i) for c, i in z_idx.items()})

#         # Merge cube and pipe tiling ids to minimum of the connected group
#         # Build coord2id (patch grouping) later, after pipes are processed
#         # TODO: Implement this algorithm キョウプロかんがある
#         # アルゴリズムは以下の通り
#         """
#         for (source_pos, sink_pos), p in pipe_tilings.items():
#             min_id = min(
#                 cube_tilings[source_pos].id_, cube_tilings[sink_pos].id_, p.id_
#             )
#             cube_tilings[source_pos].set_ids(min_id)
#             cube_tilings[sink_pos].set_ids(min_id)
#             p.set_ids(min_id)
#         end
#         """
#         # このアルゴリズムには問題があり、それは
#         # ...です
#         self.coord2id = {}
#         for p in pipes:
#             # ここsource ,sinkは元のTilingには入っていないが、
#             # "同じpatch groupならくっつける"をどうやって実装しようか悩む
#             source, sink = p.source, p.sink
#         for c in cubes:
#             self.coord2id.update(c.coord2id)
#         for p in pipes:
#             self.coord2id.update(p.coord2id)

#         #

#         # Fast node maps using the precomputed index maps
#         self.node_maps = {
#             "data": {c: data_idx[c] for t in self.parts for c in t.data_coords},
#             "x": {c: x_idx[c] for t in self.parts for c in t.x_coords},
#             "z": {c: z_idx[c] for t in self.parts for c in t.z_coords},
#         }


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
        raise ValueError(
            "ConnectedTiling coordinate collisions: " + "; ".join(problems)
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
