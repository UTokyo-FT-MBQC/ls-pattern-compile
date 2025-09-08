from dataclasses import dataclass, field
from itertools import count

from lspattern.mytype import QubitGroupIdLocal, QubitIndex, TilingCoord2D


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

    id_: QubitGroupIdLocal = field(init=False)  # unique identifier, auto-assigned on init
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
        self.coord2qubitindex = {c: QubitIndex(int(qi) + delta) for c, qi in self.coord2qubitindex.items()}
