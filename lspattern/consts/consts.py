# This class stores some constants

# 2D spatial orientations
from __future__ import annotations

import enum

DIRECTIONS2D: list[tuple[int, int]] = [
    (-1, -1),  # down left
    (-1, +1),  # down right
    (+1, -1),  # up left
    (+1, +1),  # up right
]


# 3D spatial orientations
DIRECTIONS3D: list[tuple[int, int, int]] = [
    # left, right, up, down, diagonal upright, upleft, downright, downleft
    # (-1, 0, 0),  # left
    # (1, 0, 0),  # right
    # (0, 1, 0),  # up
    # (0, -1, 0),  # down
    (1, 1, 0),  # diagonal upright
    (-1, 1, 0),  # diagonal upleft
    (1, -1, 0),  # diagonal downright
    (-1, -1, 0),  # diagonal downleft
]


class PIPEDIRECTION(enum.Enum):
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    UP = 4
    DOWN = 5


# ---------------------------------------------------------------------
# Type-safe string enums for constants used across the codebase
# Using str mixin allows enums to be used in string contexts
# ---------------------------------------------------------------------


class EdgeSpecValue(str, enum.Enum):  # noqa: UP042
    """Edge specification values for quantum patch boundaries.

    X: X-type stabilizer boundary
    Z: Z-type stabilizer boundary
    O: Open or trimmed boundary
    """

    X = "X"
    Z = "Z"
    O = "O"  # noqa: E741


class BoundarySide(str, enum.Enum):  # noqa: UP042
    """Spatial boundary sides for patch edges.

    Used to specify which side of a patch a boundary condition applies to.
    """

    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


class NodeRole(str, enum.Enum):  # noqa: UP042
    """Role of a node in the RHG lattice.

    DATA: Logical data qubit
    ANCILLA_X: X-type ancilla qubit
    ANCILLA_Z: Z-type ancilla qubit
    """

    DATA = "data"
    ANCILLA_X = "ancilla_x"
    ANCILLA_Z = "ancilla_z"


class CoordinateSystem(str, enum.Enum):  # noqa: UP042
    """Coordinate system identifiers for spatial transformations.

    TILING_2D: 2D tiling coordinate system
    PHYS_3D: 3D physical qubit coordinate system
    PATCH_3D: 3D patch coordinate system
    """

    TILING_2D = "tiling2d"
    PHYS_3D = "phys3d"
    PATCH_3D = "patch3d"


class VisualizationKind(str, enum.Enum):  # noqa: UP042
    """Visualization kind for parity and flow visualizers.

    BOTH: Show both X and Z components
    X: Show only X components
    Z: Show only Z components
    """

    BOTH = "both"
    X = "x"
    Z = "z"


class VisualizationMode(str, enum.Enum):  # noqa: UP042
    """Visualization mode for schedule visualizers.

    HIST: Histogram mode
    SLICES: Slice-by-slice mode
    """

    HIST = "hist"
    SLICES = "slices"


class InitializationState(str, enum.Enum):  # noqa: UP042
    """Initialization state for quantum state preparation.

    PLUS: |+⟩ state initialization
    ZERO: |0⟩ state initialization
    """

    PLUS = "plus"
    ZERO = "zero"


class Observable(str, enum.Enum):  # noqa: UP042
    """Observable type for quantum measurements.

    X: X-type Pauli observable
    Z: Z-type Pauli observable
    """

    X = "X"
    Z = "Z"
