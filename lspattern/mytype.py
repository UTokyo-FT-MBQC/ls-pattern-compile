# coordinate system
"""Typed aliases for coordinates and ids (local/global frames).

This module defines type names for the six coordinate/id kinds discussed in
the codebase:

- Patch coordinates (local/global): 2D anchors where a logical patch sits.
- Physical qubit coordinates (local/global): 3D (x, y, z) coordinates used in RHG.
- Node indices (local/global): integer ids for graph nodes.

These are aliases and NewTypes to aid static checking/documentation without
changing runtime representations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Mapping, NewType, Set, Tuple

# ---------------------------------------------------------------------
# Core scalar ids (NewType for static distinction)
# ---------------------------------------------------------------------
NodeIdLocal = NewType("NodeIdLocal", int)
NodeIdGlobal = NewType("NodeIdGlobal", int)

LogicalIndex = NewType("LogicalIndex", int)
QubitIndex = NewType("QubitIndex", int)

# ---------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------
# Tiling coordinate

TilingCoord2D = NewType("TilingCoord2D", Tuple[int, int])
TilingConsistentQubitId = NewType("TilingConsistentQubitId", int)

# Patch coordinates are 2D integer anchors (x0, y0).
PatchCoordLocal2D = NewType("PatchCoordLocal2D", Tuple[int, int])
PatchCoordGlobal3D = NewType("PatchCoordGlobal3D", Tuple[int, int, int])
PipeCoordGlobal3D = NewType(
    "PipeCoordGlobal3D", Tuple[Tuple[int, int, int], Tuple[int, int, int]]
)

# Physical qubit coordinates are 3D integer positions (x, y, z).
PhysCoordLocal2D = NewType("PhysCoordLocal2D", Tuple[int, int])  # (x, y)
PhysCoordLocal3D = NewType("PhysCoordLocal3D", Tuple[int, int, int])  # (x, y, z)
PhysCoordGlobal3D = NewType("PhysCoordGlobal3D", Tuple[int, int, int])

# Convenience aliases for collections
NodeSetLocal = Set[NodeIdLocal]
NodeSetGlobal = Set[NodeIdGlobal]

# Ports and q-index mappings (LOCAL frame on blocks)
InPortsLocal = Dict[LogicalIndex, NodeSetLocal]
OutPortsLocal = Dict[LogicalIndex, NodeSetLocal]
OutQMapLocal = Dict[LogicalIndex, Dict[NodeIdLocal, QubitIndex]]

# Schedule and flow (LOCAL)
LocalTime = NewType("LocalTime", int)
ScheduleTuplesLocal = List[Tuple[LocalTime, NodeSetLocal]]
FlowLocal = Dict[NodeIdLocal, NodeSetLocal]

# Parity caps linking PREV global center to CURR local nodes
ParityCapsLocal = List[Tuple[NodeIdGlobal, List[NodeIdLocal]]]

# Block kind
BlockKindstr = tuple[str, str, str]


__all__ = [
    # ids
    "NodeIdLocal",
    "NodeIdGlobal",
    "LogicalIndex",
    "QubitIndex",
    # coords
    "PatchCoordLocal2D",
    "PatchCoordGlobal3D",
    "PipeCoordGlobal3D",
    "PhysCoordLocal3D",
    "PhysCoordGlobal3D",
    # sets/maps
    "NodeSetLocal",
    "NodeSetGlobal",
    # block IO
    "InPortsLocal",
    "OutPortsLocal",
    "OutQMapLocal",
    # schedule/flow
    "LocalTime",
    "ScheduleTuplesLocal",
    "FlowLocal",
    # parity
    "ParityCapsLocal",
    # kinds
    "BlockKindstr",
    # directions
]

# ---------------------------------------------------------------------
# Boundary enums and types (for per-side face specification)
# ---------------------------------------------------------------------


class BoundarySide(str, Enum):
    TOP = "TOP"  # +Y
    BOTTOM = "BOTTOM"  # -Y
    LEFT = "LEFT"  # -X
    RIGHT = "RIGHT"  # +X
    UP = "UP"  # +Z
    DOWN = "DOWN"  # -Z


# Allowed edge boundary value type and helper set
EdgeSpecValue = Literal["X", "Z", "O"]


class _EdgeSpecMeta(type):
    _allowed_keys = {"TOP", "BOTTOM", "LEFT", "RIGHT"}  # , "UP", "DOWN"
    _allowed_vals = {"X", "Z", "O"}
    _values: Dict[str, str] = {k: "O" for k in _allowed_keys}

    def __getattr__(cls, name: str) -> str:  # type: ignore[override]
        if name in cls._allowed_keys:
            return cls._values[name]
        raise AttributeError(name)

    def __setattr__(cls, name: str, value) -> None:  # type: ignore[override]
        # Intercept assignment to the four sides and validate
        if name in ("_allowed_keys", "_allowed_vals", "_values"):
            return super().__setattr__(name, value)
        if name in cls._allowed_keys:
            if isinstance(value, str):
                v = value.upper()
            else:
                raise TypeError("EdgeSpec values must be 'X', 'Z', or 'O' (str)")
            if v not in cls._allowed_vals:
                raise ValueError("EdgeSpec value must be one of 'X', 'Z', 'O'")
            cls._values[name] = v
            return
        # Fallback to regular class attribute set
        return super().__setattr__(name, value)

    def __call__(cls, *args, **kwargs):  # type: ignore[override]
        """Factory: allow `EdgeSpec("X","Z","Z","O")` or kwargs.

        Positional args order (4 or 6 values):
        - 4 args: (TOP, BOTTOM, LEFT, RIGHT)
        - 6 args: (TOP, BOTTOM, LEFT, RIGHT, UP, DOWN)

        Keyword args (case-insensitive keys) can specify any subset of
        TOP/BOTTOM/LEFT/RIGHT/UP/DOWN. Unspecified sides default to "O".
        Returns an instance with attribute access (e.g., `es.TOP`).
        """
        mapping: Dict[str, str] = {k: "O" for k in self._allowed_keys}  # type: ignore[name-defined]
        # Positional handling
        if args:
            if len(args) == 4:
                keys = ("TOP", "BOTTOM", "LEFT", "RIGHT")
            elif len(args) == 6:
                keys = ("TOP", "BOTTOM", "LEFT", "RIGHT", "UP", "DOWN")
            else:
                raise TypeError("EdgeSpec(...) expects 4 or 6 positional values")
            for k, v in zip(keys, args):
                if not isinstance(v, str):
                    raise TypeError("EdgeSpec values must be strings 'X'/'Z'/'O'")
                vv = v.upper()
                if vv not in self._allowed_vals:  # type: ignore[attr-defined]
                    raise ValueError("EdgeSpec value must be one of 'X','Z','O'")
                mapping[k] = vv
        # Keyword handling
        for k, v in kwargs.items():
            kk = str(k).upper()
            if kk not in self._allowed_keys:  # type: ignore[attr-defined]
                raise ValueError(f"Unknown EdgeSpec side: {k}")
            if not isinstance(v, str):
                raise TypeError("EdgeSpec values must be strings 'X'/'Z'/'O'")
            vv = v.upper()
            if vv not in self._allowed_vals:  # type: ignore[attr-defined]
                raise ValueError("EdgeSpec value must be one of 'X','Z','O'")
            mapping[kk] = vv
        return EdgeSpecInstance(mapping)


class EdgeSpec(metaclass=_EdgeSpecMeta):
    """Class-level container for per-side edge specifications.

    Usage
    -----
    - Set per-side spec:  EdgeSpec.TOP = "X"
    - Read current spec:  EdgeSpec.TOP  -> "X"
    - Allowed values: "X", "Z", "O" only.
    - Sides handled: TOP, BOTTOM, LEFT, RIGHT, UP, DOWN.

    Helper methods
    --------------
    - EdgeSpec.as_dict() -> dict[str, EdgeSpecValue]
    - EdgeSpec.update({...}) to set multiple at once.
    """

    # Provide annotations for better IDE/type hints
    TOP: str
    BOTTOM: str
    LEFT: str
    RIGHT: str
    UP: str
    DOWN: str

    @classmethod
    def as_dict(cls) -> Dict[str, EdgeSpecValue]:
        return cls._values.copy()  # type: ignore[return-value]

    @classmethod
    def update(cls, mapping: Mapping[str, str]) -> None:
        for k, v in mapping.items():
            setattr(cls, k, v)


class EdgeSpecInstance:
    """Instance-style edge spec with attribute access and validation.

    Supports attributes: TOP, BOTTOM, LEFT, RIGHT, UP, DOWN; plus
    `.as_dict()` and `.update({...})` like the class-level EdgeSpec.
    """

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        base = {k: "O" for k in _EdgeSpecMeta._allowed_keys}  # type: ignore[attr-defined]
        if values:
            for k, v in values.items():
                kk = str(k).upper()
                vv = str(v).upper()
                if kk not in _EdgeSpecMeta._allowed_keys:  # type: ignore[attr-defined]
                    raise ValueError(f"Unknown EdgeSpec side: {k}")
                if vv not in _EdgeSpecMeta._allowed_vals:  # type: ignore[attr-defined]
                    raise ValueError("EdgeSpec value must be one of 'X','Z','O'")
                base[kk] = vv
        self._values: Dict[str, str] = base

    # Properties for each side
    @property
    def TOP(self) -> str:  # noqa: N802
        return self._values["TOP"]

    @TOP.setter
    def TOP(self, v: str) -> None:  # noqa: N802
        self._values["TOP"] = self._validate(v)

    @property
    def BOTTOM(self) -> str:  # noqa: N802
        return self._values["BOTTOM"]

    @BOTTOM.setter
    def BOTTOM(self, v: str) -> None:  # noqa: N802
        self._values["BOTTOM"] = self._validate(v)

    @property
    def LEFT(self) -> str:  # noqa: N802
        return self._values["LEFT"]

    @LEFT.setter
    def LEFT(self, v: str) -> None:  # noqa: N802
        self._values["LEFT"] = self._validate(v)

    @property
    def RIGHT(self) -> str:  # noqa: N802
        return self._values["RIGHT"]

    @RIGHT.setter
    def RIGHT(self, v: str) -> None:  # noqa: N802
        self._values["RIGHT"] = self._validate(v)

    @property
    def UP(self) -> str:  # noqa: N802
        return self._values["UP"]

    @UP.setter
    def UP(self, v: str) -> None:  # noqa: N802
        self._values["UP"] = self._validate(v)

    @property
    def DOWN(self) -> str:  # noqa: N802
        return self._values["DOWN"]

    @DOWN.setter
    def DOWN(self, v: str) -> None:  # noqa: N802
        self._values["DOWN"] = self._validate(v)

    def as_dict(self) -> Dict[str, EdgeSpecValue]:
        return dict(self._values)  # type: ignore[return-value]

    def update(self, mapping: Mapping[str, str]) -> None:
        for k, v in mapping.items():
            setattr(self, str(k).upper(), v)

    @staticmethod
    def _validate(v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("EdgeSpec values must be strings 'X'/'Z'/'O'")
        vv = v.upper()
        if vv not in _EdgeSpecMeta._allowed_vals:  # type: ignore[attr-defined]
            raise ValueError("EdgeSpec value must be one of 'X','Z','O'")
        return vv

# Convenience alias (user requested Edgespec)
Edgespec = EdgeSpec


# Mapping from side to boundary spec
BoundarySpec = Dict[BoundarySide, EdgeSpecValue]

__all__ += [
    "BoundarySide",
    "EdgeSpec",
    "EdgeSpecInstance",
    "EdgeSpecValue",
    "BoundarySpec",
]
