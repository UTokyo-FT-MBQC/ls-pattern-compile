"""Global corner ancilla analysis for canvas-wide coordination.

This module provides canvas-wide analysis of corner ancillas to determine
which far corner ancillas should be removed based on global topology.

In rotated surface codes, corner ancillas (positioned outside the main bounds)
may need to be removed when adjacent cubes/pipes share those positions.
This module analyzes the global canvas layout to make coordinated decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

from lspattern.consts import BoundarySide, EdgeSpecValue

if TYPE_CHECKING:
    from lspattern.canvas_loader import CanvasCubeSpec, CanvasPipeSpec, CanvasSpec
    from lspattern.mytype import Coord2D, Coord3D


class CornerPosition(NamedTuple):
    """Corner identifier (combination of two boundary sides).

    A corner is defined by the intersection of two perpendicular boundary sides.
    For example, TOP + LEFT defines the top-left corner.

    Attributes
    ----------
    side1 : BoundarySide
        First boundary side (TOP, BOTTOM, LEFT, RIGHT).
    side2 : BoundarySide
        Second boundary side (TOP, BOTTOM, LEFT, RIGHT).
        Must be perpendicular to side1.

    Examples
    --------
    >>> corner = CornerPosition(BoundarySide.TOP, BoundarySide.LEFT)
    >>> corner.side1
    <BoundarySide.TOP: 'TOP'>
    """

    side1: BoundarySide
    side2: BoundarySide

    def normalized(self) -> CornerPosition:
        """Return a normalized version with sides in canonical order.

        Canonical order ensures TOP/BOTTOM comes before LEFT/RIGHT.

        Returns
        -------
        CornerPosition
            Normalized corner position.
        """
        tb = {BoundarySide.TOP, BoundarySide.BOTTOM}
        if self.side1 in tb:
            return self
        if self.side2 in tb:
            return CornerPosition(self.side2, self.side1)
        # Both are LEFT/RIGHT - keep as-is (shouldn't happen for valid corners)
        return self


# All valid corner positions in normalized form
CORNER_TOP_LEFT = CornerPosition(BoundarySide.TOP, BoundarySide.LEFT)
CORNER_TOP_RIGHT = CornerPosition(BoundarySide.TOP, BoundarySide.RIGHT)
CORNER_BOTTOM_LEFT = CornerPosition(BoundarySide.BOTTOM, BoundarySide.LEFT)
CORNER_BOTTOM_RIGHT = CornerPosition(BoundarySide.BOTTOM, BoundarySide.RIGHT)

ALL_CORNERS: tuple[CornerPosition, ...] = (
    CORNER_TOP_LEFT,
    CORNER_TOP_RIGHT,
    CORNER_BOTTOM_LEFT,
    CORNER_BOTTOM_RIGHT,
)


@dataclass(frozen=True, slots=True)
class CornerAncillaDecision:
    """Decision for a specific corner's far ancillas.

    Each corner can have up to two far ancillas (X and Z type) that may need
    to be removed or added based on global canvas topology.

    Attributes
    ----------
    remove_far_x_ancilla : bool
        If True, remove the far X-type ancilla at this corner.
    remove_far_z_ancilla : bool
        If True, remove the far Z-type ancilla at this corner.
    has_x_ancilla : bool
        If True, an X-type ancilla should exist at this corner.
    has_z_ancilla : bool
        If True, a Z-type ancilla should exist at this corner.
    """

    remove_far_x_ancilla: bool = False
    remove_far_z_ancilla: bool = False
    has_x_ancilla: bool = False
    has_z_ancilla: bool = False

    def with_removal(self, ancilla_type: Literal["x", "z"], *, remove: bool = True) -> CornerAncillaDecision:
        """Return a new decision with the specified ancilla type marked for removal.

        Parameters
        ----------
        ancilla_type : Literal["x", "z"]
            The type of ancilla to mark for removal.
        remove : bool
            Whether to remove (True) or keep (False) the ancilla.

        Returns
        -------
        CornerAncillaDecision
            New decision with updated removal flag.
        """
        if ancilla_type == "x":
            return CornerAncillaDecision(
                remove_far_x_ancilla=remove,
                remove_far_z_ancilla=self.remove_far_z_ancilla,
                has_x_ancilla=self.has_x_ancilla,
                has_z_ancilla=self.has_z_ancilla,
            )
        return CornerAncillaDecision(
            remove_far_x_ancilla=self.remove_far_x_ancilla,
            remove_far_z_ancilla=remove,
            has_x_ancilla=self.has_x_ancilla,
            has_z_ancilla=self.has_z_ancilla,
        )


@dataclass(slots=True)
class CornerAnalysisResult:
    """Canvas-wide corner ancilla analysis result.

    Contains per-block decisions for which corner ancillas should be removed
    based on global canvas topology analysis.

    Attributes
    ----------
    cube_decisions : dict[Coord3D, dict[CornerPosition, CornerAncillaDecision]]
        Per-cube decisions indexed by cube position and corner.
    pipe_decisions : dict[tuple[Coord3D, Coord3D], dict[CornerPosition, CornerAncillaDecision]]
        Per-pipe decisions indexed by (start, end) positions and corner.
    """

    cube_decisions: dict[Coord3D, dict[CornerPosition, CornerAncillaDecision]] = field(default_factory=dict)
    pipe_decisions: dict[tuple[Coord3D, Coord3D], dict[CornerPosition, CornerAncillaDecision]] = field(
        default_factory=dict
    )

    def get_cube_decision(
        self,
        position: Coord3D,
        corner: CornerPosition,
    ) -> CornerAncillaDecision | None:
        """Get the decision for a specific cube corner.

        Parameters
        ----------
        position : Coord3D
            Global position of the cube.
        corner : CornerPosition
            The corner to get the decision for.

        Returns
        -------
        CornerAncillaDecision | None
            The decision if found, None otherwise.
        """
        cube_map = self.cube_decisions.get(position)
        if cube_map is None:
            return None
        return cube_map.get(corner.normalized())

    def get_pipe_decision(
        self,
        start: Coord3D,
        end: Coord3D,
        corner: CornerPosition,
    ) -> CornerAncillaDecision | None:
        """Get the decision for a specific pipe corner.

        Parameters
        ----------
        start : Coord3D
            Start position of the pipe.
        end : Coord3D
            End position of the pipe.
        corner : CornerPosition
            The corner to get the decision for.

        Returns
        -------
        CornerAncillaDecision | None
            The decision if found, None otherwise.
        """
        pipe_map = self.pipe_decisions.get((start, end))
        if pipe_map is None:
            return None
        return pipe_map.get(corner.normalized())

    def set_cube_decision(
        self,
        position: Coord3D,
        corner: CornerPosition,
        decision: CornerAncillaDecision,
    ) -> None:
        """Set the decision for a specific cube corner.

        Parameters
        ----------
        position : Coord3D
            Global position of the cube.
        corner : CornerPosition
            The corner to set the decision for.
        decision : CornerAncillaDecision
            The decision to set.
        """
        if position not in self.cube_decisions:
            self.cube_decisions[position] = {}
        self.cube_decisions[position][corner.normalized()] = decision

    def set_pipe_decision(
        self,
        start: Coord3D,
        end: Coord3D,
        corner: CornerPosition,
        decision: CornerAncillaDecision,
    ) -> None:
        """Set the decision for a specific pipe corner.

        Parameters
        ----------
        start : Coord3D
            Start position of the pipe.
        end : Coord3D
            End position of the pipe.
        corner : CornerPosition
            The corner to set the decision for.
        decision : CornerAncillaDecision
            The decision to set.
        """
        key = (start, end)
        if key not in self.pipe_decisions:
            self.pipe_decisions[key] = {}
        self.pipe_decisions[key][corner.normalized()] = decision


def _get_pipe_sides_for_position(
    dx: int,
    dy: int,
    *,
    is_start: bool,
) -> BoundarySide | None:
    """Get the boundary side for a position based on pipe direction.

    Parameters
    ----------
    dx : int
        X difference from start to end.
    dy : int
        Y difference from start to end.
    is_start : bool
        True if computing for start position, False for end.

    Returns
    -------
    BoundarySide | None
        The boundary side, or None if invalid.
    """
    if is_start:
        if dx == 1:
            return BoundarySide.RIGHT
        if dx == -1:
            return BoundarySide.LEFT
        if dy == 1:
            return BoundarySide.BOTTOM
        if dy == -1:
            return BoundarySide.TOP
    else:
        if dx == 1:
            return BoundarySide.LEFT
        if dx == -1:
            return BoundarySide.RIGHT
        if dy == 1:
            return BoundarySide.TOP
        if dy == -1:
            return BoundarySide.BOTTOM
    return None


def _collect_adjacent_corner_info(
    spec: CanvasSpec,
) -> dict[Coord3D, set[BoundarySide]]:
    """Collect information about which sides of each cube have adjacent pipes.

    Parameters
    ----------
    spec : CanvasSpec
        Canvas specification containing cubes and pipes.

    Returns
    -------
    dict[Coord3D, set[BoundarySide]]
        For each cube position, the set of sides that have adjacent pipes.
    """
    result: dict[Coord3D, set[BoundarySide]] = {}

    for cube in spec.cubes:
        result.setdefault(cube.position, set())

    for pipe in spec.pipes:
        dx = pipe.end.x - pipe.start.x
        dy = pipe.end.y - pipe.start.y

        start_side = _get_pipe_sides_for_position(dx, dy, is_start=True)
        end_side = _get_pipe_sides_for_position(dx, dy, is_start=False)

        if start_side is not None:
            result.setdefault(pipe.start, set()).add(start_side)
        if end_side is not None:
            result.setdefault(pipe.end, set()).add(end_side)

    return result


def _corner_shares_pipe(
    corner: CornerPosition,
    adjacent_sides: set[BoundarySide],
) -> bool:
    """Check if a corner's far ancilla position might be shared with a pipe.

    A corner's far ancilla is at the position farthest from the cube center.
    It may need removal if both adjacent sides have pipes (meaning the corner
    region is "surrounded" by connectivity).

    Parameters
    ----------
    corner : CornerPosition
        The corner to check.
    adjacent_sides : set[BoundarySide]
        Set of sides that have adjacent pipes.

    Returns
    -------
    bool
        True if both sides of the corner have adjacent pipes.
    """
    normalized = corner.normalized()
    return normalized.side1 in adjacent_sides and normalized.side2 in adjacent_sides


@dataclass
class AdjacentPipeInfo:
    """Information about a pipe adjacent to a cube on a specific side."""

    pipe: CanvasPipeSpec
    boundary: dict[BoundarySide, EdgeSpecValue]


def _get_adjacent_pipes_for_cube(
    cube: CanvasCubeSpec,
    pipes: list[CanvasPipeSpec],
) -> dict[BoundarySide, AdjacentPipeInfo]:
    """Get pipes adjacent to each side of a cube.

    Parameters
    ----------
    cube : CanvasCubeSpec
        The cube specification.
    pipes : list[CanvasPipeSpec]
        All pipes in the canvas.

    Returns
    -------
    dict[BoundarySide, AdjacentPipeInfo]
        Mapping from boundary side to adjacent pipe info.
    """
    result: dict[BoundarySide, AdjacentPipeInfo] = {}

    for pipe in pipes:
        # Check if pipe connects to this cube
        if pipe.start == cube.position:
            dx = pipe.end.x - pipe.start.x
            dy = pipe.end.y - pipe.start.y
            side = _get_pipe_sides_for_position(dx, dy, is_start=True)
            if side is not None:
                result[side] = AdjacentPipeInfo(pipe=pipe, boundary=pipe.boundary)
        elif pipe.end == cube.position:
            dx = pipe.end.x - pipe.start.x
            dy = pipe.end.y - pipe.start.y
            side = _get_pipe_sides_for_position(dx, dy, is_start=False)
            if side is not None:
                result[side] = AdjacentPipeInfo(pipe=pipe, boundary=pipe.boundary)

    return result


def _get_pipe_boundary_for_corner(
    pipe_info: AdjacentPipeInfo,
    cube_side: BoundarySide,
    corner: CornerPosition,
) -> EdgeSpecValue:
    """Get the pipe boundary value relevant for corner analysis.

    When a pipe connects to a cube, the connection boundary is always O.
    For corner analysis, we need to look at the pipe's boundary in the
    direction perpendicular to the connection (the corner direction).

    Parameters
    ----------
    pipe_info : AdjacentPipeInfo
        Information about the adjacent pipe.
    cube_side : BoundarySide
        The side of the cube where the pipe connects.
    corner : CornerPosition
        The corner being analyzed.

    Returns
    -------
    EdgeSpecValue
        The relevant boundary value from the pipe.

    Examples
    --------
    For TOP-RIGHT corner with pipe on TOP side:
        -> Return pipe's RIGHT boundary
    For TOP-RIGHT corner with pipe on RIGHT side:
        -> Return pipe's TOP boundary
    """
    normalized = corner.normalized()

    # Find the other side of the corner (not the connection side)
    other_side = normalized.side2 if normalized.side1 == cube_side else normalized.side1

    return pipe_info.boundary[other_side]


def _get_cube_corner_coordinate(
    cube: CanvasCubeSpec,
    corner: CornerPosition,
    code_distance: int,
) -> Coord2D:
    """Get the physical coordinate of a corner ancilla position.

    Corner coordinates are outside the main cube bounds:
    - TOP-LEFT: (x_min-1, y_min-1)
    - TOP-RIGHT: (x_max+1, y_min-1)
    - BOTTOM-LEFT: (x_min-1, y_max+1)
    - BOTTOM-RIGHT: (x_max+1, y_max+1)

    Parameters
    ----------
    cube : CanvasCubeSpec
        The cube specification.
    corner : CornerPosition
        The corner to get coordinates for.
    code_distance : int
        The code distance.

    Returns
    -------
    Coord2D
        The (x, y) coordinate of the corner ancilla position.
    """
    from lspattern.mytype import Coord2D  # noqa: PLC0415

    # Calculate cube offset (same as in layout)
    offset_x = 2 * (code_distance + 1) * cube.position.x
    offset_y = 2 * (code_distance + 1) * cube.position.y

    # Calculate bounds
    x_min = offset_x
    x_max = offset_x + 2 * (code_distance - 1)
    y_min = offset_y
    y_max = offset_y + 2 * (code_distance - 1)

    normalized = corner.normalized()

    # Determine corner coordinate based on corner position
    if normalized == CORNER_TOP_LEFT:
        return Coord2D(x_min - 1, y_min - 1)
    if normalized == CORNER_TOP_RIGHT:
        return Coord2D(x_max + 1, y_min - 1)
    if normalized == CORNER_BOTTOM_LEFT:
        return Coord2D(x_min - 1, y_max + 1)
    if normalized == CORNER_BOTTOM_RIGHT:
        return Coord2D(x_max + 1, y_max + 1)

    msg = f"Invalid corner: {corner}"
    raise ValueError(msg)


def _check_corner_ancilla_parity(
    corner_coord: Coord2D,
    ancilla_type: Literal["x", "z"],
) -> bool:
    """Check if a corner coordinate satisfies the ancilla parity condition.

    Bulk ancilla placement follows checkerboard pattern:
    - X ancilla: x % 2 == 1 and y % 2 == 1 and (x + y) % 4 == 0
    - Z ancilla: x % 2 == 1 and y % 2 == 1 and (x + y) % 4 == 2

    Parameters
    ----------
    corner_coord : Coord2D
        The corner coordinate to check.
    ancilla_type : Literal["x", "z"]
        The type of ancilla to check for.

    Returns
    -------
    bool
        True if the coordinate satisfies the parity condition.
    """
    x, y = corner_coord.x, corner_coord.y

    # Corner coordinates should be at odd positions (outside data bounds)
    if x % 2 != 1 or y % 2 != 1:
        return False

    if ancilla_type == "x":
        return (x + y) % 4 == 0
    # ancilla_type == "z"
    return (x + y) % 4 == 2  # noqa: PLR2004


def _evaluate_corner(
    cube: CanvasCubeSpec,
    corner: CornerPosition,
    adjacent_pipes: dict[BoundarySide, AdjacentPipeInfo],
    code_distance: int,
) -> CornerAncillaDecision:
    """Evaluate a corner to determine if ancillas should exist there.

    The decision is based on:
    1. For each side of the corner, find the "effective" non-O boundary:
       - If cube's boundary is non-O -> use cube's boundary (source = "cube")
       - If cube's boundary is O -> use pipe's conjugate boundary (source = "pipe")
    2. Source combination determines if ancilla is needed:
       - (cube, cube): Both from cube -> no ancilla
       - (cube, pipe) or (pipe, pipe): At least one from pipe -> possible ancilla
    3. Both effective boundaries must be the same type (XX or ZZ)
    4. Corner coordinate must satisfy parity condition for that ancilla type

    Parameters
    ----------
    cube : CanvasCubeSpec
        The cube specification.
    corner : CornerPosition
        The corner to evaluate.
    adjacent_pipes : dict[BoundarySide, AdjacentPipeInfo]
        Pipes adjacent to each side of the cube.
    code_distance : int
        The code distance.

    Returns
    -------
    CornerAncillaDecision
        Decision indicating if X/Z ancillas should exist at this corner.
    """
    normalized = corner.normalized()
    side1, side2 = normalized.side1, normalized.side2

    # Get cube's original boundaries
    cube_boundary1 = cube.boundary[side1]
    cube_boundary2 = cube.boundary[side2]
    pipe1 = adjacent_pipes.get(side1)
    pipe2 = adjacent_pipes.get(side2)

    # For each side, determine effective boundary and source
    # If cube boundary is O, use pipe's conjugate boundary
    # Otherwise, use cube's boundary
    if cube_boundary1 == EdgeSpecValue.O:
        if pipe1 is None:
            msg = f"Cube {cube.position} has O boundary on {side1} but no adjacent pipe."
            raise ValueError(msg)
        boundary1 = _get_pipe_boundary_for_corner(pipe1, side1, corner)
        source1 = "pipe"
    else:
        boundary1 = cube_boundary1
        source1 = "cube"

    if cube_boundary2 == EdgeSpecValue.O:
        if pipe2 is None:
            msg = f"Cube {cube.position} has O boundary on {side2} but no adjacent pipe."
            raise ValueError(msg)
        boundary2 = _get_pipe_boundary_for_corner(pipe2, side2, corner)
        source2 = "pipe"
    else:
        boundary2 = cube_boundary2
        source2 = "cube"

    # (cube, cube) combination -> no corner ancilla added
    if source1 == "cube" and source2 == "cube":
        return CornerAncillaDecision()

    # (cube, pipe) or (pipe, pipe): Check if boundaries match
    if boundary1 != boundary2:
        # Different types (XZ or ZX) -> no ancilla
        return CornerAncillaDecision()

    # Get corner coordinate and check parity
    corner_coord = _get_cube_corner_coordinate(cube, corner, code_distance)

    has_x = False
    has_z = False

    # XX boundaries -> check X ancilla parity
    if boundary1 == EdgeSpecValue.X and _check_corner_ancilla_parity(corner_coord, "x"):
        has_x = True
    # ZZ boundaries -> check Z ancilla parity
    elif boundary1 == EdgeSpecValue.Z and _check_corner_ancilla_parity(corner_coord, "z"):
        has_z = True

    return CornerAncillaDecision(has_x_ancilla=has_x, has_z_ancilla=has_z)


def analyze_corner_ancillas(
    spec: CanvasSpec,
    *,
    code_distance: int,
) -> CornerAnalysisResult:
    """Analyze canvas topology to determine corner ancilla decisions.

    This function analyzes the global canvas layout and determines which
    corner ancillas should exist at each cube corner based on adjacent
    pipe boundaries.

    Corner Ancilla Rules:
    1. For each corner of a cube, check the two adjacent boundary sides
    2. If both sides are cube boundaries (no pipe): no corner ancilla
    3. If at least one side has an adjacent pipe:
       - Get boundary value from pipe (perpendicular direction) or cube
       - Both boundaries must be same type (XX or ZZ)
       - Corner coordinate must satisfy checkerboard parity
    4. Resulting ancilla type matches the boundary type (XX -> X, ZZ -> Z)

    Parameters
    ----------
    spec : CanvasSpec
        Canvas specification containing cubes and pipes.
    code_distance : int
        Code distance of the surface code.

    Returns
    -------
    CornerAnalysisResult
        Analysis result containing per-cube corner ancilla decisions.
        Each decision indicates if X/Z ancillas should exist at corners.
    """
    result = CornerAnalysisResult()

    # Analyze each cube's corners
    for cube in spec.cubes:
        # Get all adjacent pipes for this cube
        adjacent_pipes = _get_adjacent_pipes_for_cube(cube, spec.pipes)

        # Evaluate each corner
        for corner in ALL_CORNERS:
            decision = _evaluate_corner(cube, corner, adjacent_pipes, code_distance)

            # Only store decisions that have ancillas
            if decision.has_x_ancilla or decision.has_z_ancilla:
                result.set_cube_decision(cube.position, corner, decision)

    return result


def get_cube_corner_decisions(
    result: CornerAnalysisResult,
    position: Coord3D,
) -> dict[CornerPosition, CornerAncillaDecision]:
    """Get all corner decisions for a specific cube.

    Parameters
    ----------
    result : CornerAnalysisResult
        The analysis result.
    position : Coord3D
        Global position of the cube.

    Returns
    -------
    dict[CornerPosition, CornerAncillaDecision]
        Dictionary mapping corner positions to their decisions.
        Returns empty dict if no decisions exist for this cube.
    """
    return result.cube_decisions.get(position, {})


def get_pipe_corner_decisions(
    result: CornerAnalysisResult,
    start: Coord3D,
    end: Coord3D,
) -> dict[CornerPosition, CornerAncillaDecision]:
    """Get all corner decisions for a specific pipe.

    Parameters
    ----------
    result : CornerAnalysisResult
        The analysis result.
    start : Coord3D
        Start position of the pipe.
    end : Coord3D
        End position of the pipe.

    Returns
    -------
    dict[CornerPosition, CornerAncillaDecision]
        Dictionary mapping corner positions to their decisions.
        Returns empty dict if no decisions exist for this pipe.
    """
    return result.pipe_decisions.get((start, end), {})
