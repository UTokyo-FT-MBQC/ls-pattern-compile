"""Patch region data structures and coordinate generation for rotated surface codes.

This module provides clean abstractions for generating qubit coordinates in
rotated surface code layouts, separating the logic into bulk, boundary, and
corner components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.layout.base import TopologicalCodeLayoutBuilder
from lspattern.layout.checkerboard import generate_checkerboard_coords
from lspattern.layout.coordinates import PatchBounds, PatchCoordinates
from lspattern.mytype import AxisDIRECTION2D, Coord2D, Coord3D

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lspattern.init_flow_analysis import AdjacentPipeData


# =============================================================================
# Constants
# =============================================================================

# Ancilla edge offsets for X-type stabilizers (order optimized for distance)
ANCILLA_EDGE_X: tuple[tuple[int, int], ...] = (
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
)

# Ancilla edge offsets for Z-type stabilizers (order optimized for distance)
ANCILLA_EDGE_Z: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


# =============================================================================
# Corner Rules (Module-level constants)
# =============================================================================

# Corner removal rules: (side1, side2) -> required EdgeSpecValue pairs
_CORNER_DATA_REMOVAL_RULES: dict[
    tuple[BoundarySide, BoundarySide],
    list[tuple[EdgeSpecValue, EdgeSpecValue]],
] = {
    (BoundarySide.TOP, BoundarySide.RIGHT): [(EdgeSpecValue.Z, EdgeSpecValue.Z), (EdgeSpecValue.X, EdgeSpecValue.X)],
    (BoundarySide.BOTTOM, BoundarySide.LEFT): [(EdgeSpecValue.Z, EdgeSpecValue.Z), (EdgeSpecValue.X, EdgeSpecValue.X)],
    (BoundarySide.TOP, BoundarySide.LEFT): [(EdgeSpecValue.Z, EdgeSpecValue.Z), (EdgeSpecValue.X, EdgeSpecValue.X)],
    (BoundarySide.BOTTOM, BoundarySide.RIGHT): [(EdgeSpecValue.Z, EdgeSpecValue.Z), (EdgeSpecValue.X, EdgeSpecValue.X)],
}

# Corner ancilla removal offsets: (side1, side2, ancilla_type) -> list of (dx, dy)
# dx, dy are relative offsets from the corner data coordinate
_CORNER_ANCILLA_REMOVAL_OFFSETS: dict[
    tuple[BoundarySide, BoundarySide, Literal["x", "z"]],
    list[tuple[int, int]],
] = {
    (BoundarySide.TOP, BoundarySide.RIGHT, "z"): [
        (-1, -1),
        (1, 1),
    ],
    (BoundarySide.BOTTOM, BoundarySide.LEFT, "z"): [
        (-1, -1),
        (1, 1),
    ],
    (BoundarySide.TOP, BoundarySide.LEFT, "x"): [
        (1, -1),
        (-1, 1),
    ],
    (BoundarySide.BOTTOM, BoundarySide.RIGHT, "x"): [
        (-1, 1),
        (1, -1),
    ],
}

# Corner coordinate positions relative to bounds: (x_attr, y_attr)
_CORNER_POSITIONS: dict[tuple[BoundarySide, BoundarySide], tuple[str, str]] = {
    (BoundarySide.TOP, BoundarySide.RIGHT): ("x_max", "y_min"),
    (BoundarySide.BOTTOM, BoundarySide.LEFT): ("x_min", "y_max"),
    (BoundarySide.TOP, BoundarySide.LEFT): ("x_min", "y_min"),
    (BoundarySide.BOTTOM, BoundarySide.RIGHT): ("x_max", "y_max"),
}


# =============================================================================
# Rotated Surface Code Layout Implementation (Instance-based)
# =============================================================================


class RotatedSurfaceCodeLayout(TopologicalCodeLayoutBuilder):
    """Instance-based implementation of rotated surface code layout builder.

    This class implements the TopologicalCodeLayoutBuilder interface for
    rotated surface codes. It can be used directly or via the static
    facade RotatedSurfaceCodeLayoutBuilder.

    Examples
    --------
    >>> from lspattern.consts import BoundarySide, EdgeSpecValue
    >>> from lspattern.mytype import Coord2D
    >>> layout = RotatedSurfaceCodeLayout()
    >>> boundary = {
    ...     BoundarySide.TOP: EdgeSpecValue.X,
    ...     BoundarySide.BOTTOM: EdgeSpecValue.X,
    ...     BoundarySide.LEFT: EdgeSpecValue.Z,
    ...     BoundarySide.RIGHT: EdgeSpecValue.Z,
    ... }
    >>> coords = layout.cube(code_distance=3, global_pos=Coord2D(0, 0), boundary=boundary)
    >>> len(coords.data) > 0
    True
    """

    # =========================================================================
    # CoordinateGenerator Implementation
    # =========================================================================

    def cube(
        self,
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> PatchCoordinates:
        """Build complete cube layout coordinates.

        This method generates all qubit coordinates for a rotated surface code
        cube patch, including bulk, boundary, and corner qubits.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.

        Returns
        -------
        PatchCoordinates
            Complete coordinate sets for the cube.
        """
        offset = self.cube_offset(code_distance, global_pos)
        bounds = self.cube_bounds(code_distance, offset)

        # Generate components
        bulk = generate_checkerboard_coords(bounds)
        corner_data_remove = self._get_corner_data_to_remove(bounds, boundary)
        corner_x_remove, corner_z_remove = self._get_corner_ancillas_to_remove(bounds, boundary)
        boundary_x, boundary_z = self._generate_cube_boundary_ancillas(bounds, boundary)
        corner_x, corner_z = self._get_corner_ancillas(bounds, boundary)

        return PatchCoordinates(
            data=bulk.data - corner_data_remove,
            ancilla_x=(bulk.ancilla_x | boundary_x | corner_x) - corner_x_remove,
            ancilla_z=(bulk.ancilla_z | boundary_z | corner_z) - corner_z_remove,
        )

    def pipe(
        self,
        code_distance: int,
        global_pos_source: Coord3D,
        global_pos_target: Coord3D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> PatchCoordinates:
        """Build complete pipe layout coordinates.

        This method generates all qubit coordinates for a rotated surface code
        pipe patch, including bulk and boundary qubits.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos_source : Coord3D
            Global (x, y, z) position of the pipe source.
        global_pos_target : Coord3D
            Global (x, y, z) position of the pipe target.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the pipe.

        Returns
        -------
        PatchCoordinates
            Complete coordinate sets for the pipe.
        """
        # Determine pipe direction from source/target positions
        pipe_offset_dir = self.pipe_offset(global_pos_source, global_pos_target)
        pipe_dir = self.pipe_axis_from_offset(pipe_offset_dir)

        offset = self._compute_pipe_offset(code_distance, global_pos_source, pipe_offset_dir)
        bounds = self.pipe_bounds(code_distance, offset, pipe_dir)

        # Generate components
        bulk = generate_checkerboard_coords(bounds)
        boundary_x, boundary_z = self._generate_pipe_boundary_ancillas(bounds, boundary, pipe_dir)

        return PatchCoordinates(
            data=bulk.data,
            ancilla_x=bulk.ancilla_x | boundary_x,
            ancilla_z=bulk.ancilla_z | boundary_z,
        )

    # =========================================================================
    # BoundsCalculator Implementation
    # =========================================================================

    def cube_bounds(self, code_distance: int, offset: Coord2D) -> PatchBounds:
        """Create bounds for a cube patch."""
        return PatchBounds(
            x_min=offset.x,
            x_max=offset.x + 2 * (code_distance - 1),
            y_min=offset.y,
            y_max=offset.y + 2 * (code_distance - 1),
        )

    def pipe_bounds(
        self,
        code_distance: int,
        offset: Coord2D,
        direction: AxisDIRECTION2D,
    ) -> PatchBounds:
        """Create bounds for a pipe patch."""
        if direction == AxisDIRECTION2D.H:
            # Horizontal pipe: narrow in x (3 units), long in y (d data qubits)
            return PatchBounds(
                x_min=offset.x - 1,
                x_max=offset.x + 1,
                y_min=offset.y,
                y_max=offset.y + 2 * (code_distance - 1),
            )
        # Vertical pipe: long in x (d data qubits), narrow in y (3 units)
        return PatchBounds(
            x_min=offset.x,
            x_max=offset.x + 2 * (code_distance - 1),
            y_min=offset.y - 1,
            y_max=offset.y + 1,
        )

    def cube_offset(self, code_distance: int, global_pos: Coord2D) -> Coord2D:
        """Compute the offset for a cube patch based on global position."""
        return Coord2D(
            x=2 * (code_distance + 1) * global_pos.x,
            y=2 * (code_distance + 1) * global_pos.y,
        )

    # =========================================================================
    # BoundaryPathCalculator Implementation
    # =========================================================================

    def cube_boundary_path(
        self,
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side_a: BoundarySide,
        side_b: BoundarySide,
    ) -> list[Coord2D]:
        """Get data-qubit path from side_a to side_b through cube center.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        side_a : BoundarySide
            Starting boundary side.
        side_b : BoundarySide
            Ending boundary side.

        Returns
        -------
        list[Coord2D]
            Ordered data-qubit coordinates from side_a to side_b.
        """
        offset = self.cube_offset(code_distance, global_pos)
        bounds = self.cube_bounds(code_distance, offset)
        coords = self.cube(code_distance, global_pos, boundary)
        return self._boundary_path(coords.data, bounds, side_a, side_b)

    def pipe_boundary_path(
        self,
        code_distance: int,
        global_pos_source: Coord3D,
        global_pos_target: Coord3D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side_a: BoundarySide,
        side_b: BoundarySide,
    ) -> list[Coord2D]:
        """Get data-qubit path from side_a to side_b through pipe center.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos_source : Coord3D
            Global (x, y, z) position of the pipe source.
        global_pos_target : Coord3D
            Global (x, y, z) position of the pipe target.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the pipe.
        side_a : BoundarySide
            Starting boundary side.
        side_b : BoundarySide
            Ending boundary side.

        Returns
        -------
        list[Coord2D]
            Ordered data-qubit coordinates from side_a to side_b.
        """
        pipe_offset_dir = self.pipe_offset(global_pos_source, global_pos_target)
        pipe_dir = self.pipe_axis_from_offset(pipe_offset_dir)
        offset = self._compute_pipe_offset(code_distance, global_pos_source, pipe_offset_dir)
        bounds = self.pipe_bounds(code_distance, offset, pipe_dir)
        coords = self.pipe(code_distance, global_pos_source, global_pos_target, boundary)
        return self._boundary_path(coords.data, bounds, side_a, side_b)

    # =========================================================================
    # BoundaryAncillaRetriever Implementation
    # =========================================================================

    def cube_boundary_ancillas_for_side(
        self,
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Get ancilla coordinates for a specific boundary side of a cube.

        This method returns all ancilla qubits located on the specified boundary
        side, including corner ancillas. Each corner belongs to both adjacent
        sides (e.g., top_left corner belongs to both TOP and LEFT sides).

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        side : BoundarySide
            The boundary side to get ancillas for (TOP, BOTTOM, LEFT, RIGHT).

        Returns
        -------
        tuple[frozenset[Coord2D], frozenset[Coord2D]]
            (X ancillas, Z ancillas) for the specified boundary side,
            including corner ancillas that belong to this side.

        Examples
        --------
        >>> from lspattern.consts import BoundarySide, EdgeSpecValue
        >>> from lspattern.mytype import Coord2D
        >>> layout = RotatedSurfaceCodeLayout()
        >>> boundary = {
        ...     BoundarySide.TOP: EdgeSpecValue.X,
        ...     BoundarySide.BOTTOM: EdgeSpecValue.X,
        ...     BoundarySide.LEFT: EdgeSpecValue.Z,
        ...     BoundarySide.RIGHT: EdgeSpecValue.Z,
        ... }
        >>> x_anc, z_anc = layout.cube_boundary_ancillas_for_side(
        ...     code_distance=3,
        ...     global_pos=Coord2D(0, 0),
        ...     boundary=boundary,
        ...     side=BoundarySide.TOP,
        ... )
        """
        # Calculate bounds internally
        offset = self.cube_offset(code_distance, global_pos)
        bounds = self.cube_bounds(code_distance, offset)

        # Get edge ancillas for this side
        edge_x, edge_z = self._generate_boundary_ancillas_for_side(bounds, boundary, side)

        # Get corner ancillas for this side
        corner_x, corner_z = self._get_corner_ancillas_for_side(bounds, boundary, side)

        # Combine edge and corner ancillas
        return frozenset(edge_x | corner_x), frozenset(edge_z | corner_z)

    # =========================================================================
    # AncillaFlowConstructor Implementation
    # =========================================================================

    def construct_initial_ancilla_flow(
        self,
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        ancilla_type: EdgeSpecValue,
        move_vec: Coord2D,
        *,
        adjacent_data: AdjacentPipeData | None = None,
    ) -> dict[Coord2D, set[Coord2D]]:
        """Construct flow mapping for initial ancilla qubits.

        This method computes the flow relationships for ancilla qubits in
        initialization layers. The flow determines the causal dependencies
        between ancilla measurements.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        ancilla_type : EdgeSpecValue
            Type of ancilla qubit. "Z" for layer1 (Z-stabilizer), "X" for layer2 (X-stabilizer).
        move_vec : Coord2D
            Signed movement direction (one of (0, 1), (0, -1), (-1, 0), (1, 0)).
        adjacent_data : AdjacentPipeData | None
            Optional per-boundary-side data qubit coordinates from adjacent pipes.
            Used for cubes with O (open) boundaries to find flow targets in pipes.

        Returns
        -------
        dict[Coord2D, set[Coord2D]]
            Mapping from source 2D coordinate to target 2D coordinates for ancilla flow.
            Each source coordinate maps to a set of target coordinates.
        """
        coords = self.cube(code_distance, global_pos, boundary)
        data2d = set(coords.data)

        # Merge adjacent pipe data for the target boundary side
        if adjacent_data:
            target_side = self._move_vec_to_side(move_vec)
            if target_side is not None and target_side in adjacent_data:
                data2d |= adjacent_data[target_side]

        if ancilla_type == EdgeSpecValue.X:
            ancilla_nodes = coords.ancilla_x
        elif ancilla_type == EdgeSpecValue.Z:
            ancilla_nodes = coords.ancilla_z
        else:
            msg = f"Invalid ancilla type for flow: {ancilla_type}."
            raise ValueError(msg)

        if (move_vec.x, move_vec.y) not in {(0, 1), (0, -1), (-1, 0), (1, 0)}:
            msg = f"Invalid move_vec for init flow: {move_vec}."
            raise ValueError(msg)

        flow_map: dict[Coord2D, set[Coord2D]] = {}
        for node in ancilla_nodes:
            target = self._determine_flow(
                node,
                frozenset(data2d),
                ancilla_type,
                move_vec,
            )
            flow_map.setdefault(node, set()).add(target)

        return flow_map

    def _move_vec_to_side(self, move_vec: Coord2D) -> BoundarySide | None:
        """Convert movement vector to boundary side.

        Parameters
        ----------
        move_vec : Coord2D
            Movement direction vector.

        Returns
        -------
        BoundarySide | None
            The boundary side corresponding to the move direction, or None.
        """
        mapping: dict[tuple[int, int], BoundarySide] = {
            (0, -1): BoundarySide.TOP,
            (0, 1): BoundarySide.BOTTOM,
            (-1, 0): BoundarySide.LEFT,
            (1, 0): BoundarySide.RIGHT,
        }
        return mapping.get((move_vec.x, move_vec.y))

    # =========================================================================
    # PipeDirectionHelper Implementation
    # =========================================================================

    def pipe_offset(
        self,
        global_pos_source: Coord3D,
        global_pos_target: Coord3D,
    ) -> BoundarySide:
        """Calculate pipe offset direction from source to target positions.

        Parameters
        ----------
        global_pos_source : Coord3D
            Global (x, y, z) position of the pipe source.
        global_pos_target : Coord3D
            Global (x, y, z) position of the pipe target.

        Returns
        -------
        BoundarySide
            The direction from source to target.

        Raises
        ------
        ValueError
            If source and target positions don't form a valid pipe offset.
        """
        dx = global_pos_target.x - global_pos_source.x
        dy = global_pos_target.y - global_pos_source.y

        if dx == 1 and dy == 0:
            return BoundarySide.RIGHT
        if dx == -1 and dy == 0:
            return BoundarySide.LEFT
        if dx == 0 and dy == 1:
            return BoundarySide.BOTTOM  # y increases toward BOTTOM (y_max side)
        if dx == 0 and dy == -1:
            return BoundarySide.TOP  # y decreases toward TOP (y_min side)

        msg = f"Invalid pipe offset: source {global_pos_source}, target {global_pos_target}."
        raise ValueError(msg)

    def pipe_axis_from_offset(self, offset_dir: BoundarySide) -> AxisDIRECTION2D:
        """Derive pipe axis direction from offset direction.

        Parameters
        ----------
        offset_dir : BoundarySide
            The direction from source to target (from pipe_offset()).

        Returns
        -------
        AxisDIRECTION2D
            H for horizontal pipe (RIGHT/LEFT), V for vertical pipe (TOP/BOTTOM).
        """
        if offset_dir in {BoundarySide.RIGHT, BoundarySide.LEFT}:
            return AxisDIRECTION2D.H
        return AxisDIRECTION2D.V

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _compute_pipe_offset(
        self,
        code_distance: int,
        global_pos_source: Coord3D,
        pipe_dir: BoundarySide,
    ) -> Coord2D:
        """Compute the offset for a pipe patch based on source position and direction."""
        offset_x = 2 * (code_distance + 1) * global_pos_source.x
        offset_y = 2 * (code_distance + 1) * global_pos_source.y

        if pipe_dir == BoundarySide.RIGHT:
            offset_x += 2 * code_distance
        elif pipe_dir == BoundarySide.LEFT:
            offset_x -= 2
        elif pipe_dir == BoundarySide.TOP:
            offset_y -= 2
        elif pipe_dir == BoundarySide.BOTTOM:
            offset_y += 2 * code_distance

        return Coord2D(offset_x, offset_y)

    def _get_corner_data_to_remove(
        self,
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> frozenset[Coord2D]:
        """Determine which corner data qubits should be removed based on boundary conditions."""
        to_remove: set[Coord2D] = set()

        for (side1, side2), rules in _CORNER_DATA_REMOVAL_RULES.items():
            for expected1, expected2 in rules:
                if (boundary[side1], boundary[side2]) == (expected1, expected2):
                    x_attr, y_attr = _CORNER_POSITIONS[side1, side2]
                    coord = Coord2D(getattr(bounds, x_attr), getattr(bounds, y_attr))
                    to_remove.add(coord)

        return frozenset(to_remove)

    def _get_corner_ancillas_to_remove(
        self,
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Determine which corner ancilla qubits should be removed based on boundary conditions.

        Returns
        -------
        tuple[frozenset[Coord2D], frozenset[Coord2D]]
            (X ancillas to remove, Z ancillas to remove)
        """
        x_to_remove: set[Coord2D] = set()
        z_to_remove: set[Coord2D] = set()

        for (side1, side2), rules in _CORNER_DATA_REMOVAL_RULES.items():
            for expected1, expected2 in rules:
                if (boundary[side1], boundary[side2]) == (expected1, expected2):
                    # Get corner data coordinate
                    x_attr, y_attr = _CORNER_POSITIONS[side1, side2]
                    corner_x = getattr(bounds, x_attr)
                    corner_y = getattr(bounds, y_attr)

                    # Apply offsets to get ancilla coordinates
                    for ancilla_type in ("x", "z"):
                        key = (side1, side2, ancilla_type)
                        if key not in _CORNER_ANCILLA_REMOVAL_OFFSETS:
                            continue
                        for dx, dy in _CORNER_ANCILLA_REMOVAL_OFFSETS[key]:
                            coord = Coord2D(corner_x + dx, corner_y + dy)
                            if ancilla_type == "x":
                                x_to_remove.add(coord)
                            else:
                                z_to_remove.add(coord)

        return frozenset(x_to_remove), frozenset(z_to_remove)

    def _get_corner_ancillas(
        self,
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate corner ancilla coordinates for 'O' (open) boundaries."""
        x_ancillas: set[Coord2D] = set()
        z_ancillas: set[Coord2D] = set()

        # Corner positions (outside the main bounds)
        corner_coords = {
            "top_right": Coord2D(bounds.x_max + 1, bounds.y_min - 1),
            "bottom_right": Coord2D(bounds.x_max + 1, bounds.y_max + 1),
            "top_left": Coord2D(bounds.x_min - 1, bounds.y_min - 1),
            "bottom_left": Coord2D(bounds.x_min - 1, bounds.y_max + 1),
        }

        # Rules: (primary_side, secondary_side, secondary_vals, ancilla_type, corner_key)
        xo_set = {EdgeSpecValue.X, EdgeSpecValue.O}
        zo_set = {EdgeSpecValue.Z, EdgeSpecValue.O}

        rules: list[tuple[BoundarySide, BoundarySide, set[EdgeSpecValue], str, str]] = [
            # RIGHT == O rules
            (BoundarySide.RIGHT, BoundarySide.TOP, xo_set, "x", "top_right"),
            (BoundarySide.RIGHT, BoundarySide.BOTTOM, zo_set, "z", "bottom_right"),
            # LEFT == O rules
            (BoundarySide.LEFT, BoundarySide.TOP, zo_set, "z", "top_left"),
            (BoundarySide.LEFT, BoundarySide.BOTTOM, xo_set, "x", "bottom_left"),
            # TOP == O rules
            (BoundarySide.TOP, BoundarySide.LEFT, zo_set, "z", "top_left"),
            (BoundarySide.TOP, BoundarySide.RIGHT, xo_set, "x", "top_right"),
            # BOTTOM == O rules
            (BoundarySide.BOTTOM, BoundarySide.LEFT, xo_set, "x", "bottom_left"),
            (BoundarySide.BOTTOM, BoundarySide.RIGHT, zo_set, "z", "bottom_right"),
        ]

        for primary_side, secondary_side, secondary_vals, ancilla_type, corner_key in rules:
            if boundary[primary_side] == EdgeSpecValue.O and boundary[secondary_side] in secondary_vals:
                coord = corner_coords[corner_key]
                if ancilla_type == "x":
                    x_ancillas.add(coord)
                else:
                    z_ancillas.add(coord)

        return frozenset(x_ancillas), frozenset(z_ancillas)

    def _get_corner_ancillas_for_side(
        self,
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate corner ancilla coordinates for a specific side.

        Each corner belongs to two adjacent sides. This method returns
        the corners that are adjacent to the specified side.

        Parameters
        ----------
        bounds : PatchBounds
            Bounding box for the patch region.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        side : BoundarySide
            The boundary side to get corner ancillas for.

        Returns
        -------
        tuple[frozenset[Coord2D], frozenset[Coord2D]]
            (X ancillas, Z ancillas) for corners adjacent to the specified side.
        """
        x_ancillas: set[Coord2D] = set()
        z_ancillas: set[Coord2D] = set()

        # Corner positions (outside the main bounds)
        corner_coords = {
            "top_right": Coord2D(bounds.x_max + 1, bounds.y_min - 1),
            "bottom_right": Coord2D(bounds.x_max + 1, bounds.y_max + 1),
            "top_left": Coord2D(bounds.x_min - 1, bounds.y_min - 1),
            "bottom_left": Coord2D(bounds.x_min - 1, bounds.y_max + 1),
        }

        # Corners adjacent to each side
        side_corners: dict[BoundarySide, tuple[str, str]] = {
            BoundarySide.TOP: ("top_left", "top_right"),
            BoundarySide.BOTTOM: ("bottom_left", "bottom_right"),
            BoundarySide.LEFT: ("top_left", "bottom_left"),
            BoundarySide.RIGHT: ("top_right", "bottom_right"),
        }

        xo_set = {EdgeSpecValue.X, EdgeSpecValue.O}
        zo_set = {EdgeSpecValue.Z, EdgeSpecValue.O}

        # All rules: (primary_side, secondary_side, secondary_vals, ancilla_type, corner_key)
        rules: list[tuple[BoundarySide, BoundarySide, set[EdgeSpecValue], str, str]] = [
            # RIGHT == O rules
            (BoundarySide.RIGHT, BoundarySide.TOP, xo_set, "x", "top_right"),
            (BoundarySide.RIGHT, BoundarySide.BOTTOM, zo_set, "z", "bottom_right"),
            # LEFT == O rules
            (BoundarySide.LEFT, BoundarySide.TOP, zo_set, "z", "top_left"),
            (BoundarySide.LEFT, BoundarySide.BOTTOM, xo_set, "x", "bottom_left"),
            # TOP == O rules
            (BoundarySide.TOP, BoundarySide.LEFT, zo_set, "z", "top_left"),
            (BoundarySide.TOP, BoundarySide.RIGHT, xo_set, "x", "top_right"),
            # BOTTOM == O rules
            (BoundarySide.BOTTOM, BoundarySide.LEFT, xo_set, "x", "bottom_left"),
            (BoundarySide.BOTTOM, BoundarySide.RIGHT, zo_set, "z", "bottom_right"),
        ]

        # Get corners that belong to this side
        valid_corners = side_corners[side]

        for primary_side, secondary_side, secondary_vals, ancilla_type, corner_key in rules:
            # Only include corners adjacent to the specified side
            if corner_key not in valid_corners:
                continue
            if boundary[primary_side] == EdgeSpecValue.O and boundary[secondary_side] in secondary_vals:
                coord = corner_coords[corner_key]
                if ancilla_type == "x":
                    x_ancillas.add(coord)
                else:
                    z_ancillas.add(coord)

        return frozenset(x_ancillas), frozenset(z_ancillas)

    def _should_add_boundary_ancilla(
        self,
        edge_spec: EdgeSpecValue,
        target_type: EdgeSpecValue,
        position_mod4: int,
        expected_mod4: int,
    ) -> bool:
        """Check if a boundary ancilla should be added at the given position."""
        return edge_spec in {target_type, EdgeSpecValue.O} and position_mod4 == expected_mod4

    def _generate_cube_boundary_ancillas(  # noqa: C901
        self,
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate boundary ancilla coordinates for a cube layout."""
        x_ancillas: set[Coord2D] = set()
        z_ancillas: set[Coord2D] = set()
        check = self._should_add_boundary_ancilla

        # Horizontal boundaries (TOP and BOTTOM)
        for x in range(bounds.x_min + 1, bounds.x_max):
            rel_x = x - bounds.x_min
            rel_x_mod4 = rel_x % 4

            # TOP boundary (y = y_min - 1)
            if check(boundary[BoundarySide.TOP], EdgeSpecValue.X, rel_x_mod4, 1):
                x_ancillas.add(Coord2D(x, bounds.y_min - 1))
            if check(boundary[BoundarySide.TOP], EdgeSpecValue.Z, rel_x_mod4, 3):
                z_ancillas.add(Coord2D(x, bounds.y_min - 1))

            # BOTTOM boundary (y = y_max + 1)
            if check(boundary[BoundarySide.BOTTOM], EdgeSpecValue.X, rel_x_mod4, 3):
                x_ancillas.add(Coord2D(x, bounds.y_max + 1))
            if check(boundary[BoundarySide.BOTTOM], EdgeSpecValue.Z, rel_x_mod4, 1):
                z_ancillas.add(Coord2D(x, bounds.y_max + 1))

        # Vertical boundaries (LEFT and RIGHT)
        for y in range(bounds.y_min + 1, bounds.y_max):
            rel_y = y - bounds.y_min
            rel_y_mod4 = rel_y % 4

            # LEFT boundary (x = x_min - 1)
            if check(boundary[BoundarySide.LEFT], EdgeSpecValue.X, rel_y_mod4, 1):
                x_ancillas.add(Coord2D(bounds.x_min - 1, y))
            if check(boundary[BoundarySide.LEFT], EdgeSpecValue.Z, rel_y_mod4, 3):
                z_ancillas.add(Coord2D(bounds.x_min - 1, y))

            # RIGHT boundary (x = x_max + 1)
            if check(boundary[BoundarySide.RIGHT], EdgeSpecValue.X, rel_y_mod4, 3):
                x_ancillas.add(Coord2D(bounds.x_max + 1, y))
            if check(boundary[BoundarySide.RIGHT], EdgeSpecValue.Z, rel_y_mod4, 1):
                z_ancillas.add(Coord2D(bounds.x_max + 1, y))

        return frozenset(x_ancillas), frozenset(z_ancillas)

    def _generate_boundary_ancillas_for_side(  # noqa: C901
        self,
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate boundary ancilla coordinates for a specific side of a cube.

        This method generates edge ancillas (not corners) for the specified side.

        Parameters
        ----------
        bounds : PatchBounds
            Bounding box for the patch region.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        side : BoundarySide
            The boundary side to generate ancillas for.

        Returns
        -------
        tuple[frozenset[Coord2D], frozenset[Coord2D]]
            (X ancillas, Z ancillas) for the specified side (edges only, no corners).
        """
        x_ancillas: set[Coord2D] = set()
        z_ancillas: set[Coord2D] = set()
        check = self._should_add_boundary_ancilla
        edge_spec = boundary[side]

        if side == BoundarySide.TOP:
            # TOP boundary (y = y_min - 1)
            y = bounds.y_min - 1
            for x in range(bounds.x_min + 1, bounds.x_max):
                rel_x = x - bounds.x_min
                rel_x_mod4 = rel_x % 4
                if check(edge_spec, EdgeSpecValue.X, rel_x_mod4, 1):
                    x_ancillas.add(Coord2D(x, y))
                if check(edge_spec, EdgeSpecValue.Z, rel_x_mod4, 3):
                    z_ancillas.add(Coord2D(x, y))

        elif side == BoundarySide.BOTTOM:
            # BOTTOM boundary (y = y_max + 1)
            y = bounds.y_max + 1
            for x in range(bounds.x_min + 1, bounds.x_max):
                rel_x = x - bounds.x_min
                rel_x_mod4 = rel_x % 4
                if check(edge_spec, EdgeSpecValue.X, rel_x_mod4, 3):
                    x_ancillas.add(Coord2D(x, y))
                if check(edge_spec, EdgeSpecValue.Z, rel_x_mod4, 1):
                    z_ancillas.add(Coord2D(x, y))

        elif side == BoundarySide.LEFT:
            # LEFT boundary (x = x_min - 1)
            x = bounds.x_min - 1
            for y in range(bounds.y_min + 1, bounds.y_max):
                rel_y = y - bounds.y_min
                rel_y_mod4 = rel_y % 4
                if check(edge_spec, EdgeSpecValue.X, rel_y_mod4, 1):
                    x_ancillas.add(Coord2D(x, y))
                if check(edge_spec, EdgeSpecValue.Z, rel_y_mod4, 3):
                    z_ancillas.add(Coord2D(x, y))

        elif side == BoundarySide.RIGHT:
            # RIGHT boundary (x = x_max + 1)
            x = bounds.x_max + 1
            for y in range(bounds.y_min + 1, bounds.y_max):
                rel_y = y - bounds.y_min
                rel_y_mod4 = rel_y % 4
                if check(edge_spec, EdgeSpecValue.X, rel_y_mod4, 3):
                    x_ancillas.add(Coord2D(x, y))
                if check(edge_spec, EdgeSpecValue.Z, rel_y_mod4, 1):
                    z_ancillas.add(Coord2D(x, y))

        return frozenset(x_ancillas), frozenset(z_ancillas)

    def _generate_pipe_boundary_ancillas(  # noqa: C901
        self,
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        direction: AxisDIRECTION2D,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate boundary ancilla coordinates for a pipe layout."""
        x_ancillas: set[Coord2D] = set()
        z_ancillas: set[Coord2D] = set()

        if direction == AxisDIRECTION2D.H:
            # Horizontal pipe: check TOP and BOTTOM boundaries
            center_x = (bounds.x_min + bounds.x_max) // 2
            for i in (-1, 1):
                x = center_x + i
                y_top = bounds.y_min - 1
                y_bottom = bounds.y_max + 1

                if boundary[BoundarySide.TOP] == EdgeSpecValue.X and (x + y_top) % 4 == 0:
                    x_ancillas.add(Coord2D(x, y_top))
                if boundary[BoundarySide.TOP] == EdgeSpecValue.Z and (x + y_top) % 4 == 2:  # noqa: PLR2004
                    z_ancillas.add(Coord2D(x, y_top))
                if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.X and (x + y_bottom) % 4 == 0:
                    x_ancillas.add(Coord2D(x, y_bottom))
                if boundary[BoundarySide.BOTTOM] == EdgeSpecValue.Z and (x + y_bottom) % 4 == 2:  # noqa: PLR2004
                    z_ancillas.add(Coord2D(x, y_bottom))
        else:  # direction == AxisDIRECTION2D.V
            # Vertical pipe: check LEFT and RIGHT boundaries
            center_y = (bounds.y_min + bounds.y_max) // 2
            for i in (-1, 1):
                x_left = bounds.x_min - 1
                x_right = bounds.x_max + 1
                y = center_y + i

                if boundary[BoundarySide.LEFT] == EdgeSpecValue.X and (x_left + y) % 4 == 0:
                    x_ancillas.add(Coord2D(x_left, y))
                if boundary[BoundarySide.LEFT] == EdgeSpecValue.Z and (x_left + y) % 4 == 2:  # noqa: PLR2004
                    z_ancillas.add(Coord2D(x_left, y))
                if boundary[BoundarySide.RIGHT] == EdgeSpecValue.X and (x_right + y) % 4 == 0:
                    x_ancillas.add(Coord2D(x_right, y))
                if boundary[BoundarySide.RIGHT] == EdgeSpecValue.Z and (x_right + y) % 4 == 2:  # noqa: PLR2004
                    z_ancillas.add(Coord2D(x_right, y))

        return frozenset(x_ancillas), frozenset(z_ancillas)

    def _boundary_path(  # noqa: C901
        self,
        data: frozenset[Coord2D],
        bounds: PatchBounds,
        side_a: BoundarySide,
        side_b: BoundarySide,
    ) -> list[Coord2D]:
        """Compute path between two boundaries using bounds for center calculation.

        Parameters
        ----------
        data : frozenset[Coord2D]
            Data qubit coordinates.
        bounds : PatchBounds
            Bounds of the patch (used for center calculation).
        side_a : BoundarySide
            Starting boundary side.
        side_b : BoundarySide
            Ending boundary side.

        Returns
        -------
        list[Coord2D]
            Ordered data-qubit coordinates from side_a to side_b through the center.
        """
        if not data:
            return []

        # Compute boundary positions and center from actual data coordinates
        # (bounds parameter kept for API consistency but not used here)
        _ = bounds  # Silence unused parameter warning
        xs = sorted({c.x for c in data})
        ys = sorted({c.y for c in data})
        min_x, max_x = xs[0], xs[-1]
        min_y, max_y = ys[0], ys[-1]
        center_x = xs[len(xs) // 2]
        center_y = ys[len(ys) // 2]

        boundary_pos = {
            BoundarySide.TOP: min_y,
            BoundarySide.BOTTOM: max_y,
            BoundarySide.LEFT: min_x,
            BoundarySide.RIGHT: max_x,
        }

        vertical = {BoundarySide.TOP, BoundarySide.BOTTOM}
        horizontal = {BoundarySide.LEFT, BoundarySide.RIGHT}

        def range_step2(start: int, end: int) -> list[int]:
            """Inclusive range in steps of 2."""
            if start == end:
                return [start]
            step = 2 if end > start else -2
            return list(range(start, end + step, step))

        def vertical_segment(src: BoundarySide, dst: BoundarySide) -> list[Coord2D]:
            y_start = boundary_pos[src]
            y_end = boundary_pos[dst]
            return [Coord2D(center_x, y) for y in range_step2(y_start, y_end) if Coord2D(center_x, y) in data]

        def horizontal_segment(src: BoundarySide, dst: BoundarySide) -> list[Coord2D]:
            x_start = boundary_pos[src]
            x_end = boundary_pos[dst]
            return [Coord2D(x, center_y) for x in range_step2(x_start, x_end) if Coord2D(x, center_y) in data]

        # Opposite sides -> straight line
        if side_a in vertical and side_b in vertical:
            return vertical_segment(side_a, side_b)
        if side_a in horizontal and side_b in horizontal:
            return horizontal_segment(side_a, side_b)

        # Adjacent sides -> L via center
        path: list[Coord2D] = []
        if side_a in vertical and side_b in horizontal:
            v_seg = [
                Coord2D(center_x, y)
                for y in range_step2(boundary_pos[side_a], center_y)
                if Coord2D(center_x, y) in data
            ]
            h_seg = [
                Coord2D(x, center_y)
                for x in range_step2(center_x, boundary_pos[side_b])
                if Coord2D(x, center_y) in data
            ]
            path.extend(v_seg)
            if h_seg and path and path[-1] == h_seg[0]:
                path.extend(h_seg[1:])
            else:
                path.extend(h_seg)
            return path

        if side_a in horizontal and side_b in vertical:
            h_seg = [
                Coord2D(x, center_y)
                for x in range_step2(boundary_pos[side_a], center_x)
                if Coord2D(x, center_y) in data
            ]
            v_seg = [
                Coord2D(center_x, y)
                for y in range_step2(center_y, boundary_pos[side_b])
                if Coord2D(center_x, y) in data
            ]
            path.extend(h_seg)
            if v_seg and path and path[-1] == v_seg[0]:
                path.extend(v_seg[1:])
            else:
                path.extend(v_seg)
            return path

        msg = f"Unsupported boundary pair: {side_a}, {side_b}."
        raise ValueError(msg)

    def _determine_move_vec(
        self,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        ancilla_type: EdgeSpecValue,
    ) -> AxisDIRECTION2D:
        """Determine the global movement vector direction for ancilla flow.

        Based on the boundary conditions and ancilla type, this method determines
        which axis the ancilla flow should move along. The flow moves orthogonally
        to the logical chain direction.

        This method only supports standard rotated surface code boundaries where
        X and Z are placed on opposite pairs of edges (e.g., TOP/BOTTOM=X and
        LEFT/RIGHT=Z, or vice versa).

        Parameters
        ----------
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        ancilla_type : EdgeSpecValue
            Type of ancilla (X or Z) to determine flow for.

        Returns
        -------
        AxisDIRECTION2D
            H (horizontal) if TOP/BOTTOM boundaries match ancilla_type,
            V (vertical) if LEFT/RIGHT boundaries match ancilla_type.

        Raises
        ------
        ValueError
            If the boundary conditions are not standard rotated surface code
            boundaries (X and Z on opposite pairs).
        """
        top = boundary[BoundarySide.TOP]
        bottom = boundary[BoundarySide.BOTTOM]
        left = boundary[BoundarySide.LEFT]
        right = boundary[BoundarySide.RIGHT]

        # Check for standard rotated surface code boundary:
        # X and Z must each be on opposite pairs of edges
        top_bottom_same = top == bottom
        left_right_same = left == right
        is_standard = top_bottom_same and left_right_same and {top, left} == {EdgeSpecValue.X, EdgeSpecValue.Z}

        if not is_standard:
            msg = (
                "Only standard rotated surface code boundaries are supported: "
                "X and Z must each be on opposite pairs of edges "
                "(e.g., TOP/BOTTOM=X and LEFT/RIGHT=Z, or vice versa)."
            )
            raise ValueError(msg)

        # Return movement direction based on ancilla type
        if top == ancilla_type:
            return AxisDIRECTION2D.H  # ancilla type matches TOP/BOTTOM, move horizontally
        return AxisDIRECTION2D.V  # ancilla type matches LEFT/RIGHT, move vertically

    def _determine_flow(
        self,
        node: Coord2D,
        data2d: frozenset[Coord2D],
        ancilla_type: EdgeSpecValue,
        move_vec: Coord2D,
    ) -> Coord2D:
        """Determine the flow target for a single ancilla qubit.

        Given an ancilla qubit position, this method finds a valid data qubit
        target based on the ancilla type's edge pattern and movement direction.
        The flow represents the causal dependency between ancilla measurements.

        Parameters
        ----------
        node : Coord2D
            The 2D coordinate of the ancilla qubit.
        data2d : frozenset[Coord2D]
            Set of all data qubit coordinates in the patch.
        ancilla_type : EdgeSpecValue
            Type of ancilla (X or Z), which determines the edge pattern used.
        move_vec : Coord2D
            Signed movement direction (one of (0, 1), (0, -1), (-1, 0), (1, 0)).

        Returns
        -------
        Coord2D
            A valid data qubit coordinate that the ancilla flows to.

        Raises
        ------
        ValueError
            If ancilla_type is neither X nor Z.
        ValueError
            If no valid target data qubit is found for the given ancilla.
        """
        if ancilla_type == EdgeSpecValue.X:
            edge_pattern = ANCILLA_EDGE_X
        elif ancilla_type == EdgeSpecValue.Z:
            edge_pattern = ANCILLA_EDGE_Z
        else:
            msg = f"Invalid ancilla type: {ancilla_type}."
            raise ValueError(msg)

        candidates = set()
        for dx, dy in edge_pattern:
            if move_vec.x != 0 and dx == move_vec.x:
                candidates.add(Coord2D(node.x + dx, node.y + dy))
            if move_vec.y != 0 and dy == move_vec.y:
                candidates.add(Coord2D(node.x + dx, node.y + dy))

        valid_targets = candidates.intersection(data2d)
        if not valid_targets:
            msg = f"No valid target found for ancilla at {node}."
            raise ValueError(msg)

        return min(valid_targets)  # Return deterministic result


# =============================================================================
# Singleton instance for backward compatibility
# =============================================================================

_default_layout = RotatedSurfaceCodeLayout()


# =============================================================================
# Static Facade (Backward Compatible API)
# =============================================================================


class RotatedSurfaceCodeLayoutBuilder:
    """Builder for rotated surface code patch layouts.

    This class provides static methods to generate qubit coordinates for
    cube and pipe patches in a rotated surface code layout. It serves as
    a backward-compatible facade delegating to RotatedSurfaceCodeLayout.

    For new code, consider using RotatedSurfaceCodeLayout directly for
    better testability and dependency injection.

    Examples
    --------
    >>> from lspattern.consts import BoundarySide, EdgeSpecValue
    >>> from lspattern.mytype import Coord2D, Coord3D
    >>> boundary = {
    ...     BoundarySide.TOP: EdgeSpecValue.X,
    ...     BoundarySide.BOTTOM: EdgeSpecValue.X,
    ...     BoundarySide.LEFT: EdgeSpecValue.Z,
    ...     BoundarySide.RIGHT: EdgeSpecValue.Z,
    ... }
    >>> coords = RotatedSurfaceCodeLayoutBuilder.cube(
    ...     code_distance=3,
    ...     global_pos=Coord2D(0, 0),
    ...     boundary=boundary,
    ... )
    >>> len(coords.data) > 0
    True
    """

    @staticmethod
    def cube(
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> PatchCoordinates:
        """Build complete cube layout coordinates.

        This method generates all qubit coordinates for a rotated surface code
        cube patch, including bulk, boundary, and corner qubits.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.

        Returns
        -------
        PatchCoordinates
            Complete coordinate sets for the cube.
        """
        return _default_layout.cube(code_distance, global_pos, boundary)

    @staticmethod
    def pipe(
        code_distance: int,
        global_pos_source: Coord3D,
        global_pos_target: Coord3D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> PatchCoordinates:
        """Build complete pipe layout coordinates.

        This method generates all qubit coordinates for a rotated surface code
        pipe patch, including bulk and boundary qubits.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos_source : Coord3D
            Global (x, y, z) position of the pipe source.
        global_pos_target : Coord3D
            Global (x, y, z) position of the pipe target.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the pipe.

        Returns
        -------
        PatchCoordinates
            Complete coordinate sets for the pipe.
        """
        return _default_layout.pipe(code_distance, global_pos_source, global_pos_target, boundary)

    # =========================================================================
    # Bounds Calculation (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def _cube_bounds(code_distance: int, offset: Coord2D) -> PatchBounds:
        """Create bounds for a cube patch."""
        return _default_layout.cube_bounds(code_distance, offset)

    @staticmethod
    def _pipe_bounds(
        code_distance: int,
        offset: Coord2D,
        direction: AxisDIRECTION2D,
    ) -> PatchBounds:
        """Create bounds for a pipe patch."""
        return _default_layout.pipe_bounds(code_distance, offset, direction)

    # =========================================================================
    # Offset Calculation (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def _compute_cube_offset(code_distance: int, global_pos: Coord2D) -> Coord2D:
        """Compute the offset for a cube patch based on global position."""
        return _default_layout.cube_offset(code_distance, global_pos)

    @staticmethod
    def _compute_pipe_offset(
        code_distance: int,
        global_pos_source: Coord3D,
        pipe_dir: BoundarySide,
    ) -> Coord2D:
        """Compute the offset for a pipe patch based on source position and direction."""
        return _default_layout._compute_pipe_offset(code_distance, global_pos_source, pipe_dir)

    # =========================================================================
    # Bulk Coordinate Generation (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def _generate_bulk_coords(bounds: PatchBounds) -> PatchCoordinates:
        """Generate bulk coordinates using the checkerboard pattern.

        The checkerboard pattern places qubits as follows:
        - Data qubits: absolute even x, absolute even y
        - X ancillas: absolute odd x, absolute odd y, (rel_x + rel_y) % 4 == 0
        - Z ancillas: absolute odd x, absolute odd y, (rel_x + rel_y) % 4 == 2
        """
        return generate_checkerboard_coords(bounds)

    # =========================================================================
    # Corner Handling (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def _get_corner_data_to_remove(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> frozenset[Coord2D]:
        """Determine which corner data qubits should be removed based on boundary conditions."""
        return _default_layout._get_corner_data_to_remove(bounds, boundary)

    @staticmethod
    def _get_corner_ancillas(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate corner ancilla coordinates for 'O' (open) boundaries."""
        return _default_layout._get_corner_ancillas(bounds, boundary)

    @staticmethod
    def _get_corner_ancillas_to_remove(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Determine which corner ancilla qubits should be removed based on boundary conditions."""
        return _default_layout._get_corner_ancillas_to_remove(bounds, boundary)

    @staticmethod
    def _get_corner_ancillas_for_side(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate corner ancilla coordinates for a specific side."""
        return _default_layout._get_corner_ancillas_for_side(bounds, boundary, side)

    # =========================================================================
    # Boundary Ancilla Generation (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def _should_add_boundary_ancilla(
        edge_spec: EdgeSpecValue,
        target_type: EdgeSpecValue,
        position_mod4: int,
        expected_mod4: int,
    ) -> bool:
        """Check if a boundary ancilla should be added at the given position."""
        return _default_layout._should_add_boundary_ancilla(edge_spec, target_type, position_mod4, expected_mod4)

    @staticmethod
    def _generate_cube_boundary_ancillas(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate boundary ancilla coordinates for a cube layout."""
        return _default_layout._generate_cube_boundary_ancillas(bounds, boundary)

    @staticmethod
    def _generate_boundary_ancillas_for_side(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate boundary ancilla coordinates for a specific side of a cube."""
        return _default_layout._generate_boundary_ancillas_for_side(bounds, boundary, side)

    @staticmethod
    def _generate_pipe_boundary_ancillas(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        direction: AxisDIRECTION2D,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate boundary ancilla coordinates for a pipe layout."""
        return _default_layout._generate_pipe_boundary_ancillas(bounds, boundary, direction)

    # =========================================================================
    # Pipe Direction Helpers (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def _pipe_axis_from_offset(offset_dir: BoundarySide) -> AxisDIRECTION2D:
        """Derive pipe axis direction from offset direction.

        Parameters
        ----------
        offset_dir : BoundarySide
            The direction from source to target (from pipe_offset()).

        Returns
        -------
        AxisDIRECTION2D
            H for horizontal pipe (RIGHT/LEFT), V for vertical pipe (TOP/BOTTOM).
        """
        return _default_layout.pipe_axis_from_offset(offset_dir)

    @staticmethod
    def pipe_offset(
        global_pos_source: Coord3D,
        global_pos_target: Coord3D,
    ) -> BoundarySide:
        """Calculate pipe offset direction from source to target positions.

        Parameters
        ----------
        global_pos_source : Coord3D
            Global (x, y, z) position of the pipe source.
        global_pos_target : Coord3D
            Global (x, y, z) position of the pipe target.

        Returns
        -------
        BoundarySide
            The direction from source to target.

        Raises
        ------
        ValueError
            If source and target positions don't form a valid pipe offset.
        """
        return _default_layout.pipe_offset(global_pos_source, global_pos_target)

    # =========================================================================
    # Boundary Path Methods (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def _boundary_path(
        data: frozenset[Coord2D],
        bounds: PatchBounds,
        side_a: BoundarySide,
        side_b: BoundarySide,
    ) -> list[Coord2D]:
        """Compute path between two boundaries using bounds for center calculation."""
        return _default_layout._boundary_path(data, bounds, side_a, side_b)

    @staticmethod
    def cube_boundary_path(
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side_a: BoundarySide,
        side_b: BoundarySide,
    ) -> list[Coord2D]:
        """Get data-qubit path from side_a to side_b through cube center.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        side_a : BoundarySide
            Starting boundary side.
        side_b : BoundarySide
            Ending boundary side.

        Returns
        -------
        list[Coord2D]
            Ordered data-qubit coordinates from side_a to side_b.
        """
        return _default_layout.cube_boundary_path(code_distance, global_pos, boundary, side_a, side_b)

    @staticmethod
    def cube_boundary_ancillas_for_side(
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Get ancilla coordinates for a specific boundary side of a cube.

        This method returns all ancilla qubits located on the specified boundary
        side, including corner ancillas. Each corner belongs to both adjacent
        sides (e.g., top_left corner belongs to both TOP and LEFT sides).

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        side : BoundarySide
            The boundary side to get ancillas for (TOP, BOTTOM, LEFT, RIGHT).

        Returns
        -------
        tuple[frozenset[Coord2D], frozenset[Coord2D]]
            (X ancillas, Z ancillas) for the specified boundary side,
            including corner ancillas that belong to this side.

        Examples
        --------
        >>> from lspattern.consts import BoundarySide, EdgeSpecValue
        >>> from lspattern.mytype import Coord2D
        >>> boundary = {
        ...     BoundarySide.TOP: EdgeSpecValue.X,
        ...     BoundarySide.BOTTOM: EdgeSpecValue.X,
        ...     BoundarySide.LEFT: EdgeSpecValue.Z,
        ...     BoundarySide.RIGHT: EdgeSpecValue.Z,
        ... }
        >>> x_anc, z_anc = RotatedSurfaceCodeLayoutBuilder.cube_boundary_ancillas_for_side(
        ...     code_distance=3,
        ...     global_pos=Coord2D(0, 0),
        ...     boundary=boundary,
        ...     side=BoundarySide.TOP,
        ... )
        """
        return _default_layout.cube_boundary_ancillas_for_side(code_distance, global_pos, boundary, side)

    @staticmethod
    def pipe_boundary_path(
        code_distance: int,
        global_pos_source: Coord3D,
        global_pos_target: Coord3D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side_a: BoundarySide,
        side_b: BoundarySide,
    ) -> list[Coord2D]:
        """Get data-qubit path from side_a to side_b through pipe center.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos_source : Coord3D
            Global (x, y, z) position of the pipe source.
        global_pos_target : Coord3D
            Global (x, y, z) position of the pipe target.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the pipe.
        side_a : BoundarySide
            Starting boundary side.
        side_b : BoundarySide
            Ending boundary side.

        Returns
        -------
        list[Coord2D]
            Ordered data-qubit coordinates from side_a to side_b.
        """
        return _default_layout.pipe_boundary_path(
            code_distance, global_pos_source, global_pos_target, boundary, side_a, side_b
        )

    # =========================================================================
    # Initial Ancilla Flow (Static methods for backward compatibility)
    # =========================================================================

    @staticmethod
    def construct_initial_ancilla_flow(
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        ancilla_type: EdgeSpecValue,
        move_vec: Coord2D,
        *,
        adjacent_data: AdjacentPipeData | None = None,
    ) -> dict[Coord2D, set[Coord2D]]:
        """Construct flow mapping for initial ancilla qubits.

        This method computes the flow relationships for ancilla qubits in
        initialization layers. The flow determines the causal dependencies
        between ancilla measurements.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        ancilla_type : EdgeSpecValue
            Type of ancilla qubit. "Z" for layer1 (Z-stabilizer), "X" for layer2 (X-stabilizer).
        move_vec : Coord2D
            Signed movement direction (one of (0, 1), (0, -1), (-1, 0), (1, 0)).
        adjacent_data : AdjacentPipeData | None
            Optional per-boundary-side data qubit coordinates from adjacent pipes.
            Used for cubes with O (open) boundaries to find flow targets in pipes.

        Returns
        -------
        dict[Coord2D, set[Coord2D]]
            Mapping from source 2D coordinate to target 2D coordinates for ancilla flow.
            Each source coordinate maps to a set of target coordinates.
        """
        return _default_layout.construct_initial_ancilla_flow(
            code_distance,
            global_pos,
            boundary,
            ancilla_type,
            move_vec,
            adjacent_data=adjacent_data,
        )

    @staticmethod
    def _determine_flow(
        node: Coord2D,
        data2d: frozenset[Coord2D],
        ancilla_type: EdgeSpecValue,
        move_vec: Coord2D,
    ) -> Coord2D:
        """Determine the flow target for a single ancilla qubit.

        Given an ancilla qubit position, this method finds a valid data qubit
        target based on the ancilla type's edge pattern and movement direction.
        The flow represents the causal dependency between ancilla measurements.

        Parameters
        ----------
        node : Coord2D
            The 2D coordinate of the ancilla qubit.
        data2d : frozenset[Coord2D]
            Set of all data qubit coordinates in the patch.
        ancilla_type : EdgeSpecValue
            Type of ancilla (X or Z), which determines the edge pattern used.
        move_vec : Coord2D
            Signed movement direction (one of (0, 1), (0, -1), (-1, 0), (1, 0)).

        Returns
        -------
        Coord2D
            A valid data qubit coordinate that the ancilla flows to.

        Raises
        ------
        ValueError
            If ancilla_type is neither X nor Z.
        ValueError
            If no valid target data qubit is found for the given ancilla.
        """
        if ancilla_type == EdgeSpecValue.X:
            edge_pattern = ANCILLA_EDGE_X
        elif ancilla_type == EdgeSpecValue.Z:
            edge_pattern = ANCILLA_EDGE_Z
        else:
            msg = f"Invalid ancilla type: {ancilla_type}."
            raise ValueError(msg)

        candidates = set()
        for dx, dy in edge_pattern:
            if move_vec.x != 0 and dx == move_vec.x:
                candidates.add(Coord2D(node.x + dx, node.y + dy))
            if move_vec.y != 0 and dy == move_vec.y:
                candidates.add(Coord2D(node.x + dx, node.y + dy))

        valid_targets = candidates.intersection(data2d)
        if not valid_targets:
            msg = f"No valid target found for ancilla at {node}."
            raise ValueError(msg)

        return min(valid_targets)  # Return deterministic result
