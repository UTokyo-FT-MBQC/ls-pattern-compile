"""Patch region data structures and coordinate generation for rotated surface codes.

This module provides clean abstractions for generating qubit coordinates in
rotated surface code layouts, separating the logic into bulk, boundary, and
corner components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.mytype import AxisDIRECTION2D, Coord2D, Coord3D

if TYPE_CHECKING:
    from collections.abc import Mapping


# =============================================================================
# Constants
# =============================================================================

# Ancilla edge offsets for X-type stabilizers (order optimized for distance)
ANCILLA_EDGE_X: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, -1),
    (-1, -1),
    (-1, 1),
)

# Ancilla edge offsets for Z-type stabilizers (order optimized for distance)
ANCILLA_EDGE_Z: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class PatchBounds:
    """Bounding box for a patch region in 2D coordinates.

    Attributes
    ----------
    x_min : int
        Minimum x coordinate (inclusive).
    x_max : int
        Maximum x coordinate (inclusive).
    y_min : int
        Minimum y coordinate (inclusive).
    y_max : int
        Maximum y coordinate (inclusive).
    """

    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def center_x(self) -> int:
        """Center x coordinate (nearest even value for data qubits)."""
        raw_center = (self.x_min + self.x_max) // 2
        return raw_center if raw_center % 2 == 0 else raw_center + 1

    @property
    def center_y(self) -> int:
        """Center y coordinate (nearest even value for data qubits)."""
        raw_center = (self.y_min + self.y_max) // 2
        return raw_center if raw_center % 2 == 0 else raw_center + 1


@dataclass(frozen=True, slots=True)
class PatchCoordinates:
    """Complete coordinate sets for a patch region.

    This immutable container holds the generated coordinates for
    data qubits, X ancillas, and Z ancillas.

    Attributes
    ----------
    data : frozenset[Coord2D]
        Coordinates of data qubits.
    ancilla_x : frozenset[Coord2D]
        Coordinates of X-type ancilla qubits.
    ancilla_z : frozenset[Coord2D]
        Coordinates of Z-type ancilla qubits.
    """

    data: frozenset[Coord2D]
    ancilla_x: frozenset[Coord2D]
    ancilla_z: frozenset[Coord2D]

    def to_mutable_sets(self) -> tuple[set[Coord2D], set[Coord2D], set[Coord2D]]:
        """Convert to mutable sets for backward compatibility.

        Returns
        -------
        tuple[set[Coord2D], set[Coord2D], set[Coord2D]]
            (data_coords, x_ancilla_coords, z_ancilla_coords)
        """
        return set(self.data), set(self.ancilla_x), set(self.ancilla_z)


# =============================================================================
# Corner Rules (Module-level constants)
# =============================================================================

# Corner removal rules: (side1, side2) -> required EdgeSpecValue pairs
_CORNER_DATA_REMOVAL_RULES: dict[
    tuple[BoundarySide, BoundarySide],
    tuple[EdgeSpecValue, EdgeSpecValue],
] = {
    (BoundarySide.TOP, BoundarySide.RIGHT): (EdgeSpecValue.Z, EdgeSpecValue.Z),
    (BoundarySide.BOTTOM, BoundarySide.LEFT): (EdgeSpecValue.Z, EdgeSpecValue.Z),
    (BoundarySide.TOP, BoundarySide.LEFT): (EdgeSpecValue.X, EdgeSpecValue.X),
    (BoundarySide.BOTTOM, BoundarySide.RIGHT): (EdgeSpecValue.X, EdgeSpecValue.X),
}

# Corner coordinate positions relative to bounds: (x_attr, y_attr)
_CORNER_POSITIONS: dict[tuple[BoundarySide, BoundarySide], tuple[str, str]] = {
    (BoundarySide.TOP, BoundarySide.RIGHT): ("x_max", "y_min"),
    (BoundarySide.BOTTOM, BoundarySide.LEFT): ("x_min", "y_max"),
    (BoundarySide.TOP, BoundarySide.LEFT): ("x_min", "y_min"),
    (BoundarySide.BOTTOM, BoundarySide.RIGHT): ("x_max", "y_max"),
}


# =============================================================================
# Rotated Surface Code Layout Builder
# =============================================================================


class RotatedSurfaceCodeLayoutBuilder:
    """Builder for rotated surface code patch layouts.

    This class provides static methods to generate qubit coordinates for
    cube and pipe patches in a rotated surface code layout.

    Examples
    --------
    >>> from lspattern.consts import BoundarySide, EdgeSpecValue
    >>> from lspattern.new_blocks.mytype import Coord2D, Coord3D
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
        offset = RotatedSurfaceCodeLayoutBuilder._compute_cube_offset(code_distance, global_pos)
        bounds = RotatedSurfaceCodeLayoutBuilder._cube_bounds(code_distance, offset)

        # Generate components
        bulk = RotatedSurfaceCodeLayoutBuilder._generate_bulk_coords(bounds)
        corner_remove = RotatedSurfaceCodeLayoutBuilder._get_corner_data_to_remove(bounds, boundary)
        boundary_x, boundary_z = RotatedSurfaceCodeLayoutBuilder._generate_cube_boundary_ancillas(bounds, boundary)
        corner_x, corner_z = RotatedSurfaceCodeLayoutBuilder._get_corner_ancillas(bounds, boundary)

        return PatchCoordinates(
            data=bulk.data - corner_remove,
            ancilla_x=bulk.ancilla_x | boundary_x | corner_x,
            ancilla_z=bulk.ancilla_z | boundary_z | corner_z,
        )

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
        # Determine pipe direction from boundary or positions
        pipe_dir = RotatedSurfaceCodeLayoutBuilder.pipe_direction(boundary)
        print(f"Determined pipe direction: {pipe_dir}")
        pipe_offset_dir = RotatedSurfaceCodeLayoutBuilder.pipe_offset(global_pos_source, global_pos_target)

        offset = RotatedSurfaceCodeLayoutBuilder._compute_pipe_offset(code_distance, global_pos_source, pipe_offset_dir)
        bounds = RotatedSurfaceCodeLayoutBuilder._pipe_bounds(code_distance, offset, pipe_dir)

        # Generate components
        bulk = RotatedSurfaceCodeLayoutBuilder._generate_bulk_coords(bounds)
        boundary_x, boundary_z = RotatedSurfaceCodeLayoutBuilder._generate_pipe_boundary_ancillas(
            bounds, boundary, pipe_dir
        )

        return PatchCoordinates(
            data=bulk.data,
            ancilla_x=bulk.ancilla_x | boundary_x,
            ancilla_z=bulk.ancilla_z | boundary_z,
        )

    # =========================================================================
    # Bounds Calculation
    # =========================================================================

    @staticmethod
    def _cube_bounds(code_distance: int, offset: Coord2D) -> PatchBounds:
        """Create bounds for a cube patch."""
        return PatchBounds(
            x_min=offset.x,
            x_max=offset.x + 2 * (code_distance - 1),
            y_min=offset.y,
            y_max=offset.y + 2 * (code_distance - 1),
        )

    @staticmethod
    def _pipe_bounds(
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

    # =========================================================================
    # Offset Calculation
    # =========================================================================

    @staticmethod
    def _compute_cube_offset(code_distance: int, global_pos: Coord2D) -> Coord2D:
        """Compute the offset for a cube patch based on global position."""
        return Coord2D(
            x=2 * (code_distance + 1) * global_pos.x,
            y=2 * (code_distance + 1) * global_pos.y,
        )

    @staticmethod
    def _compute_pipe_offset(
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

    # =========================================================================
    # Bulk Coordinate Generation
    # =========================================================================

    @staticmethod
    def _generate_bulk_coords(bounds: PatchBounds) -> PatchCoordinates:
        """Generate bulk coordinates using the checkerboard pattern.

        The checkerboard pattern places qubits as follows:
        - Data qubits: absolute even x, absolute even y
        - X ancillas: absolute odd x, absolute odd y, (rel_x + rel_y) % 4 == 0
        - Z ancillas: absolute odd x, absolute odd y, (rel_x + rel_y) % 4 == 2
        """
        data: set[Coord2D] = set()
        ancilla_x: set[Coord2D] = set()
        ancilla_z: set[Coord2D] = set()

        for x in range(bounds.x_min, bounds.x_max + 1):
            for y in range(bounds.y_min, bounds.y_max + 1):
                # Data qubits use absolute even coordinates
                if x % 2 == 0 and y % 2 == 0:
                    data.add(Coord2D(x, y))
                # Ancillas use absolute odd coordinates but relative pattern for X/Z distinction
                elif x % 2 == 1 and y % 2 == 1:
                    if (x + y) % 4 == 0:
                        ancilla_x.add(Coord2D(x, y))
                    elif (x + y) % 4 == 2:  # noqa: PLR2004
                        ancilla_z.add(Coord2D(x, y))

        return PatchCoordinates(
            data=frozenset(data),
            ancilla_x=frozenset(ancilla_x),
            ancilla_z=frozenset(ancilla_z),
        )

    # =========================================================================
    # Corner Handling
    # =========================================================================

    @staticmethod
    def _get_corner_data_to_remove(
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> frozenset[Coord2D]:
        """Determine which corner data qubits should be removed based on boundary conditions."""
        to_remove: set[Coord2D] = set()

        for (side1, side2), (expected1, expected2) in _CORNER_DATA_REMOVAL_RULES.items():
            if (boundary[side1], boundary[side2]) == (expected1, expected2):
                x_attr, y_attr = _CORNER_POSITIONS[side1, side2]
                coord = Coord2D(getattr(bounds, x_attr), getattr(bounds, y_attr))
                to_remove.add(coord)

        return frozenset(to_remove)

    @staticmethod
    def _get_corner_ancillas(
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

    @staticmethod
    def _get_corner_ancillas_for_side(
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

    # =========================================================================
    # Boundary Ancilla Generation
    # =========================================================================

    @staticmethod
    def _should_add_boundary_ancilla(
        edge_spec: EdgeSpecValue,
        target_type: EdgeSpecValue,
        position_mod4: int,
        expected_mod4: int,
    ) -> bool:
        """Check if a boundary ancilla should be added at the given position."""
        return edge_spec in {target_type, EdgeSpecValue.O} and position_mod4 == expected_mod4

    @staticmethod
    def _generate_cube_boundary_ancillas(  # noqa: C901
        bounds: PatchBounds,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Generate boundary ancilla coordinates for a cube layout."""
        x_ancillas: set[Coord2D] = set()
        z_ancillas: set[Coord2D] = set()
        check = RotatedSurfaceCodeLayoutBuilder._should_add_boundary_ancilla

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

    @staticmethod
    def _generate_boundary_ancillas_for_side(  # noqa: C901
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
        check = RotatedSurfaceCodeLayoutBuilder._should_add_boundary_ancilla
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

    @staticmethod
    def _generate_pipe_boundary_ancillas(  # noqa: C901
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

    # =========================================================================
    # Pipe Direction Helpers
    # =========================================================================

    @staticmethod
    def pipe_direction(
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> AxisDIRECTION2D:
        """Determine pipe direction from boundary specifications.

        Parameters
        ----------
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the pipe.

        Returns
        -------
        AxisDIRECTION2D
            The pipe direction (H for horizontal, V for vertical).

        Raises
        ------
        ValueError
            If boundary specifications don't match a valid pipe direction.
        """
        vertical = boundary[BoundarySide.TOP] == EdgeSpecValue.O and boundary[BoundarySide.BOTTOM] == EdgeSpecValue.O
        horizontal = boundary[BoundarySide.LEFT] == EdgeSpecValue.O and boundary[BoundarySide.RIGHT] == EdgeSpecValue.O

        if horizontal and not vertical:
            return AxisDIRECTION2D.H
        if vertical and not horizontal:
            return AxisDIRECTION2D.V
        if horizontal and vertical:
            msg = "Both horizontal and vertical boundaries cannot be open for pipe layout."
            raise ValueError(msg)
        msg = "Either top-bottom or left-right boundaries must be open for pipe layout."
        raise ValueError(msg)

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

    # =========================================================================
    # Boundary Path Methods
    # =========================================================================

    @staticmethod
    def _boundary_path(  # noqa: C901
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
        offset = RotatedSurfaceCodeLayoutBuilder._compute_cube_offset(code_distance, global_pos)
        bounds = RotatedSurfaceCodeLayoutBuilder._cube_bounds(code_distance, offset)
        coords = RotatedSurfaceCodeLayoutBuilder.cube(code_distance, global_pos, boundary)
        return RotatedSurfaceCodeLayoutBuilder._boundary_path(coords.data, bounds, side_a, side_b)

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
        >>> from lspattern.new_blocks.mytype import Coord2D
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
        # Calculate bounds internally
        offset = RotatedSurfaceCodeLayoutBuilder._compute_cube_offset(code_distance, global_pos)
        bounds = RotatedSurfaceCodeLayoutBuilder._cube_bounds(code_distance, offset)

        # Get edge ancillas for this side
        edge_x, edge_z = RotatedSurfaceCodeLayoutBuilder._generate_boundary_ancillas_for_side(bounds, boundary, side)

        # Get corner ancillas for this side
        corner_x, corner_z = RotatedSurfaceCodeLayoutBuilder._get_corner_ancillas_for_side(bounds, boundary, side)

        # Combine edge and corner ancillas
        return frozenset(edge_x | corner_x), frozenset(edge_z | corner_z)

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
        pipe_dir = RotatedSurfaceCodeLayoutBuilder.pipe_direction(boundary)
        pipe_offset_dir = RotatedSurfaceCodeLayoutBuilder.pipe_offset(global_pos_source, global_pos_target)
        offset = RotatedSurfaceCodeLayoutBuilder._compute_pipe_offset(code_distance, global_pos_source, pipe_offset_dir)
        bounds = RotatedSurfaceCodeLayoutBuilder._pipe_bounds(code_distance, offset, pipe_dir)
        coords = RotatedSurfaceCodeLayoutBuilder.pipe(code_distance, global_pos_source, global_pos_target, boundary)
        return RotatedSurfaceCodeLayoutBuilder._boundary_path(coords.data, bounds, side_a, side_b)
