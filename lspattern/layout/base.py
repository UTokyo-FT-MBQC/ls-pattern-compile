"""Abstract base classes for topological code layout builders.

This module defines the interface contracts for topological code layout
generation, enabling different topological code variants (surface code,
color code, etc.) to be implemented as interchangeable components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lspattern.consts import BoundarySide, EdgeSpecValue
    from lspattern.layout.coordinates import PatchBounds, PatchCoordinates
    from lspattern.mytype import AxisDIRECTION2D, Coord2D, Coord3D


class CoordinateGenerator(ABC):
    """Abstract base class for coordinate generation.

    Responsible for generating complete coordinate sets for cube and pipe
    patches in a topological code layout.
    """

    @abstractmethod
    def cube(
        self,
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> PatchCoordinates:
        """Build complete cube layout coordinates.

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
        ...

    @abstractmethod
    def pipe(
        self,
        code_distance: int,
        global_pos_source: Coord3D,
        global_pos_target: Coord3D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
    ) -> PatchCoordinates:
        """Build complete pipe layout coordinates.

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
        ...


class BoundsCalculator(ABC):
    """Abstract base class for bounds calculation.

    Responsible for calculating bounding boxes and offsets for patches.
    """

    @abstractmethod
    def cube_bounds(self, code_distance: int, offset: Coord2D) -> PatchBounds:
        """Create bounds for a cube patch.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        offset : Coord2D
            Offset for the cube position.

        Returns
        -------
        PatchBounds
            Bounding box for the cube.
        """
        ...

    @abstractmethod
    def pipe_bounds(
        self,
        code_distance: int,
        offset: Coord2D,
        direction: AxisDIRECTION2D,
    ) -> PatchBounds:
        """Create bounds for a pipe patch.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        offset : Coord2D
            Offset for the pipe position.
        direction : AxisDIRECTION2D
            Direction of the pipe (H or V).

        Returns
        -------
        PatchBounds
            Bounding box for the pipe.
        """
        ...

    @abstractmethod
    def cube_offset(self, code_distance: int, global_pos: Coord2D) -> Coord2D:
        """Compute the offset for a cube patch based on global position.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global position of the cube.

        Returns
        -------
        Coord2D
            Offset coordinates for the cube.
        """
        ...


class BoundaryPathCalculator(ABC):
    """Abstract base class for boundary path calculation.

    Responsible for computing data qubit paths between boundaries.
    """

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...


class BoundaryAncillaRetriever(ABC):
    """Abstract base class for boundary ancilla retrieval.

    Responsible for retrieving ancilla coordinates for specific boundary sides.
    """

    @abstractmethod
    def cube_boundary_ancillas_for_side(
        self,
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        side: BoundarySide,
    ) -> tuple[frozenset[Coord2D], frozenset[Coord2D]]:
        """Get ancilla coordinates for a specific boundary side of a cube.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        side : BoundarySide
            The boundary side to get ancillas for.

        Returns
        -------
        tuple[frozenset[Coord2D], frozenset[Coord2D]]
            (X ancillas, Z ancillas) for the specified boundary side.
        """
        ...


class AncillaFlowConstructor(ABC):
    """Abstract base class for ancilla flow construction.

    Responsible for constructing flow mappings for ancilla qubits.
    """

    @abstractmethod
    def construct_initial_ancilla_flow(
        self,
        code_distance: int,
        global_pos: Coord2D,
        boundary: Mapping[BoundarySide, EdgeSpecValue],
        ancilla_type: EdgeSpecValue,
        move_vec: Coord2D,
    ) -> dict[Coord2D, set[Coord2D]]:
        """Construct flow mapping for initial ancilla qubits.

        Parameters
        ----------
        code_distance : int
            Code distance of the surface code.
        global_pos : Coord2D
            Global (x, y) position of the cube.
        boundary : Mapping[BoundarySide, EdgeSpecValue]
            Boundary specifications for the cube.
        ancilla_type : EdgeSpecValue
            Type of ancilla qubit (X or Z).
        move_vec : Coord2D
            Signed movement direction (one of (0, 1), (0, -1), (-1, 0), (1, 0)).

        Returns
        -------
        dict[Coord2D, set[Coord2D]]
            Mapping from source 2D coordinate to target 2D coordinates.
        """
        ...


class PipeDirectionHelper(ABC):
    """Abstract base class for pipe direction calculations.

    Responsible for computing pipe offset directions and axis information.
    """

    @abstractmethod
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
        """
        ...

    @abstractmethod
    def pipe_axis_from_offset(self, offset_dir: BoundarySide) -> AxisDIRECTION2D:
        """Derive pipe axis direction from offset direction.

        Parameters
        ----------
        offset_dir : BoundarySide
            The direction from source to target.

        Returns
        -------
        AxisDIRECTION2D
            H for horizontal pipe, V for vertical pipe.
        """
        ...


class TopologicalCodeLayoutBuilder(
    CoordinateGenerator,
    BoundsCalculator,
    BoundaryPathCalculator,
    BoundaryAncillaRetriever,
    AncillaFlowConstructor,
    PipeDirectionHelper,
):
    """Complete interface for topological code layout builders.

    This abstract class combines all layout-related responsibilities into
    a single interface. Implementations provide complete topological code
    layout generation capabilities.

    Subclasses must implement all abstract methods from:
    - CoordinateGenerator: cube(), pipe()
    - BoundsCalculator: cube_bounds(), pipe_bounds(), cube_offset()
    - BoundaryPathCalculator: cube_boundary_path(), pipe_boundary_path()
    - BoundaryAncillaRetriever: cube_boundary_ancillas_for_side()
    - AncillaFlowConstructor: construct_initial_ancilla_flow()
    - PipeDirectionHelper: pipe_offset(), pipe_axis_from_offset()
    """
