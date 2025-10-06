"""Seam edge generation module for RHG canvas compilation.

This module provides functionality for generating CZ edges across cube-pipe seams
within temporal layers. Seam edges connect nodes at the boundaries between cubes
and pipes, ensuring proper entanglement in the quantum graph state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lspattern.blocks.pipes.measure import _MeasurePipeBase
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.mytype import (
    NodeIdLocal,
    PhysCoordGlobal3D,
    QubitGroupIdGlobal,
    TilingId,
)
from lspattern.utils import is_allowed_pair

# Constants
EDGE_TUPLE_SIZE = 2

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphix_zx.graphstate import BaseGraphState, GraphState

    from lspattern.blocks.cubes.base import RHGCube
    from lspattern.blocks.pipes.base import RHGPipe
    from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D


class SeamGenerator:
    """Generator for seam edges across cube-pipe boundaries.

    This class encapsulates the logic for generating CZ edges that connect
    nodes at the boundaries between cubes and pipes within a temporal layer.
    Seam edges are only created between allowed tiling ID pairs and follow
    specific geometric constraints.

    Attributes
    ----------
    cubes_ : dict[PatchCoordGlobal3D, RHGCube]
        Dictionary mapping patch coordinates to cube blocks.
    pipes_ : dict[PipeCoordGlobal3D, RHGPipe]
        Dictionary mapping pipe coordinates to pipe blocks.
    node2coord : dict[NodeIdLocal, PhysCoordGlobal3D]
        Bidirectional mapping from node IDs to physical 3D coordinates.
    coord2node : dict[PhysCoordGlobal3D, NodeIdLocal]
        Bidirectional mapping from physical 3D coordinates to node IDs.
    allowed_gid_pairs : set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]]
        Set of allowed tiling ID pairs for seam edge generation.
    """

    def __init__(
        self,
        cubes: dict[PatchCoordGlobal3D, RHGCube],
        pipes: dict[PipeCoordGlobal3D, RHGPipe],
        node2coord: dict[NodeIdLocal, PhysCoordGlobal3D],
        coord2node: dict[PhysCoordGlobal3D, NodeIdLocal],
        allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]],
    ) -> None:
        """Initialize the SeamGenerator.

        Parameters
        ----------
        cubes : dict[PatchCoordGlobal3D, RHGCube]
            Dictionary of cube blocks indexed by patch coordinates.
        pipes : dict[PipeCoordGlobal3D, RHGPipe]
            Dictionary of pipe blocks indexed by pipe coordinates.
        node2coord : dict[NodeIdLocal, PhysCoordGlobal3D]
            Mapping from node IDs to physical coordinates.
        coord2node : dict[PhysCoordGlobal3D, NodeIdLocal]
            Mapping from physical coordinates to node IDs.
        allowed_gid_pairs : set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]]
            Set of allowed tiling ID pairs for connections.
        """
        self.cubes_ = cubes
        self.pipes_ = pipes
        self.node2coord = node2coord
        self.coord2node = coord2node
        self.allowed_gid_pairs = allowed_gid_pairs

    def add_seam_edges(
        self, g: BaseGraphState, coord_gid_2d: Mapping[tuple[int, int], QubitGroupIdGlobal]
    ) -> GraphState:
        """Add CZ edges across cube-pipe seams within the same temporal layer.

        Iterates through all nodes in the graph and creates edges between nodes
        at cube-pipe boundaries, respecting allowed tiling ID pairs and geometric
        constraints.

        Parameters
        ----------
        g : BaseGraphState
            The quantum graph state to add seam edges to.
        coord_gid_2d : Mapping[tuple[int, int], QubitGroupIdGlobal]
            2D coordinate to tiling group ID mapping.

        Returns
        -------
        GraphState
            The updated graph state with seam edges added.
        """
        # Build XY regions for cubes
        cube_xy_all = self._build_xy_regions(dict(coord_gid_2d))
        existing = self._get_existing_edges(g)

        for u, coord_u in list(self.node2coord.items()):
            xy_u = (int(coord_u[0]), int(coord_u[1]))
            gid_u = coord_gid_2d.get(xy_u)
            if gid_u is None:
                continue

            self._process_neighbor_connections(u, coord_u, gid_u, cube_xy_all, coord_gid_2d, g, existing)

        return cast("GraphState", g)  # TODO: use graphstate constructor once available

    def _build_xy_regions(self, coord_gid_2d: dict[tuple[int, int], QubitGroupIdGlobal]) -> set[tuple[int, int]]:
        """Build XY coordinate sets for cubes and update group ID mapping.

        Populates coord_gid_2d with tiling group IDs for all cube and pipe coordinates,
        and returns the set of XY coordinates belonging to cubes.

        Parameters
        ----------
        coord_gid_2d : dict[tuple[int, int], QubitGroupIdGlobal]
            Dictionary to populate with XY coordinate to group ID mappings.

        Returns
        -------
        set[tuple[int, int]]
            Set of XY coordinates belonging to cube regions.
        """
        cube_xy_all: set[tuple[int, int]] = set()

        # Build cube XY regions
        for blk in self.cubes_.values():
            t = blk.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                for x, y in coord_list or []:
                    xy = (int(x), int(y))
                    cube_xy_all.add(xy)
                    coord_gid_2d[xy] = QubitGroupIdGlobal(blk.get_tiling_id())

        # Build pipe XY regions
        for pipe in self.pipes_.values():
            t = pipe.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                for x, y in coord_list or []:
                    xy = (int(x), int(y))
                    coord_gid_2d[xy] = QubitGroupIdGlobal(pipe.get_tiling_id())

        return cube_xy_all

    @staticmethod
    def _get_existing_edges(g: BaseGraphState) -> set[tuple[int, int]]:
        """Get existing edges from graph to avoid duplicates.

        Parameters
        ----------
        g : BaseGraphState
            The graph state to extract edges from.

        Returns
        -------
        set[tuple[int, int]]
            Set of existing edges in canonical (sorted) form.
        """
        edges = g.physical_edges
        result: set[tuple[int, int]] = set()
        for u, v in edges:
            edge = tuple(sorted((int(u), int(v))))
            result.add((edge[0], edge[1]))
        return result

    def _is_measure_pipe_node(
        self,
        xy: tuple[int, int],
        cube_xy_all: set[tuple[int, int]],
    ) -> bool:
        """Check if a node belongs to a measure pipe.

        Measure pipe nodes should not have seam edges added to them.

        Parameters
        ----------
        xy : tuple[int, int]
            The XY coordinate to check.
        cube_xy_all : set[tuple[int, int]]
            Set of all XY coordinates belonging to cubes.

        Returns
        -------
        bool
            True if the node belongs to a measure pipe, False otherwise.
        """
        # If the node is in cube region, it's not a pipe node
        if xy in cube_xy_all:
            return False

        # Check if this XY coordinate belongs to any measure pipe
        for pipe in self.pipes_.values():
            if isinstance(pipe, _MeasurePipeBase):
                # Check if this xy coordinate is in the pipe's template
                for coord in pipe.template.data_coords or []:
                    if (int(coord[0]), int(coord[1])) == xy:
                        return True
        return False

    def _should_connect_nodes(
        self,
        xy_u: tuple[int, int],
        xy_v: tuple[int, int],
        cube_xy_all: set[tuple[int, int]],
        gid_u: QubitGroupIdGlobal,
        gid_v: QubitGroupIdGlobal,
    ) -> bool:
        """Check if two nodes should be connected based on XY regions and group IDs.

        Seam edges are only created between nodes where:
        1. One node is in a cube region and the other in a pipe region
        2. Neither node belongs to a measure pipe
        3. The tiling IDs form an allowed pair

        Parameters
        ----------
        xy_u : tuple[int, int]
            XY coordinate of first node.
        xy_v : tuple[int, int]
            XY coordinate of second node.
        cube_xy_all : set[tuple[int, int]]
            Set of all XY coordinates belonging to cubes.
        gid_u : QubitGroupIdGlobal
            Tiling group ID of first node.
        gid_v : QubitGroupIdGlobal
            Tiling group ID of second node.

        Returns
        -------
        bool
            True if the nodes should be connected, False otherwise.
        """
        u_in_cube = xy_u in cube_xy_all
        v_in_cube = xy_v in cube_xy_all

        # Skip connection if either node belongs to a measure pipe
        if self._is_measure_pipe_node(xy_u, cube_xy_all) or self._is_measure_pipe_node(xy_v, cube_xy_all):
            return False

        # Connect iff one is in cube region, other in pipe region, and allowed pair
        return u_in_cube != v_in_cube and is_allowed_pair(
            TilingId(int(gid_u)),
            TilingId(int(gid_v)),
            {(TilingId(int(a)), TilingId(int(b))) for a, b in self.allowed_gid_pairs},
        )

    def _process_neighbor_connections(
        self,
        u: NodeIdLocal,
        coord_u: PhysCoordGlobal3D,
        gid_u: QubitGroupIdGlobal,
        cube_xy_all: set[tuple[int, int]],
        coord_gid_2d: Mapping[tuple[int, int], QubitGroupIdGlobal],
        g: BaseGraphState,
        existing: set[tuple[int, int]],
    ) -> None:
        """Process connections to neighboring nodes.

        For a given node, checks all 2D neighbors (same z-plane) and creates
        seam edges where appropriate.

        Parameters
        ----------
        u : NodeIdLocal
            The source node ID.
        coord_u : PhysCoordGlobal3D
            3D coordinate of the source node.
        gid_u : QubitGroupIdGlobal
            Tiling group ID of the source node.
        cube_xy_all : set[tuple[int, int]]
            Set of all XY coordinates belonging to cubes.
        coord_gid_2d : Mapping[tuple[int, int], QubitGroupIdGlobal]
            2D coordinate to group ID mapping.
        g : BaseGraphState
            The graph state to add edges to.
        existing : set[tuple[int, int]]
            Set of existing edges to avoid duplicates.
        """
        xu, yu, zu = int(coord_u[0]), int(coord_u[1]), int(coord_u[2])
        xy_u = (xu, yu)

        for dx, dy, dz in DIRECTIONS3D:
            if dz != 0:
                continue  # we only connect within the same z plane

            xv, yv, zv = xu + int(dx), yu + int(dy), zu
            coord_v = PhysCoordGlobal3D((xv, yv, zv))
            v = self.coord2node.get(coord_v)
            if v is None or v == u:
                continue

            xy_v = (xv, yv)
            gid_v = coord_gid_2d.get(xy_v)
            if gid_v is None:
                continue

            if not self._should_connect_nodes(xy_u, xy_v, cube_xy_all, gid_u, gid_v):
                continue

            self._add_edge_if_valid(u, v, g, existing)

    @staticmethod
    def _add_edge_if_valid(
        u: NodeIdLocal,
        v: NodeIdLocal,
        g: BaseGraphState,
        existing: set[tuple[int, int]],
    ) -> None:
        """Add edge if valid and not duplicate.

        Does not add edges to output nodes or duplicate edges.

        Parameters
        ----------
        u : NodeIdLocal
            First node ID.
        v : NodeIdLocal
            Second node ID.
        g : BaseGraphState
            The graph state to add edge to.
        existing : set[tuple[int, int]]
            Set of existing edges to check for duplicates.
        """
        # Check if either node is an output node - if so, don't add edge
        if int(u) in g.output_node_indices or int(v) in g.output_node_indices:
            return

        # Avoid duplicates by canonical edge ordering
        sorted_edge = tuple(sorted((int(u), int(v))))
        if len(sorted_edge) == EDGE_TUPLE_SIZE:
            edge = (sorted_edge[0], sorted_edge[1])
            if edge not in existing:
                g.add_physical_edge(u, v)
                existing.add(edge)
