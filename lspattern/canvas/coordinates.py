"""Coordinate mapping utilities for RHG canvas.

This module provides the CoordinateMapper class for managing bidirectional mappings
between node IDs and physical coordinates in 3D space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class CoordinateMapper:
    """Manages bidirectional mappings between node IDs and physical coordinates.

    This class handles the mapping of node IDs to 3D physical coordinates and vice versa,
    along with node role assignments. It provides utilities for coordinate transformations,
    boundary detection, and role-based node classification.

    Attributes
    ----------
    node2coord : dict[NodeIdLocal, PhysCoordGlobal3D]
        Mapping from node IDs to their physical coordinates.
    coord2node : dict[PhysCoordGlobal3D, NodeIdLocal]
        Mapping from physical coordinates to node IDs (inverse of node2coord).
    node2role : dict[NodeIdLocal, str]
        Mapping from node IDs to their roles ('data', 'ancilla_x', 'ancilla_z').

    Examples
    --------
    >>> mapper = CoordinateMapper()
    >>> mapper.add_node(NodeIdLocal(1), PhysCoordGlobal3D((0, 0, 0)), "data")
    >>> mapper.get_coordinate(NodeIdLocal(1))
    PhysCoordGlobal3D((0, 0, 0))
    >>> mapper.get_node(PhysCoordGlobal3D((0, 0, 0)))
    NodeIdLocal(1)
    """

    def __init__(self) -> None:
        """Initialize an empty CoordinateMapper."""
        self.node2coord: dict[NodeIdLocal, PhysCoordGlobal3D] = {}
        self.coord2node: dict[PhysCoordGlobal3D, NodeIdLocal] = {}
        self.node2role: dict[NodeIdLocal, str] = {}

    def add_node(self, node_id: NodeIdLocal, coord: PhysCoordGlobal3D, role: str | None = None) -> None:
        """Add a node with its coordinate and optional role.

        Parameters
        ----------
        node_id : NodeIdLocal
            The node identifier.
        coord : PhysCoordGlobal3D
            The physical coordinate in 3D space.
        role : str or None, optional
            The role of the node ('data', 'ancilla_x', 'ancilla_z'), by default None.
        """
        # Remove old coordinate mapping if node already exists
        if node_id in self.node2coord:
            old_coord = self.node2coord[node_id]
            if old_coord in self.coord2node and self.coord2node[old_coord] == node_id:
                del self.coord2node[old_coord]

        self.node2coord[node_id] = coord
        self.coord2node[coord] = node_id
        if role is not None:
            self.node2role[node_id] = role

    def get_coordinate(self, node_id: NodeIdLocal) -> PhysCoordGlobal3D | None:
        """Get the coordinate for a given node ID.

        Parameters
        ----------
        node_id : NodeIdLocal
            The node identifier.

        Returns
        -------
        PhysCoordGlobal3D or None
            The coordinate if found, None otherwise.
        """
        return self.node2coord.get(node_id)

    def get_node(self, coord: PhysCoordGlobal3D) -> NodeIdLocal | None:
        """Get the node ID for a given coordinate.

        Parameters
        ----------
        coord : PhysCoordGlobal3D
            The physical coordinate.

        Returns
        -------
        NodeIdLocal or None
            The node ID if found, None otherwise.
        """
        return self.coord2node.get(coord)

    def get_role(self, node_id: NodeIdLocal) -> str | None:
        """Get the role for a given node ID.

        Parameters
        ----------
        node_id : NodeIdLocal
            The node identifier.

        Returns
        -------
        str or None
            The role if assigned, None otherwise.
        """
        return self.node2role.get(node_id)

    def remap_nodes(self, node_map: Mapping[int, int]) -> None:
        """Remap node IDs according to the given mapping.

        Updates all internal mappings to reflect the new node IDs.

        Parameters
        ----------
        node_map : Mapping[int, int]
            Mapping from old node IDs to new node IDs.
        """
        # Remap node2coord
        self.node2coord = {NodeIdLocal(node_map.get(int(n), int(n))): c for n, c in self.node2coord.items()}
        # Remap coord2node
        self.coord2node = {c: NodeIdLocal(node_map.get(int(n), int(n))) for c, n in self.coord2node.items()}
        # Remap node2role
        self.node2role = {NodeIdLocal(node_map.get(int(n), int(n))): r for n, r in self.node2role.items()}

    def get_coordinate_bounds(self) -> tuple[int, int, int, int, int, int]:
        """Get min/max bounds for all coordinates.

        Returns
        -------
        tuple[int, int, int, int, int, int]
            A tuple (xmin, xmax, ymin, ymax, zmin, zmax).

        Raises
        ------
        ValueError
            If no coordinates are registered.
        """
        if not self.node2coord:
            msg = "No coordinates available to compute bounds"
            raise ValueError(msg)

        coords = list(self.node2coord.values())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    @staticmethod
    def create_face_checker(
        face: str, bounds: tuple[int, int, int, int, int, int], depths: list[int]
    ) -> Callable[[tuple[int, int, int]], bool]:
        """Create a function to check if a coordinate is on a requested face.

        Parameters
        ----------
        face : str
            The face identifier ('x+', 'x-', 'y+', 'y-', 'z+', 'z-').
        bounds : tuple[int, int, int, int, int, int]
            Coordinate bounds (xmin, xmax, ymin, ymax, zmin, zmax).
        depths : list[int]
            Depths from the face to include (e.g., [0] for surface, [0, 1] for two layers).

        Returns
        -------
        Callable[[tuple[int, int, int]], bool]
            A function that checks if a coordinate is on the specified face.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        f = face.strip().lower()

        def on_face(c: tuple[int, int, int]) -> bool:
            x, y, z = c
            if f == "x+":
                return x in {xmax - d for d in depths}
            if f == "x-":
                return x in {xmin + d for d in depths}
            if f == "y+":
                return y in {ymax - d for d in depths}
            if f == "y-":
                return y in {ymin + d for d in depths}
            if f == "z+":
                return z in {zmax - d for d in depths}
            # f == 'z-'
            return z in {zmin + d for d in depths}

        return on_face

    def classify_nodes_by_role(
        self, on_face_checker: Callable[[tuple[int, int, int]], bool]
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Classify nodes by their role on a specified face.

        Parameters
        ----------
        on_face_checker : Callable[[tuple[int, int, int]], bool]
            A function that checks if a coordinate is on the target face.

        Returns
        -------
        dict[str, list[PhysCoordGlobal3D]]
            A dictionary with keys 'data', 'xcheck', 'zcheck' mapping to lists of coordinates.
        """
        data: list[PhysCoordGlobal3D] = []
        xcheck: list[PhysCoordGlobal3D] = []
        zcheck: list[PhysCoordGlobal3D] = []

        for nid, c in self.node2coord.items():
            if not on_face_checker(c):
                continue
            role = (self.node2role.get(nid) or "").lower()
            if role == "ancilla_x":
                xcheck.append(c)
            elif role == "ancilla_z":
                zcheck.append(c)
            else:
                # Default to data if role is missing or 'data'
                data.append(c)

        return {"data": data, "xcheck": xcheck, "zcheck": zcheck}

    def copy(self) -> CoordinateMapper:
        """Create a deep copy of this CoordinateMapper.

        Returns
        -------
        CoordinateMapper
            A new CoordinateMapper with copied mappings.
        """
        new_mapper = CoordinateMapper()
        new_mapper.node2coord = self.node2coord.copy()
        new_mapper.coord2node = self.coord2node.copy()
        new_mapper.node2role = self.node2role.copy()
        return new_mapper

    def merge(
        self, other: CoordinateMapper, self_node_map: Mapping[int, int], other_node_map: Mapping[int, int]
    ) -> CoordinateMapper:
        """Merge this mapper with another, applying node remappings.

        Parameters
        ----------
        other : CoordinateMapper
            The other CoordinateMapper to merge.
        self_node_map : Mapping[int, int]
            Node remapping for self.
        other_node_map : Mapping[int, int]
            Node remapping for other.

        Returns
        -------
        CoordinateMapper
            A new merged CoordinateMapper.
        """
        merged = CoordinateMapper()

        # Remap and add from self
        for node_id, coord in self.node2coord.items():
            new_id = NodeIdLocal(self_node_map.get(int(node_id), int(node_id)))
            merged.node2coord[new_id] = coord
            merged.coord2node[coord] = new_id
            if node_id in self.node2role:
                merged.node2role[new_id] = self.node2role[node_id]

        # Remap and add from other
        for node_id, coord in other.node2coord.items():
            new_id = NodeIdLocal(other_node_map.get(int(node_id), int(node_id)))
            merged.node2coord[new_id] = coord
            merged.coord2node[coord] = new_id
            if node_id in other.node2role:
                merged.node2role[new_id] = other.node2role[node_id]

        return merged

    def clear(self) -> None:
        """Clear all mappings."""
        self.node2coord.clear()
        self.coord2node.clear()
        self.node2role.clear()
