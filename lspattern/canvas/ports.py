"""Port management for RHG canvas.

This module provides the PortManager class for handling input/output/cout ports
in temporal layers and compiled canvases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D, PipeCoordGlobal3D

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class PortManager:
    """Manages input, output, and cout ports for RHG blocks in canvas.

    This class encapsulates all port-related operations including:
    - Port registration (in/out/cout)
    - Port grouping (cout_port_groups_cube/pipe)
    - Port lookup (cout_group_lookup_cube/pipe)
    - Node remapping for ports

    Attributes
    ----------
    in_portset : dict[PatchCoordGlobal3D, list[NodeIdLocal]]
        Input ports for cubes organized by patch coordinate
    out_portset : dict[PatchCoordGlobal3D, list[NodeIdLocal]]
        Output ports for cubes organized by patch coordinate
    in_portset_pipe : dict[PipeCoordGlobal3D, list[NodeIdLocal]]
        Input ports for pipes organized by pipe coordinate
    out_portset_pipe : dict[PipeCoordGlobal3D, list[NodeIdLocal]]
        Output ports for pipes organized by pipe coordinate
    cout_portset_cube : dict[PatchCoordGlobal3D, list[NodeIdLocal]]
        Cout (ancilla output) ports for cubes organized by patch coordinate (flat list)
    cout_portset_pipe : dict[PipeCoordGlobal3D, list[NodeIdLocal]]
        Cout (ancilla output) ports for pipes organized by pipe coordinate (flat list)
    cout_port_groups_cube : dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]]
        Grouped cout ports per cube patch for logical observable extraction
    cout_port_groups_pipe : dict[PipeCoordGlobal3D, list[list[NodeIdLocal]]]
        Grouped cout ports per pipe for logical observable extraction
    cout_group_lookup_cube : dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]]
        Reverse index from node id to (cube patch, group index)
    cout_group_lookup_pipe : dict[NodeIdLocal, tuple[PipeCoordGlobal3D, int]]
        Reverse index from node id to (pipe coord, group index)
    in_ports : list[NodeIdLocal]
        Flattened list of all input ports (from both cubes and pipes)
    out_ports : list[NodeIdLocal]
        Flattened list of all output ports (from both cubes and pipes)
    """

    def __init__(self) -> None:
        """Initialize empty port manager."""
        # Cube in/out ports
        self.in_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = {}
        self.out_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = {}

        # Pipe in/out ports
        self.in_portset_pipe: dict[PipeCoordGlobal3D, list[NodeIdLocal]] = {}
        self.out_portset_pipe: dict[PipeCoordGlobal3D, list[NodeIdLocal]] = {}

        # Cube cout ports
        self.cout_portset_cube: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = {}
        self.cout_port_groups_cube: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]] = {}
        self.cout_group_lookup_cube: dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]] = {}

        # Pipe cout ports
        self.cout_portset_pipe: dict[PipeCoordGlobal3D, list[NodeIdLocal]] = {}
        self.cout_port_groups_pipe: dict[PipeCoordGlobal3D, list[list[NodeIdLocal]]] = {}
        self.cout_group_lookup_pipe: dict[NodeIdLocal, tuple[PipeCoordGlobal3D, int]] = {}

        self.in_ports: list[NodeIdLocal] = []
        self.out_ports: list[NodeIdLocal] = []

    def register_cout_group_cube(
        self,
        patch_pos: PatchCoordGlobal3D,
        nodes: Sequence[NodeIdLocal],
    ) -> None:
        """Record a cout group for the given cube patch and keep caches in sync.

        Parameters
        ----------
        patch_pos : PatchCoordGlobal3D
            The patch coordinate where this cout group belongs
        nodes : list[NodeIdLocal]
            List of node IDs in this cout group
        """
        group_nodes = [NodeIdLocal(int(n)) for n in nodes if n is not None]
        if not group_nodes:
            return
        groups = self.cout_port_groups_cube.setdefault(patch_pos, [])
        index = len(groups)
        groups.append(group_nodes)
        flat = self.cout_portset_cube.setdefault(patch_pos, [])
        flat.extend(group_nodes)
        for node in group_nodes:
            self.cout_group_lookup_cube[node] = (patch_pos, index)

    def register_cout_group_pipe(
        self,
        pipe_coord: PipeCoordGlobal3D,
        nodes: Sequence[NodeIdLocal],
    ) -> None:
        """Record a cout group for the given pipe and keep caches in sync.

        Parameters
        ----------
        pipe_coord : PipeCoordGlobal3D
            The pipe coordinate (source, sink) where this cout group belongs
        nodes : list[NodeIdLocal]
            List of node IDs in this cout group
        """
        group_nodes = [NodeIdLocal(int(n)) for n in nodes if n is not None]
        if not group_nodes:
            return
        groups = self.cout_port_groups_pipe.setdefault(pipe_coord, [])
        index = len(groups)
        groups.append(group_nodes)
        flat = self.cout_portset_pipe.setdefault(pipe_coord, [])
        flat.extend(group_nodes)
        for node in group_nodes:
            self.cout_group_lookup_pipe[node] = (pipe_coord, index)

    def rebuild_cout_group_cache(self) -> None:  # noqa: C901
        """Recompute flat cout caches from grouped data.

        This method rebuilds cout_portset_cube/pipe and cout_group_lookup_cube/pipe
        from the authoritative cout_port_groups_cube/pipe data.
        """
        # Rebuild cube caches
        self.cout_portset_cube = {}
        self.cout_group_lookup_cube = {}
        for patch_pos, groups in self.cout_port_groups_cube.items():
            flat: list[NodeIdLocal] = []
            for idx, group in enumerate(groups):
                normalized = [NodeIdLocal(int(n)) for n in group if n is not None]
                if not normalized:
                    continue
                self.cout_port_groups_cube[patch_pos][idx] = normalized
                flat.extend(normalized)
                for node in normalized:
                    self.cout_group_lookup_cube[node] = (patch_pos, idx)
            if flat:
                self.cout_portset_cube[patch_pos] = flat

        # Rebuild pipe caches
        self.cout_portset_pipe = {}
        self.cout_group_lookup_pipe = {}
        for pipe_coord, groups in self.cout_port_groups_pipe.items():
            flat_pipe: list[NodeIdLocal] = []
            for idx, group in enumerate(groups):
                normalized = [NodeIdLocal(int(n)) for n in group if n is not None]
                if not normalized:
                    continue
                self.cout_port_groups_pipe[pipe_coord][idx] = normalized
                flat_pipe.extend(normalized)
                for node in normalized:
                    self.cout_group_lookup_pipe[node] = (pipe_coord, idx)
            if flat_pipe:
                self.cout_portset_pipe[pipe_coord] = flat_pipe

    def remap_ports(self, node_map: Mapping[int, int]) -> None:
        """Remap all ports with given node mapping.

        Parameters
        ----------
        node_map : Mapping[int, int]
            Mapping from old node IDs to new node IDs

        Notes
        -----
        This method remaps in_portset, out_portset, in_portset_pipe, out_portset_pipe,
        in_ports, out_ports, and cout-related structures using the provided node mapping.
        """
        # Remap cube in/out ports
        for p, nodes in self.in_portset.items():
            self.in_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        for p, nodes in self.out_portset.items():
            self.out_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]

        # Remap pipe in/out ports
        for pipe_coord, nodes in self.in_portset_pipe.items():
            self.in_portset_pipe[pipe_coord] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        for pipe_coord, nodes in self.out_portset_pipe.items():
            self.out_portset_pipe[pipe_coord] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]

        # Remap cube cout ports
        new_groups_cube: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]] = {}
        for patch_pos, groups in self.cout_port_groups_cube.items():
            remapped_groups: list[list[NodeIdLocal]] = []
            for group in groups:
                remapped = [NodeIdLocal(node_map.get(n, n)) for n in group]
                remapped_groups.append(remapped)
            new_groups_cube[patch_pos] = remapped_groups
        self.cout_port_groups_cube = new_groups_cube

        # Remap pipe cout ports
        new_groups_pipe: dict[PipeCoordGlobal3D, list[list[NodeIdLocal]]] = {}
        for pipe_coord, groups in self.cout_port_groups_pipe.items():
            remapped_groups_pipe: list[list[NodeIdLocal]] = []
            for group in groups:
                remapped = [NodeIdLocal(node_map.get(n, n)) for n in group]
                remapped_groups_pipe.append(remapped)
            new_groups_pipe[pipe_coord] = remapped_groups_pipe
        self.cout_port_groups_pipe = new_groups_pipe

        # Rebuild caches
        self.rebuild_cout_group_cache()

        # Remap flat port lists
        self.in_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.in_ports]
        self.out_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.out_ports]

    def add_in_ports(self, patch_pos: PatchCoordGlobal3D, nodes: Sequence[NodeIdLocal]) -> None:
        """Add input ports for a patch.

        Parameters
        ----------
        patch_pos : PatchCoordGlobal3D
            The patch coordinate
        nodes : list[NodeIdLocal]
            List of node IDs to add as input ports
        """
        # Filter out None values for consistency with register_cout_group
        valid_nodes = [NodeIdLocal(int(n)) for n in nodes if n is not None]
        if not valid_nodes:
            return
        self.in_portset.setdefault(patch_pos, []).extend(valid_nodes)
        self.in_ports.extend(valid_nodes)

    def add_out_ports(self, patch_pos: PatchCoordGlobal3D, nodes: Sequence[NodeIdLocal]) -> None:
        """Add output ports for a patch.

        Parameters
        ----------
        patch_pos : PatchCoordGlobal3D
            The patch coordinate
        nodes : list[NodeIdLocal]
            List of node IDs to add as output ports
        """
        # Filter out None values for consistency with register_cout_group
        valid_nodes = [NodeIdLocal(int(n)) for n in nodes if n is not None]
        if not valid_nodes:
            return
        self.out_portset.setdefault(patch_pos, []).extend(valid_nodes)
        self.out_ports.extend(valid_nodes)

    def add_in_ports_pipe(self, pipe_coord: PipeCoordGlobal3D, nodes: Sequence[NodeIdLocal]) -> None:
        """Add input ports for a pipe.

        Parameters
        ----------
        pipe_coord : PipeCoordGlobal3D
            The pipe coordinate (source, sink)
        nodes : list[NodeIdLocal]
            List of node IDs to add as input ports
        """
        # Filter out None values for consistency with register_cout_group
        valid_nodes = [NodeIdLocal(int(n)) for n in nodes if n is not None]
        if not valid_nodes:
            return
        self.in_portset_pipe.setdefault(pipe_coord, []).extend(valid_nodes)
        self.in_ports.extend(valid_nodes)

    def add_out_ports_pipe(self, pipe_coord: PipeCoordGlobal3D, nodes: Sequence[NodeIdLocal]) -> None:
        """Add output ports for a pipe.

        Parameters
        ----------
        pipe_coord : PipeCoordGlobal3D
            The pipe coordinate (source, sink)
        nodes : list[NodeIdLocal]
            List of node IDs to add as output ports
        """
        # Filter out None values for consistency with register_cout_group
        valid_nodes = [NodeIdLocal(int(n)) for n in nodes if n is not None]
        if not valid_nodes:
            return
        self.out_portset_pipe.setdefault(pipe_coord, []).extend(valid_nodes)
        self.out_ports.extend(valid_nodes)

    def get_cout_group_by_node(
        self, node: NodeIdLocal
    ) -> tuple[PatchCoordGlobal3D | PipeCoordGlobal3D, list[NodeIdLocal]] | None:
        """Get the cout group containing the given node.

        Parameters
        ----------
        node : NodeIdLocal
            The node ID to look up

        Returns
        -------
        tuple[PatchCoordGlobal3D | PipeCoordGlobal3D, list[NodeIdLocal]] | None
            Tuple of (coord, group_nodes) if found, None otherwise.
            coord is PatchCoordGlobal3D for cube groups, PipeCoordGlobal3D for pipe groups.
        """
        # Check cube groups first
        mapping = self.cout_group_lookup_cube.get(node)
        if mapping is not None:
            patch_pos, group_idx = mapping
            groups = self.cout_port_groups_cube.get(patch_pos)
            if groups is not None and group_idx < len(groups):
                return patch_pos, list(groups[group_idx])

        # Check pipe groups
        mapping_pipe = self.cout_group_lookup_pipe.get(node)
        if mapping_pipe is not None:
            pipe_coord, group_idx = mapping_pipe
            groups_pipe = self.cout_port_groups_pipe.get(pipe_coord)
            if groups_pipe is not None and group_idx < len(groups_pipe):
                return pipe_coord, list(groups_pipe[group_idx])

        return None

    def merge(  # noqa: C901
        self,
        other: PortManager,
        self_node_map: Mapping[int, int],
        other_node_map: Mapping[int, int],
        *,
        in_ports_from: str = "other",
    ) -> PortManager:
        """Merge this PortManager with another for temporal composition.

        Parameters
        ----------
        other : PortManager
            The other PortManager to merge with
        self_node_map : Mapping[int, int]
            Node mapping to apply to self's ports
        other_node_map : Mapping[int, int]
            Node mapping to apply to other's ports
        in_ports_from : str, optional
            Which source to take in_ports from ("self", "other", or "both").
            Default is "other" (temporal composition pattern).

        Returns
        -------
        PortManager
            A new merged PortManager

        Notes
        -----
        This method is designed for temporal layer composition where:
        - in_ports typically come from the next layer (other)
        - out_ports are merged from both layers
        - cout_port_groups_cube/pipe are merged from both layers
        """
        merged = PortManager()

        # Remap self and other
        self_remapped = self.copy()
        self_remapped.remap_ports(self_node_map)

        other_remapped = other.copy()
        other_remapped.remap_ports(other_node_map)

        # Merge in_ports based on strategy
        if in_ports_from == "other":
            for pos, nodes in other_remapped.in_portset.items():
                merged.add_in_ports(pos, nodes)
        elif in_ports_from == "self":
            for pos, nodes in self_remapped.in_portset.items():
                merged.add_in_ports(pos, nodes)
        elif in_ports_from == "both":
            for pos, nodes in self_remapped.in_portset.items():
                merged.add_in_ports(pos, nodes)
            for pos, nodes in other_remapped.in_portset.items():
                merged.add_in_ports(pos, nodes)
        else:
            msg = f"Invalid in_ports_from: {in_ports_from}. Must be 'self', 'other', or 'both'."
            raise ValueError(msg)

        # Merge out_ports from both
        for pos, nodes in self_remapped.out_portset.items():
            merged.add_out_ports(pos, nodes)
        for pos, nodes in other_remapped.out_portset.items():
            merged.add_out_ports(pos, nodes)

        # Merge pipe in_ports from both
        for pipe_coord, nodes in self_remapped.in_portset_pipe.items():
            merged.add_in_ports_pipe(pipe_coord, nodes)
        for pipe_coord, nodes in other_remapped.in_portset_pipe.items():
            merged.add_in_ports_pipe(pipe_coord, nodes)

        # Merge pipe out_ports from both
        for pipe_coord, nodes in self_remapped.out_portset_pipe.items():
            merged.add_out_ports_pipe(pipe_coord, nodes)
        for pipe_coord, nodes in other_remapped.out_portset_pipe.items():
            merged.add_out_ports_pipe(pipe_coord, nodes)

        # Merge cout_port_groups_cube from both
        for pos, groups in self_remapped.cout_port_groups_cube.items():
            for group in groups:
                merged.register_cout_group_cube(pos, group)
        for pos, groups in other_remapped.cout_port_groups_cube.items():
            for group in groups:
                merged.register_cout_group_cube(pos, group)

        # Merge cout_port_groups_pipe from both
        for pipe_coord, groups in self_remapped.cout_port_groups_pipe.items():
            for group in groups:
                merged.register_cout_group_pipe(pipe_coord, group)
        for pipe_coord, groups in other_remapped.cout_port_groups_pipe.items():
            for group in groups:
                merged.register_cout_group_pipe(pipe_coord, group)

        return merged

    def copy(self) -> PortManager:
        """Create a deep copy of this PortManager.

        Returns
        -------
        PortManager
            A new PortManager with copied data

        Notes
        -----
        When adding new fields to PortManager, remember to update this method
        to ensure they are properly copied.
        """
        new_manager = PortManager()
        # Copy cube in/out ports
        new_manager.in_portset = {k: list(v) for k, v in self.in_portset.items()}
        new_manager.out_portset = {k: list(v) for k, v in self.out_portset.items()}

        # Copy pipe in/out ports
        new_manager.in_portset_pipe = {k: list(v) for k, v in self.in_portset_pipe.items()}
        new_manager.out_portset_pipe = {k: list(v) for k, v in self.out_portset_pipe.items()}

        # Copy cube cout ports
        new_manager.cout_portset_cube = {k: list(v) for k, v in self.cout_portset_cube.items()}
        new_manager.cout_port_groups_cube = {k: [list(g) for g in v] for k, v in self.cout_port_groups_cube.items()}
        new_manager.cout_group_lookup_cube = dict(self.cout_group_lookup_cube)

        # Copy pipe cout ports
        new_manager.cout_portset_pipe = {k: list(v) for k, v in self.cout_portset_pipe.items()}
        new_manager.cout_port_groups_pipe = {k: [list(g) for g in v] for k, v in self.cout_port_groups_pipe.items()}
        new_manager.cout_group_lookup_pipe = dict(self.cout_group_lookup_pipe)

        new_manager.in_ports = list(self.in_ports)
        new_manager.out_ports = list(self.out_ports)
        return new_manager
