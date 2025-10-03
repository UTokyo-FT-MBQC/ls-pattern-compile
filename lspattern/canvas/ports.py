"""Port management for RHG canvas.

This module provides the PortManager class for handling input/output/cout ports
in temporal layers and compiled canvases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D

if TYPE_CHECKING:
    from collections.abc import Mapping


class PortManager:
    """Manages input, output, and cout ports for RHG blocks in canvas.

    This class encapsulates all port-related operations including:
    - Port registration (in/out/cout)
    - Port grouping (cout_port_groups)
    - Port lookup (cout_group_lookup)
    - Node remapping for ports

    Attributes
    ----------
    in_portset : dict[PatchCoordGlobal3D, list[NodeIdLocal]]
        Input ports organized by patch coordinate
    out_portset : dict[PatchCoordGlobal3D, list[NodeIdLocal]]
        Output ports organized by patch coordinate
    cout_portset : dict[PatchCoordGlobal3D, list[NodeIdLocal]]
        Cout (ancilla output) ports organized by patch coordinate (flat list)
    cout_port_groups : dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]]
        Grouped cout ports per patch for logical observable extraction
    cout_group_lookup : dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]]
        Reverse index from node id to (patch, group index)
    in_ports : list[NodeIdLocal]
        Flattened list of all input ports
    out_ports : list[NodeIdLocal]
        Flattened list of all output ports
    cout_ports : list[NodeIdLocal]
        Flattened list of all cout ports
    """

    def __init__(self) -> None:
        """Initialize empty port manager."""
        self.in_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = {}
        self.out_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = {}
        self.cout_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = {}
        self.cout_port_groups: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]] = {}
        self.cout_group_lookup: dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]] = {}
        self.in_ports: list[NodeIdLocal] = []
        self.out_ports: list[NodeIdLocal] = []
        self.cout_ports: list[NodeIdLocal] = []

    def register_cout_group(
        self,
        patch_pos: PatchCoordGlobal3D,
        nodes: list[NodeIdLocal],
    ) -> None:
        """Record a cout group for the given patch and keep caches in sync.

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
        groups = self.cout_port_groups.setdefault(patch_pos, [])
        index = len(groups)
        groups.append(group_nodes)
        flat = self.cout_portset.setdefault(patch_pos, [])
        flat.extend(group_nodes)
        self.cout_ports.extend(group_nodes)
        for node in group_nodes:
            self.cout_group_lookup[node] = (patch_pos, index)

    def rebuild_cout_group_cache(self) -> None:
        """Recompute flat cout caches from grouped data.

        This method rebuilds cout_portset, cout_ports, and cout_group_lookup
        from the authoritative cout_port_groups data.
        """
        self.cout_portset = {}
        self.cout_ports = []
        self.cout_group_lookup = {}
        for patch_pos, groups in self.cout_port_groups.items():
            flat: list[NodeIdLocal] = []
            for idx, group in enumerate(groups):
                normalized = [NodeIdLocal(int(n)) for n in group if n is not None]
                if not normalized:
                    continue
                self.cout_port_groups[patch_pos][idx] = normalized
                flat.extend(normalized)
                self.cout_ports.extend(normalized)
                for node in normalized:
                    self.cout_group_lookup[node] = (patch_pos, idx)
            if flat:
                self.cout_portset[patch_pos] = flat

    def remap_ports(self, node_map: Mapping[int, int]) -> None:
        """Remap all ports with given node mapping.

        Parameters
        ----------
        node_map : Mapping[int, int]
            Mapping from old node IDs to new node IDs

        Notes
        -----
        This method remaps in_portset, out_portset, in_ports, out_ports,
        and cout-related structures using the provided node mapping.

        For cout ports, the behavior depends on whether cout_port_groups is populated:
        - If cout_port_groups exists: Uses grouped structure and rebuilds flat caches
        - Otherwise: Remaps cout_portset and cout_ports directly

        This dual behavior ensures compatibility with both grouped and non-grouped
        cout port management patterns.
        """
        for p, nodes in self.in_portset.items():
            self.in_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        for p, nodes in self.out_portset.items():
            self.out_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        if self.cout_port_groups:
            new_groups: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]] = {}
            for patch_pos, groups in self.cout_port_groups.items():
                remapped_groups: list[list[NodeIdLocal]] = []
                for group in groups:
                    remapped = [NodeIdLocal(node_map.get(n, n)) for n in group]
                    remapped_groups.append(remapped)
                new_groups[patch_pos] = remapped_groups
            self.cout_port_groups = new_groups
            self.rebuild_cout_group_cache()
        else:
            for p, nodes in self.cout_portset.items():
                self.cout_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
            self.cout_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.cout_ports]
        self.in_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.in_ports]
        self.out_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.out_ports]

    def add_in_ports(self, patch_pos: PatchCoordGlobal3D, nodes: list[NodeIdLocal]) -> None:
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

    def add_out_ports(self, patch_pos: PatchCoordGlobal3D, nodes: list[NodeIdLocal]) -> None:
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

    def get_cout_group_by_node(self, node: NodeIdLocal) -> tuple[PatchCoordGlobal3D, list[NodeIdLocal]] | None:
        """Get the cout group containing the given node.

        Parameters
        ----------
        node : NodeIdLocal
            The node ID to look up

        Returns
        -------
        tuple[PatchCoordGlobal3D, list[NodeIdLocal]] | None
            Tuple of (patch_pos, group_nodes) if found, None otherwise
        """
        mapping = self.cout_group_lookup.get(node)
        if mapping is None:
            return None
        patch_pos, group_idx = mapping
        groups = self.cout_port_groups.get(patch_pos)
        if groups is None or group_idx >= len(groups):
            return None
        return patch_pos, list(groups[group_idx])

    def merge(
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
        - cout_port_groups are merged from both layers
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

        # Merge cout_port_groups from both
        for pos, groups in self_remapped.cout_port_groups.items():
            for group in groups:
                merged.register_cout_group(pos, group)
        for pos, groups in other_remapped.cout_port_groups.items():
            for group in groups:
                merged.register_cout_group(pos, group)

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
        # Copy all port-related data structures
        new_manager.in_portset = {k: list(v) for k, v in self.in_portset.items()}
        new_manager.out_portset = {k: list(v) for k, v in self.out_portset.items()}
        new_manager.cout_portset = {k: list(v) for k, v in self.cout_portset.items()}
        new_manager.cout_port_groups = {k: [list(g) for g in v] for k, v in self.cout_port_groups.items()}
        new_manager.cout_group_lookup = dict(self.cout_group_lookup)
        new_manager.in_ports = list(self.in_ports)
        new_manager.out_ports = list(self.out_ports)
        new_manager.cout_ports = list(self.cout_ports)
        return new_manager
