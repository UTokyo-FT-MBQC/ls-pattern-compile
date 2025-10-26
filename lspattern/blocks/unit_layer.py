"""Abstract base class for unit layers in RHG blocks.

A unit layer represents the fundamental building block for constructing RHG cubes
and pipes: two physical layers containing one X-check and one Z-check cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.accumulator import (
    FlowAccumulator,
    ParityAccumulator,
    ScheduleAccumulator,
)
from lspattern.consts import DIRECTIONS2D, NodeRole
from lspattern.mytype import NodeIdGlobal, NodeIdLocal

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from graphqomb.graphstate import GraphState

    from lspattern.tiling.template import ScalableTemplate


@dataclass
class LayerData:
    """Data structure representing a single unit layer (2 physical layers).

    Attributes
    ----------
    nodes_by_z : dict[int, dict[tuple[int, int], int]]
        Mapping from z-coordinate to {(x,y): node_id} for nodes at that layer.
    node2coord : dict[int, tuple[int, int, int]]
        Mapping from node ID to (x, y, z) coordinate.
    coord2node : dict[tuple[int, int, int], int]
        Mapping from (x, y, z) coordinate to node ID.
    node2role : dict[int, NodeRole]
        Mapping from node ID to role ('data', 'ancilla_x', 'ancilla_z').
    schedule : ScheduleAccumulator
        Measurement schedule for this layer.
    flow : FlowAccumulator
        Flow accumulator for this layer.
    parity : ParityAccumulator
        Parity accumulator for this layer.
    """

    nodes_by_z: dict[int, dict[tuple[int, int], int]] = field(default_factory=dict)
    node2coord: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    coord2node: dict[tuple[int, int, int], int] = field(default_factory=dict)
    node2role: dict[int, NodeRole] = field(default_factory=dict)
    schedule: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(default_factory=ParityAccumulator)


class UnitLayer(ABC):
    """Abstract base class for 2-layer unit (1 Z-check + 1 X-check cycle).

    A UnitLayer encapsulates the logic for building two physical layers of an RHG
    block, typically consisting of:
    - Layer 1: Data qubits + Z-check ancillas (even z-coordinate)
    - Layer 2: Data qubits + X-check ancillas (odd z-coordinate)

    Subclasses should implement the `build_layer` method to define the specific
    structure and connectivity for different block types (memory, initialization, etc.).
    """

    @abstractmethod
    def build_layer(
        self,
        graph: GraphState,
        z_offset: int,
        template: ScalableTemplate,
    ) -> LayerData:
        """Build a single unit layer at the given z_offset.

        This method should add nodes and edges to the provided graph and return
        the LayerData containing all metadata for this layer.

        Parameters
        ----------
        graph : GraphState
            The graph state to add nodes and edges to.
        z_offset : int
            Starting z-coordinate for this layer. The layer will occupy z_offset
            and z_offset+1.
        template : ScalableTemplate
            Template providing data/ancilla coordinates in 2D. The template should
            have been evaluated (to_tiling() called) before being passed here.

        Returns
        -------
        LayerData
            Layer data including nodes, coordinates, roles, and accumulators.
        """

    @staticmethod
    def _assign_nodes_at_z(
        graph: GraphState,
        z: int,
        data2d: Sequence[tuple[int, int]],
        ancilla2d: Sequence[tuple[int, int]],
        ancilla_role: NodeRole,
        node2coord: MutableMapping[int, tuple[int, int, int]],
        coord2node: MutableMapping[tuple[int, int, int], int],
        node2role: MutableMapping[int, NodeRole],
    ) -> dict[tuple[int, int], int]:
        """Assign nodes for a single z-layer.

        Parameters
        ----------
        graph : GraphState
            Graph to add nodes to.
        z : int
            Z-coordinate for this layer.
        data2d : Sequence[tuple[int, int]]
            2D coordinates for data qubits.
        ancilla2d : Sequence[tuple[int, int]]
            2D coordinates for ancilla qubits.
        ancilla_role : NodeRole
            Role for ancilla qubits (NodeRole.ANCILLA_X or NodeRole.ANCILLA_Z).
        node2coord : MutableMapping[int, tuple[int, int, int]]
            Mapping to populate with node ID -> coordinate.
        coord2node : MutableMapping[tuple[int, int, int], int]
            Mapping to populate with coordinate -> node ID.
        node2role : MutableMapping[int, NodeRole]
            Mapping to populate with node ID -> role.

        Returns
        -------
        dict[tuple[int, int], int]
            Mapping from (x, y) to node ID for this layer.
        """
        layer_nodes: dict[tuple[int, int], int] = {}

        # Add data nodes
        for x, y in data2d:
            n = graph.add_physical_node()
            node2coord[n] = (int(x), int(y), int(z))
            coord2node[int(x), int(y), int(z)] = n
            node2role[n] = NodeRole.DATA
            layer_nodes[int(x), int(y)] = n

        # Add ancilla nodes
        for x, y in ancilla2d:
            n = graph.add_physical_node()
            node2coord[n] = (int(x), int(y), int(z))
            coord2node[int(x), int(y), int(z)] = n
            node2role[n] = ancilla_role
            layer_nodes[int(x), int(y)] = n

        return layer_nodes

    @staticmethod
    def add_spatial_edges(
        graph: GraphState,
        layer_nodes: Mapping[tuple[int, int], int],
    ) -> None:
        """Add intra-layer spatial edges.

        Parameters
        ----------
        graph : GraphState
            Graph to add edges to.
        layer_nodes : collections.abc.Mapping[tuple[int, int], int]
            Mapping from (x, y) to node ID for this layer.
        """
        for (x, y), u in layer_nodes.items():
            for dx, dy in DIRECTIONS2D:
                xy2 = (x + dx, y + dy)
                v = layer_nodes.get(xy2)
                if v is not None and v > u:
                    graph.add_physical_edge(u, v)

    @staticmethod
    def _add_temporal_edges(
        graph: GraphState,
        curr_layer: Mapping[tuple[int, int], int],
        prev_layer: Mapping[tuple[int, int], int] | None,
        flow: FlowAccumulator,
    ) -> None:
        """Add inter-layer temporal edges.

        Parameters
        ----------
        graph : GraphState
            Graph to add edges to.
        curr_layer : collections.abc.Mapping[tuple[int, int], int]
            Current layer nodes {(x,y): node_id}.
        prev_layer : collections.abc.Mapping[tuple[int, int], int] | None
            Previous layer nodes {(x,y): node_id}, or None if this is the first layer.
        flow : FlowAccumulator
            Flow accumulator to populate with temporal dependencies.
        """
        if prev_layer is None:
            return

        for xy, u in curr_layer.items():
            v = prev_layer.get(xy)
            if v is not None:
                graph.add_physical_edge(u, v)
                flow.flow.setdefault(NodeIdLocal(v), set()).add(NodeIdLocal(u))

    @staticmethod
    def _construct_schedule(
        nodes_by_z: Mapping[int, Mapping[tuple[int, int], int]],
        node2role: Mapping[int, NodeRole],
    ) -> ScheduleAccumulator:
        """Construct measurement schedule for this layer.

        Parameters
        ----------
        nodes_by_z : collections.abc.Mapping[int, collections.abc.Mapping[tuple[int, int], int]]
            Nodes organized by z-coordinate.
        node2role : collections.abc.Mapping[int, NodeRole]
            Node role mapping.

        Returns
        -------
        ScheduleAccumulator
            Schedule accumulator with ancillas at 2*t and data at 2*t+1.
        """
        schedule = ScheduleAccumulator()
        for t, nodes in nodes_by_z.items():
            # Separate nodes by role: ancillas first, then data
            ancilla_nodes = []
            data_nodes = []

            for node_id in nodes.values():
                role = node2role.get(NodeIdLocal(node_id), "")
                if role in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}:
                    ancilla_nodes.append(node_id)
                else:  # data or other roles
                    data_nodes.append(node_id)

            # Schedule ancillas and data at different time slots
            # ancillas at 2*t, data at 2*t+1 to ensure temporal separation
            if ancilla_nodes:
                ancilla_global_nodes = {NodeIdGlobal(node_id) for node_id in ancilla_nodes}
                schedule.schedule[2 * t] = ancilla_global_nodes

            if data_nodes:
                data_global_nodes = {NodeIdGlobal(node_id) for node_id in data_nodes}
                schedule.schedule[2 * t + 1] = data_global_nodes

        return schedule
