"""Initialization unit layer implementations.

Initialization layers prepare qubits in specific quantum states (|+⟩ or |0⟩).
These are typically used at the start of a quantum computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.accumulator import FlowAccumulator, ParityAccumulator
from lspattern.blocks.unit_layer import LayerData, UnitLayer
from lspattern.consts import NodeRole
from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D, PhysCoordLocal2D

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix_zx.graphstate import GraphState

    from lspattern.tiling.template import ScalableTemplate


class InitPlusUnitLayer(UnitLayer):
    """Initialization layer for |+⟩ state preparation.

    This layer initializes qubits in the |+⟩ state. The structure is:
    - Layer z: Data qubits + Z-check ancillas
    - Layer z+1: Data qubits + X-check ancillas

    Note: The first layer of ancillas is not deterministic (no previous layer
    to connect to), so they form dangling parity checks.
    """

    def build_layer(
        self,
        graph: GraphState,
        z_offset: int,
        template: ScalableTemplate,
    ) -> LayerData:
        """Build an initialization unit layer for |+⟩ state.

        Parameters
        ----------
        graph : GraphState
            The graph state to add nodes and edges to.
        z_offset : int
            Starting z-coordinate for this layer.
        template : ScalableTemplate
            Template providing data/ancilla coordinates.

        Returns
        -------
        LayerData
            Layer data with nodes, edges, and accumulators for this initialization layer.
        """
        # Extract 2D coordinates from template
        data2d = list(template.data_coords or [])
        x2d = list(template.x_coords or [])
        z2d = list(template.z_coords or [])

        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, NodeRole] = {}
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}

        # Layer 0 (z_offset): Data + Z-check ancillas
        z0 = z_offset
        layer0 = self._assign_nodes_at_z(graph, z0, data2d, z2d, NodeRole.ANCILLA_Z, node2coord, coord2node, node2role)
        nodes_by_z[z0] = layer0
        self.add_spatial_edges(graph, layer0)

        # Layer 1 (z_offset + 1): Data + X-check ancillas
        z1 = z_offset + 1
        layer1 = self._assign_nodes_at_z(graph, z1, data2d, x2d, NodeRole.ANCILLA_X, node2coord, coord2node, node2role)
        nodes_by_z[z1] = layer1
        self.add_spatial_edges(graph, layer1)

        # Add temporal edges between layers
        flow = FlowAccumulator()
        self._add_temporal_edges(graph, layer1, layer0, flow)

        # Construct schedule
        schedule = self._construct_schedule(nodes_by_z, node2role)

        # Construct parity checks (first layer ancillas are not deterministic)
        parity = self._construct_parity_init(z_offset, x2d, z2d, coord2node)

        return LayerData(
            nodes_by_z=nodes_by_z,
            node2coord=node2coord,
            coord2node=coord2node,
            node2role=node2role,
            schedule=schedule,
            flow=flow,
            parity=parity,
        )

    @staticmethod
    def _construct_parity_init(
        z_offset: int,
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],
        coord2node: Mapping[tuple[int, int, int], int],
    ) -> ParityAccumulator:
        """Construct parity checks for initialization layer.

        The first layer of ancillas is not deterministic (no previous layer to
        connect to), so they are stored as dangling parity.

        Parameters
        ----------
        z_offset : int
            Starting z-coordinate for this layer.
        x2d : collections.abc.Sequence[tuple[int, int]]
            X-check ancilla coordinates.
        z2d : collections.abc.Sequence[tuple[int, int]]
            Z-check ancilla coordinates.
        coord2node : collections.abc.Mapping[tuple[int, int, int], int]
            Coordinate to node mapping.

        Returns
        -------
        ParityAccumulator
            Parity accumulator with dangling parity for the first layer.
        """
        parity = ParityAccumulator()
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # First layer ancillas (z_offset) are not deterministic
        for x, y in z2d:
            node_id = coord2node.get(PhysCoordGlobal3D((x, y, z_offset)))
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {NodeIdLocal(node_id)}
            parity.ignore_dangling[PhysCoordLocal2D((x, y))] = True  # Mark as ignored

        # X-check layer (z_offset + 1)
        for x, y in x2d:
            node_id = coord2node.get(PhysCoordGlobal3D((x, y, z_offset + 1)))
            if node_id is None:
                continue
            coord = PhysCoordLocal2D((x, y))
            node_group = {NodeIdLocal(node_id)}
            parity.checks.setdefault(coord, {})[z_offset + 1] = node_group
            dangling_detectors[coord] = {NodeIdLocal(node_id)}

        # Add dangling detectors for connectivity to next layer
        for coord, nodes in dangling_detectors.items():
            parity.dangling_parity[coord] = nodes

        return parity


class InitZeroUnitLayer(UnitLayer):
    """Initialization layer for |0⟩ state preparation.

    This layer initializes qubits in the |0⟩ state. The structure is a single
    X-check layer (odd z-coordinate).

    Note: This produces a thinner layer (1 physical layer instead of 2) compared
    to standard memory layers.
    """

    def build_layer(
        self,
        graph: GraphState,
        z_offset: int,
        template: ScalableTemplate,
    ) -> LayerData:
        """Build an initialization unit layer for |0⟩ state.

        Parameters
        ----------
        graph : GraphState
            The graph state to add nodes and edges to.
        z_offset : int
            Starting z-coordinate for this layer (should be odd for |0⟩ init).
        template : ScalableTemplate
            Template providing data/ancilla coordinates.

        Returns
        -------
        LayerData
            Layer data with nodes, edges, and accumulators for this initialization layer.
        """
        # Extract 2D coordinates from template
        data2d = list(template.data_coords or [])
        x2d = list(template.x_coords or [])
        z2d = list(template.z_coords or [])

        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, NodeRole] = {}
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}

        # Single layer (z_offset + 1): Data + X-check ancillas
        z0 = z_offset + 1
        layer0 = self._assign_nodes_at_z(graph, z0, data2d, x2d, NodeRole.ANCILLA_X, node2coord, coord2node, node2role)
        nodes_by_z[z0] = layer0
        self.add_spatial_edges(graph, layer0)

        # No temporal edges for a single-layer initialization
        flow = FlowAccumulator()

        # Construct schedule
        schedule = self._construct_schedule(nodes_by_z, node2role)

        # Construct parity checks
        parity = self._construct_parity_zero(z0, x2d, z2d, coord2node)

        return LayerData(
            nodes_by_z=nodes_by_z,
            node2coord=node2coord,
            coord2node=coord2node,
            node2role=node2role,
            schedule=schedule,
            flow=flow,
            parity=parity,
        )

    @staticmethod
    def _construct_parity_zero(
        z_offset: int,
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],  # noqa: ARG004
        coord2node: Mapping[tuple[int, int, int], int],
    ) -> ParityAccumulator:
        """Construct parity checks for |0⟩ initialization layer.

        Parameters
        ----------
        z_offset : int
            Starting z-coordinate for this layer.
        x2d : collections.abc.Sequence[tuple[int, int]]
            X-check ancilla coordinates.
        z2d : collections.abc.Sequence[tuple[int, int]]
            Z-check ancilla coordinates (unused for |0⟩ init).
        coord2node : collections.abc.Mapping[tuple[int, int, int], int]
            Coordinate to node mapping.

        Returns
        -------
        ParityAccumulator
            Parity accumulator with dangling parity for connection to next layer.
        """
        parity = ParityAccumulator()
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # X-check layer (z_offset)
        for x, y in x2d:
            node_id = coord2node.get(PhysCoordGlobal3D((x, y, z_offset)))
            if node_id is None:
                continue
            dangling_detectors[PhysCoordLocal2D((x, y))] = {NodeIdLocal(node_id)}
            parity.ignore_dangling[PhysCoordLocal2D((x, y))] = True  # Mark as ignored

        # Add dangling detectors for connectivity to next layer
        for coord, nodes in dangling_detectors.items():
            parity.dangling_parity[coord] = nodes

        return parity
