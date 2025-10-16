"""Measurement unit layer implementations.

Measurement layers perform final measurements of qubits in either X or Z basis.
These are typically used at the end of a quantum computation to read out results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.accumulator import FlowAccumulator, ParityAccumulator
from lspattern.blocks.unit_layer import LayerData, UnitLayer
from lspattern.consts import DIRECTIONS2D, NodeRole
from lspattern.mytype import NodeIdLocal, PhysCoordGlobal3D, PhysCoordLocal2D

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix_zx.graphstate import GraphState

    from lspattern.tiling.template import ScalableTemplate


class MeasureXUnitLayer(UnitLayer):
    """Measurement layer for X-basis measurement.

    This layer measures qubits in the X basis. The structure is:
    - Single layer at z_offset: Data qubits only
    - No ancilla qubits are created
    - Detectors are constructed based on X-check ancilla coordinates
    """

    def build_layer(
        self,
        graph: GraphState,
        z_offset: int,
        template: ScalableTemplate,
    ) -> LayerData:
        """Build a measurement unit layer for X-basis measurement.

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
            Layer data with nodes, edges, and accumulators for this measurement layer.
        """
        # Extract 2D coordinates from template
        data2d = list(template.data_coords or [])
        x2d = list(template.x_coords or [])

        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, NodeRole] = {}
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}

        # Single layer at z_offset: Data qubits only (no ancillas)
        z0 = z_offset
        layer0: dict[tuple[int, int], int] = {}

        # Add only data nodes
        for x, y in data2d:
            n = graph.add_physical_node()
            node2coord[n] = (int(x), int(y), int(z0))
            coord2node[int(x), int(y), int(z0)] = n
            node2role[n] = NodeRole.DATA
            layer0[int(x), int(y)] = n

        nodes_by_z[z0] = layer0
        self.add_spatial_edges(graph, layer0)

        # No temporal edges for measurement layer (it's the final layer)
        flow = FlowAccumulator()

        # Construct schedule
        schedule = self._construct_schedule(nodes_by_z, node2role)

        # Construct parity checks using X-check ancilla coordinates
        parity = self._construct_parity_measurex(z_offset, x2d, coord2node)

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
    def _construct_parity_measurex(
        z_offset: int,
        x2d: Sequence[tuple[int, int]],
        coord2node: Mapping[tuple[int, int, int], int],
    ) -> ParityAccumulator:
        """Construct parity checks for X-basis measurement layer.

        Detectors are constructed at X-check ancilla coordinates by grouping
        data qubits in the diagonal directions.

        Parameters
        ----------
        z_offset : int
            Z-coordinate for this layer.
        x2d : collections.abc.Sequence[tuple[int, int]]
            X-check ancilla coordinates.
        coord2node : collections.abc.Mapping[tuple[int, int, int], int]
            Coordinate to node mapping.

        Returns
        -------
        ParityAccumulator
            Parity accumulator with detectors for X-basis measurement.
        """
        parity = ParityAccumulator()

        # Construct detectors at X-check ancilla coordinates
        for x, y in x2d:
            node_group: set[NodeIdLocal] = set()
            for dx, dy in DIRECTIONS2D:
                node_id = coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, z_offset)))
                if node_id is not None:
                    node_group.add(NodeIdLocal(node_id))
            if node_group:
                parity.checks.setdefault(PhysCoordLocal2D((x, y)), {})[z_offset] = node_group

        return parity


class MeasureZUnitLayer(UnitLayer):
    """Measurement layer for Z-basis measurement.

    This layer measures qubits in the Z basis. The structure is:
    - Single layer at z_offset: Data qubits only
    - No ancilla qubits are created
    - Detectors are constructed based on Z-check ancilla coordinates
    """

    def build_layer(
        self,
        graph: GraphState,
        z_offset: int,
        template: ScalableTemplate,
    ) -> LayerData:
        """Build a measurement unit layer for Z-basis measurement.

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
            Layer data with nodes, edges, and accumulators for this measurement layer.
        """
        # Extract 2D coordinates from template
        data2d = list(template.data_coords or [])
        z2d = list(template.z_coords or [])

        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, NodeRole] = {}
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}

        # Single layer at z_offset: Data qubits only (no ancillas)
        z0 = z_offset
        layer0: dict[tuple[int, int], int] = {}

        # Add only data nodes
        for x, y in data2d:
            n = graph.add_physical_node()
            node2coord[n] = (int(x), int(y), int(z0))
            coord2node[int(x), int(y), int(z0)] = n
            node2role[n] = NodeRole.DATA
            layer0[int(x), int(y)] = n

        nodes_by_z[z0] = layer0
        self.add_spatial_edges(graph, layer0)

        # No temporal edges for measurement layer (it's the final layer)
        flow = FlowAccumulator()

        # Construct schedule
        schedule = self._construct_schedule(nodes_by_z, node2role)

        # Construct parity checks using Z-check ancilla coordinates
        parity = self._construct_parity_measurez(z_offset, z2d, coord2node)

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
    def _construct_parity_measurez(
        z_offset: int,
        z2d: Sequence[tuple[int, int]],
        coord2node: Mapping[tuple[int, int, int], int],
    ) -> ParityAccumulator:
        """Construct parity checks for Z-basis measurement layer.

        Detectors are constructed at Z-check ancilla coordinates by grouping
        data qubits in the diagonal directions.

        Parameters
        ----------
        z_offset : int
            Z-coordinate for this layer.
        z2d : collections.abc.Sequence[tuple[int, int]]
            Z-check ancilla coordinates.
        coord2node : collections.abc.Mapping[tuple[int, int, int], int]
            Coordinate to node mapping.

        Returns
        -------
        ParityAccumulator
            Parity accumulator with detectors for Z-basis measurement.
        """
        parity = ParityAccumulator()

        # Construct detectors at Z-check ancilla coordinates
        for x, y in z2d:
            node_group: set[NodeIdLocal] = set()
            for dx, dy in DIRECTIONS2D:
                node_id = coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, z_offset)))
                if node_id is not None:
                    node_group.add(NodeIdLocal(node_id))
            if node_group:
                parity.checks.setdefault(PhysCoordLocal2D((x, y)), {})[z_offset] = node_group

        return parity
