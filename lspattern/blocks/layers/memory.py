"""Memory unit layer implementation.

Memory layers preserve quantum information through standard X and Z stabilizer checks.
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


class MemoryUnitLayer(UnitLayer):
    """Standard memory layer: Z-check (even z) followed by X-check (odd z).

    This layer preserves quantum information by alternating X and Z stabilizer
    measurements. The structure is:
    - Layer z: Data qubits + Z-check ancillas (even z-coordinate)
    - Layer z+1: Data qubits + X-check ancillas (odd z-coordinate)
    """

    def build_layer(
        self,
        graph: GraphState,
        z_offset: int,
        template: ScalableTemplate,
    ) -> LayerData:
        """Build a memory unit layer at the given z_offset.

        Parameters
        ----------
        graph : GraphState
            The graph state to add nodes and edges to.
        z_offset : int
            Starting z-coordinate for this layer (must be even for standard memory).
        template : ScalableTemplate
            Template providing data/ancilla coordinates.

        Returns
        -------
        LayerData
            Layer data with nodes, edges, and accumulators for this memory layer.
        """
        # Extract 2D coordinates from template
        data2d = list(template.data_coords or [])
        x2d = list(template.x_coords or [])
        z2d = list(template.z_coords or [])

        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, NodeRole] = {}
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}

        for height in (0, 1):
            z0 = z_offset + height
            if z0 % 2 == 0:
                # Even layer: Data + Z-check ancillas
                layer = self._assign_nodes_at_z(
                    graph, z0, data2d, z2d, NodeRole.ANCILLA_Z, node2coord, coord2node, node2role
                )
            else:
                # Odd layer: Data + X-check ancillas
                layer = self._assign_nodes_at_z(
                    graph, z0, data2d, x2d, NodeRole.ANCILLA_X, node2coord, coord2node, node2role
                )
            nodes_by_z[z0] = layer
            self.add_spatial_edges(graph, layer)

        # Add temporal edges between layers
        flow = FlowAccumulator()
        self._add_temporal_edges(graph, nodes_by_z[z_offset + 1], nodes_by_z[z_offset], flow)

        # Construct schedule
        schedule = self._construct_schedule(nodes_by_z, node2role)

        # Construct parity checks (detectors)
        parity = self._construct_parity(z_offset, x2d, z2d, coord2node)

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
    def _construct_parity(
        z_offset: int,
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],
        coord2node: Mapping[tuple[int, int, int], int],
    ) -> ParityAccumulator:
        """Construct parity checks for this memory layer.

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
            Parity accumulator with checks and dangling parity for connection to next layer.
        """
        parity = ParityAccumulator()
        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        for height in (0, 1):
            current_z = z_offset + height
            coord_list = z2d if current_z % 2 == 0 else x2d
            for x, y in coord_list:
                node_id = coord2node.get(PhysCoordGlobal3D((x, y, current_z)))
                if node_id is None:
                    continue
                local_coord = PhysCoordLocal2D((x, y))
                node_group = {NodeIdLocal(node_id)}
                parity.checks.setdefault(local_coord, {})[current_z] = node_group
                dangling_detectors[local_coord] = {NodeIdLocal(node_id)}

        # Add dangling detectors for connectivity to next layer
        for coord, nodes in dangling_detectors.items():
            parity.dangling_parity[coord] = nodes

        return parity
