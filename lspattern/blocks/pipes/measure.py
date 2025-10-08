from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

from graphix_zx.common import Axis, AxisMeasBasis, Sign
from graphix_zx.graphstate import GraphState

from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D, PhysCoordGlobal3D, PhysCoordLocal2D, SpatialEdgeSpec
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from lspattern.consts.consts import PIPEDIRECTION

ANCILLA_TARGET_DIRECTION2D = {(1, 1), (1, -1), (-1, 1), (-1, -1)}


class _MeasurePipeBase(RHGPipe):
    """Base class for MBQC measurement pipes.

    Measurement pipes have spatial extents of either 1 x d x 1 or d x 1 x 1 (xyz).
    They consume logical qubits without producing outputs.
    """

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
        basis: Axis,
    ) -> None:
        edge_spec = edgespec or {}
        super().__init__(d=d, edge_spec=edge_spec)
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edge_spec)
        self.meas_basis = AxisMeasBasis(basis, Sign.PLUS)

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Set input ports from template data indices."""
        if patch_coord is not None and self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices(source_2d, patch_type="pipe", sink_patch=sink_2d)
        else:
            idx_map = self.template.get_data_indices()
        self.in_ports = set(idx_map.values())

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Set output ports from template data indices."""
        super().set_out_ports(patch_coord)

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Measurement pipes do not have classical output ports."""
        return super().set_cout_ports(patch_coord)

    def _assign_meas_bases(self, g: GraphState, meas_basis: object) -> None:  # noqa: ARG002
        """Assign measurement basis to all nodes."""
        for node in g.physical_nodes:
            g.assign_meas_basis(node, self.meas_basis)

    def _build_3d_graph(
        self,
    ) -> tuple[GraphState, dict[int, tuple[int, int, int]], dict[tuple[int, int, int], int], dict[int, str]]:
        """Build 3D RHG graph structure optimized for measurement pipes.

        Measurement pipes only need a single layer (thickness=1) with data qubits only.
        This overrides the base implementation to avoid creating unnecessary temporal layers.
        """
        # Collect 2D coordinates from the evaluated template
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # For measurement pipes, use only single layer (max_t = 0)
        max_t = 0
        z0 = int(self.source[2]) * 2 * self.d

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for single time slice only
        nodes_by_z = self._assign_nodes_by_timeslice(g, data2d, x2d, z2d, max_t, z0, node2coord, coord2node, node2role)

        self._assign_meas_bases(g, self.meas_basis)

        self._construct_schedule(nodes_by_z, node2role)

        # Add spatial edges only (no temporal edges needed for single layer)
        self._add_spatial_edges(g, nodes_by_z)
        # Skip temporal edges for single layer measurement

        return g, node2coord, coord2node, node2role

    def _assign_nodes_by_timeslice(  # noqa: PLR6301
        self,
        g: GraphState,
        data2d: Sequence[tuple[int, int]],
        x2d: Sequence[tuple[int, int]],  # Unused for measurement pipes
        z2d: Sequence[tuple[int, int]],  # Unused for measurement pipes
        max_t: int,
        z0: int,
        node2coord: MutableMapping[int, tuple[int, int, int]],
        coord2node: MutableMapping[tuple[int, int, int], int],
        node2role: MutableMapping[int, str],
    ) -> dict[int, dict[tuple[int, int], int]]:
        """Assign nodes for measurement pipes - data qubits only, no ancillas.

        Note: x2d and z2d parameters are kept for interface compatibility but unused
        since measurement pipes don't need ancilla qubits.
        """
        _ = x2d, z2d  # Mark as intentionally unused
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}

        # For measurement pipes, only create data nodes at t=0
        for t_local in range(max_t + 1):
            t = z0 + t_local
            cur: dict[tuple[int, int], int] = {}

            # Only add data nodes, no ancilla nodes for measurement pipes
            for x, y in data2d:
                n = g.add_physical_node()
                node2coord[n] = (int(x), int(y), int(t))
                coord2node[int(x), int(y), int(t)] = n
                node2role[n] = "data"
                cur[int(x), int(y)] = n

            nodes_by_z[t] = cur

        return nodes_by_z


@dataclass
class MeasureXPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for X-basis measurement pipes."""

    @overload
    def to_block(self) -> MeasureXPipe: ...

    @overload
    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> MeasureXPipe: ...

    def to_block(
        self, source: PatchCoordGlobal3D | None = None, sink: PatchCoordGlobal3D | None = None
    ) -> MeasureXPipe:
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = MeasureXPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        block.source = source
        block.sink = sink
        block.final_layer = "MX"
        return block


@dataclass
class MeasureZPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for Z-basis measurement pipes."""

    @overload
    def to_block(self) -> MeasureZPipe: ...

    @overload
    def to_block(self, source: PatchCoordGlobal3D, sink: PatchCoordGlobal3D) -> MeasureZPipe: ...

    def to_block(
        self, source: PatchCoordGlobal3D | None = None, sink: PatchCoordGlobal3D | None = None
    ) -> MeasureZPipe:
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = MeasureZPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        block.source = source
        block.sink = sink
        block.final_layer = "MZ"
        return block


class MeasureXPipe(_MeasurePipeBase):
    """X-basis measurement pipe with spatial extent 1 x d x 1 or d x 1 x 1."""

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        super().__init__(d, edgespec, direction, Axis.X)

    def _construct_detectors(self) -> None:
        """Construct Z-stabilizer detectors for X measurement."""
        x2d = self.template.x_coords

        z_offset = self.source[2] * (2 * self.d)
        height = max({coord[2] for coord in self.coord2node}, default=0) - z_offset + 1

        for z in range(height):
            for x, y in x2d:
                node_group: set[NodeIdLocal] = set()
                for dx, dy in ANCILLA_TARGET_DIRECTION2D:
                    node_id = self.coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, z + z_offset)))
                    if node_id is not None:
                        node_group.add(node_id)
                if node_group:
                    self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), {})[z + z_offset + 1] = (
                        node_group  # To group with neighboring X ancilla
                    )


class MeasureZPipe(_MeasurePipeBase):
    """Z-basis measurement pipe with spatial extent 1 x d x 1 or d x 1 x 1."""

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        super().__init__(d, edgespec, direction, Axis.Z)

    def _construct_detectors(self) -> None:
        """Construct X-stabilizer detectors for Z measurement."""
        z2d = self.template.z_coords

        z_offset = self.source[2] * (2 * self.d)
        height = max({coord[2] for coord in self.coord2node}, default=0) - z_offset + 1

        for z in range(height):
            for x, y in z2d:
                node_group: set[NodeIdLocal] = set()
                for dx, dy in ANCILLA_TARGET_DIRECTION2D:
                    node_id = self.coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, z + z_offset)))
                    if node_id is not None:
                        node_group.add(node_id)
                if node_group:
                    self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), {})[z + z_offset] = node_group
