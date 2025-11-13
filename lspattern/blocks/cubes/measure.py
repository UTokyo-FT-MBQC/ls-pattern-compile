from __future__ import annotations

import operator
from typing import TYPE_CHECKING, ClassVar, cast

from graphqomb.common import Axis, AxisMeasBasis, MeasBasis, Sign
from graphqomb.graphstate import GraphState

from lspattern.blocks.base import compute_logical_op_direction
from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
from lspattern.consts import DIRECTIONS2D, BoundarySide, EdgeSpecValue, NodeRole, Observable, TemporalBoundarySpecValue
from lspattern.mytype import (
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PhysCoordLocal2D,
    QubitIndexLocal,
)
from lspattern.tiling.template import ScalableTemplate

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from lspattern.canvas import RHGCanvas


class _MeasureBase(RHGCube):
    """MBQC measurement block on the latest DATA layer (RHG parity-aware).

    Behavior
    --------
    - Determine the latest DATA layer footprint from the canvas logical boundary.
    - Create readout nodes only on DATA sites at that z-layer.
    - Register each readout node as both MBQC input and output (same q_index).
    - `out_ports` is empty: this block consumes the logical boundary.
    - Provide X-cap parity directives that close the top with the previous X layer.
    """

    def __init__(self, logical: int, basis: Axis, **kwargs: object) -> None:
        # Extract specific arguments for the parent dataclass
        d = cast("int", kwargs.pop("d", 3))
        edge_spec = cast("dict[BoundarySide, EdgeSpecValue] | None", kwargs.pop("edge_spec", None))
        source = cast("PatchCoordGlobal3D", kwargs.pop("source", PatchCoordGlobal3D((0, 0, 0))))
        sink = cast("PatchCoordGlobal3D | None", kwargs.pop("sink", None))
        template = cast(
            "ScalableTemplate",
            kwargs.pop("template", ScalableTemplate(d=3, edgespec={})),
        )
        in_ports = cast("set[QubitIndexLocal]", kwargs.pop("in_ports", set()))
        out_ports = cast("set[QubitIndexLocal]", kwargs.pop("out_ports", set()))
        cout_ports = cast("list[set[NodeIdLocal]]", kwargs.pop("cout_ports", []))

        # Initialize parent with explicit arguments
        super().__init__(
            d=d,
            edge_spec=edge_spec,
            source=source,
            sink=sink,
            template=template,
            in_ports=in_ports,
            out_ports=out_ports,
            cout_ports=cout_ports,
        )
        self.logical = logical
        self.meas_basis = AxisMeasBasis(basis, Sign.PLUS)  # is it actually override the base class's meas_basis?

    def emit(self, canvas: RHGCanvas) -> None:
        # This detailed implementation is out of scope for this milestone.
        # Kept as a placeholder to satisfy imports without runtime use.
        msg = "Measure blocks are not implemented in this build"
        raise NotImplementedError(msg)

    def _build_3d_graph(
        self,
    ) -> tuple[
        GraphState,
        dict[int, tuple[int, int, int]],
        dict[tuple[int, int, int], int],
        dict[int, str],
    ]:
        """Build 3D RHG graph structure optimized for measurement blocks.

        Measurement blocks only need a single layer (thickness=1) with data qubits only.
        This overrides the base implementation to avoid creating unnecessary temporal layers.
        """
        # Collect 2D coordinates from the evaluated template
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # For measurement blocks, use only single layer (max_t = 0)
        max_t = 0
        z0 = int(self.source[2]) * (2 * self.d)

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
        x2d: Sequence[tuple[int, int]],  # Unused for measurement blocks
        z2d: Sequence[tuple[int, int]],  # Unused for measurement blocks
        max_t: int,
        z0: int,
        node2coord: MutableMapping[int, tuple[int, int, int]],
        coord2node: MutableMapping[tuple[int, int, int], int],
        node2role: MutableMapping[int, str],
    ) -> dict[int, dict[tuple[int, int], int]]:
        """Assign nodes for measurement blocks - data qubits only, no ancillas.

        Note: x2d and z2d parameters are kept for interface compatibility but unused
        since measurement blocks don't need ancilla qubits.
        """
        _ = x2d, z2d  # Mark as intentionally unused
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}

        # For measurement blocks, only create data nodes at t=0
        for t_local in range(max_t + 1):
            t = z0 + t_local
            cur: dict[tuple[int, int], int] = {}

            # Only add data nodes, no ancilla nodes for measurement blocks
            for x, y in data2d:
                n = g.add_physical_node()
                node2coord[n] = (int(x), int(y), int(t))
                coord2node[int(x), int(y), int(t)] = n
                node2role[n] = NodeRole.DATA
                cur[int(x), int(y)] = n

            nodes_by_z[t] = cur

        return nodes_by_z

    def _assign_meas_bases(self, g: GraphState, meas_basis: MeasBasis) -> None:  # noqa: PLR6301
        """Assign measurement basis for non-output nodes."""
        for node in g.physical_nodes:
            g.assign_meas_basis(node, meas_basis)

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.in_ports = set(idx_map.values())

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        return super().set_out_ports(patch_coord)


class MeasureX(_MeasureBase):
    """Measure a logical block in the X basis."""

    def __init__(self, logical: int, **kwargs: object) -> None:
        super().__init__(logical, Axis.X, **kwargs)

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.in_ports = set(idx_map.values())

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # no out_ports for measurement blocks
        super().set_out_ports(patch_coord)

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:  # noqa: ARG002
        z_pos = self.source[2] * (2 * self.d)

        if self.edgespec is None:
            msg = f"edgespec must be defined to determine logical operator direction at {self.source}"
            raise ValueError(msg)
        direction = compute_logical_op_direction(self.edgespec, Observable.X)

        # Get actual data coordinates from template (after any shifts)
        data_coords = sorted(self.template.data_coords) if self.template.data_coords else []

        if direction == "V":
            # For vertical direction, select coords with y=min_y at regular x intervals
            min_y = min(y for _, y in data_coords) if data_coords else 0
            target_coords = [(x, y) for x, y in data_coords if y == min_y]
            target_coords = sorted(target_coords)[: self.d]  # Take first d coordinates
        else:  # direction == "H"
            # For horizontal direction, select coords with x=min_x at regular y intervals
            min_x = min(x for x, _ in data_coords) if data_coords else 0
            target_coords = [(x, y) for x, y in data_coords if x == min_x]
            target_coords = sorted(target_coords, key=operator.itemgetter(1))[: self.d]  # Sort by y, take first d

        cout_coords = [(x, y, z_pos) for x, y in target_coords]
        cout_group = {
            self.coord2node[PhysCoordGlobal3D(coord)]
            for coord in cout_coords
            if PhysCoordGlobal3D(coord) in self.coord2node
        }

        self.cout_ports = [cout_group]

    def _construct_detectors(self) -> None:
        x2d = self.template.x_coords

        # Use the actual z-coordinate where nodes are placed
        z0 = int(self.source[2]) * (2 * self.d)

        for x, y in x2d:
            node_group: set[NodeIdLocal] = set()
            for dx, dy in DIRECTIONS2D:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, z0)))
                if node_id is not None:
                    node_group.add(node_id)
            self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), {})[z0] = node_group


class MeasureZ(_MeasureBase):
    """Measure a logical block in the Z basis."""

    def __init__(self, logical: int, **kwargs: object) -> None:
        super().__init__(logical, Axis.Z, **kwargs)

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.in_ports = set(idx_map.values())

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        # no out_ports for measurement blocks
        super().set_out_ports(patch_coord)

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:  # noqa: ARG002
        z_pos = self.source[2] * (2 * self.d)

        if self.edgespec is None:
            msg = f"edgespec must be defined to determine logical operator direction at {self.source}"
            raise ValueError(msg)
        direction = compute_logical_op_direction(self.edgespec, Observable.Z)

        # Get actual data coordinates from template (after any shifts)
        data_coords = sorted(self.template.data_coords) if self.template.data_coords else []

        if direction == "V":
            # For vertical direction, select coords with y=min_y at regular x intervals
            min_y = min(y for _, y in data_coords) if data_coords else 0
            target_coords = [(x, y) for x, y in data_coords if y == min_y]
            target_coords = sorted(target_coords)[: self.d]  # Take first d coordinates
        else:  # direction == "H"
            # For horizontal direction, select coords with x=min_x at regular y intervals
            min_x = min(x for x, _ in data_coords) if data_coords else 0
            target_coords = [(x, y) for x, y in data_coords if x == min_x]
            target_coords = sorted(target_coords, key=operator.itemgetter(1))[: self.d]  # Sort by y, take first d

        cout_coords = [(x, y, z_pos) for x, y in target_coords]
        cout_group = {
            self.coord2node[PhysCoordGlobal3D(coord)]
            for coord in cout_coords
            if PhysCoordGlobal3D(coord) in self.coord2node
        }

        self.cout_ports = [cout_group]

    def _construct_detectors(self) -> None:
        z2d = self.template.z_coords

        # Use the actual z-coordinate where nodes are placed
        z0 = int(self.source[2]) * (2 * self.d)

        for x, y in z2d:
            node_group: set[NodeIdLocal] = set()
            for dx, dy in DIRECTIONS2D:
                node_id = self.coord2node.get(PhysCoordGlobal3D((x + dx, y + dy, z0)))
                if node_id is not None:
                    node_group.add(node_id)
            self.parity.checks.setdefault(PhysCoordLocal2D((x, y)), {})[z0] = node_group


class MeasureXSkeleton(RHGCubeSkeleton):
    """Skeleton for X-basis measurement blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "MeasureXSkelton"

    def to_block(self) -> MeasureX:
        """Materialize to a MeasureX (template evaluated, no local graph yet)."""
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)
        # Evaluate template coordinates
        self.template.to_tiling()

        block = MeasureX(
            logical=self.d,
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )
        block.final_layer = TemporalBoundarySpecValue.MX
        return block


class MeasureZSkeleton(RHGCubeSkeleton):
    """Skeleton for Z-basis measurement blocks in cube-shaped RHG structures."""

    name: ClassVar[str] = "MeasureZSkelton"

    def to_block(self) -> MeasureZ:
        """Materialize to a MeasureZ (template evaluated, no local graph yet)."""
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)
        # Evaluate template coordinates
        self.template.to_tiling()

        block = MeasureZ(
            logical=self.d,
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
        )
        block.final_layer = TemporalBoundarySpecValue.MZ
        return block
