from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphqomb.graphstate import GraphState

from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton
from lspattern.consts import TemporalBoundarySpecValue
from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D, PhysCoordGlobal3D, PhysCoordLocal2D, SpatialEdgeSpec
from lspattern.tiling.template import RotatedPlanarPipetemplate
from lspattern.utils import get_direction

# Type alias for the return type of _build_3d_graph method
Build3DGraphReturn = tuple[
    GraphState,
    dict[int, tuple[int, int, int]],
    dict[tuple[int, int, int], int],
    dict[int, str],
]

if TYPE_CHECKING:
    from lspattern.consts.consts import PIPEDIRECTION


class _InitPipeBase(RHGPipe):
    """Base class for initialization pipes."""

    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec | None,
        direction: PIPEDIRECTION,
    ) -> None:
        edge_spec = edgespec or {}
        super().__init__(d=d, edge_spec=edge_spec)
        self.direction = direction
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edge_spec, direction=direction)

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Initialization pipes do not consume logical inputs."""
        super().set_in_ports(patch_coord)

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:  # noqa: ARG002
        """Expose all data indices from the template as logical outputs."""
        if self.source is not None and self.sink is not None:
            source_2d = (self.source[0], self.source[1])
            sink_2d = (self.sink[0], self.sink[1])
            idx_map = self.template.get_data_indices_pipe(source_2d, sink_2d)
        else:
            idx_map = self.template.get_data_indices_cube()
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Initialization pipes do not emit classical outputs."""
        super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Build X/Z parity detectors, deferring the first X seam ancilla pairing."""
        x2d = self.template.x_coords
        z2d = self.template.z_coords

        zmin = min({coord[2] for coord in self.coord2node}, default=0)
        zmax = max({coord[2] for coord in self.coord2node}, default=0)
        height = zmax - zmin + 1

        dangling_detectors: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        for dz in range(height):
            for x, y in x2d + z2d:
                coord = PhysCoordLocal2D((x, y))
                node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, zmin + dz)))
                if node_id is None:
                    continue
                if dz == 0:
                    # ancillas of first layer is not deterministic
                    dangling_detectors[coord] = {node_id}
                    self.parity.ignore_dangling[coord] = True
                else:
                    node_group = {node_id} | dangling_detectors.pop(coord, set())
                    self.parity.checks.setdefault(coord, {})[zmin + dz] = node_group
                    dangling_detectors[coord] = {node_id}

        # add dangling detectors for connectivity to next block
        for coord, nodes in dangling_detectors.items():
            self.parity.dangling_parity[coord] = nodes


@dataclass
class InitPlusPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for an InitPlus-style pipe.

    Behavior
    - If ``edgespec`` is ``None``, downstream components use direction-specific defaults:
      - Horizontal (RIGHT/LEFT): {TOP: 'O', BOTTOM: 'O', LEFT: 'X', RIGHT: 'Z'}
      - Vertical   (TOP/BOTTOM): {LEFT: 'O', RIGHT: 'O', TOP: 'X', BOTTOM: 'Z'}
    - Direction is inferred from ``source`` and ``sink`` in ``to_block`` via
      ``get_direction``.
    """

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitPlusPipe:
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitPlusPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = TemporalBoundarySpecValue.O
        return block


class InitPlusPipe(_InitPipeBase):
    """Plus-state initialization pipe with default RHG thickness."""

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:  # noqa: ARG002
        """Set classical outport for init plus

        Parameters
        ----------
        patch_coord : tuple[int, int] | None, optional
            global patch coordinates, by default None
        """
        z0 = int(self.source[2]) * (2 * int(self.d))  # Base z-offset per block
        ancilla_coords = self.template.z_coords if z0 % 2 == 0 else self.template.x_coords
        print("Initplus pipe coords", self.template)
        cout_port_set = set()
        for x, y in ancilla_coords:
            node_id = self.coord2node.get(PhysCoordGlobal3D((x, y, z0)))
            if node_id is not None:
                cout_port_set.add(node_id)
        self.cout_ports.append(cout_port_set)


@dataclass
class InitPlusPipeThinLayerSkeleton(RHGPipeSkeleton):
    """Skeleton for thin-layer Plus State initialization pipes in pipe-shaped RHG structures."""

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitPlusThinLayerPipe:
        """
        Return a template-holding block for single-layer initialization.

        Returns
        -------
        InitPlusThinLayerPipe
            A block containing the template with no local graph state.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitPlusThinLayerPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = TemporalBoundarySpecValue.O
        return block


class InitPlusThinLayerPipe(_InitPipeBase):
    """Thin-layer Plus State initialization pipe (height=3) for compose-based initialization."""

    def _build_3d_graph(self) -> Build3DGraphReturn:
        """Override to create single-layer graph with only 13 nodes (9 data + 4 ancilla) at z=2*d."""
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Calculate z-coordinate based on source position and 2*d
        d_val = int(self.d)
        z0 = int(self.source[2]) * (2 * d_val)  # Base z-offset per block
        start_layer_z = z0 + (2 * d_val) - 2
        max_t = 2

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for each time slice
        nodes_by_z = self._assign_nodes_by_timeslice(
            g, data2d, x2d, z2d, max_t, start_layer_z, node2coord, coord2node, node2role
        )

        self._construct_schedule(nodes_by_z, node2role)

        self._add_spatial_edges(g, nodes_by_z)
        self._add_temporal_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role


@dataclass
class InitZeroPipeSkeleton(RHGPipeSkeleton):
    """Skeleton for an InitZero-style pipe.

    Behavior
    - If ``edgespec`` is ``None``, downstream components use direction-specific defaults:
      - Horizontal (RIGHT/LEFT): {TOP: 'O', BOTTOM: 'O', LEFT: 'X', RIGHT: 'Z'}
      - Vertical   (TOP/BOTTOM): {LEFT: 'O', RIGHT: 'O', TOP: 'X', BOTTOM: 'Z'}
    - Direction is inferred from ``source`` and ``sink`` in ``to_block`` via
      ``get_direction``.
    """

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitZeroPipe:
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitZeroPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = TemporalBoundarySpecValue.O
        return block


class InitZeroPipe(_InitPipeBase):
    """Zero-state initialization pipe with standard RHG depth."""

    def _build_3d_graph(self) -> Build3DGraphReturn:
        """Override to create single-layer graph with only 13 nodes (9 data + 4 ancilla) at z=2*d."""
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Calculate z-coordinate based on source position and 2*d
        d_val = int(self.d)
        z0 = int(self.source[2]) * (2 * d_val)  # Base z-offset per block
        start_layer_z = z0 + 1
        max_t = 2 * self.d - 1

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for each time slice
        nodes_by_z = self._assign_nodes_by_timeslice(
            g, data2d, x2d, z2d, max_t, start_layer_z, node2coord, coord2node, node2role
        )

        self._construct_schedule(nodes_by_z, node2role)

        self._add_spatial_edges(g, nodes_by_z)
        self._add_temporal_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role


@dataclass
class InitZeroPipeThinLayerSkeleton(RHGPipeSkeleton):
    """Skeleton for thin-layer Zero State initialization pipes in pipe-shaped RHG structures."""

    def to_block(
        self,
        source: PatchCoordGlobal3D | None = None,
        sink: PatchCoordGlobal3D | None = None,
    ) -> InitZeroThinLayerPipe:
        """
        Return a template-holding block for single-layer initialization.

        Returns
        -------
        InitZeroThinLayerPipe
            A block containing the template with no local graph state.
        """
        # Default values if not provided
        if source is None:
            source = PatchCoordGlobal3D((0, 0, 0))
        if sink is None:
            sink = PatchCoordGlobal3D((1, 0, 0))

        direction = get_direction(source, sink)

        block = InitZeroThinLayerPipe(
            d=self.d,
            edgespec=self.edgespec,
            direction=direction,
        )
        # Set source and sink for boundary-based qindex calculation
        block.source = source
        block.sink = sink
        # Init blocks: final layer is open (O) without measurement
        block.final_layer = TemporalBoundarySpecValue.O
        return block


class InitZeroThinLayerPipe(_InitPipeBase):
    """Thin-layer Zero State initialization pipe (height=2) for compose-based initialization."""

    def _build_3d_graph(self) -> Build3DGraphReturn:
        """Override to create single-layer graph with only 13 nodes (9 data + 4 ancilla) at z=2*d."""
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Calculate z-coordinate based on source position and 2*d
        d_val = int(self.d)
        z0 = int(self.source[2]) * (2 * d_val)  # Base z-offset per block
        start_layer_z = z0 + (2 * d_val) - 1
        max_t = 1

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for each time slice
        nodes_by_z = self._assign_nodes_by_timeslice(
            g, data2d, x2d, z2d, max_t, start_layer_z, node2coord, coord2node, node2role
        )

        self._construct_schedule(nodes_by_z, node2role)

        self._add_spatial_edges(g, nodes_by_z)
        self._add_temporal_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role
