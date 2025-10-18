"""Temporal layer module for RHG canvas.

This module provides the TemporalLayer class for representing a single temporal layer
in the RHG canvas, along with utilities for creating temporal layers from cubes and pipes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.accumulator import FlowAccumulator, ParityAccumulator, ScheduleAccumulator
from lspattern.blocks.cubes.base import RHGCube
from lspattern.blocks.pipes.base import RHGPipe
from lspattern.canvas.composition import GraphComposer
from lspattern.canvas.coordinates import CoordinateMapper
from lspattern.canvas.ports import PortManager
from lspattern.canvas.seams import SeamGenerator
from lspattern.consts import EDGE_TUPLE_SIZE, CoordinateSystem
from lspattern.mytype import (
    NodeIdGlobal,
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PipeCoordGlobal3D,
    QubitGroupIdGlobal,
)
from lspattern.tiling.template import cube_offset_xy, pipe_offset_xy
from lspattern.utils import UnionFind, get_direction

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from graphqomb.graphstate import GraphState


@dataclass
class TemporalLayer:
    """
    Represents a single temporal layer in the RHG canvas.

    Contains cubes, pipes, and associated graph state
    for a given time step (z). Handles the construction and mapping of physical nodes, ports, and accumulators.
    """

    z: int
    qubit_count: int = 0
    patches: list[PatchCoordGlobal3D] = field(default_factory=list)
    lines: list[PipeCoordGlobal3D] = field(default_factory=list)

    # Port management delegated to PortManager
    port_manager: PortManager = field(default_factory=PortManager)

    # Coordinate mapping delegated to CoordinateMapper
    coord_mapper: CoordinateMapper = field(default_factory=CoordinateMapper)

    # Graph composition delegated to GraphComposer
    graph_composer: GraphComposer = field(init=False)

    schedule: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(default_factory=ParityAccumulator)

    local_graph: GraphState | None = None

    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    tiling_node_maps: dict[str, dict[int, tuple[int, int]]] = field(default_factory=dict)

    coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal] = field(default_factory=dict)
    allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]] = field(default_factory=set)

    def __post_init__(self) -> None:
        # Initialize graph_composer after coord_mapper and port_manager are set
        self.graph_composer = GraphComposer(self.coord_mapper, self.port_manager)

    # Backward compatibility properties for port management
    @property
    def in_portset(self) -> dict[PatchCoordGlobal3D, list[NodeIdLocal]]:
        """Get in_portset from port_manager."""
        return self.port_manager.in_portset

    @property
    def out_portset(self) -> dict[PatchCoordGlobal3D, list[NodeIdLocal]]:
        """Get out_portset from port_manager."""
        return self.port_manager.out_portset

    @property
    def cout_portset(self) -> dict[PatchCoordGlobal3D, list[NodeIdLocal]]:
        """Get cout_portset from port_manager."""
        return self.port_manager.cout_portset

    @property
    def cout_port_groups(self) -> dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]]:
        """Get cout_port_groups from port_manager."""
        return self.port_manager.cout_port_groups

    @property
    def cout_group_lookup(self) -> dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]]:
        """Get cout_group_lookup from port_manager."""
        return self.port_manager.cout_group_lookup

    @property
    def in_ports(self) -> list[NodeIdLocal]:
        """Get in_ports from port_manager."""
        return self.port_manager.in_ports

    @property
    def out_ports(self) -> list[NodeIdLocal]:
        """Get out_ports from port_manager."""
        return self.port_manager.out_ports

    @property
    def cout_ports(self) -> list[NodeIdLocal]:
        """Get cout_ports from port_manager."""
        return self.port_manager.cout_ports

    # Backward compatibility properties for coordinate mapping
    @property
    def node2coord(self) -> dict[NodeIdLocal, PhysCoordGlobal3D]:
        """Get node2coord from coord_mapper."""
        return self.coord_mapper.node2coord

    @property
    def coord2node(self) -> dict[PhysCoordGlobal3D, NodeIdLocal]:
        """Get coord2node from coord_mapper."""
        return self.coord_mapper.coord2node

    @property
    def node2role(self) -> dict[NodeIdLocal, str]:
        """Get node2role from coord_mapper."""
        return self.coord_mapper.node2role

    def add_cubes(self, cubes: Mapping[PatchCoordGlobal3D, RHGCube]) -> None:
        """Add multiple cubes to this temporal layer."""
        for pos, cube in cubes.items():
            self.add_cube(pos, cube)

    def add_pipes(self, pipes: Mapping[PipeCoordGlobal3D, RHGPipe]) -> None:
        """Add multiple pipes to this temporal layer."""
        for pipe_coord, pipe in pipes.items():
            source, sink = pipe_coord
            self.add_pipe(source, sink, pipe)

    def add_cube(self, pos: PatchCoordGlobal3D, cube: RHGCube) -> None:
        """Add a materialized cube to this temporal layer and place it at `pos`."""
        self.cubes_[pos] = cube
        self.patches.append(pos)

    def add_pipe(
        self,
        source: PatchCoordGlobal3D,
        sink: PatchCoordGlobal3D,
        spatial_pipe: RHGPipe,
    ) -> None:
        """Register a spatial pipe within this layer between `source` and `sink`."""
        pipe_coord = PipeCoordGlobal3D((source, sink))
        self.pipes_[pipe_coord] = spatial_pipe
        self.lines.append(pipe_coord)

    def _setup_union_find(self) -> tuple[UnionFind, Mapping[PhysCoordGlobal3D, QubitGroupIdGlobal]]:
        """Initialize Union-Find structure and process tiling IDs.

        Returns:
            Tuple of (union_find_structure, coordinate_to_gid_mapping)
        """
        # Union-Find (DSU) over tiling ids to compute connected groups
        uf = UnionFind()

        # 1) Initialize with all tiling IDs
        for c in self.cubes_.values():
            uf.add(QubitGroupIdGlobal(c.get_tiling_id()))
        for pipe in self.pipes_.values():
            uf.add(QubitGroupIdGlobal(pipe.get_tiling_id()))

        # 2) Union-Find: Unify cube<->pipe<->cube per spatial pipe
        for pipe_coord in self.pipes_:
            source, sink = pipe_coord
            uf.union(
                QubitGroupIdGlobal(self.cubes_[source].get_tiling_id()),
                QubitGroupIdGlobal(self.pipes_[pipe_coord].get_tiling_id()),
            )
            uf.union(
                QubitGroupIdGlobal(self.pipes_[pipe_coord].get_tiling_id()),
                QubitGroupIdGlobal(self.cubes_[sink].get_tiling_id()),
            )

        # 3) Set final tiling IDs and build coordinate mapping
        coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal] = {}

        for pos, cube in self.cubes_.items():
            cube.set_tiling_id(uf.find(cube.get_tiling_id()))
            self.cubes_[pos] = cube
            coord2gid.update({PhysCoordGlobal3D(k): QubitGroupIdGlobal(v) for k, v in cube.coord2gid.items()})
        for pipe_pos, pipe in self.pipes_.items():
            pipe.set_tiling_id(uf.find(pipe.get_tiling_id()))
            self.pipes_[pipe_pos] = pipe
            coord2gid.update({PhysCoordGlobal3D(k): QubitGroupIdGlobal(v) for k, v in pipe.coord2gid.items()})

        return uf, coord2gid

    def _remap_node_mappings(self, node_map: Mapping[int, int]) -> None:
        """Remap node mappings with given node map."""
        if not node_map:
            return
        self.coord_mapper.remap_nodes(node_map)

    def _remap_portsets(self, node_map: Mapping[int, int]) -> None:
        """Remap portsets with given node map."""
        self.port_manager.remap_ports(node_map)

    def _register_cout_group(
        self,
        patch_pos: PatchCoordGlobal3D,
        nodes: list[NodeIdLocal],
    ) -> None:
        """Record a cout group for the given patch and keep caches in sync."""
        self.port_manager.register_cout_group(patch_pos, nodes)

    def _rebuild_cout_group_cache(self) -> None:
        """Recompute flat cout caches from grouped data."""
        self.port_manager.rebuild_cout_group_cache()

    def _build_graph_from_blocks(self) -> GraphState:
        """Build the quantum graph state from cubes and pipes."""
        return self.graph_composer.build_graph_from_blocks(self.cubes_, self.pipes_)

    def compile(self) -> None:
        """Compile the temporal layer into a quantum pattern.

        Aggregates coordinates and patch groups, processes cubes and pipes,
        and builds the quantum graph state with proper port mappings.
        """
        # Setup Union-Find structure and coordinate mappings
        _, coord2gid = self._setup_union_find()

        # Set up allowed pairs for spatial pipes
        allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]] = set()
        allowed_gid_pairs.update(
            (
                QubitGroupIdGlobal(self.cubes_[source].get_tiling_id()),
                QubitGroupIdGlobal(self.cubes_[sink].get_tiling_id()),
            )
            for source, sink in self.pipes_
        )

        self.coord2gid = dict(coord2gid)
        self.allowed_gid_pairs = allowed_gid_pairs

        # Build GraphState by composing pre-materialized block graphs
        self.coord_mapper.clear()
        g = self._build_graph_from_blocks()

        # Add CZ edges across cube-pipe seams within the same temporal layer
        seam_generator = SeamGenerator(
            cubes=self.cubes_,
            pipes=self.pipes_,
            node2coord=self.node2coord,
            coord2node=self.coord2node,
            allowed_gid_pairs=allowed_gid_pairs,
        )
        g = seam_generator.add_seam_edges(g)

        # Finalize
        self.local_graph = g
        self.qubit_count = len(g.physical_nodes)

        # Preserve simple XY maps for inspection
        cube_xy_all: set[tuple[int, int]] = set()
        pipe_xy_all: set[tuple[int, int]] = set()
        for blk in self.cubes_.values():
            t = blk.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                cube_xy_all.update((int(x), int(y)) for x, y in coord_list or [])
        for pipe in self.pipes_.values():
            t = pipe.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                pipe_xy_all.update((int(x), int(y)) for x, y in coord_list or [])
        data2d = sorted(cube_xy_all.union(pipe_xy_all))
        self.tiling_node_maps = {
            "xy": dict(enumerate(data2d)),
        }

        # Merge accumulators from all blocks using node mappings from graph composition
        for cube in self.cubes_.values():
            # Get node mapping for this cube (stored during graph composition)
            node_map = cube.node_map_global
            if node_map:
                # Remap cube's accumulators to global node space
                remapped_schedule = cube.schedule.remap_nodes(
                    {NodeIdGlobal(k): NodeIdGlobal(v) for k, v in node_map.items()}
                )
                remapped_flow = cube.flow.remap_nodes(node_map)
                remapped_parity = cube.parity.remap_nodes(node_map)
            else:
                # Single cube case or no remapping needed
                remapped_schedule = cube.schedule
                remapped_flow = cube.flow
                remapped_parity = cube.parity

            self.schedule = self.schedule.compose_parallel(remapped_schedule)
            self.flow = self.flow.merge_with(remapped_flow)
            # Use parallel merge for same temporal layer (horizontal merge with XOR)
            self.parity = self.parity.merge_parallel(remapped_parity)

        for pipe in self.pipes_.values():
            # Get node mapping for this pipe (stored during graph composition)
            node_map = pipe.node_map_global
            if node_map:
                # Remap pipe's accumulators to global node space
                remapped_schedule = pipe.schedule.remap_nodes(
                    {NodeIdGlobal(k): NodeIdGlobal(v) for k, v in node_map.items()}
                )
                remapped_flow = pipe.flow.remap_nodes(node_map)
                remapped_parity = pipe.parity.remap_nodes(node_map)
            else:
                # Single pipe case or no remapping needed
                remapped_schedule = pipe.schedule
                remapped_flow = pipe.flow
                remapped_parity = pipe.parity

            self.schedule = self.schedule.compose_parallel(remapped_schedule)
            self.flow = self.flow.merge_with(remapped_flow)
            # Use parallel merge for same temporal layer (horizontal merge with XOR)
            self.parity = self.parity.merge_parallel(remapped_parity)

    def _get_coordinate_bounds(self) -> tuple[int, int, int, int, int, int]:
        """Get min/max bounds for all coordinates."""
        return self.coord_mapper.get_coordinate_bounds()

    @staticmethod
    def _create_face_checker(
        face: str, bounds: tuple[int, int, int, int, int, int], depths: Sequence[int]
    ) -> Callable[[tuple[int, int, int]], bool]:
        """Create function to check if coordinate is on requested face."""
        return CoordinateMapper.create_face_checker(face, bounds, list(depths))

    def _classify_nodes_by_role(
        self, on_face_checker: Callable[[tuple[int, int, int]], bool]
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Classify nodes by their role (data, xcheck, zcheck)."""
        return self.coord_mapper.classify_nodes_by_role(on_face_checker)

    def get_boundary_nodes(
        self,
        *,
        face: str,
        depth: Sequence[int] | None = None,
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Return nodes on a given face at the requested depths, grouped by role."""
        if not self.node2coord:
            return {"data": [], "xcheck": [], "zcheck": []}

        # Validate face parameter
        f = face.strip().lower()
        if f not in {"x+", "x-", "y+", "y-", "z+", "z-"}:
            error_msg = "face must be one of: x+/x-/y+/y-/z+/z-"
            raise ValueError(error_msg)

        depths = [max(int(d), 0) for d in (depth or [0])]
        bounds = self._get_coordinate_bounds()
        on_face_checker = self._create_face_checker(face, bounds, depths)

        return self._classify_nodes_by_role(on_face_checker)

    def get_node_maps(self) -> dict[str, dict[int, tuple[int, int]]]:
        """
        Return node_maps from ConnectedTiling (compute lazily if missing).

        Returns
        -------
            dict[str, dict[int, tuple[int, int]]]: The node_maps from ConnectedTiling.
        """
        if not self.tiling_node_maps:
            self.compile()
        return self.tiling_node_maps


def to_temporal_layer(
    z: int,
    cubes: dict[PatchCoordGlobal3D, RHGCube],
    pipes: dict[PipeCoordGlobal3D, RHGPipe],
) -> TemporalLayer:
    """Convert cubes and pipes to a temporal layer.

    Parameters
    ----------
    z : int
        The z-coordinate (time step) for this layer.
    cubes : dict[PatchCoordGlobal3D, RHGCube]
        Dictionary of cube positions to cubes.
    pipes : dict[PipeCoordGlobal3D, RHGPipe]
        Dictionary of pipe coordinates to pipes.

    Returns
    -------
    TemporalLayer
        The compiled temporal layer.
    """
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer(z)

    # shift position before materialization(テンプレートは ScalableTemplate を保持したままXY移動)
    for pos, c in cubes.items():
        dx, dy = cube_offset_xy(c.d, pos)
        # directory move the template (inplace=True)
        c.template.shift_coords((dx, dy), coordinate=CoordinateSystem.TILING_2D, inplace=True)
    for pipe_coord, p in pipes.items():
        coord_tuple = tuple(pipe_coord)
        if len(coord_tuple) != EDGE_TUPLE_SIZE:
            continue
        source, sink = coord_tuple
        direction = get_direction(source, sink)
        dx, dy = pipe_offset_xy(p.d, source, sink, direction)
        # directory move the template (inplace=True)
        p.template.shift_coords((dx, dy), coordinate=CoordinateSystem.TILING_2D, inplace=True)

    # materialize blocks before adding
    cubes_mat = {}
    for pos, blk in cubes.items():
        materialized = blk.materialize()
        if not isinstance(materialized, RHGCube):
            msg = f"Expected RHGCube after materialization, got {type(materialized)}"
            raise TypeError(msg)
        cubes_mat[pos] = materialized

    pipes_mat = {}
    for pipe_coord, p in pipes.items():
        materialized = p.materialize()
        if not isinstance(materialized, RHGPipe):
            msg = f"Expected RHGPipe after materialization, got {type(materialized)}"
            raise TypeError(msg)
        pipes_mat[pipe_coord] = materialized

    layer.add_cubes(cubes_mat)
    layer.add_pipes(pipes_mat)

    # compile this layer
    layer.compile()
    return layer
