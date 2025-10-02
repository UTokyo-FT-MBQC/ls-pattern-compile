"""Canvas module for RHG (Random Hamiltonian Graph) compilation.

This module provides the main compilation framework for converting RHG blocks
into executable quantum patterns with proper temporal layering and flow management.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from operator import itemgetter
from typing import TYPE_CHECKING

from graphix_zx.graphstate import (
    BaseGraphState,
    GraphState,
    compose,
)

from lspattern.accumulator import FlowAccumulator, ParityAccumulator, ScheduleAccumulator
from lspattern.blocks.cubes.base import RHGCube
from lspattern.blocks.pipes.base import RHGPipe
from lspattern.blocks.pipes.measure import _MeasurePipeBase
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.mytype import (
    NodeIdGlobal,
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PipeCoordGlobal3D,
    QubitGroupIdGlobal,
    TilingId,
)
from lspattern.tiling.template import ScalableTemplate, cube_offset_xy, pipe_offset_xy
from lspattern.utils import UnionFind, get_direction, is_allowed_pair

# Constants
EDGE_TUPLE_SIZE = 2

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lspattern.blocks.cubes.base import RHGCubeSkeleton
    from lspattern.blocks.pipes.base import RHGPipeSkeleton


class MixedCodeDistanceError(Exception):
    """Raised when mixed code distances are detected in TemporalLayer.materialize."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TemporalLayer:
    """
    Represents a single temporal layer in the RHG canvas.

    Contains cubes, pipes, and associated graph state
    for a given time step (z). Handles the construction and mapping of physical nodes, ports, and accumulators.
    """

    z: int
    qubit_count: int
    patches: list[PatchCoordGlobal3D]
    lines: list[PipeCoordGlobal3D]

    in_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]
    out_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]
    cout_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]
    cout_port_groups: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]]
    cout_group_lookup: dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]]

    in_ports: list[NodeIdLocal]
    out_ports: list[NodeIdLocal]
    cout_ports: list[NodeIdLocal]

    schedule: ScheduleAccumulator
    flow: FlowAccumulator
    parity: ParityAccumulator

    local_graph: BaseGraphState | None
    node2coord: dict[NodeIdLocal, PhysCoordGlobal3D]
    coord2node: dict[PhysCoordGlobal3D, NodeIdLocal]
    node2role: dict[NodeIdLocal, str]

    cubes_: dict[PatchCoordGlobal3D, RHGCube]
    pipes_: dict[PipeCoordGlobal3D, RHGPipe]
    tiling_node_maps: dict[str, dict[int, tuple[int, int]]]

    coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal]
    allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]]

    def __init__(self, z: int) -> None:
        self.z = z
        self.qubit_count = 0
        self.patches = []
        self.lines = []
        self.in_portset = {}
        self.out_portset = {}
        self.cout_portset = {}
        self.cout_port_groups = {}
        self.cout_group_lookup = {}
        self.in_ports = []
        self.out_ports = []
        self.cout_ports = []
        self.local_graph = None
        self.node2coord = {}
        self.coord2node = {}
        self.node2role = {}
        # accumulators
        self.schedule = ScheduleAccumulator()
        self.flow = FlowAccumulator()
        self.parity = ParityAccumulator()

        self.cubes_ = {}
        self.pipes_ = {}
        self.tiling_node_maps = {}

        self.coord2gid = {}
        self.allowed_gid_pairs = set()

    def __post_init__(self) -> None:
        """Post-initialization hook."""

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
        self.node2coord = {NodeIdLocal(node_map.get(n, n)): c for n, c in self.node2coord.items()}
        self.coord2node = {c: NodeIdLocal(node_map.get(n, n)) for c, n in self.coord2node.items()}
        self.node2role = {NodeIdLocal(node_map.get(n, n)): r for n, r in self.node2role.items()}

    def _remap_portsets(self, node_map: Mapping[int, int]) -> None:
        """Remap portsets with given node map."""
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
            self._rebuild_cout_group_cache()
        else:
            for p, nodes in self.cout_portset.items():
                self.cout_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
            self.cout_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.cout_ports]
        self.in_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.in_ports]
        self.out_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.out_ports]

    def _register_cout_group(
        self,
        patch_pos: PatchCoordGlobal3D,
        nodes: list[NodeIdLocal],
    ) -> None:
        """Record a cout group for the given patch and keep caches in sync."""
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

    def _rebuild_cout_group_cache(self) -> None:
        """Recompute flat cout caches from grouped data."""
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

    @staticmethod
    def _compose_single_cube(
        pos: PatchCoordGlobal3D,  # noqa: ARG004
        blk: RHGCube,
        g: BaseGraphState,
    ) -> tuple[BaseGraphState, Mapping[int, int], Mapping[int, int]]:
        """Compose a single cube into the graph."""
        g2 = blk.local_graph

        g_new, node_map1, node_map2 = compose(g, g2)
        return g_new, node_map1, node_map2

    def _process_cube_coordinates(self, blk: RHGCube, pos: tuple[int, int, int], node_map2: Mapping[int, int]) -> None:  # noqa: ARG002
        """Process cube coordinates and roles."""
        # All blocks now use absolute coordinates - no z_shift needed
        # Ingest coords/roles
        for old_n, coord in blk.node2coord.items():
            new_n = node_map2.get(old_n)
            if new_n is None:
                continue
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            c_new = PhysCoordGlobal3D((x, y, z))
            self.node2coord[NodeIdLocal(new_n)] = c_new
            self.coord2node[c_new] = NodeIdLocal(new_n)

        for old_n, role in blk.node2role.items():
            new_n = node_map2.get(old_n)
            if new_n is not None:
                self.node2role[NodeIdLocal(new_n)] = role

    def _process_pipe_ports(self, pipe_coord: PipeCoordGlobal3D, pipe: RHGPipe, node_map2: Mapping[int, int]) -> None:
        """Process pipe ports with node mapping."""
        source, sink = pipe_coord
        if pipe.in_ports:
            patch_pos = PatchCoordGlobal3D(source)
            self.in_portset[patch_pos] = [NodeIdLocal(node_map2[n]) for n in pipe.in_ports if n in node_map2]
            self.in_ports.extend(self.in_portset[patch_pos])
        if pipe.out_ports:
            patch_pos = PatchCoordGlobal3D(sink)
            mapped_nodes = [NodeIdLocal(node_map2[n]) for n in pipe.out_ports if n in node_map2]
            self.out_portset[patch_pos] = mapped_nodes
            self.out_ports.extend(self.out_portset[patch_pos])
        if pipe.cout_ports:
            patch_pos = PatchCoordGlobal3D(sink)
            for group in pipe.cout_ports:
                mapped_group: list[NodeIdLocal] = []
                for node in group:
                    new_id = node_map2.get(int(node))
                    if new_id is None:
                        continue
                    mapped_group.append(NodeIdLocal(new_id))
                self._register_cout_group(patch_pos, mapped_group)

    def _process_cube_ports(self, pos: tuple[int, int, int], blk: RHGCube, node_map2: Mapping[int, int]) -> None:
        """Process cube ports."""
        if blk.in_ports:
            patch_pos = PatchCoordGlobal3D(pos)
            self.in_portset[patch_pos] = [NodeIdLocal(node_map2[n]) for n in blk.in_ports if n in node_map2]
            self.in_ports.extend(self.in_portset[patch_pos])
        if blk.out_ports:
            patch_pos = PatchCoordGlobal3D(pos)
            self.out_portset[patch_pos] = [NodeIdLocal(node_map2[n]) for n in blk.out_ports if n in node_map2]
            self.out_ports.extend(self.out_portset[patch_pos])
        if blk.cout_ports:
            patch_pos = PatchCoordGlobal3D(pos)
            for group in blk.cout_ports:
                mapped_group: list[NodeIdLocal] = []
                for node in group:
                    new_id = node_map2.get(int(node))
                    if new_id is None:
                        continue
                    mapped_group.append(NodeIdLocal(new_id))
                self._register_cout_group(patch_pos, mapped_group)

    def _build_graph_from_blocks(self) -> BaseGraphState:
        """Build the quantum graph state from cubes and pipes."""
        # Special case: single block - use its GraphState directly to preserve q_indices
        if len(self.cubes_) == 1 and len(self.pipes_) == 0:
            pos, blk = next(iter(self.cubes_.items()))
            g = blk.local_graph
            # Process coordinates and ports without composition
            self._process_cube_coordinates_direct(blk, pos)
            self._process_cube_ports_direct(pos, blk)
            return g

        # Multiple blocks case - compose as before
        g_state: GraphState = GraphState()

        # Compose cube graphs
        for pos, blk in self.cubes_.items():
            g_state, node_map1, node_map2 = self._compose_single_cube(pos, blk, g_state)
            # Store node mapping for later use in accumulator merging
            blk.node_map_global = {NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map2.items()}
            self._remap_node_mappings(node_map1)
            self._remap_portsets(node_map1)
            self._process_cube_coordinates(blk, pos, node_map2)
            self._process_cube_ports(pos, blk, node_map2)

        # Compose pipe graphs (spatial pipes in this layer)
        return self._compose_pipe_graphs(g_state)

    def _process_cube_coordinates_direct(self, blk: RHGCube, pos: tuple[int, int, int]) -> None:  # noqa: ARG002
        """Process cube coordinates directly without node mapping."""
        # All blocks now use absolute coordinates - no z_shift needed
        # Directly use node coordinates
        for node, coord in blk.node2coord.items():
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            c_new = PhysCoordGlobal3D((x, y, z))
            self.node2coord[node] = c_new
            self.coord2node[c_new] = node

        for node, role in blk.node2role.items():
            self.node2role[node] = role

    def _process_cube_ports_direct(self, pos: PatchCoordGlobal3D, blk: RHGCube) -> None:
        """Process cube ports directly without node mapping."""
        # Process input ports
        input_port_nodes = [node for node, _ in blk.local_graph.input_node_indices.items()]
        self.in_portset.setdefault(pos, []).extend(input_port_nodes)
        self.in_ports.extend(input_port_nodes)

        # Process output ports
        output_port_nodes = [node for node, _ in blk.local_graph.output_node_indices.items()]
        self.out_portset.setdefault(pos, []).extend(output_port_nodes)
        self.out_ports.extend(output_port_nodes)

        if blk.cout_ports:
            patch_pos = PatchCoordGlobal3D(pos)
            for group in blk.cout_ports:
                mapped_group = [NodeIdLocal(int(node)) for node in group]
                self._register_cout_group(patch_pos, mapped_group)

    def _compose_pipe_graphs(self, g: BaseGraphState) -> BaseGraphState:
        """Compose pipe graphs into the main graph state."""
        for pipe_coord, pipe in self.pipes_.items():
            # Use materialized pipe if local_graph is None
            pipe_block = pipe
            g2 = pipe.local_graph

            g_new, node_map1, node_map2 = compose(g, g2)
            # Store node mapping for later use in accumulator merging
            pipe.node_map_global = {NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map2.items()}
            self._remap_node_mappings(node_map1)
            self._remap_portsets(node_map1)
            g = g_new

            # All blocks now use absolute coordinates - no z_shift needed
            for old_n, coord in pipe_block.node2coord.items():
                new_n = node_map2.get(old_n)
                if new_n is None:
                    continue
                # XY and Z are already in absolute coordinates (shifted in to_temporal_layer)
                x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                c_new = PhysCoordGlobal3D((x, y, z))
                self.node2coord[NodeIdLocal(new_n)] = c_new
                self.coord2node[c_new] = NodeIdLocal(new_n)
            for old_n, role in pipe_block.node2role.items():
                new_n = node_map2.get(old_n)
                if new_n is not None:
                    self.node2role[NodeIdLocal(new_n)] = role

            # Process pipe ports with node mapping
            self._process_pipe_ports(pipe_coord, pipe_block, node_map2)

        return g

    def _build_xy_regions(self, coord_gid_2d: dict[tuple[int, int], QubitGroupIdGlobal]) -> set[tuple[int, int]]:
        """Build XY coordinate sets for cubes and pipes."""
        cube_xy_all: set[tuple[int, int]] = set()

        for blk in self.cubes_.values():
            t = blk.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                for x, y in coord_list or []:
                    xy = (int(x), int(y))
                    cube_xy_all.add(xy)
                    coord_gid_2d[xy] = QubitGroupIdGlobal(blk.get_tiling_id())

        for pipe in self.pipes_.values():
            t = pipe.template
            for coord_list in (t.data_coords, t.x_coords, t.z_coords):
                for x, y in coord_list or []:
                    xy = (int(x), int(y))
                    coord_gid_2d[xy] = QubitGroupIdGlobal(pipe.get_tiling_id())

        return cube_xy_all

    @staticmethod
    def _get_existing_edges(g: BaseGraphState) -> set[tuple[int, int]]:
        """Get existing edges from graph to avoid duplicates."""
        edges = g.physical_edges
        result: set[tuple[int, int]] = set()
        for u, v in edges:
            edge = tuple(sorted((int(u), int(v))))
            if len(edge) == EDGE_TUPLE_SIZE:
                result.add((edge[0], edge[1]))
        return result

    def _is_measure_pipe_node(
        self,
        xy: tuple[int, int],
        cube_xy_all: set[tuple[int, int]],
    ) -> bool:
        """Check if a node belongs to a measure pipe."""
        # If the node is in cube region, it's not a pipe node
        if xy in cube_xy_all:
            return False

        # Check if this XY coordinate belongs to any measure pipe
        for pipe in self.pipes_.values():
            if isinstance(pipe, _MeasurePipeBase):
                # Check if this xy coordinate is in the pipe's template
                for coord in pipe.template.data_coords or []:
                    if (int(coord[0]), int(coord[1])) == xy:
                        return True
        return False

    def _should_connect_nodes(
        self,
        xy_u: tuple[int, int],
        xy_v: tuple[int, int],
        cube_xy_all: set[tuple[int, int]],
        gid_u: QubitGroupIdGlobal,
        gid_v: QubitGroupIdGlobal,
    ) -> bool:
        """Check if two nodes should be connected based on XY regions and group IDs."""
        u_in_cube = xy_u in cube_xy_all
        v_in_cube = xy_v in cube_xy_all

        # Skip connection if either node belongs to a measure pipe
        if self._is_measure_pipe_node(xy_u, cube_xy_all) or self._is_measure_pipe_node(xy_v, cube_xy_all):
            return False

        # Connect iff one is in cube region, other in pipe region, and allowed pair
        return u_in_cube != v_in_cube and is_allowed_pair(
            TilingId(int(gid_u)),
            TilingId(int(gid_v)),
            {(TilingId(int(a)), TilingId(int(b))) for a, b in self.allowed_gid_pairs},
        )

    def _process_neighbor_connections(
        self,
        u: NodeIdLocal,
        coord_u: PhysCoordGlobal3D,
        gid_u: QubitGroupIdGlobal,
        cube_xy_all: set[tuple[int, int]],
        coord_gid_2d: Mapping[tuple[int, int], QubitGroupIdGlobal],
        g: BaseGraphState,
        existing: set[tuple[int, int]],
    ) -> None:
        """Process connections to neighboring nodes."""
        xu, yu, zu = int(coord_u[0]), int(coord_u[1]), int(coord_u[2])
        xy_u = (xu, yu)

        for dx, dy, dz in DIRECTIONS3D:
            if dz != 0:
                continue  # we only connect within the same z plane

            xv, yv, zv = xu + int(dx), yu + int(dy), zu
            coord_v = PhysCoordGlobal3D((xv, yv, zv))
            v = self.coord2node.get(coord_v)
            if v is None or v == u:
                continue

            xy_v = (xv, yv)
            gid_v = coord_gid_2d.get(xy_v)
            if gid_v is None:
                continue

            if not self._should_connect_nodes(xy_u, xy_v, cube_xy_all, gid_u, gid_v):
                continue

            TemporalLayer._add_edge_if_valid(u, v, g, existing)

    @staticmethod
    def _add_edge_if_valid(
        u: NodeIdLocal,
        v: NodeIdLocal,
        g: BaseGraphState,
        existing: set[tuple[int, int]],
    ) -> None:
        """Add edge if valid and not duplicate."""
        # Check if either node is an output node - if so, don't add edge
        if int(u) in g.output_node_indices or int(v) in g.output_node_indices:
            return

        # Avoid duplicates by canonical edge ordering
        sorted_edge = tuple(sorted((int(u), int(v))))
        if len(sorted_edge) == EDGE_TUPLE_SIZE:
            edge = (sorted_edge[0], sorted_edge[1])
            if edge not in existing:
                g.add_physical_edge(u, v)
                existing.add(edge)

    def _add_seam_edges(
        self, g: BaseGraphState, coord_gid_2d: Mapping[tuple[int, int], QubitGroupIdGlobal]
    ) -> BaseGraphState:
        """Add CZ edges across cube-pipe seams within the same temporal layer."""
        # Build XY regions
        cube_xy_all = self._build_xy_regions(dict(coord_gid_2d))
        existing = self._get_existing_edges(g)

        for u, coord_u in list(self.node2coord.items()):
            xy_u = (int(coord_u[0]), int(coord_u[1]))
            gid_u = coord_gid_2d.get(xy_u)
            if gid_u is None:
                continue

            self._process_neighbor_connections(u, coord_u, gid_u, cube_xy_all, coord_gid_2d, g, existing)

        return g

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
        self.node2coord = {}
        self.coord2node = {}
        self.node2role = {}
        g = self._build_graph_from_blocks()

        # Add CZ edges across cube-pipe seams within the same temporal layer
        coord_gid_2d = {(x, y): gid for (x, y, _), gid in coord2gid.items()}
        g = self._add_seam_edges(g, coord_gid_2d)

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
        coords = list(self.node2coord.values())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    @staticmethod
    def _create_face_checker(
        face: str, bounds: tuple[int, int, int, int, int, int], depths: Sequence[int]
    ) -> Callable[[tuple[int, int, int]], bool]:
        """Create function to check if coordinate is on requested face."""
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        f = face.strip().lower()

        def on_face(c: tuple[int, int, int]) -> bool:
            x, y, z = c
            if f == "x+":
                return x in {xmax - d for d in depths}
            if f == "x-":
                return x in {xmin + d for d in depths}
            if f == "y+":
                return y in {ymax - d for d in depths}
            if f == "y-":
                return y in {ymin + d for d in depths}
            if f == "z+":
                return z in {zmax - d for d in depths}
            # f == 'z-'
            return z in {zmin + d for d in depths}

        return on_face

    def _classify_nodes_by_role(
        self, on_face_checker: Callable[[tuple[int, int, int]], bool]
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Classify nodes by their role (data, xcheck, zcheck)."""
        roles = self.node2role or {}
        data: list[PhysCoordGlobal3D] = []
        xcheck: list[PhysCoordGlobal3D] = []
        zcheck: list[PhysCoordGlobal3D] = []

        for nid, c in self.node2coord.items():
            if not on_face_checker(c):
                continue
            role = (roles.get(nid) or "").lower()
            if role == "ancilla_x":
                xcheck.append(c)
            elif role == "ancilla_z":
                zcheck.append(c)
            else:
                data.append(c)

        return {"data": data, "xcheck": xcheck, "zcheck": zcheck}

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


@dataclass
class CompiledRHGCanvas:
    """
    Represents a compiled RHG canvas, containing temporal layers, the global graph,
    coordinate mappings, port sets, and accumulators for schedule, flow, and parity.

    Attributes
    ----------
    layers : list[TemporalLayer]
        The temporal layers of the canvas.
    global_graph : BaseGraphState | None
        The global graph state after compilation.
    coord2node : dict[PhysCoordGlobal3D, int]
        Mapping from physical coordinates to node IDs.
    node2role : dict[int, str]
        Mapping from node IDs to their roles.
    in_portset : dict[PatchCoordGlobal3D, list[int]]
        Input port sets for each patch.
    out_portset : dict[PatchCoordGlobal3D, list[int]]
        Output port sets for each patch.
    cout_portset : dict[PatchCoordGlobal3D, list[int]]
        Ancilla output port sets for each patch.
    cout_port_groups : dict[PatchCoordGlobal3D, list[list[int]]]
        Grouped cout nodes per patch for logical observable extraction.
    cout_group_lookup : dict[int, tuple[PatchCoordGlobal3D, int]]
        Reverse index from node id to (patch, group index).
    schedule : ScheduleAccumulator
        Accumulator for scheduling information.
    flow : FlowAccumulator
        Accumulator for flow information.
    parity : ParityAccumulator
        Accumulator for parity checks.
    zlist : list[int]
        The current temporal layer indices.
    """

    # Non-default fields must come first
    layers: list[TemporalLayer]

    # Optional/defaulted fields follow
    global_graph: BaseGraphState | None = None
    coord2node: dict[PhysCoordGlobal3D, NodeIdLocal] = field(default_factory=dict)
    node2role: dict[NodeIdLocal, str] = field(default_factory=dict)

    in_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = field(default_factory=dict)
    out_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = field(default_factory=dict)
    cout_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = field(default_factory=dict)
    cout_port_groups: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]] = field(default_factory=dict)
    cout_group_lookup: dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]] = field(default_factory=dict)

    # Give defaults to satisfy dataclass ordering; caller may override later
    schedule: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(default_factory=ParityAccumulator)

    def _register_cout_group(
        self,
        patch_pos: PatchCoordGlobal3D,
        nodes: list[NodeIdLocal],
    ) -> None:
        """Record a cout group on the compiled canvas and sync caches."""
        group_nodes = [NodeIdLocal(int(n)) for n in nodes if n is not None]
        if not group_nodes:
            return
        groups = self.cout_port_groups.setdefault(patch_pos, [])
        index = len(groups)
        groups.append(group_nodes)
        flat = self.cout_portset.setdefault(patch_pos, [])
        flat.extend(group_nodes)
        for node in group_nodes:
            self.cout_group_lookup[node] = (patch_pos, index)

    def _rebuild_cout_group_cache(self) -> None:
        """Recompute flat cout caches from grouped data."""
        self.cout_portset = {}
        self.cout_group_lookup = {}
        for patch_pos, groups in self.cout_port_groups.items():
            flat: list[NodeIdLocal] = []
            for idx, group in enumerate(groups):
                normalized = [NodeIdLocal(int(n)) for n in group if n is not None]
                if not normalized:
                    continue
                self.cout_port_groups[patch_pos][idx] = normalized
                flat.extend(normalized)
                for node in normalized:
                    self.cout_group_lookup[node] = (patch_pos, idx)
            if flat:
                self.cout_portset[patch_pos] = flat

    def get_cout_group_by_coord(
        self,
        coord: PhysCoordGlobal3D,
    ) -> tuple[PatchCoordGlobal3D, list[NodeIdLocal]] | None:
        """Return the cout group for the node at `coord`, if present."""
        node = self.coord2node.get(coord)
        if node is None:
            return None
        mapping = self.cout_group_lookup.get(node)
        if mapping is None:
            return None
        patch_pos, group_idx = mapping
        groups = self.cout_port_groups.get(patch_pos)
        if groups is None or group_idx >= len(groups):
            return None
        return patch_pos, list(groups[group_idx])

    def resolve_cout_groups(
        self,
        groups_by_key: Mapping[str, Sequence[PhysCoordGlobal3D]],
    ) -> dict[str, list[NodeIdLocal]]:
        """Resolve logical observable keys to cout node groups using coordinates."""
        resolved: dict[str, list[NodeIdLocal]] = {}
        missing: dict[str, list[PhysCoordGlobal3D]] = {}
        for key, coords in groups_by_key.items():
            group_nodes: list[NodeIdLocal] = []
            missing_coords: list[PhysCoordGlobal3D] = []
            for coord in coords:
                result = self.get_cout_group_by_coord(coord)
                if result is None:
                    missing_coords.append(coord)
                    continue
                _, nodes = result
                group_nodes.extend(nodes)
            if missing_coords:
                missing[key] = missing_coords
                continue
            resolved[key] = group_nodes
        if missing:
            detail_parts = []
            for key, coords in missing.items():
                rendered = ", ".join(str(tuple(c)) for c in coords)
                detail_parts.append(f"{key}: [{rendered}]")
            details = "; ".join(detail_parts)
            msg = f"cout group not found for coordinates -> {details}"
            raise KeyError(msg)
        return resolved

    zlist: list[int] = field(default_factory=list)

    # Optional placeholders
    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    # (deprecated) debug seam pairs: removed

    # def generate_stim_circuit(self) -> stim.Circuit:
    #     pass

    @staticmethod
    def _remap_graph_nodes(
        gsrc: BaseGraphState, nmap: dict[NodeIdLocal, NodeIdLocal]
    ) -> tuple[dict[int, int], GraphState]:
        """Create new nodes in destination graph."""
        gdst = GraphState()
        created: dict[int, int] = {}
        for old in gsrc.physical_nodes:
            new_id = nmap.get(NodeIdLocal(old), NodeIdLocal(old))
            if int(new_id) in created:
                continue
            created[int(new_id)] = gdst.add_physical_node()
        return created, gdst

    @staticmethod
    def _remap_measurement_bases(
        gsrc: BaseGraphState,
        gdst: BaseGraphState,
        nmap: dict[NodeIdLocal, NodeIdLocal],
        created: dict[int, int],
    ) -> None:
        """Remap measurement bases."""
        for old, new_id in nmap.items():
            mb = gsrc.meas_bases.get(int(old))
            if mb is not None:
                gdst.assign_meas_basis(created.get(int(new_id), int(new_id)), mb)

    @staticmethod
    def _remap_graph_edges(
        gsrc: BaseGraphState,
        gdst: BaseGraphState,
        nmap: dict[NodeIdLocal, NodeIdLocal],
        created: dict[int, int],
    ) -> None:
        """Remap graph edges."""
        for u, v in gsrc.physical_edges:
            nu = nmap.get(NodeIdLocal(u), NodeIdLocal(u))
            nv = nmap.get(NodeIdLocal(v), NodeIdLocal(v))
            gdst.add_physical_edge(created.get(int(nu), int(nu)), created.get(int(nv), int(nv)))

    @staticmethod
    def _create_remapped_graphstate(
        gsrc: BaseGraphState | None, nmap: dict[NodeIdLocal, NodeIdLocal]
    ) -> BaseGraphState | None:
        """Create a remapped GraphState."""
        if gsrc is None:
            return None
        created, gdst = CompiledRHGCanvas._remap_graph_nodes(gsrc, nmap)
        CompiledRHGCanvas._remap_measurement_bases(gsrc, gdst, nmap, created)
        CompiledRHGCanvas._remap_graph_edges(gsrc, gdst, nmap, created)
        return gdst

    @staticmethod
    def _remap_layer(layer: TemporalLayer, node_map: Mapping[NodeIdLocal, NodeIdLocal]) -> TemporalLayer:
        """Remap a single temporal layer."""
        # Create a copy of the layer and remap its node mappings
        remapped_layer = TemporalLayer(layer.z)
        remapped_layer.qubit_count = layer.qubit_count
        remapped_layer.patches = layer.patches.copy()
        remapped_layer.lines = layer.lines.copy()

        # Remap layer's node mappings
        remapped_layer.coord2node = {c: node_map.get(n, n) for c, n in layer.coord2node.items()}
        remapped_layer.node2coord = {node_map.get(n, n): c for n, c in layer.node2coord.items()}
        remapped_layer.node2role = {node_map.get(n, n): r for n, r in layer.node2role.items()}

        # Remap portsets
        CompiledRHGCanvas._remap_layer_portsets(layer, remapped_layer, node_map)

        # Remap port lists
        remapped_layer.in_ports = [NodeIdLocal(node_map.get(n, n)) for n in layer.in_ports]
        remapped_layer.out_ports = [NodeIdLocal(node_map.get(n, n)) for n in layer.out_ports]
        CompiledRHGCanvas._remap_layer_cout_ports(layer, remapped_layer, node_map)

        # Copy other attributes
        remapped_layer.local_graph = layer.local_graph  # GraphState will be remapped separately

        # Remap accumulators to use new node IDs
        remapped_layer.schedule = layer.schedule.remap_nodes(
            {NodeIdGlobal(k): NodeIdGlobal(v) for k, v in node_map.items()}
        )
        remapped_layer.flow = layer.flow.remap_nodes(dict(node_map))
        remapped_layer.parity = layer.parity.remap_nodes(dict(node_map))
        remapped_layer.cubes_ = layer.cubes_.copy()
        remapped_layer.pipes_ = layer.pipes_.copy()
        remapped_layer.tiling_node_maps = layer.tiling_node_maps.copy()
        remapped_layer.coord2gid = layer.coord2gid.copy()
        remapped_layer.allowed_gid_pairs = layer.allowed_gid_pairs.copy()

        return remapped_layer

    @staticmethod
    def _remap_layer_portsets(
        layer: TemporalLayer, remapped_layer: TemporalLayer, node_map: Mapping[NodeIdLocal, NodeIdLocal]
    ) -> None:
        """Remap portsets for a layer."""
        remapped_layer.in_portset = {
            p: [NodeIdLocal(node_map.get(n, n)) for n in nodes] for p, nodes in layer.in_portset.items()
        }
        remapped_layer.out_portset = {
            p: [NodeIdLocal(node_map.get(n, n)) for n in nodes] for p, nodes in layer.out_portset.items()
        }
        if layer.cout_port_groups:
            remapped_groups: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]] = {}
            for patch_pos, groups in layer.cout_port_groups.items():
                remapped_group_list: list[list[NodeIdLocal]] = []
                for group in groups:
                    remapped_group = [NodeIdLocal(node_map.get(n, n)) for n in group]
                    remapped_group_list.append(remapped_group)
                remapped_groups[patch_pos] = remapped_group_list
            remapped_layer.cout_port_groups = remapped_groups
            remapped_layer._rebuild_cout_group_cache()  # noqa: SLF001
        else:
            remapped_layer.cout_portset = {
                p: [NodeIdLocal(node_map.get(n, n)) for n in nodes] for p, nodes in layer.cout_portset.items()
            }

    @staticmethod
    def _remap_layer_cout_ports(
        layer: TemporalLayer, remapped_layer: TemporalLayer, node_map: Mapping[NodeIdLocal, NodeIdLocal]
    ) -> None:
        """Remap cout ports for a layer."""
        if layer.cout_port_groups:
            remapped_layer.cout_ports = [
                NodeIdLocal(node_map.get(n, n))
                for groups in layer.cout_port_groups.values()
                for group in groups
                for n in group
            ]
        else:
            remapped_layer.cout_ports = [NodeIdLocal(node_map.get(n, n)) for n in layer.cout_ports]

    # TODO: this could be made more efficient by avoiding deep copies
    def remap_nodes(self, node_map: Mapping[NodeIdLocal, NodeIdLocal]) -> CompiledRHGCanvas:
        """Remap nodes according to the given node mapping."""
        # Deep copy and remap each layer
        remapped_layers = [CompiledRHGCanvas._remap_layer(layer, node_map) for layer in self.layers]

        new_cgraph = CompiledRHGCanvas(
            layers=remapped_layers,
            global_graph=self._create_remapped_graphstate(self.global_graph, dict(node_map)),
            coord2node={},
            in_portset={},
            out_portset={},
            cout_portset={},
            cout_port_groups={},
            cout_group_lookup={},
            schedule=self.schedule.remap_nodes({NodeIdGlobal(k): NodeIdGlobal(v) for k, v in node_map.items()}),
            flow=self.flow.remap_nodes(dict(node_map)),
            parity=self.parity.remap_nodes(dict(node_map)),
            cubes_=self.cubes_.copy(),
            pipes_=self.pipes_.copy(),
            zlist=list(self.zlist),
        )

        # Remap coord2node
        for coord, old_nodeid in self.coord2node.items():
            new_cgraph.coord2node[coord] = node_map[old_nodeid]
        # Remap node2role
        for old_nodeid, role in self.node2role.items():
            new_cgraph.node2role[node_map[old_nodeid]] = role

        # Remap portsets
        for pos, nodes in self.in_portset.items():
            new_cgraph.in_portset[pos] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        for pos, nodes in self.out_portset.items():
            new_cgraph.out_portset[pos] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        if self.cout_port_groups:
            remapped_global_groups: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]] = {}
            for patch_pos, groups in self.cout_port_groups.items():
                remapped_global_group_list: list[list[NodeIdLocal]] = []
                for group in groups:
                    remapped_group = [NodeIdLocal(node_map.get(n, n)) for n in group]
                    remapped_global_group_list.append(remapped_group)
                remapped_global_groups[patch_pos] = remapped_global_group_list
            new_cgraph.cout_port_groups = remapped_global_groups
            new_cgraph._rebuild_cout_group_cache()
        else:
            for pos, nodes in self.cout_portset.items():
                new_cgraph.cout_portset[pos] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]

        return new_cgraph

    def get_boundary_nodes(
        self,
        *,
        face: str,
        depth: list[int] | None = None,
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Boundary query after temporal composition on the compiled canvas.

        Operates on the global coord2node map using the same semantics as
        TemporalLayer.get_boundary_nodes.
        """
        if not self.coord2node:
            return {"data": [], "xcheck": [], "zcheck": []}

        coords = list(self.coord2node.keys())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)

        f = face.strip().lower()
        if f not in {"x+", "x-", "y+", "y-", "z+", "z-"}:
            msg = "face must be one of: x+/x-/y+/y-/z+/z-"
            raise ValueError(msg)
        depths = [max(int(d), 0) for d in (depth or [0])]

        def on_face(c: tuple[int, int, int]) -> bool:
            x, y, z = c
            if f == "x+":
                return x in {xmax - d for d in depths}
            if f == "x-":
                return x in {xmin + d for d in depths}
            if f == "y+":
                return y in {ymax - d for d in depths}
            if f == "y-":
                return y in {ymin + d for d in depths}
            if f == "z+":
                return z in {zmax - d for d in depths}
            return z in {zmin + d for d in depths}

        # Without global role info, conservatively return all as 'data'.
        selected = [c for c in coords if on_face(c)]
        return {"data": selected, "xcheck": [], "zcheck": []}

    def add_temporal_layer(self, next_layer: TemporalLayer, *, pipes: list[RHGPipe] | None = None) -> CompiledRHGCanvas:
        """Compose this compiled canvas with `next_layer`.

        Convenience instance-method wrapper around the module-level
        `add_temporal_layer` with optional `pipes` gating cross-time connections.
        """
        return add_temporal_layer(self, next_layer, list(pipes or []))


@dataclass
class RHGCanvasSkeleton:  # BlockGraph in tqec
    name: str = "Blank Canvas Skeleton"
    # Optional template placeholder for future use
    template: ScalableTemplate | None = None
    cubes_: dict[PatchCoordGlobal3D, RHGCubeSkeleton] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipeSkeleton] = field(default_factory=dict)

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCubeSkeleton) -> None:
        self.cubes_[position] = cube

    def add_pipe(self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipeSkeleton) -> None:
        pipe_coord = PipeCoordGlobal3D((start, end))
        self.pipes_[pipe_coord] = pipe

    @staticmethod
    def _get_spatial_direction(dx: int, dy: int) -> tuple[str, str] | None:
        """Get trim directions for spatial pipe."""
        if dx == 1 and dy == 0:
            return "RIGHT", "LEFT"  # X+ direction
        if dx == -1 and dy == 0:
            return "LEFT", "RIGHT"  # X- direction
        if dy == 1 and dx == 0:
            return "TOP", "BOTTOM"  # Y+ direction
        if dy == -1 and dx == 0:
            return "BOTTOM", "TOP"  # Y- direction
        return None

    def _trim_adjacent_cubes(self, u: PatchCoordGlobal3D, v: PatchCoordGlobal3D, left_dir: str, right_dir: str) -> None:
        """Trim boundaries of adjacent cubes."""
        left = self.cubes_.get(u)
        right = self.cubes_.get(v)

        if left is not None:
            left.trim_spatial_boundary(left_dir)
        if right is not None:
            right.trim_spatial_boundary(right_dir)

    def trim_spatial_boundaries(self) -> None:
        """Trim spatial boundaries of adjacent cubes."""
        for pipe_coord in list(self.pipes_.keys()):
            coord_tuple = tuple(pipe_coord)
            if len(coord_tuple) != EDGE_TUPLE_SIZE:
                continue
            u, v = coord_tuple
            ux, uy, uz = u
            vx, vy, vz = v

            # Skip temporal pipes
            if uz != vz:
                continue

            dx, dy = vx - ux, vy - uy
            directions = self._get_spatial_direction(dx, dy)

            if directions is not None:
                left_dir, right_dir = directions
                self._trim_adjacent_cubes(u, v, left_dir, right_dir)

    def to_canvas(self) -> RHGCanvas:
        self.trim_spatial_boundaries()

        trimmed_cubes_skeleton = self.cubes_.copy()
        trimmed_pipes_skeleton = self.pipes_.copy()

        cubes_: dict[PatchCoordGlobal3D, RHGCube] = {}
        for pos, c in trimmed_cubes_skeleton.items():
            # Materialize block and attach its 3D anchor so z-offset is correct
            blk = c.to_block()
            blk.source = pos
            if not isinstance(blk, RHGCube):
                msg = f"Expected RHGCube, got {type(blk)}"
                raise TypeError(msg)
            cubes_[pos] = blk
        pipes_: dict[PipeCoordGlobal3D, RHGPipe] = {}
        for pipe_coord, p in trimmed_pipes_skeleton.items():
            source, sink = pipe_coord
            # Type: ignore because pipe skeletons override to_block with source/sink args
            block = p.to_block(source, sink)  # type: ignore[call-arg]
            if not isinstance(block, RHGPipe):
                msg = f"Expected RHGPipe, got {type(block)}"
                raise TypeError(msg)
            pipes_[pipe_coord] = block

        return RHGCanvas(
            name=self.name,
            cubes_=cubes_,
            pipes_=pipes_,
        )


@dataclass
class RHGCanvas:  # TopologicalComputationGraph in tqec
    name: str = "Blank Canvas"

    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    layers: list[TemporalLayer] = field(default_factory=list)

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCube) -> None:
        self.cubes_[position] = cube
        # Reset one-shot guard so layers can be rebuilt after topology changes
        with suppress(AttributeError):
            self._to_temporal_layers_called = False

    def add_pipe(self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipe) -> None:
        pipe_coord = PipeCoordGlobal3D((start, end))
        self.pipes_[pipe_coord] = pipe
        # Reset one-shot guard so layers can be rebuilt after topology changes
        with suppress(AttributeError):
            self._to_temporal_layers_called = False

    def to_temporal_layers(self) -> dict[int, TemporalLayer]:
        # Disallow multiple calls to prevent duplicate XY shifts on templates.
        if getattr(self, "_to_temporal_layers_called", False):
            msg = (
                "RHGCanvas.to_temporal_layers() can be called at most once per canvas. "
                "Rebuild the canvas (or use RHGCanvasSkeleton.to_canvas()) before calling again."
            )
            raise RuntimeError(msg)
        temporal_layers: dict[int, TemporalLayer] = {}

        for z in range(max(self.cubes_.keys(), key=itemgetter(2))[2] + 1):
            cubes = {pos: c for pos, c in self.cubes_.items() if pos[2] == z}
            pipes = {}
            for pipe_coord, p in self.pipes_.items():
                coord_tuple = tuple(pipe_coord)
                if len(coord_tuple) == EDGE_TUPLE_SIZE:
                    start, end = coord_tuple
                    if start[2] == z and end[2] == z:
                        pipes[pipe_coord] = p

            layer = to_temporal_layer(z, cubes, pipes)
            temporal_layers[z] = layer

        with suppress(AttributeError):
            self._to_temporal_layers_called = True
        return temporal_layers

    def compile(self) -> CompiledRHGCanvas:
        temporal_layers = self.to_temporal_layers()
        # Initialize an empty compiled canvas with required accumulators
        initial_parity = ParityAccumulator()

        cgraph = CompiledRHGCanvas(
            layers=[],
            global_graph=None,
            coord2node={},
            in_portset={},
            out_portset={},
            cout_portset={},
            schedule=ScheduleAccumulator(),
            flow=FlowAccumulator(),
            parity=initial_parity,
            zlist=[],
        )

        # Note: q_index consistency is now automatically ensured by patch coordinate-based calculation

        # Compose layers in increasing temporal order, wiring any cross-layer pipes
        for z in sorted(temporal_layers.keys()):
            layer = temporal_layers[z]
            # Select pipes whose start.z is the last compiled z and end.z is this layer z
            prev_z = cgraph.zlist[-1] if cgraph.zlist else None
            if prev_z is None:
                pipes: list[RHGPipe] = []
            else:
                pipes = []
                for pipe_coord, pipe in self.pipes_.items():
                    u, v = pipe_coord
                    if u[2] == prev_z and v[2] == z:
                        # 明示的に端点情報を埋めて渡す(skeleton->block で保持されないため)
                        with suppress(Exception):
                            pipe.source = u
                            pipe.sink = v
                            pipe.direction = get_direction(u, v)
                        pipes.append(pipe)
            cgraph = add_temporal_layer(cgraph, layer, pipes)
        return cgraph


def _create_first_layer_canvas(next_layer: TemporalLayer) -> CompiledRHGCanvas:
    """Create compiled canvas for the first temporal layer."""

    compiled = CompiledRHGCanvas(
        layers=[next_layer],
        global_graph=next_layer.local_graph,
        coord2node={k: NodeIdLocal(v) for k, v in next_layer.coord2node.items()},
        node2role={NodeIdLocal(k): v for k, v in next_layer.node2role.items()},
        in_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in next_layer.in_portset.items()},
        out_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in next_layer.out_portset.items()},
        cout_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in next_layer.cout_portset.items()},
        cout_port_groups={
            k: [[NodeIdLocal(int(v)) for v in group] for group in groups]
            for k, groups in next_layer.cout_port_groups.items()
        },
        cout_group_lookup={},
        schedule=next_layer.schedule,
        parity=next_layer.parity,
        flow=next_layer.flow,
        zlist=[next_layer.z],
        cubes_=next_layer.cubes_,
        pipes_=next_layer.pipes_,
    )
    if compiled.cout_port_groups:
        compiled._rebuild_cout_group_cache()  # noqa: SLF001
    return compiled


def to_temporal_layer(
    z: int,
    cubes: dict[PatchCoordGlobal3D, RHGCube],
    pipes: dict[PipeCoordGlobal3D, RHGPipe],
) -> TemporalLayer:
    # 1) Make empty TemporalLayer instance
    layer = TemporalLayer(z)

    # shift position before materialization(テンプレートは ScalableTemplate を保持したままXY移動)
    for pos, c in cubes.items():
        dx, dy = cube_offset_xy(c.d, pos)
        # directory move the template (inplace=True)
        c.template.shift_coords((dx, dy), coordinate="tiling2d", inplace=True)
    for pipe_coord, p in pipes.items():
        coord_tuple = tuple(pipe_coord)
        if len(coord_tuple) != EDGE_TUPLE_SIZE:
            continue
        source, sink = coord_tuple
        direction = get_direction(source, sink)
        dx, dy = pipe_offset_xy(p.d, source, sink, direction)
        # directory move the template (inplace=True)
        p.template.shift_coords((dx, dy), coordinate="tiling2d", inplace=True)

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


def _remap_layer_mappings(next_layer: TemporalLayer, node_map2: Mapping[int, int]) -> None:
    """Remap next layer mappings."""
    next_layer.coord2node = {c: NodeIdLocal(node_map2.get(int(n), int(n))) for c, n in next_layer.coord2node.items()}
    next_layer.node2coord = {NodeIdLocal(node_map2.get(int(n), int(n))): c for n, c in next_layer.node2coord.items()}
    next_layer.node2role = {NodeIdLocal(node_map2.get(int(n), int(n))): r for n, r in next_layer.node2role.items()}

    # Also remap accumulators
    local_node_map = {NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map2.items()}
    next_layer.schedule = next_layer.schedule.remap_nodes(
        {NodeIdGlobal(k): NodeIdGlobal(v) for k, v in node_map2.items()}
    )
    next_layer.flow = next_layer.flow.remap_nodes(local_node_map)
    next_layer.parity = next_layer.parity.remap_nodes(local_node_map)


def _build_merged_coord2node(cgraph: CompiledRHGCanvas, next_layer: TemporalLayer) -> dict[PhysCoordGlobal3D, int]:
    """Build merged coordinate to node mapping."""
    return {
        **cgraph.coord2node,
        **next_layer.coord2node,
    }


def _extend_cout_port_groups(
    target: dict[PatchCoordGlobal3D, list[list[int]]],
    source: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]],
    node_map: dict[int, int],
) -> None:
    """Extend cout port groups with remapped nodes."""
    for patch_pos, groups in source.items():
        mapped_groups: list[list[int]] = []
        for group in groups:
            mapped_group = [node_map.get(int(n), int(n)) for n in group]
            if mapped_group:
                mapped_groups.append(mapped_group)
        if mapped_groups:
            target.setdefault(patch_pos, []).extend(mapped_groups)


def _remap_temporal_portsets(
    cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, node_map1: dict[int, int], node_map2: dict[int, int]
) -> tuple[
    dict[PatchCoordGlobal3D, list[int]],
    dict[PatchCoordGlobal3D, list[int]],
    dict[PatchCoordGlobal3D, list[int]],
    dict[PatchCoordGlobal3D, list[list[int]]],
]:
    """Remap portsets for temporal composition."""
    in_portset = {pos: [node_map2[int(n)] for n in nodes] for pos, nodes in next_layer.in_portset.items()}

    # NOTE: if int(n) in node_map1 sentence is unexpected handling. It might suggest a bug somewhere else.
    out_portset = {
        **{pos: [node_map1[int(n)] for n in nodes if int(n) in node_map1] for pos, nodes in cgraph.out_portset.items()},
        **{pos: [node_map2[int(n)] for n in nodes] for pos, nodes in next_layer.out_portset.items()},
    }
    cout_portset = {
        **{
            pos: [node_map1[int(n)] for n in nodes if int(n) in node_map1] for pos, nodes in cgraph.cout_portset.items()
        },
        **{pos: [node_map2[int(n)] for n in nodes] for pos, nodes in next_layer.cout_portset.items()},
    }

    cout_port_groups: dict[PatchCoordGlobal3D, list[list[int]]] = {}
    _extend_cout_port_groups(cout_port_groups, cgraph.cout_port_groups, node_map1)
    _extend_cout_port_groups(cout_port_groups, next_layer.cout_port_groups, node_map2)

    if not cgraph.cout_port_groups:
        for patch_pos, nodes in cgraph.cout_portset.items():
            mapped = [node_map1.get(int(n), int(n)) for n in nodes]
            if mapped:
                cout_port_groups.setdefault(patch_pos, []).append(mapped)
    if not next_layer.cout_port_groups:
        for patch_pos, nodes in next_layer.cout_portset.items():
            mapped = [node_map2.get(int(n), int(n)) for n in nodes]
            if mapped:
                cout_port_groups.setdefault(patch_pos, []).append(mapped)

    return in_portset, out_portset, cout_portset, cout_port_groups


def _build_coordinate_gid_mapping(
    cgraph: CompiledRHGCanvas, next_layer: TemporalLayer
) -> dict[PhysCoordGlobal3D, QubitGroupIdGlobal]:
    """Build coordinate to group ID mapping."""
    new_coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal] = {}
    for cube in [*cgraph.cubes_.values(), *next_layer.cubes_.values()]:
        new_coord2gid.update({PhysCoordGlobal3D(k): QubitGroupIdGlobal(v) for k, v in cube.coord2gid.items()})
    for pipe in [*cgraph.pipes_.values(), *next_layer.pipes_.values()]:
        new_coord2gid.update({PhysCoordGlobal3D(k): QubitGroupIdGlobal(v) for k, v in pipe.coord2gid.items()})
    return new_coord2gid


def _setup_temporal_connections(
    pipes: list[RHGPipe],
    cgraph: CompiledRHGCanvas,
    next_layer: TemporalLayer,
    new_graph: BaseGraphState,
    new_coord2node: dict[PhysCoordGlobal3D, int],
    new_coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal],
) -> None:
    """Setup temporal connections between layers."""
    allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]] = set()

    allowed_gid_pairs.update(
        (
            QubitGroupIdGlobal(cgraph.cubes_[p.source].get_tiling_id()),
            QubitGroupIdGlobal(next_layer.cubes_[p.sink].get_tiling_id()),
        )
        for p in pipes
        if p.sink is not None
    )

    for source in next_layer.get_boundary_nodes(face="z-", depth=[-1])["data"]:
        sink_coord = PhysCoordGlobal3D((source[0], source[1], source[2] - 1))
        source_gid = new_coord2gid.get(PhysCoordGlobal3D(source))
        sink_gid = new_coord2gid.get(sink_coord)

        if (
            source_gid is not None
            and sink_gid is not None
            and is_allowed_pair(
                TilingId(int(source_gid)),
                TilingId(int(sink_gid)),
                {(TilingId(int(a)), TilingId(int(b))) for a, b in allowed_gid_pairs},
            )
        ):
            source_node = new_coord2node.get(PhysCoordGlobal3D(source))
            sink_node = new_coord2node.get(sink_coord)
            if source_node is not None and sink_node is not None and sink_node not in new_graph.neighbors(source_node):
                new_graph.add_physical_edge(source_node, sink_node)


def add_temporal_layer(cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, pipes: list[RHGPipe]) -> CompiledRHGCanvas:
    """Compose the compiled canvas with the next temporal layer."""

    if cgraph.global_graph is None:
        return _create_first_layer_canvas(next_layer)

    # Compose graphs and remap
    new_graph, node_map1, node_map2 = compose(cgraph.global_graph, next_layer.local_graph)

    # Only remap if node mapping actually changes node IDs
    if any(k != v for k, v in node_map1.items()):
        cgraph = cgraph.remap_nodes({NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map1.items()})

    _remap_layer_mappings(next_layer, node_map2)

    # Build merged mappings
    new_coord2node = _build_merged_coord2node(cgraph, next_layer)
    in_portset, out_portset, cout_portset, cout_port_groups = _remap_temporal_portsets(
        cgraph, next_layer, node_map1, node_map2
    )
    new_coord2gid = _build_coordinate_gid_mapping(cgraph, next_layer)

    # Setup temporal connections
    _setup_temporal_connections(pipes, cgraph, next_layer, new_graph, new_coord2node, new_coord2gid)

    new_layers = [*cgraph.layers, next_layer]

    # Update accumulators
    last_nodes = set(next_layer.in_ports)
    # remap to global node IDs
    last_nodes_remapped = {node_map2[int(n)] for n in last_nodes}
    cgraph_filtered_schedule = cgraph.schedule.exclude_nodes(last_nodes_remapped)
    new_schedule = cgraph_filtered_schedule.compose_sequential(next_layer.schedule, exclude_nodes=None)
    # TODO: Fix flow merge to handle connected q_indices properly
    try:
        merged_flow = cgraph.flow.merge_with(next_layer.flow)
    except ValueError as e:
        if "Flow merge conflict" in str(e):
            # Temporary workaround: use the first layer's flow
            merged_flow = cgraph.flow
        else:
            raise
    new_parity = cgraph.parity.merge_with(next_layer.parity)

    # TODO: should add boundary checks?

    compiled = CompiledRHGCanvas(
        layers=new_layers,
        global_graph=new_graph,
        coord2node={k: NodeIdLocal(v) for k, v in new_coord2node.items()},
        in_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in in_portset.items()},
        out_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in out_portset.items()},
        cout_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in cout_portset.items()},
        cout_port_groups={
            k: [[NodeIdLocal(v) for v in group] for group in groups] for k, groups in cout_port_groups.items()
        },
        cout_group_lookup={},
        schedule=new_schedule,
        flow=merged_flow,
        parity=new_parity,
        zlist=[*list(cgraph.zlist), next_layer.z],
    )
    if compiled.cout_port_groups:
        compiled._rebuild_cout_group_cache()  # noqa: SLF001
    return compiled
