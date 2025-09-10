"""Canvas module for RHG (Random Hamiltonian Graph) compilation.

This module provides the main compilation framework for converting RHG blocks
into executable quantum patterns with proper temporal layering and flow management.
"""

from __future__ import annotations

from contextlib import suppress

# import layout is intentionally non-standard due to optional deps fallback
from dataclasses import dataclass, field
from operator import itemgetter
from typing import TYPE_CHECKING

from graphix_zx.graphstate import (
    BaseGraphState,
    GraphState,
    compose_in_parallel,
    compose_sequentially,
)

from lspattern.accumulator import FlowAccumulator, ParityAccumulator, ScheduleAccumulator
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
from lspattern.tiling.template import cube_offset_xy, pipe_offset_xy
from lspattern.utils import UnionFind, get_direction, is_allowed_pair

# Constants
EDGE_TUPLE_SIZE = 2

if TYPE_CHECKING:
    from collections.abc import Callable

    from lspattern.blocks.cubes.base import RHGCube, RHGCubeSkeleton
    from lspattern.blocks.pipes.base import RHGPipe, RHGPipeSkeleton


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

    coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal]
    allowed_gid_pairs: set[tuple[QubitGroupIdGlobal, QubitGroupIdGlobal]]

    def __init__(self, z: int) -> None:
        self.z = z
        self.qubit_count = 0
        self.d: set[int] = set()
        self.patches = []
        self.lines = []
        self.in_portset = {}
        self.out_portset = {}
        self.cout_portset = {}
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
        self.tiling_node_maps: dict[str, dict[int, int]] = {}

        self.coord2tid: dict[PatchCoordGlobal3D, str] = {}
        self.coord2gid = {}
        self.allowed_gid_pairs = set()

    def __post_init__(self) -> None:
        """Post-initialization hook."""

    def add_cubes(self, cubes: dict[PatchCoordGlobal3D, RHGCube]) -> None:
        """Add multiple cubes to this temporal layer."""
        for pos, cube in cubes.items():
            self.add_cube(pos, cube)

    def add_pipes(self, pipes: dict[PipeCoordGlobal3D, RHGPipe]) -> None:
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

    def _setup_union_find(self) -> tuple[UnionFind, dict[PhysCoordGlobal3D, QubitGroupIdGlobal]]:
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

    def _remap_node_mappings(self, node_map: dict[int, int]) -> None:
        """Remap node mappings with given node map."""
        if not node_map:
            return
        self.node2coord = {NodeIdLocal(node_map.get(n, n)): c for n, c in self.node2coord.items()}
        self.coord2node = {c: NodeIdLocal(node_map.get(n, n)) for c, n in self.coord2node.items()}
        self.node2role = {NodeIdLocal(node_map.get(n, n)): r for n, r in self.node2role.items()}

    def _remap_portsets(self, node_map: dict[int, int]) -> None:
        """Remap portsets with given node map."""
        for p, nodes in list(self.in_portset.items()):
            self.in_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        for p, nodes in list(self.out_portset.items()):
            self.out_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        for p, nodes in list(self.cout_portset.items()):
            self.cout_portset[p] = [NodeIdLocal(node_map.get(n, n)) for n in nodes]
        self.in_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.in_ports]
        self.out_ports = [NodeIdLocal(node_map.get(n, n)) for n in self.out_ports]

    @staticmethod
    def _compose_single_cube(
        _pos: tuple[int, int, int], blk: object, g: BaseGraphState | None
    ) -> tuple[BaseGraphState, dict[int, int], dict[int, int]]:
        """Compose a single cube into the graph."""
        g2 = blk.local_graph  # type: ignore[attr-defined]

        if g is None:
            return g2, {}, {n: n for n in getattr(g2, "physical_nodes", [])}

        g_new, node_map1, node_map2 = compose_in_parallel(g, g2)
        return g_new, node_map1, node_map2

    def _process_cube_coordinates(self, blk: object, pos: tuple[int, int, int], node_map2: dict[int, int]) -> None:
        """Process cube coordinates and roles."""
        d_val = int(blk.d)  # type: ignore[attr-defined]
        z_base = int(pos[2]) * (2 * d_val)

        # Compute z-shift
        try:
            bmin_z = min(c[2] for c in blk.node2coord.values())  # type: ignore[attr-defined]
        except ValueError:
            bmin_z = z_base
        z_shift = int(z_base - bmin_z)

        # Ingest coords/roles
        for old_n, coord in blk.node2coord.items():  # type: ignore[attr-defined]
            new_n = node_map2.get(old_n)
            if new_n is None:
                continue
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2]) + z_shift
            c_new = PhysCoordGlobal3D((x, y, z))
            self.node2coord[NodeIdLocal(new_n)] = c_new
            self.coord2node[c_new] = NodeIdLocal(new_n)

        for old_n, role in blk.node2role.items():  # type: ignore[attr-defined]
            new_n = node_map2.get(old_n)
            if new_n is not None:
                self.node2role[NodeIdLocal(new_n)] = role

    def _process_cube_ports(self, pos: tuple[int, int, int], blk: object, node_map2: dict[int, int]) -> None:
        """Process cube ports."""
        if getattr(blk, "in_ports", None):
            patch_pos = PatchCoordGlobal3D(pos)
            self.in_portset[patch_pos] = [NodeIdLocal(node_map2[n]) for n in blk.in_ports if n in node_map2]  # type: ignore[attr-defined]
            self.in_ports.extend(self.in_portset[patch_pos])
        if getattr(blk, "out_ports", None):
            patch_pos = PatchCoordGlobal3D(pos)
            self.out_portset[patch_pos] = [NodeIdLocal(node_map2[n]) for n in blk.out_ports if n in node_map2]  # type: ignore[attr-defined]
            self.out_ports.extend(self.out_portset[patch_pos])
        if getattr(blk, "cout_ports", None):
            patch_pos = PatchCoordGlobal3D(pos)
            self.cout_portset[patch_pos] = [
                NodeIdLocal(node_map2[n])
                for s in blk.cout_ports  # type: ignore[attr-defined]
                for n in s
                if n in node_map2
            ]

    def _build_graph_from_blocks(self) -> BaseGraphState | None:
        """Build the quantum graph state from cubes and pipes."""
        g: BaseGraphState | None = None

        # Compose cube graphs
        for pos, blk in self.cubes_.items():
            g, node_map1, node_map2 = self._compose_single_cube(pos, blk, g)
            self._remap_node_mappings(node_map1)
            self._remap_portsets(node_map1)
            self._process_cube_coordinates(blk, pos, node_map2)
            self._process_cube_ports(pos, blk, node_map2)

        # Compose pipe graphs (spatial pipes in this layer)
        return self._compose_pipe_graphs(g)

    def _compose_pipe_graphs(self, g: BaseGraphState | None) -> BaseGraphState | None:  # noqa: C901
        """Compose pipe graphs into the main graph state."""
        for pipe_coord, pipe in self.pipes_.items():
            source, _sink = pipe_coord
            d_val = int(pipe.d)
            z_base = int(source[2]) * (2 * d_val)

            # Use materialized pipe if local_graph is None
            pipe_block = pipe
            g2 = pipe.local_graph
            if g2 is None:
                materialized = pipe.materialize()
                if hasattr(materialized, "local_graph"):
                    pipe_block = materialized  # type: ignore[assignment]
                    g2 = pipe_block.local_graph
                else:
                    g2 = None

            if g is None:
                g = g2
                node_map1: dict[int, int] = {}
                node_map2: dict[int, int] = {n: n for n in getattr(g2, "physical_nodes", [])}
            elif g2 is not None:
                g_new, node_map1, node_map2 = compose_in_parallel(g, g2)
                self._remap_node_mappings(node_map1)
                self._remap_portsets(node_map1)
                g = g_new
            else:
                node_map1 = {}
                node_map2 = {}

            try:
                bmin_z = min(c[2] for c in pipe_block.node2coord.values())
            except ValueError:
                bmin_z = z_base
            z_shift = int(z_base - bmin_z)

            for old_n, coord in pipe_block.node2coord.items():
                new_n = node_map2.get(old_n)
                if new_n is None:
                    continue
                # XY はテンプレートで既に絶対座標化済み(to_temporal_layer で shift 済み)
                x, y, z = (
                    int(coord[0]),
                    int(coord[1]),
                    int(coord[2]) + z_shift,
                )
                c_new = PhysCoordGlobal3D((x, y, z))
                self.node2coord[NodeIdLocal(new_n)] = c_new
                self.coord2node[c_new] = NodeIdLocal(new_n)
            for old_n, role in pipe_block.node2role.items():
                new_n = node_map2.get(old_n)
                if new_n is not None:
                    self.node2role[NodeIdLocal(new_n)] = role

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
        try:
            edges = getattr(g, "physical_edges", []) or []
            result: set[tuple[int, int]] = set()
            for u, v in edges:
                edge = tuple(sorted((int(u), int(v))))
                if len(edge) == EDGE_TUPLE_SIZE:
                    result.add((edge[0], edge[1]))
        except (AttributeError, TypeError, ValueError):
            return set()
        else:
            return result

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
        coord_gid_2d: dict[tuple[int, int], QubitGroupIdGlobal],
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
        # Avoid duplicates by canonical edge ordering
        sorted_edge = tuple(sorted((int(u), int(v))))
        if len(sorted_edge) == EDGE_TUPLE_SIZE:
            edge = (sorted_edge[0], sorted_edge[1])
            if edge not in existing:
                g.add_physical_edge(u, v)
                existing.add(edge)

    def _add_seam_edges(
        self, g: BaseGraphState | None, coord_gid_2d: dict[tuple[int, int], QubitGroupIdGlobal]
    ) -> BaseGraphState:
        """Add CZ edges across cube-pipe seams within the same temporal layer."""
        if g is None:
            g = GraphState()

        # Build XY regions
        cube_xy_all = self._build_xy_regions(coord_gid_2d)
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

        self.coord2gid = coord2gid
        self.allowed_gid_pairs = allowed_gid_pairs

        # Build GraphState by composing pre-materialized block graphs
        self.node2coord = {}
        self.coord2node = {}
        self.node2role = {}
        g = self._build_graph_from_blocks()

        # Add CZ edges across cube-pipe seams within the same temporal layer
        coord_gid_2d: dict[tuple[int, int], QubitGroupIdGlobal] = {}
        g = self._add_seam_edges(g, coord_gid_2d)

        # Finalize
        if g is None:
            g = GraphState()
        self.local_graph = g
        self.qubit_count = len(getattr(g, "physical_nodes", []) or [])

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
            "xy": dict(enumerate(data2d)),  # type: ignore[arg-type]
        }

    def _get_coordinate_bounds(self) -> tuple[int, int, int, int, int, int]:
        """Get min/max bounds for all coordinates."""
        coords = list(self.node2coord.values())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    @staticmethod
    def _create_face_checker(
        face: str, bounds: tuple[int, int, int, int, int, int], depths: list[int]
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

    # ---- T25: boundary queries ---------------------------------------------
    def get_boundary_nodes(
        self,
        *,
        face: str,
        depth: list[int] | None = None,
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
        return self.tiling_node_maps  # type: ignore[return-value]


# (removed duplicate CompiledRHGCanvas definition)


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
    in_portset : dict[PatchCoordGlobal3D, list[int]]
        Input port sets for each patch.
    out_portset : dict[PatchCoordGlobal3D, list[int]]
        Output port sets for each patch.
    cout_portset : dict[PatchCoordGlobal3D, list[int]]
        Ancilla output port sets for each patch.
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

    in_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = field(default_factory=dict)
    out_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = field(default_factory=dict)
    cout_portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]] = field(default_factory=dict)

    # Give defaults to satisfy dataclass ordering; caller may override later
    schedule: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(default_factory=ParityAccumulator)
    zlist: list[int] = field(default_factory=list)

    # Optional placeholders
    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    # (deprecated) debug seam pairs: removed

    # def generate_stim_circuit(self) -> stim.Circuit:
    #     pass

    @staticmethod
    def _remap_graph_nodes(gsrc: BaseGraphState, nmap: dict[NodeIdLocal, NodeIdLocal]) -> dict[int, int]:
        """Create new nodes in destination graph."""
        gdst = GraphState()
        created: dict[int, int] = {}
        for old in gsrc.physical_nodes:
            new_id = nmap.get(NodeIdLocal(old), NodeIdLocal(old))
            if int(new_id) in created:
                continue
            created[int(new_id)] = gdst.add_physical_node()
        return created

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
                with suppress(Exception):
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
            with suppress(Exception):
                gdst.add_physical_edge(created.get(int(nu), int(nu)), created.get(int(nv), int(nv)))

    @staticmethod
    def _create_remapped_graphstate(
        gsrc: BaseGraphState | None, nmap: dict[NodeIdLocal, NodeIdLocal]
    ) -> BaseGraphState | None:
        """Create a remapped GraphState."""
        if gsrc is None:
            return None
        gdst = GraphState()
        created = CompiledRHGCanvas._remap_graph_nodes(gsrc, nmap)
        CompiledRHGCanvas._remap_measurement_bases(gsrc, gdst, nmap, created)
        CompiledRHGCanvas._remap_graph_edges(gsrc, gdst, nmap, created)
        return gdst

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> CompiledRHGCanvas:
        """Remap nodes according to the given node mapping."""

        new_cgraph = CompiledRHGCanvas(
            layers=self.layers.copy(),
            global_graph=self._create_remapped_graphstate(self.global_graph, node_map),
            coord2node={},
            in_portset={},
            out_portset={},
            cout_portset={},
            schedule=self.schedule.remap_nodes({NodeIdGlobal(k): NodeIdGlobal(v) for k, v in node_map.items()}),
            flow=self.flow.remap_nodes(node_map),
            parity=self.parity.remap_nodes(node_map),
            cubes_=self.cubes_.copy(),
            pipes_=self.pipes_.copy(),
            zlist=list(self.zlist),
        )

        # Remap coord2node
        for coord, old_nodeid in self.coord2node.items():
            new_cgraph.coord2node[coord] = node_map[old_nodeid]

        # Remap portsets
        for pos, nodes in self.in_portset.items():
            new_cgraph.in_portset[pos] = [node_map[n] for n in nodes]
        for pos, nodes in self.out_portset.items():
            new_cgraph.out_portset[pos] = [node_map[n] for n in nodes]
        for pos, nodes in self.cout_portset.items():
            new_cgraph.cout_portset[pos] = [node_map[n] for n in nodes]

        return new_cgraph

    # ---- T25: boundary queries ---------------------------------------------
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

    # ---- T25: method form of temporal composition --------------------------
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
    template: object | None = None
    # {(0,0,0): InitPlusSkeleton(), ..}
    cubes_: dict[PatchCoordGlobal3D, RHGCubeSkeleton] = field(default_factory=dict)
    # {((0,0,0),(1,0,0)): StabilizeSkeleton(), ((0,0,0), (0,0,1)): MeasureSkeleton(basis=X)}
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

        cubes_ = {}
        for pos, c in trimmed_cubes_skeleton.items():
            # Materialize block and attach its 3D anchor so z-offset is correct
            blk = c.to_block()
            with suppress(Exception):
                # Ensure blocks know their placement (x, y, z)
                blk.source = pos
            cubes_[pos] = blk
        pipes_: dict[PipeCoordGlobal3D, RHGPipe] = {}
        for pipe_coord, p in trimmed_pipes_skeleton.items():
            coord_tuple = tuple(pipe_coord)
            if len(coord_tuple) != EDGE_TUPLE_SIZE:
                continue
            _start, _end = coord_tuple
            block = p.to_block()
            if hasattr(block, "local_graph"):
                pipes_[pipe_coord] = block  # type: ignore[assignment]

        cubes_filtered = {k: v for k, v in cubes_.items() if hasattr(v, "local_graph")}
        return RHGCanvas(
            name=self.name,
            cubes_=cubes_filtered,  # type: ignore[arg-type]
            pipes_=pipes_,
        )


@dataclass
class RHGCanvas:  # TopologicalComputationGraph in tqec
    name: str = "Blank Canvas"

    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    layers: list[TemporalLayer] | None = None

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
        cgraph = CompiledRHGCanvas(
            layers=[],
            global_graph=None,
            coord2node={},
            in_portset={},
            out_portset={},
            cout_portset={},
            schedule=ScheduleAccumulator(),
            flow=FlowAccumulator(),
            parity=ParityAccumulator(),
            zlist=[],
        )

        # Compose layers in increasing temporal order, wiring any cross-layer pipes
        for z in sorted(temporal_layers.keys()):
            layer = temporal_layers[z]
            # Select pipes whose start.z is the last compiled z and end.z is this layer z
            prev_z = cgraph.zlist[-1] if getattr(cgraph, "zlist", []) else None
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
    return CompiledRHGCanvas(
        layers=[next_layer],
        global_graph=next_layer.local_graph,
        coord2node={k: NodeIdLocal(v) for k, v in next_layer.coord2node.items()},
        in_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in next_layer.in_portset.items()},
        out_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in next_layer.out_portset.items()},
        cout_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in next_layer.cout_portset.items()},
        schedule=next_layer.schedule,
        zlist=[next_layer.z],
        cubes_=next_layer.cubes_,
        pipes_=next_layer.pipes_,
    )


def _compose_graphs_sequentially(
    graph1: BaseGraphState, graph2: BaseGraphState
) -> tuple[BaseGraphState, dict[int, int], dict[int, int]]:
    """Compose two graphs sequentially with fallback to manual composition."""
    try:
        result = compose_sequentially(graph1, graph2)
        expected_tuple_size = 3
        if isinstance(result, tuple) and len(result) == expected_tuple_size:
            return result
        return _manual_graph_composition(graph1, graph2)
    except (ValueError, TypeError, AttributeError):
        return _manual_graph_composition(graph1, graph2)


# TODO: should be removed after compose_sequentially is fixed
def _manual_graph_composition(
    graph1: BaseGraphState, graph2: BaseGraphState
) -> tuple[BaseGraphState, dict[int, int], dict[int, int]]:
    """Manually compose two graphs when canonical composition fails."""
    g = GraphState()
    node_map1 = {}
    node_map2 = {}

    # Copy graph1 nodes
    for n in graph1.physical_nodes:
        nn = g.add_physical_node()
        mb = graph1.meas_bases.get(n)
        if mb is not None:
            g.assign_meas_basis(nn, mb)
        node_map1[n] = nn

    # Copy graph2 nodes
    for n in graph2.physical_nodes:
        nn = g.add_physical_node()
        mb = graph2.meas_bases.get(n)
        if mb is not None:
            g.assign_meas_basis(nn, mb)
        node_map2[n] = nn

    # Copy edges
    for u, v in graph1.physical_edges:
        g.add_physical_edge(node_map1[u], node_map1[v])
    for u, v in graph2.physical_edges:
        g.add_physical_edge(node_map2[u], node_map2[v])

    return g, node_map1, node_map2


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
        # 直接テンプレートをXY移動(inplace=True)
        c.template.shift_coords((dx, dy), coordinate="tiling2d", inplace=True)
    for pipe_coord, p in pipes.items():
        coord_tuple = tuple(pipe_coord)
        if len(coord_tuple) != EDGE_TUPLE_SIZE:
            continue
        source, sink = coord_tuple
        direction = get_direction(source, sink)
        dx, dy = pipe_offset_xy(p.d, source, sink, direction)
        # 直接テンプレートをXY移動(inplace=True)
        p.template.shift_coords((dx, dy), coordinate="tiling2d", inplace=True)

    # materialize blocks before adding
    cubes_mat = {pos: blk.materialize() for pos, blk in cubes.items()}
    pipes_mat = {pipe_coord: p.materialize() for pipe_coord, p in pipes.items()}

    # Ensure proper typing for layer addition
    cubes_typed = {pos: cube for pos, cube in cubes_mat.items() if hasattr(cube, "local_graph")}
    pipes_typed = {pipe_coord: pipe for pipe_coord, pipe in pipes_mat.items() if hasattr(pipe, "local_graph")}

    layer.add_cubes(cubes_typed)  # type: ignore[arg-type]
    layer.add_pipes(pipes_typed)  # type: ignore[arg-type]

    # compile this layer
    layer.compile()
    return layer


def _remap_layer_mappings(next_layer: TemporalLayer, node_map2: dict[int, int]) -> None:
    """Remap next layer mappings."""
    next_layer.coord2node = {c: NodeIdLocal(node_map2.get(int(n), int(n))) for c, n in next_layer.coord2node.items()}
    next_layer.node2coord = {NodeIdLocal(node_map2.get(int(n), int(n))): c for n, c in next_layer.node2coord.items()}
    next_layer.node2role = {NodeIdLocal(node_map2.get(int(n), int(n))): r for n, r in next_layer.node2role.items()}


def _build_merged_coord2node(cgraph: CompiledRHGCanvas, next_layer: TemporalLayer) -> dict[PhysCoordGlobal3D, int]:
    """Build merged coordinate to node mapping."""
    return {
        **cgraph.coord2node,
        **next_layer.coord2node,
    }


def _remap_temporal_portsets(
    cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, node_map1: dict[int, int], node_map2: dict[int, int]
) -> tuple[
    dict[PatchCoordGlobal3D, list[int]],
    dict[PatchCoordGlobal3D, list[int]],
    dict[PatchCoordGlobal3D, list[int]],
]:
    """Remap portsets for temporal composition."""
    in_portset = {pos: [node_map2[int(n)] for n in nodes] for pos, nodes in next_layer.in_portset.items()}
    out_portset = {pos: [node_map1[int(n)] for n in nodes] for pos, nodes in cgraph.out_portset.items()}
    cout_portset = {
        **{pos: [node_map1[int(n)] for n in nodes] for pos, nodes in cgraph.cout_portset.items()},
        **{pos: [node_map2[int(n)] for n in nodes] for pos, nodes in next_layer.cout_portset.items()},
    }
    return in_portset, out_portset, cout_portset


def _build_coordinate_gid_mapping(
    cgraph: CompiledRHGCanvas, next_layer: TemporalLayer
) -> dict[PhysCoordGlobal3D, QubitGroupIdGlobal]:
    """Build coordinate to group ID mapping."""
    new_coord2gid: dict[PhysCoordGlobal3D, QubitGroupIdGlobal] = {}
    for _pos, cube in [*cgraph.cubes_.items(), *next_layer.cubes_.items()]:
        new_coord2gid.update({PhysCoordGlobal3D(k): QubitGroupIdGlobal(v) for k, v in cube.coord2gid.items()})
    for _pipe_pos, pipe in [*cgraph.pipes_.items(), *next_layer.pipes_.items()]:
        new_coord2gid.update({PhysCoordGlobal3D(k): QubitGroupIdGlobal(v) for k, v in pipe.coord2gid.items()})
    return new_coord2gid


def _update_accumulators(cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, new_graph: BaseGraphState) -> None:
    """Update accumulators at layer boundaries."""
    try:
        z_minus_ancillas = []
        bn = next_layer.get_boundary_nodes(face="z-", depth=[0])

        # Collect anchor node IDs for ancilla X/Z
        for c in bn.get("xcheck", []):
            nid = next_layer.coord2node.get(c)
            if nid is not None:
                z_minus_ancillas.append(nid)
        for c in bn.get("zcheck", []):
            nid = next_layer.coord2node.get(c)
            if nid is not None:
                z_minus_ancillas.append(nid)

        # Update accumulators
        for anchor in z_minus_ancillas:
            cgraph.schedule.update_at(anchor, new_graph)
            cgraph.parity.update_at(anchor, new_graph)
            cgraph.flow.update_at(anchor, new_graph)
    except (ValueError, KeyError, AttributeError):
        pass  # Be tolerant: boundary queries are best-effort


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

    for p in pipes:
        if hasattr(p, "source") and p.source and hasattr(p, "sink") and p.sink:
            allowed_gid_pairs.add(
                (
                    QubitGroupIdGlobal(cgraph.cubes_[p.source].get_tiling_id()),
                    QubitGroupIdGlobal(next_layer.cubes_[p.sink].get_tiling_id()),
                )
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
            if source_node is not None and sink_node is not None:
                new_graph.add_physical_edge(source_node, sink_node)


def add_temporal_layer(cgraph: CompiledRHGCanvas, next_layer: TemporalLayer, pipes: list[RHGPipe]) -> CompiledRHGCanvas:
    """Compose the compiled canvas with the next temporal layer."""
    if cgraph.global_graph is None:
        return _create_first_layer_canvas(next_layer)

    # Compose graphs and remap
    if next_layer.local_graph is None:
        error_msg = "next_layer.local_graph cannot be None"
        raise ValueError(error_msg)
    new_graph, node_map1, node_map2 = _compose_graphs_sequentially(cgraph.global_graph, next_layer.local_graph)
    cgraph = cgraph.remap_nodes({NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map1.items()})
    _remap_layer_mappings(next_layer, node_map2)

    # Build merged mappings
    new_coord2node = _build_merged_coord2node(cgraph, next_layer)
    in_portset, out_portset, cout_portset = _remap_temporal_portsets(cgraph, next_layer, node_map1, node_map2)
    new_coord2gid = _build_coordinate_gid_mapping(cgraph, next_layer)

    # Setup temporal connections
    _setup_temporal_connections(pipes, cgraph, next_layer, new_graph, new_coord2node, new_coord2gid)

    new_layers = [*cgraph.layers, next_layer]

    # Update accumulators
    _update_accumulators(cgraph, next_layer, new_graph)

    # Merge schedule, flow, parity
    new_schedule = cgraph.schedule.compose_sequential(next_layer.schedule)

    # Start from unions of existing x/z flows
    xflow_combined: dict[int, set[int]] = {}
    zflow_combined: dict[int, set[int]] = {}
    for src, dsts in cgraph.flow.xflow.items():
        xflow_combined[src] = set(dsts)
    for src, dsts in next_layer.flow.xflow.items():
        xflow_combined.setdefault(src, set()).update(dsts)
    for src, dsts in cgraph.flow.zflow.items():
        zflow_combined[src] = set(dsts)
    for src, dsts in next_layer.flow.zflow.items():
        zflow_combined.setdefault(src, set()).update(dsts)
    # No explicit seam corrections here; physical edges were added when pipes exist.

    # Convert int keys to NodeIdLocal for FlowAccumulator
    xflow_typed = {NodeIdLocal(k): {NodeIdLocal(v) for v in vs} for k, vs in xflow_combined.items()}
    zflow_typed = {NodeIdLocal(k): {NodeIdLocal(v) for v in vs} for k, vs in zflow_combined.items()}
    merged_flow = FlowAccumulator(xflow=xflow_typed, zflow=zflow_typed)
    new_parity = ParityAccumulator(
        x_checks=cgraph.parity.x_checks + next_layer.parity.x_checks,
        z_checks=cgraph.parity.z_checks + next_layer.parity.z_checks,
    )

    return CompiledRHGCanvas(
        layers=new_layers,
        global_graph=new_graph,
        coord2node={k: NodeIdLocal(v) for k, v in new_coord2node.items()},
        in_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in in_portset.items()},
        out_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in out_portset.items()},
        cout_portset={k: [NodeIdLocal(v) for v in vs] for k, vs in cout_portset.items()},
        schedule=new_schedule,
        flow=merged_flow,
        parity=new_parity,
        zlist=[*list(getattr(cgraph, "zlist", [])), next_layer.z],
    )
