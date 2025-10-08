"""Compiled RHG canvas module.

This module provides the CompiledRHGCanvas class for representing a compiled
RHG canvas with multiple temporal layers, along with utilities for composing
temporal layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from graphix_zx.graphstate import BaseGraphState, GraphState, compose

from lspattern.accumulator import FlowAccumulator, ParityAccumulator, ScheduleAccumulator
from lspattern.canvas.layer import TemporalLayer
from lspattern.canvas.ports import PortManager
from lspattern.mytype import (
    NodeIdGlobal,
    NodeIdLocal,
    PatchCoordGlobal3D,
    PhysCoordGlobal3D,
    PipeCoordGlobal3D,
    QubitGroupIdGlobal,
    TilingId,
)
from lspattern.utils import is_allowed_pair

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lspattern.blocks.cubes.base import RHGCube
    from lspattern.blocks.pipes.base import RHGPipe


@dataclass
class CompiledRHGCanvas:
    """
    Represents a compiled RHG canvas, containing temporal layers, the global graph,
    coordinate mappings, port sets, and accumulators for schedule, flow, and parity.

    Attributes
    ----------
    layers : list[TemporalLayer]
        The temporal layers of the canvas.
    global_graph : GraphState | None
        The global graph state after compilation.
    coord2node : dict[PhysCoordGlobal3D, int]
        Mapping from physical coordinates to node IDs.
    node2role : dict[int, str]
        Mapping from node IDs to their roles.
    port_manager : PortManager
        Manages all port-related data (in/out/cout ports).
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
    global_graph: GraphState | None = None
    coord2node: dict[PhysCoordGlobal3D, NodeIdLocal] = field(default_factory=dict)
    node2role: dict[NodeIdLocal, str] = field(default_factory=dict)

    # Port management delegated to PortManager
    port_manager: PortManager = field(default_factory=PortManager)

    # Give defaults to satisfy dataclass ordering; caller may override later
    schedule: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(default_factory=ParityAccumulator)

    # Backward compatibility properties
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

    def _register_cout_group(
        self,
        patch_pos: PatchCoordGlobal3D,
        nodes: list[NodeIdLocal],
    ) -> None:
        """Record a cout group on the compiled canvas and sync caches."""
        self.port_manager.register_cout_group(patch_pos, nodes)

    def _rebuild_cout_group_cache(self) -> None:
        """Recompute flat cout caches from grouped data."""
        self.port_manager.rebuild_cout_group_cache()

    def get_cout_group_by_coord(
        self,
        coord: PhysCoordGlobal3D,
    ) -> tuple[PatchCoordGlobal3D, list[NodeIdLocal]] | None:
        """Return the cout group for the node at `coord`, if present."""
        node = self.coord2node.get(coord)
        if node is None:
            return None
        return self.port_manager.get_cout_group_by_node(node)

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
    ) -> GraphState | None:
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

        # Remap layer's coordinate mapper
        remapped_layer.coord_mapper = layer.coord_mapper.copy()
        remapped_layer.coord_mapper.remap_nodes({int(k): int(v) for k, v in node_map.items()})

        # Remap portsets (including in_ports, out_ports, cout_ports via PortManager)
        CompiledRHGCanvas._remap_layer_portsets(layer, remapped_layer, node_map)

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
        # Copy and remap the port_manager
        remapped_layer.port_manager = layer.port_manager.copy()
        remapped_layer.port_manager.remap_ports({int(k): int(v) for k, v in node_map.items()})

    # TODO: this could be made more efficient by avoiding deep copies
    def remap_nodes(self, node_map: Mapping[NodeIdLocal, NodeIdLocal]) -> CompiledRHGCanvas:
        """Remap nodes according to the given node mapping."""
        # Deep copy and remap each layer
        remapped_layers = [CompiledRHGCanvas._remap_layer(layer, node_map) for layer in self.layers]

        # Copy and remap port_manager
        remapped_port_manager = self.port_manager.copy()
        remapped_port_manager.remap_ports({int(k): int(v) for k, v in node_map.items()})

        new_cgraph = CompiledRHGCanvas(
            layers=remapped_layers,
            global_graph=self._create_remapped_graphstate(self.global_graph, dict(node_map)),
            coord2node={},
            port_manager=remapped_port_manager,
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


def _create_first_layer_canvas(next_layer: TemporalLayer) -> CompiledRHGCanvas:
    """Create compiled canvas for the first temporal layer."""

    return CompiledRHGCanvas(
        layers=[next_layer],
        global_graph=next_layer.local_graph,
        coord2node={k: NodeIdLocal(v) for k, v in next_layer.coord2node.items()},
        node2role={NodeIdLocal(k): v for k, v in next_layer.node2role.items()},
        port_manager=next_layer.port_manager.copy(),
        schedule=next_layer.schedule,
        parity=next_layer.parity,
        flow=next_layer.flow,
        zlist=[next_layer.z],
        cubes_=next_layer.cubes_,
        pipes_=next_layer.pipes_,
    )


def _remap_layer_mappings(next_layer: TemporalLayer, node_map2: Mapping[int, int]) -> None:
    """Remap next layer mappings."""
    next_layer.coord_mapper.remap_nodes(node_map2)

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
    """Compose the compiled canvas with the next temporal layer.

    Parameters
    ----------
    cgraph : CompiledRHGCanvas
        The current compiled canvas.
    next_layer : TemporalLayer
        The next temporal layer to add.
    pipes : list[RHGPipe]
        List of temporal pipes connecting the layers.

    Returns
    -------
    CompiledRHGCanvas
        The new compiled canvas with the added layer.
    """

    if cgraph.global_graph is None:
        return _create_first_layer_canvas(next_layer)

    # Compose graphs and remap
    new_graph, node_map1, node_map2 = compose(cgraph.global_graph, next_layer.local_graph)  # pyright: ignore[reportArgumentType]

    # Only remap if node mapping actually changes node IDs
    if any(k != v for k, v in node_map1.items()):
        cgraph = cgraph.remap_nodes({NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map1.items()})

    _remap_layer_mappings(next_layer, node_map2)

    # Build merged mappings
    new_coord2node = _build_merged_coord2node(cgraph, next_layer)
    merged_port_manager = cgraph.port_manager.merge(next_layer.port_manager, node_map1, node_map2)
    new_coord2gid = _build_coordinate_gid_mapping(cgraph, next_layer)

    # Setup temporal connections
    _setup_temporal_connections(pipes, cgraph, next_layer, new_graph, new_coord2node, new_coord2gid)

    new_layers = [*cgraph.layers, next_layer]

    # Update accumulators
    last_nodes = set(next_layer.port_manager.in_ports)
    # remap to global node IDs
    last_nodes_remapped = {NodeIdGlobal(node_map2[int(n)]) for n in last_nodes}
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

    return CompiledRHGCanvas(
        layers=new_layers,
        global_graph=new_graph,
        coord2node={k: NodeIdLocal(v) for k, v in new_coord2node.items()},
        port_manager=merged_port_manager,
        schedule=new_schedule,
        flow=merged_flow,
        parity=new_parity,
        zlist=[*list(cgraph.zlist), next_layer.z],
    )
