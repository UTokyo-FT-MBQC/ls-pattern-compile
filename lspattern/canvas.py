
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional, Iterable, Mapping, Any

# graphix_zx pieces
from graphix_zx.graphstate import BaseGraphState, GraphState, compose_sequentially

from .blocks.base import BlockDelta, RHGBlock, choose_port_node
from .compile import compile_canvas
from .geom.tiler import PatchTiler


# ----------------------------
# small helpers
# ----------------------------
def _remap_set(nodes: Iterable[int], node_map: Mapping[int, int]) -> set[int]:
    return { node_map[n] for n in nodes }

def _remap_list_of_sets(sets: Iterable[Iterable[int]], node_map: Mapping[int, int]) -> list[set[int]]:
    return [ _remap_set(s, node_map) for s in sets ]

@dataclass
class ParityLast:
    z: int
    by_xy: Dict[Tuple[int,int], int]   # (x,y) -> GLOBAL node id

@dataclass
class ParityLayerRegistry:
    last_x: Dict[int, ParityLast] = field(default_factory=dict)  # logical -> last X layer
    last_z: Dict[int, ParityLast] = field(default_factory=dict)  # logical -> last Z layer

    def get_last(self, logical: int, kind: str) -> Optional[ParityLast]:
        kind = kind.upper()
        return (self.last_x if kind == 'X' else self.last_z).get(logical)

    def update_last_from_seams(
        self,
        logical: int,
        seam_last_x_g: Dict[Tuple[int,int], int],
        seam_last_z_g: Dict[Tuple[int,int], int],
        coord_to_node: Dict[Tuple[int,int,int], int],
    ) -> None:
        # node -> coord を一時生成して z を得る
        rev = { nid: coord for coord, nid in coord_to_node.items() }
        if seam_last_x_g:
            any_n = next(iter(seam_last_x_g.values()))
            z = rev[any_n][2]
            self.last_x[logical] = ParityLast(z=z, by_xy=dict(seam_last_x_g))
        if seam_last_z_g:
            any_n = next(iter(seam_last_z_g.values()))
            z = rev[any_n][2]
            self.last_z[logical] = ParityLast(z=z, by_xy=dict(seam_last_z_g))

# ----------------------------
# Registries / Accumulators
# ----------------------------
@dataclass
class LogicalRegistry:
    boundary_qidx: Dict[int, Dict[int, int]] = field(default_factory=dict)  # logical -> { GLOBAL node -> q_index }

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        self.boundary_qidx = {
            li: { node_map.get(n, n): q for n, q in qmap.items() }
            for li, qmap in self.boundary_qidx.items()
        }

    def set_boundary(self, logical: int, nodes: Set[int], qidx_map: Optional[Dict[int,int]] = None) -> None:
        if qidx_map is None:
            qidx_map = { n: i for i, n in enumerate(sorted(nodes)) }
        self.boundary_qidx[logical] = dict(qidx_map)
        
    def get_boundary_nodes(self, logical: int) -> Set[int]:
        qmap = self.boundary_qidx.get(logical, {})
        return set(qmap.keys())

    def require_boundary(self, logical: int) -> Set[int]:
        nodes = self.get_boundary_nodes(logical)
        if not nodes:
            raise ValueError(f"No boundary registered for logical {logical}.")
        return nodes


@dataclass
class ParityAccumulator:
    x_groups: list[set[int]] = field(default_factory=list)
    z_groups: list[set[int]] = field(default_factory=list)

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        self.x_groups = _remap_list_of_sets(self.x_groups, node_map)
        self.z_groups = _remap_list_of_sets(self.z_groups, node_map)

    def extend_from_delta(self, delta: BlockDelta, node_map2: Mapping[int, int]) -> None:
        self.x_groups.extend(_remap_list_of_sets(delta.x_checks, node_map2))
        self.z_groups.extend(_remap_list_of_sets(delta.z_checks, node_map2))


@dataclass
class FlowAccumulator:
    xflow: Dict[int, Set[int]] = field(default_factory=dict)

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        self.xflow = { node_map.get(k, k): { node_map.get(v, v) for v in vs } for k, vs in self.xflow.items() }

    def apply_delta(self, delta: BlockDelta, node_map2: Mapping[int, int]) -> None:
        for src_local, corr_locals in delta.flow_local.items():
            src = node_map2[src_local]
            tgts = { node_map2[v] for v in corr_locals }
            self.xflow.setdefault(src, set()).update(tgts)


@dataclass
class ScheduleAccumulator:
    measure_groups: list[set[int]] = field(default_factory=list)

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        self.measure_groups = _remap_list_of_sets(self.measure_groups, node_map)

    def extend_from_delta(self, delta: BlockDelta, node_map2: Mapping[int, int]) -> None:
        self.measure_groups.extend(_remap_list_of_sets(delta.measure_groups, node_map2))

    def as_scheduler(self, graph: BaseGraphState):
        # TODO: integrate graphix_zx.scheduler if desired
        return None


# ----------------------------
# Canvas
# ----------------------------
@dataclass
class RHGCanvas:
    """Growing RHG canvas. Each block contributes a BlockDelta that's merged in-place."""
    graph: Optional[BaseGraphState] = None
    coord_to_node: Dict[tuple[int, int, int], int] = field(default_factory=dict)

    logical_registry: LogicalRegistry = field(default_factory=LogicalRegistry)
    parity_accum: ParityAccumulator = field(default_factory=ParityAccumulator)
    flow_accum: FlowAccumulator = field(default_factory=FlowAccumulator)
    schedule_accum: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)

    parity_layers: ParityLayerRegistry = field(default_factory=ParityLayerRegistry)

    tiler: PatchTiler = field(default_factory=PatchTiler)
    z_top: int = 0

    # ---- API ----
    def append(self, block: RHGBlock) -> "RHGCanvas":
        delta = block.emit(self)
        if self.graph is None:
            self._adopt_initial_delta(delta)
        else:
            self._merge_delta(delta)
        return self

    def compile(self):
        if self.graph is None:
            raise ValueError("Nothing to compile: canvas is empty.")
        return compile_canvas(
            graph=self.graph,
            xflow=self.flow_accum.xflow,
            zflow=None,
            x_parity=self.parity_accum.x_groups or None,
            z_parity=self.parity_accum.z_groups or None,
            scheduler=self.schedule_accum.as_scheduler(self.graph),
            correct_output=True,
        )

    # ---- internals ----

    def _adopt_initial_delta(self, delta: BlockDelta) -> None:
            self.graph = delta.local_graph

            # coords
            for n_local, coord in delta.node_coords.items():
                self.coord_to_node[coord] = n_local

            # identity remap for initial graph
            initial_nodes = getattr(self.graph, "physical_nodes", set())
            id_map = { n: n for n in initial_nodes }

            # accumulators
            self.parity_accum.extend_from_delta(delta, id_map)
            self.flow_accum.apply_delta(delta, id_map)
            self.schedule_accum.extend_from_delta(delta, id_map)

            # logical boundary
            for lidx, out_nodes_local in delta.out_ports.items():
                qmap = delta.out_qmap.get(lidx)
                self.logical_registry.set_boundary(lidx, set(out_nodes_local), qidx_map=qmap)

            # z_top heuristic
            if delta.node_coords:
                self.z_top = max(z for (_, _, z) in delta.node_coords.values())

    def _merge_delta(self, delta: BlockDelta) -> None:
        print("cap_top:", delta.cap_top, "frontier_z:", self.anc_frontier_z, "frontier_x:", self.anc_frontier_x)
        assert self.graph is not None
        # compose; expect (composed, node_map1, node_map2)
        composed, node_map1, node_map2 = compose_sequentially(self.graph, delta.local_graph)

        # remap existing state to composed ids
        self.logical_registry.remap_all(node_map1)
        self.parity_accum.remap_all(node_map1)
        self.flow_accum.remap_all(node_map1)
        self.schedule_accum.remap_all(node_map1)
        self.coord_to_node = { coord: node_map1.get(n, n) for coord, n in self.coord_to_node.items() }
        
        for n_local, coord in delta.node_coords.items():
            self.coord_to_node[coord] = node_map2[n_local]
            
        self.parity_accum.extend_from_delta(delta, node_map2)
        self.flow_accum.apply_delta(delta, node_map2)
        self.schedule_accum.extend_from_delta(delta, node_map2)  
        
        for center_g, locals_list in delta.parity_x_prev_global_curr_local:
            group = {center_g}
            group.update(node_map2[l] for l in locals_list if l in node_map2)
            if len(group) >= 2:
                self.parity_accum.x_groups.append(group)

        for center_g, locals_list in delta.parity_z_prev_global_curr_local:
            group = {center_g}
            group.update(node_map2[l] for l in locals_list if l in node_map2)
            if len(group) >= 2:
                self.parity_accum.z_groups.append(group)

        # 6) parity_layers を「最後の層」で更新（次ブロックが参照するため）
        #    seam_first_* は不要。seam_last_* のみ GLOBAL 化して反映する。
        def _seam_global(seam_local_xy2nid: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
            return {xy: node_map2[nid] for xy, nid in seam_local_xy2nid.items() if nid in node_map2}

        seam_last_x_g = _seam_global(delta.seam_last_x)
        seam_last_z_g = _seam_global(delta.seam_last_z)

        # 論理 index は out_ports 優先・無ければ in_ports から推定
        lidx: Optional[int] = None
        if delta.out_ports:
            lidx = next(iter(delta.out_ports.keys()))
        elif delta.in_ports:
            lidx = next(iter(delta.in_ports.keys()))
        if lidx is not None:
            self.parity_layers.update_last_from_seams(
                logical=lidx,
                seam_last_x_g=seam_last_x_g,
                seam_last_z_g=seam_last_z_g,
                coord_to_node=self.coord_to_node,
            )

        # 7) 論理 boundary の更新（out_ports があれば上書き）
        for lidx2, out_nodes_local in delta.out_ports.items():
            out_nodes_global = _remap_set(out_nodes_local, node_map2)
            qmap_local = delta.out_qmap.get(lidx2)
            qmap_global = None
            if qmap_local:
                qmap_global = {node_map2[n]: q for n, q in qmap_local.items() if n in node_map2}
            self.logical_registry.set_boundary(lidx2, out_nodes_global, qidx_map=qmap_global)

        # 8) 論理の消費（in_ports にはいるが out_ports に残らない論理を boundary から削除）
        consumed = set(delta.in_ports.keys()) - set(delta.out_ports.keys())
        for l in consumed:
            self.logical_registry.boundary_qidx.pop(l, None)

        # 9) z_top の更新
        if delta.node_coords:
            z_max_delta = max(z for (_, _, z) in delta.node_coords.values())
            if z_max_delta > self.z_top:
                self.z_top = z_max_delta
                
        self.graph = composed
