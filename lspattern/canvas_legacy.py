from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# graphix_zx pieces
from graphix_zx.graphstate import BaseGraphState, compose_sequentially

from lspattern.blocks.base import BlockDelta, RHGBlock
from lspattern.compile import compile_canvas
from lspattern.geom.tiler import PatchTiler


# ----------------------------
# Small helpers
# ----------------------------
def _remap_set(nodes: Iterable[int], node_map: Mapping[int, int]) -> set[int]:
    """Remap a set of LOCAL node ids to GLOBAL ids via node_map."""
    return {node_map[n] for n in nodes}


def _remap_list_of_sets(sets: Iterable[Iterable[int]], node_map: Mapping[int, int]) -> list[set[int]]:
    """Remap a list of LOCAL node-id sets to GLOBAL ids via node_map."""
    return [_remap_set(s, node_map) for s in sets]


# ----------------------------
# Parity-layer tracking
# ----------------------------
@dataclass
class ParityLast:
    """A record of the latest parity layer for a given logical index."""

    z: int
    by_xy: dict[tuple[int, int], int]  # (x, y) -> GLOBAL node id


@dataclass
class ParityLayerRegistry:
    """Keeps only the *last* X/Z parity layers per logical index."""

    last_x: dict[int, ParityLast] = field(default_factory=dict)
    last_z: dict[int, ParityLast] = field(default_factory=dict)

    def get_last(self, logical: int, kind: str) -> ParityLast | None:
        """Return the last parity layer of the given kind ('X' or 'Z') for a logical index."""
        return (self.last_x if kind.upper() == "X" else self.last_z).get(logical)

    def update_last_from_seams(
        self,
        logical: int,
        seam_last_x_g: dict[tuple[int, int], int],
        seam_last_z_g: dict[tuple[int, int], int],
        coord_to_node: dict[tuple[int, int, int], int],
    ) -> None:
        """Update the last X/Z layers from seam-last dicts (already GLOBAL ids)."""
        # Build a reverse map: GLOBAL node -> (x, y, z)
        node_to_coord: dict[int, tuple[int, int, int]] = {nid: coord for coord, nid in coord_to_node.items()}

        if seam_last_x_g:
            any_node = next(iter(seam_last_x_g.values()))
            z = node_to_coord[any_node][2]
            self.last_x[logical] = ParityLast(z=z, by_xy=dict(seam_last_x_g))

        if seam_last_z_g:
            any_node = next(iter(seam_last_z_g.values()))
            z = node_to_coord[any_node][2]
            self.last_z[logical] = ParityLast(z=z, by_xy=dict(seam_last_z_g))


# ----------------------------
# Registries / Accumulators
# ----------------------------
@dataclass
class LogicalRegistry:
    """Tracks the logical boundary as a mapping: logical -> {GLOBAL node -> q_index}."""

    boundary_qidx: dict[int, dict[int, int]] = field(default_factory=dict)

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        """Remap all stored GLOBAL node ids via node_map (compose_sequentially's node_map1)."""
        self.boundary_qidx = {
            li: {node_map.get(n, n): q for n, q in qmap.items()} for li, qmap in self.boundary_qidx.items()
        }

    def set_boundary(self, logical: int, nodes: set[int], qidx_map: dict[int, int] | None = None) -> None:
        """Set logical boundary nodes with an optional explicit q_index map."""
        if qidx_map is None:
            qidx_map = {n: i for i, n in enumerate(sorted(nodes))}
        self.boundary_qidx[logical] = dict(qidx_map)

    def get_boundary_nodes(self, logical: int) -> set[int]:
        """Return the set of boundary GLOBAL nodes for the logical index (empty set if absent)."""
        return set(self.boundary_qidx.get(logical, {}).keys())

    def require_boundary(self, logical: int) -> set[int]:
        """Return boundary nodes or raise if not present."""
        nodes = self.get_boundary_nodes(logical)
        if not nodes:
            raise ValueError(f"No boundary registered for logical {logical}.")
        return nodes


@dataclass
class ParityAccumulator:
    """Collects X/Z parity-check groups (GLOBAL node-id sets)."""

    x_groups: list[set[int]] = field(default_factory=list)
    z_groups: list[set[int]] = field(default_factory=list)

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        self.x_groups = _remap_list_of_sets(self.x_groups, node_map)
        self.z_groups = _remap_list_of_sets(self.z_groups, node_map)

    def extend_from_delta(self, delta: BlockDelta, node_map2: Mapping[int, int]) -> None:
        """Append the block-local parity groups remapped to GLOBAL ids."""
        self.x_groups.extend(_remap_list_of_sets(delta.x_checks, node_map2))
        self.z_groups.extend(_remap_list_of_sets(delta.z_checks, node_map2))


@dataclass
class FlowAccumulator:
    """Collects (currently) X-flow as a dict[GLOBAL node] -> set[GLOBAL nodes]."""

    xflow: dict[int, set[int]] = field(default_factory=dict)

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        self.xflow = {node_map.get(k, k): {node_map.get(v, v) for v in vs} for k, vs in self.xflow.items()}

    def apply_delta(self, delta: BlockDelta, node_map2: Mapping[int, int]) -> None:
        """Add block-local flow entries remapped via node_map2 (LOCAL -> GLOBAL)."""
        for src_local, corr_locals in delta.flow_local.items():
            src = node_map2[src_local]
            tgts = {node_map2[v] for v in corr_locals}
            self.xflow.setdefault(src, set()).update(tgts)


@dataclass
class ScheduleAccumulator:
    """
    Global time-slice accumulation.
    Each Block returns BlockDelta.schedule_tuples = [(t_local, {local_nodes}), ..] starting at 0.
    The canvas shifts them by base_time (global head) and merges by t_global.
    """

    _timeline: dict[int, set[int]] = field(default_factory=dict)  # t_global -> GLOBAL node set
    measure_groups: list[set[int]] = field(default_factory=list)  # exposed (sorted by t_global)

    def _rebuild_groups(self) -> None:
        """Rebuild 'measure_groups' as a list sorted by t_global."""
        self.measure_groups = [self._timeline[t] for t in sorted(self._timeline)]

    def remap_all(self, node_map: Mapping[int, int]) -> None:
        """Apply a GLOBAL id remap to the entire existing timeline."""
        self._timeline = {t: {node_map.get(n, n) for n in nodes} for t, nodes in self._timeline.items()}
        self._rebuild_groups()

    def extend_from_delta_timed(self, delta: BlockDelta, node_map2: Mapping[int, int], *, base_time: int) -> None:
        """Shift (t_local, LOCAL nodes) by base_time and merge into the timeline."""
        if not delta.schedule_tuples:
            return

        for t_local, group_local in delta.schedule_tuples:
            if not group_local:
                continue
            t_global = base_time + int(t_local)
            group_global = {node_map2[n] for n in group_local if n in node_map2}
            if group_global:
                self._timeline.setdefault(t_global, set()).update(group_global)

        self._rebuild_groups()

    def as_scheduler(self, graph: BaseGraphState):
        """
        Build a graphix_zx scheduler (or a dict fallback) with:
          * prepare_time: all non-input nodes at time 0
          * measure_time: timeline's t_global as is; input nodes at min(t_global) or 1 if empty
        """
        all_nodes = set(getattr(graph, "physical_nodes", set()))
        input_nodes = set(getattr(graph, "input_node_indices", {}).keys())

        # Prepare at time 0 (non-input nodes).
        prep_time = dict.fromkeys(all_nodes - input_nodes, 0)

        # Measurements follow timeline keys; input nodes go to the earliest measurement time.
        t0 = min(self._timeline) if self._timeline else 1
        meas_time: dict[int, int] = dict.fromkeys(input_nodes, t0)
        for t in sorted(self._timeline):
            for n in self._timeline[t]:
                meas_time[n] = t

        try:
            from graphix_zx.scheduler import Scheduler  # type: ignore

            sched = Scheduler(graph)
            sched.from_manual_design(prepare_time=prep_time, measure_time=meas_time)
            return sched
        except Exception:
            # Fallback for environments without graphix_zx.scheduler.
            return {"prepare_time": prep_time, "measure_time": meas_time}


# ----------------------------
# Canvas
# ----------------------------
@dataclass
class RHGCanvas:
    """
    Growing RHG canvas. Each block contributes a BlockDelta that's merged in-place.
    Geometry/ports/flows/parity/schedule are accumulated and later compiled.
    """

    graph: BaseGraphState | None = None
    coord_to_node: dict[tuple[int, int, int], int] = field(default_factory=dict)

    logical_registry: LogicalRegistry = field(default_factory=LogicalRegistry)
    parity_accum: ParityAccumulator = field(default_factory=ParityAccumulator)
    flow_accum: FlowAccumulator = field(default_factory=FlowAccumulator)
    schedule_accum: ScheduleAccumulator = field(default_factory=ScheduleAccumulator)

    parity_layers: ParityLayerRegistry = field(default_factory=ParityLayerRegistry)

    tiler: PatchTiler = field(default_factory=PatchTiler)
    z_top: int = 0

    # Global time-slice cursor (0 is reserved for prepare; measurements start at 1).
    _time_cursor: int = 1

    # ---- Public API ----
    def append(self, block: RHGBlock) -> RHGCanvas:
        """Emit a delta from the block and merge it into the canvas."""
        delta = block.emit(self)
        if self.graph is None:
            self._adopt_initial_delta(delta)
        else:
            self._merge_delta(delta)
        return self

    def compile(self):
        """Finalize and return the compiled artifacts via compile_canvas."""
        if self.graph is None:
            raise ValueError("Nothing to compile: canvas is empty.")
        return compile_canvas(
            graph=self.graph,
            xflow=self.flow_accum.xflow,
            x_parity=self.parity_accum.x_groups,
            z_parity=self.parity_accum.z_groups,
            scheduler=self.schedule_accum.as_scheduler(self.graph),
        )

    # ---- Internals ----
    def _adopt_initial_delta(self, delta: BlockDelta) -> None:
        """Adopt the very first block delta into an empty canvas."""
        self.graph = delta.local_graph

        # Record coordinates (LOCAL ids at this point).
        for n_local, coord in delta.node_coords.items():
            self.coord_to_node[coord] = n_local

        # Identity remap for the initial graph.
        initial_nodes = getattr(self.graph, "physical_nodes", set())
        id_map = {n: n for n in initial_nodes}

        # Accumulators.
        self.parity_accum.extend_from_delta(delta, id_map)
        self.flow_accum.apply_delta(delta, id_map)

        # Schedule (shift local time-slices by the current global cursor).
        self.schedule_accum.extend_from_delta_timed(delta, id_map, base_time=self._time_cursor)
        local_max = max((t for t, _ in (delta.schedule_tuples or [])), default=-1)
        if local_max >= 0:
            self._time_cursor += local_max + 1

        # Logical boundary.
        for lidx, out_nodes_local in delta.out_ports.items():
            qmap = delta.out_qmap.get(lidx)
            self.logical_registry.set_boundary(lidx, set(out_nodes_local), qidx_map=qmap)

        # Update z_top heuristic.
        if delta.node_coords:
            self.z_top = max(z for (_, _, z) in delta.node_coords.values())

    def _merge_delta(self, delta: BlockDelta) -> None:
        """Merge a subsequent block delta into the existing canvas."""
        assert self.graph is not None

        # Compose existing graph with the new local graph.
        composed, node_map1, node_map2 = compose_sequentially(self.graph, delta.local_graph)

        # Remap existing GLOBAL state into the composed graph's ids (via node_map1).
        self.logical_registry.remap_all(node_map1)
        self.parity_accum.remap_all(node_map1)
        self.flow_accum.remap_all(node_map1)
        self.schedule_accum.remap_all(node_map1)
        self.coord_to_node = {coord: node_map1.get(n, n) for coord, n in self.coord_to_node.items()}

        # Ingest new coordinates (LOCAL -> GLOBAL).
        for n_local, coord in delta.node_coords.items():
            self.coord_to_node[coord] = node_map2[n_local]

        # Ingest parity/flow (LOCAL -> GLOBAL).
        self.parity_accum.extend_from_delta(delta, node_map2)
        self.flow_accum.apply_delta(delta, node_map2)

        # Ingest schedule (LOCAL -> GLOBAL; shift by the global time cursor).
        self.schedule_accum.extend_from_delta_timed(delta, node_map2, base_time=self._time_cursor)
        local_max = max((t for t, _ in (delta.schedule_tuples or [])), default=-1)
        if local_max >= 0:
            self._time_cursor += local_max + 1

        # Apply unified parity additions (pairs/caps) specified by the block:
        # entries are (prev_global_center, [curr_local_nodes..]).
        for center_g, locals_list in delta.parity_x_prev_global_curr_local:
            group = {center_g, *[node_map2[l] for l in locals_list if l in node_map2]}
            if len(group) >= 2:
                self.parity_accum.x_groups.append(group)

        for center_g, locals_list in delta.parity_z_prev_global_curr_local:
            group = {center_g, *[node_map2[l] for l in locals_list if l in node_map2]}
            if len(group) >= 2:
                self.parity_accum.z_groups.append(group)

        # Update last parity layers (for subsequent blocks to reference).
        def _seam_global(seam_local_xy2nid: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
            return {xy: node_map2[nid] for xy, nid in seam_local_xy2nid.items() if nid in node_map2}

        seam_last_x_g = _seam_global(delta.seam_last_x)
        seam_last_z_g = _seam_global(delta.seam_last_z)

        # Determine the logical index to update (prefer outputs; otherwise inputs).
        lidx: int | None = None
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

        # Update logical boundary if out_ports are provided.
        for lidx2, out_nodes_local in delta.out_ports.items():
            out_nodes_global = _remap_set(out_nodes_local, node_map2)
            qmap_local = delta.out_qmap.get(lidx2)
            qmap_global = {node_map2[n]: q for n, q in qmap_local.items()} if qmap_local else None
            self.logical_registry.set_boundary(lidx2, out_nodes_global, qidx_map=qmap_global)

        # Remove logicals that were consumed (present in inputs but absent in outputs).
        consumed = set(delta.in_ports.keys()) - set(delta.out_ports.keys())
        for l in consumed:
            self.logical_registry.boundary_qidx.pop(l, None)

        # Update z_top heuristic.
        if delta.node_coords:
            z_max_delta = max(z for (_, _, z) in delta.node_coords.values())
            self.z_top = max(self.z_top, z_max_delta)

        # Switch to the composed graph.
        self.graph = composed
