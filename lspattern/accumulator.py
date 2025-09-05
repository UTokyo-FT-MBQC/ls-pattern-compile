from dataclasses import dataclass, field

from lspattern.mytype import FlowLocal, NodeIdGlobal, NodeIdLocal


# Flow helpers (node maps guaranteed to contain all keys)
def _remap_flow(flow: FlowLocal, node_map: dict[NodeIdLocal, NodeIdLocal]) -> FlowLocal:
    return {node_map[src]: {node_map[dst] for dst in dsts} for src, dsts in flow.items()}


def _merge_flow(a: FlowLocal, b: FlowLocal) -> FlowLocal:
    out: FlowLocal = {}
    for src, dsts in a.items():
        if not dsts:
            continue
        out.setdefault(src, set()).update(dsts)
    for src, dsts in b.items():
        if not dsts:
            continue
        out.setdefault(src, set()).update(dsts)
    return out


# Parity groups: remap list[set[int]] via node maps and concatenate.
def _remap_groups(
    groups: list[set[NodeIdLocal]],
    node_map: dict[NodeIdLocal, NodeIdLocal],
) -> list[set[NodeIdLocal]]:
    return [
        {node_map.get(n, n) for n in grp}
        for grp in groups
        if grp  # skip empty
    ]


@dataclass
class ScheduleAccumulator:
    """Accumulator for measurement schedule data."""

    schedule: dict[int, set[NodeIdGlobal]] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> "ScheduleAccumulator":
        """Return a new accumulator with node ids remapped by `node_map`.

        Times are preserved; nodes in each time slot are mapped via `node_map`.
        Unknown nodes are kept as-is for robustness.
        """
        if not self.schedule:
            return ScheduleAccumulator()
        remapped: dict[int, set[NodeIdGlobal]] = {}
        for t, nodes in self.schedule.items():
            remapped[t] = {node_map.get(n, n) for n in nodes}
        return ScheduleAccumulator(remapped)

    def compose_parallel(self, other: "ScheduleAccumulator") -> "ScheduleAccumulator":
        """Combine two schedules in parallel, merging overlapping time slots."""
        new_schedule = self.schedule.copy()
        for t, nodes in other.schedule.items():
            if t in new_schedule:
                new_schedule[t].update(nodes)
            else:
                new_schedule[t] = nodes
        return ScheduleAccumulator(new_schedule)

    def shift_z(self, z_by: int) -> None:
        """Shift all time indices by the given offset."""
        new_schedule = {}
        for t, nodes in self.schedule.items():
            new_schedule[t + z_by] = nodes
        self.schedule = new_schedule

    def compose_sequential(self, late_schedule: "ScheduleAccumulator") -> "ScheduleAccumulator":
        """Combine two schedules sequentially, with late_schedule after self."""
        new_schedule = self.schedule.copy()
        late_schedule.shift_z(max(self.schedule.keys()) + 1)
        for t, nodes in late_schedule.schedule.items():
            new_schedule[t] = new_schedule.get(t, set()).union(nodes)
        return ScheduleAccumulator(new_schedule)


@dataclass
class ParityAccumulator:
    """Accumulator for parity check data."""

    # Parity check groups (local ids)
    x_checks: list[set[NodeIdLocal]] = field(default_factory=list)
    z_checks: list[set[NodeIdLocal]] = field(default_factory=list)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> "ParityAccumulator":
        """Return a new accumulator with node ids remapped by node_map."""
        # Fast remap via set/list comprehensions
        return ParityAccumulator(
            x_checks=_remap_groups(self.x_checks, node_map),
            z_checks=_remap_groups(self.z_checks, node_map),
        )


@dataclass
class FlowAccumulator:
    """Accumulator for flow data."""

    xflow: dict[NodeIdLocal, set[NodeIdLocal]] = field(default_factory=dict)
    zflow: dict[NodeIdLocal, set[NodeIdLocal]] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> "FlowAccumulator":
        """Return a new accumulator with node ids remapped by node_map."""
        # Remap both x/z flows using helper for speed
        return FlowAccumulator(
            xflow=_remap_flow(self.xflow, node_map),
            zflow=_remap_flow(self.zflow, node_map),
        )

    def merge_with(self, other: "FlowAccumulator") -> "FlowAccumulator":
        """Union-merge two flow accumulators (local/global-agnostic)."""
        return FlowAccumulator(
            xflow=_merge_flow(self.xflow, other.xflow),
            zflow=_merge_flow(self.zflow, other.zflow),
        )
