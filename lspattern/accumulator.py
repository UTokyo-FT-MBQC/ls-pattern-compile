"""Lightweight accumulators for schedules, parities, and flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lspattern.mytype import FlowLocal, NodeIdGlobal, NodeIdLocal, PhysCoordLocal2D


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
        if src in out:
            msg = f"Flow merge conflict at src {src}"
            raise ValueError(msg)
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
    """Collect time-indexed node sets for measurement schedule."""

    schedule: dict[int, set[NodeIdGlobal]] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdGlobal, NodeIdGlobal]) -> ScheduleAccumulator:
        """Return a new accumulator with node ids remapped by `node_map`.

        Times are preserved; nodes in each time slot are mapped via `node_map`.
        Unknown nodes are kept as-is for robustness.

        Returns
        -------
        ScheduleAccumulator
            A new instance with remapped node ids.
        """
        remapped: dict[int, set[NodeIdGlobal]] = {}
        for t, nodes in self.schedule.items():
            remapped[t] = {node_map.get(n, n) for n in nodes}
        return ScheduleAccumulator(remapped)

    def compose_parallel(self, other: ScheduleAccumulator) -> ScheduleAccumulator:
        """Merge two schedules slot-wise without shifting times."""
        new_schedule = self.schedule.copy()
        for t, nodes in other.schedule.items():
            if t in new_schedule:
                new_schedule[t].update(nodes)
            else:
                new_schedule[t] = nodes
        return ScheduleAccumulator(new_schedule)

    def shift_z(self, z_by: int) -> ScheduleAccumulator:
        """Shift all time slots by `z_by` in-place."""
        new_schedule = {}
        for t, nodes in self.schedule.items():
            new_schedule[t + z_by] = nodes
        return ScheduleAccumulator(new_schedule)

    def compose_sequential(
        self, late_schedule: ScheduleAccumulator, exclude_nodes: set[NodeIdGlobal] | None = None
    ) -> ScheduleAccumulator:
        """Concatenate schedules by placing `late_schedule` after this one.

        Parameters
        ----------
        late_schedule : ScheduleAccumulator
            The schedule to append after this one.
        exclude_nodes : set[NodeIdGlobal] | None, optional
            Set of nodes to exclude from the late_schedule when merging.

        Returns
        -------
        ScheduleAccumulator
            New accumulator with schedules concatenated.
        """
        new_schedule = self.schedule.copy()

        # Calculate the shift amount to ensure continuity
        if not self.schedule:
            # If this schedule is empty, no shift needed
            shift_amount = 0
        elif not late_schedule.schedule:
            # If late_schedule is empty, just return copy of this schedule
            return ScheduleAccumulator(new_schedule)
        else:
            # Find the next available time slot after the last occupied slot
            max_time = max(self.schedule.keys())
            min_late_time = min(late_schedule.schedule.keys())
            shift_amount = max_time + 1 - min_late_time

        shifted_late_schedule = late_schedule.shift_z(shift_amount)
        exclude_set = exclude_nodes or set()

        for t, nodes in shifted_late_schedule.schedule.items():
            # Filter out excluded nodes from the late schedule
            filtered_nodes = nodes - exclude_set
            if filtered_nodes:  # Only add if there are remaining nodes
                new_schedule[t] = new_schedule.get(t, set()).union(filtered_nodes)
        return ScheduleAccumulator(new_schedule)

    def exclude_nodes(self, nodes_to_exclude: set[NodeIdGlobal]) -> ScheduleAccumulator:
        """Remove specified nodes from the schedule.

        Parameters
        ----------
        nodes_to_exclude : set[NodeIdGlobal]
            Set of nodes to remove from the schedule.

        Returns
        -------
        ScheduleAccumulator
            New accumulator with specified nodes removed.
        """
        new_schedule = {}
        for t, nodes in self.schedule.items():
            filtered_nodes = nodes - nodes_to_exclude
            if filtered_nodes:  # Only add if there are remaining nodes
                new_schedule[t] = filtered_nodes
        return ScheduleAccumulator(new_schedule)

    def compact(self) -> ScheduleAccumulator:
        """Remove empty time slots and reindex times to be consecutive starting from 0.

        Example
        -------
        If the schedule has nodes at times [0, 2, 5, 7], this method will
        remap them to times [0, 1, 2, 3] while preserving the order.

        Returns
        -------
        ScheduleAccumulator
            New accumulator with compacted time slots.
        """
        if not self.schedule:
            return ScheduleAccumulator()

        # Get sorted list of times that actually have nodes
        occupied_times = sorted(t for t, nodes in self.schedule.items() if nodes)

        # Create mapping from old time to new consecutive time
        time_mapping = {old_time: new_time for new_time, old_time in enumerate(occupied_times)}

        # Build new schedule with compacted times
        compacted_schedule: dict[int, set[NodeIdGlobal]] = {}
        for old_time, nodes in self.schedule.items():
            if nodes and old_time in time_mapping:  # Only include non-empty slots
                new_time = time_mapping[old_time]
                compacted_schedule[new_time] = nodes.copy()

        return ScheduleAccumulator(compacted_schedule)


@dataclass
class ParityAccumulator:
    """Parity check groups for X/Z stabilizers in local id space."""

    checks: dict[PhysCoordLocal2D, list[set[NodeIdLocal]]] = field(default_factory=dict)
    dangling_parity: dict[PhysCoordLocal2D, set[NodeIdLocal]] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> ParityAccumulator:
        """Return a new parity accumulator with nodes remapped via `node_map`."""
        # Fast remap via set/list comprehensions
        new_checks = {k: _remap_groups(v, node_map) for k, v in self.checks.items()}

        # Remap dangling_parity as well
        new_dangling = {coord: {node_map.get(n, n) for n in nodes} for coord, nodes in self.dangling_parity.items()}

        return ParityAccumulator(
            checks=new_checks,
            dangling_parity=new_dangling,
        )

    def merge_with(self, other: ParityAccumulator) -> ParityAccumulator:
        """Merge two parity accumulators with dangling parity handling."""
        new_checks: dict[PhysCoordLocal2D, list[set[NodeIdLocal]]] = {}
        new_dangling: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}

        # Get all coordinates from both accumulators
        all_coords = (
            set(self.checks.keys())
            | set(self.dangling_parity.keys())
            | set(other.checks.keys())
            | set(other.dangling_parity.keys())
        )

        for coord in all_coords:
            # Start with self's completed checks
            sequence = self.checks.get(coord, [])[:]

            # Handle connection between self.dangling and other.checks
            if coord in self.dangling_parity and coord in other.checks:
                # Connect: merge self's dangling with other's first parity
                if other.checks[coord]:
                    merged = self.dangling_parity[coord].union(other.checks[coord][0])
                    sequence.append(merged)
                    # Add remaining checks from other
                    sequence.extend(other.checks[coord][1:])
            elif coord in self.dangling_parity and coord not in other.checks:
                # self has dangling but other doesn't have this coord
                # => Keep dangling for potential future connection
                new_dangling[coord] = self.dangling_parity[coord].copy()
            elif coord not in self.dangling_parity and coord in other.checks:
                # self has no dangling, other has checks
                sequence.extend(other.checks[coord])

            # Set new dangling from other (overwrites any existing)
            if coord in other.dangling_parity:
                new_dangling[coord] = other.dangling_parity[coord].copy()

            # Save sequence if it has content
            if sequence:
                new_checks[coord] = sequence

        return ParityAccumulator(
            checks=new_checks,
            dangling_parity=new_dangling,
        )


@dataclass
class FlowAccumulator:
    """Directed flow relations between nodes for X/Z types."""

    flow: FlowLocal = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> FlowAccumulator:
        """Return a new flow accumulator with ids remapped via `node_map`."""
        new_flow = _remap_flow(self.flow, node_map)
        return FlowAccumulator(
            flow=new_flow,
        )

    def merge_with(self, other: FlowAccumulator) -> FlowAccumulator:
        """Union-merge two flow accumulators (local/global-agnostic)."""
        new_flow = _merge_flow(self.flow, other.flow)
        return FlowAccumulator(
            flow=new_flow,
        )
