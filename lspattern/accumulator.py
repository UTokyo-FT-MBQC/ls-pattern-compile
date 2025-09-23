"""Lightweight accumulators for schedules, parities, and flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

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


# Parity groups: remap dict[int, set[int]] via node maps.
def _remap_groups(
    groups: Mapping[int, AbstractSet[NodeIdLocal]],
    node_map: Mapping[NodeIdLocal, NodeIdLocal],
) -> dict[int, set[NodeIdLocal]]:
    return {
        z: {node_map.get(n, n) for n in grp}
        for z, grp in groups.items()
        if grp  # skip empty
    }


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

    checks: dict[PhysCoordLocal2D, dict[int, set[NodeIdLocal]]] = field(default_factory=dict)
    dangling_parity: dict[PhysCoordLocal2D, set[NodeIdLocal]] = field(default_factory=dict)
    ignore_dangling: dict[PhysCoordLocal2D, bool] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> ParityAccumulator:
        """Return a new parity accumulator with nodes remapped via `node_map`."""
        # Fast remap via dict comprehensions
        new_checks = {k: _remap_groups(v, node_map) for k, v in self.checks.items()}

        # Remap dangling_parity as well
        new_dangling = {coord: {node_map.get(n, n) for n in nodes} for coord, nodes in self.dangling_parity.items()}

        # Preserve ignore_dangling information
        new_ignore_dangling = self.ignore_dangling.copy()

        return ParityAccumulator(
            checks=new_checks,
            dangling_parity=new_dangling,
            ignore_dangling=new_ignore_dangling,
        )

    def _handle_dangling_connection(
        self,
        coord: PhysCoordLocal2D,
        checks_dict: dict[int, set[NodeIdLocal]],
        other: ParityAccumulator,
    ) -> None:
        """Handle dangling connection logic for a specific coordinate."""
        # Check if this coordinate should ignore dangling connection
        if other.ignore_dangling.get(coord, False):
            # Don't connect
            for z, check_group in other.checks[coord].items():
                checks_dict[z] = check_group.copy()
        else:
            # Original behavior: connect dangling with first check
            other_checks = other.checks[coord]
            if other_checks:
                # Find the smallest z in other's checks
                min_z = min(other_checks.keys())
                merged = self.dangling_parity[coord].union(other_checks[min_z])
                checks_dict[min_z] = merged
                # Add remaining checks from other
                for z, check_group in other_checks.items():
                    if z > min_z:
                        checks_dict[z] = check_group.copy()

    def merge_with(self, other: ParityAccumulator) -> ParityAccumulator:
        """Merge two parity accumulators with dangling parity handling for sequential composition."""
        new_checks: dict[PhysCoordLocal2D, dict[int, set[NodeIdLocal]]] = {}
        new_dangling: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        new_ignore_dangling: dict[PhysCoordLocal2D, bool] = {}

        # Get all coordinates from both accumulators
        all_coords = (
            set(self.checks.keys())
            | set(self.dangling_parity.keys())
            | set(other.checks.keys())
            | set(other.dangling_parity.keys())
            | set(self.ignore_dangling.keys())
            | set(other.ignore_dangling.keys())
        )

        for coord in all_coords:
            # Start with self's completed checks (copy the z-dict)
            checks_dict = self.checks.get(coord, {}).copy()

            # Handle connection between self.dangling and other.checks
            if coord in self.dangling_parity and coord in other.checks:
                self._handle_dangling_connection(coord, checks_dict, other)
            elif coord in self.dangling_parity and coord not in other.checks:
                # self has dangling but other doesn't have this coord
                # => Keep dangling for potential future connection
                new_dangling[coord] = self.dangling_parity[coord].copy()
            elif coord not in self.dangling_parity and coord in other.checks:
                # self has no dangling, other has checks
                for z, check_group in other.checks[coord].items():
                    checks_dict[z] = check_group.copy()

            # Set new dangling from other (overwrites any existing)
            if coord in other.dangling_parity:
                new_dangling[coord] = other.dangling_parity[coord].copy()

            # Inherit ignore_dangling information from other (prioritize other's settings)
            if coord in self.ignore_dangling:
                new_ignore_dangling[coord] = other.ignore_dangling[coord]
            elif coord in other.ignore_dangling:
                new_ignore_dangling[coord] = self.ignore_dangling[coord]

            # Save checks dict if it has content
            if checks_dict:
                new_checks[coord] = checks_dict

        return ParityAccumulator(
            checks=new_checks,
            dangling_parity=new_dangling,
            ignore_dangling=new_ignore_dangling,
        )

    def merge_parallel(self, other: ParityAccumulator) -> ParityAccumulator:  # noqa: C901
        """Merge two parity accumulators for parallel composition with XOR merging at same z coordinates."""
        new_checks: dict[PhysCoordLocal2D, dict[int, set[NodeIdLocal]]] = {}
        new_dangling: dict[PhysCoordLocal2D, set[NodeIdLocal]] = {}
        new_ignore_dangling: dict[PhysCoordLocal2D, bool] = {}

        # Get all coordinates from both accumulators
        all_coords = (
            set(self.checks.keys())
            | set(other.checks.keys())
            | set(self.dangling_parity.keys())
            | set(other.dangling_parity.keys())
            | set(self.ignore_dangling.keys())
            | set(other.ignore_dangling.keys())
        )

        for coord in all_coords:
            checks_dict: dict[int, set[NodeIdLocal]] = {}

            # Process checks from self
            if coord in self.checks:
                for z, check_group in self.checks[coord].items():
                    checks_dict[z] = check_group.copy()

            # Process checks from other
            if coord in other.checks:
                for z, check_group in other.checks[coord].items():
                    if z in checks_dict:
                        # XOR merge for same z coordinate
                        checks_dict[z] ^= check_group
                    else:
                        checks_dict[z] = check_group.copy()

            # Remove empty checks
            checks_dict = {z: group for z, group in checks_dict.items() if group}

            # Save checks dict if it has content
            if checks_dict:
                new_checks[coord] = checks_dict

            # Process dangling parity
            if coord in other.dangling_parity:
                new_dangling[coord] = other.dangling_parity[coord].copy()
                if coord in self.dangling_parity:
                    new_dangling[coord] ^= self.dangling_parity[coord]
            elif coord in self.dangling_parity:
                new_dangling[coord] = self.dangling_parity[coord].copy()

            # Process ignore_dangling: if either accumulator ignores dangling at this coord, keep ignoring
            if self.ignore_dangling.get(coord, False) or other.ignore_dangling.get(coord, False):
                new_ignore_dangling[coord] = True

        return ParityAccumulator(
            checks=new_checks,
            dangling_parity=new_dangling,
            ignore_dangling=new_ignore_dangling,
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
