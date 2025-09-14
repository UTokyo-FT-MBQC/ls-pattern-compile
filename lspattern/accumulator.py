"""Lightweight accumulators for schedules, parities, and flows.

These helpers store simple collections during compilation and offer
remapping/combination utilities. In T23 we also add a common
``update_at(anchor, graph_local, *, allowed_pairs=None)`` API used by
TemporalLayer sweeps.

Design notes
------------
- ``graph_local`` may be either a BaseGraphState-like object (duck-typed by
  ``neighbors(node)``) or an object carrying ``local_graph``,
  ``node2coord`` and ``node2role`` (e.g., TemporalLayer). The helpers below
  attempt to extract the richest available context.
- ``allowed_pairs`` is an optional filter. In this milestone it is treated as a
  set of node-id pairs ``{(u,v), ...}`` (order-agnostic). If not provided, all
  neighbor relations are accepted.
- All ``update_at`` implementations are monotone (non-decreasing). Each method
  asserts that the total cardinality of stored relations does not shrink.
"""

# import grouping intentionally simple
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lspattern.mytype import FlowLocal, NodeIdGlobal, NodeIdLocal, PhysCoordGlobal3D, PhysCoordLocal2D

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from graphix_zx.graphstate import BaseGraphState

# -----------------------------------------------------------------------------
# Shared helpers and base class
# -----------------------------------------------------------------------------


class BaseAccumulator:
    """Common utilities for accumulator ``update_at`` implementations.

    Subclasses should implement ``update_at`` and use helper methods here to
    access graph context and enforce non-decreasing updates.
    """

    @staticmethod
    def _role_of(node: int, node2role: Mapping[int, str] | None) -> str | None:
        if node2role is None:
            return None
        try:
            return str(node2role.get(int(node)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _allows(u: int, v: int, allowed_pairs: Iterable[tuple[int, int]] | None) -> bool:
        """Return True if the pair is allowed (or no filter provided).

        This milestone treats allowed_pairs as node-id pairs. Order-agnostic.
        Future work may lift this to tiling-id pairs.
        """
        if not allowed_pairs:
            return True
        uv = (int(u), int(v))
        vu = (int(v), int(u))
        try:
            ap = {(int(a), int(b)) for (a, b) in allowed_pairs}
        except (TypeError, ValueError):
            return True
        return uv in ap or vu in ap

    # ---- monotonicity helpers ---------------------------------------------
    @staticmethod
    def _size_of_flow(flow: FlowLocal) -> int:
        return sum(len(vs) for vs in flow.values())

    @staticmethod
    def _size_of_groups(groups: list[set[int]] | list[set[NodeIdLocal]]) -> int:
        return sum(len(g) for g in groups)

    @staticmethod
    def _size_of_schedule(schedule: dict[int, set[int]] | dict[int, set[NodeIdGlobal]]) -> int:
        return sum(len(v) for v in schedule.values())


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
class ScheduleAccumulator(BaseAccumulator):
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
        if not self.schedule:
            return ScheduleAccumulator()
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

    def shift_z(self, z_by: int) -> None:
        """Shift all time slots by `z_by` in-place."""
        new_schedule = {}
        for t, nodes in self.schedule.items():
            new_schedule[t + z_by] = nodes
        self.schedule = new_schedule

    def compose_sequential(self, late_schedule: ScheduleAccumulator) -> ScheduleAccumulator:
        """Concatenate schedules by placing `late_schedule` after this one."""
        new_schedule = self.schedule.copy()
        late_schedule.shift_z(max(self.schedule.keys()) + 1)
        for t, nodes in late_schedule.schedule.items():
            new_schedule[t] = new_schedule.get(t, set()).union(nodes)
        return ScheduleAccumulator(new_schedule)

    def update_at(
        self,
        anchor: int,
        graph: BaseGraphState,
        node2coord: Mapping[int, PhysCoordGlobal3D],
    ) -> None:
        """Record the measurement of ``anchor`` at its time slice.

        Uses node2coord if available to place the node into the correct t-slot.
        Ignores classical outputs. Monotonic (non-decreasing) by construction.
        """

        if anchor in graph.output_node_indices:
            return

        before = self._size_of_schedule(self.schedule)

        t = node2coord.get(int(anchor), (0, 0, 0))[2]

        self.schedule.setdefault(t, set()).add(NodeIdGlobal(anchor))

        after = self._size_of_schedule(self.schedule)
        if after < before:
            msg = "ScheduleAccumulator must be non-decreasing"
            raise AssertionError(msg)


@dataclass
class ParityAccumulator(BaseAccumulator):
    """Parity check groups for X/Z stabilizers in local id space."""

    checks: dict[PhysCoordLocal2D, list[set[NodeIdLocal]]] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> ParityAccumulator:
        """Return a new parity accumulator with nodes remapped via `node_map`."""
        # Fast remap via set/list comprehensions
        new_checks = {k: _remap_groups(v, node_map) for k, v in self.checks.items()}

        return ParityAccumulator(
            checks=new_checks,
        )

    def merge_with(self, other: ParityAccumulator) -> ParityAccumulator:
        new_checks: dict[PhysCoordLocal2D, list[set[NodeIdLocal]]] = {}
        for coord, groups in self.checks.items():
            new_checks[coord] = groups[:-1]
            # NOTE: assumes no empty groups in self.checks
            dangling_check1 = self.checks[coord][-1]
            dangling_check2 = other.checks[coord][0]
            new_checks[coord].append(dangling_check1.union(dangling_check2))
            new_checks[coord].extend(other.checks[coord][1:])

        return ParityAccumulator(
            checks=new_checks,
        )


@dataclass
class FlowAccumulator(BaseAccumulator):
    """Directed flow relations between nodes for X/Z types."""

    flow: dict[PhysCoordLocal2D, FlowLocal] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> FlowAccumulator:
        """Return a new flow accumulator with ids remapped via `node_map`."""
        # Remap both x/z flows using helper for speed
        new_flow = {k: _remap_flow(v, node_map) for k, v in self.flow.items()}
        return FlowAccumulator(
            flow=new_flow,
        )

    def merge_with(self, other: FlowAccumulator) -> FlowAccumulator:
        """Union-merge two flow accumulators (local/global-agnostic)."""
        new_flow = {}
        for coord in set(self.flow.keys()).union(other.flow.keys()):
            f1 = self.flow.get(coord, {})
            f2 = other.flow.get(coord, {})
            new_flow[coord] = _merge_flow(f1, f2)
        return FlowAccumulator(
            flow=new_flow,
        )
