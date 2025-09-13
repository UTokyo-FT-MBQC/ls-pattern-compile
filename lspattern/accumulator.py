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

from lspattern.mytype import FlowLocal, NodeIdGlobal, NodeIdLocal, PhysCoordGlobal3D

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

    # Parity check groups (local ids)
    x_checks: list[set[NodeIdLocal]] = field(default_factory=list)
    z_checks: list[set[NodeIdLocal]] = field(default_factory=list)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> ParityAccumulator:
        """Return a new parity accumulator with nodes remapped via `node_map`."""
        # Fast remap via set/list comprehensions
        return ParityAccumulator(
            x_checks=_remap_groups(self.x_checks, node_map),
            z_checks=_remap_groups(self.z_checks, node_map),
        )

    def update_at(
        self,
        anchor: int,
        graph: BaseGraphState,
        node2coord: Mapping[int, PhysCoordGlobal3D],
        coord2node: Mapping[PhysCoordGlobal3D, int],
        node2role: Mapping[int, str],
    ) -> None:
        """Update parity groups by sweeping neighbors around an ancilla node.

        - For an X-ancilla, add the set of adjacent data nodes to ``x_checks``.
        - For a Z-ancilla, add the set of adjacent data nodes to ``z_checks``.
        - Skip classical outputs.
        - Non-decreasing is enforced by assertion.
        """
        if anchor in graph.output_node_indices:
            return

        role = (self._role_of(anchor, node2role) or "").lower()

        if not role.startswith("ancilla"):
            return  # only ancillas define parity groups

        coord = node2coord[int(anchor)]
        target_coord = (*coord[:2], coord[2] - 2)
        target_ancilla = coord2node.get(PhysCoordGlobal3D(target_coord))
        group = {NodeIdLocal(anchor)} if target_ancilla is None else {NodeIdLocal(anchor), NodeIdLocal(target_ancilla)}

        if "ancilla_x" in role:
            self.x_checks.append(group)
        elif "ancilla_z" in role:
            self.z_checks.append(group)


@dataclass
class FlowAccumulator(BaseAccumulator):
    """Directed flow relations between nodes for X/Z types."""

    xflow: dict[NodeIdLocal, set[NodeIdLocal]] = field(default_factory=dict)

    def remap_nodes(self, node_map: dict[NodeIdLocal, NodeIdLocal]) -> FlowAccumulator:
        """Return a new flow accumulator with ids remapped via `node_map`."""
        # Remap both x/z flows using helper for speed
        return FlowAccumulator(
            xflow=_remap_flow(self.xflow, node_map),
        )

    def merge_with(self, other: FlowAccumulator) -> FlowAccumulator:
        """Union-merge two flow accumulators (local/global-agnostic)."""
        return FlowAccumulator(
            xflow=_merge_flow(self.xflow, other.xflow),
        )

    def update_at(
        self,
        anchor: int,
        graph: BaseGraphState,
        node2coord: Mapping[int, PhysCoordGlobal3D],
        coord2node: Mapping[PhysCoordGlobal3D, int],
        node2role: Mapping[int, str],
    ) -> None:
        """Update X/Z flow from an ancilla to its data neighbors.

        A minimal, monotone definition suitable for T23:
        - For X-ancilla: add directed edges anchor -> data_nbr into ``xflow``.
        - For Z-ancilla: add directed edges anchor -> data_nbr into ``zflow``.
        - Skip classical outputs.
        """
        if anchor in graph.output_node_indices:
            return

        role = (self._role_of(anchor, node2role) or "").lower()
        if role.startswith("ancilla"):
            return

        coord = node2coord.get(int(anchor))
        if not coord:
            msg = f"FlowAccumulator.update_at: missing coord for node {anchor}"
            raise ValueError(msg)
        parent_coord = (*coord[:2], coord[2] - 1)
        parent = coord2node.get(PhysCoordGlobal3D(parent_coord))
        if parent is None:
            return

        self.xflow.setdefault(NodeIdLocal(parent), set()).add(NodeIdLocal(anchor))
