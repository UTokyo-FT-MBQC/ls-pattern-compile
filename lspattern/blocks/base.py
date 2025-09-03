from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from graphix_zx.graphstate import BaseGraphState

if TYPE_CHECKING:
    # Import only for type checking to avoid circular imports at runtime.
    from ..canvas import RHGCanvas


@dataclass
class RHGBlockSkeleton:
    logical: int
    d: int
    origin: Optional[Tuple[int, int]] = None


@dataclass
class RHGBlock:
    logical: int
    d: int
    origin: Optional[Tuple[int, int]] = None
    kind: tuple[str, str, str]
    template: Any

    def materialize(self, skeleton: RHGBlockSkeleton) -> None:
        pass


class Memory(RHGBlock):
    pass


class InitPlus(RHGBlock):
    pass


class MeasureX(RHGBlock):
    pass


# ---------------------------------------------------------------------
# Block delta (the unit of mutation produced by each block)
# ---------------------------------------------------------------------
@dataclass
class BlockDelta:
    """Delta produced by a block.

    Notes
    -----
    * All node ids in this object are LOCAL to `local_graph`.
      The canvas remaps them to GLOBAL ids when merging.
    * `in_ports`/`out_ports` use LOCAL ids. `out_qmap` provides LOCAL node -> q_index.
    * `schedule_tuples` is a list of (t_local, LOCAL-node-set). Each block starts at t_local=0.
    * `parity_*_prev_global_curr_local` are unified parity directives:
         (prev_global_center, [curr_local_nodes...]).
    """

    # The graph fragment contributed by the block (LOCAL ids).
    local_graph: BaseGraphState

    # MBQC interface (LOCAL ids)
    in_ports: Dict[int, Set[int]] = field(
        default_factory=dict
    )  # logical -> set of input-side boundary nodes
    out_ports: Dict[int, Set[int]] = field(
        default_factory=dict
    )  # logical -> set of output-side boundary nodes
    out_qmap: Dict[int, Dict[int, int]] = field(
        default_factory=dict
    )  # logical -> {LOCAL node -> q_index}

    # Geometry annotations (LOCAL node -> (x, y, z))
    node_coords: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    # Parity checks contributed entirely within the block (LOCAL ids)
    x_checks: List[Set[int]] = field(default_factory=list)
    z_checks: List[Set[int]] = field(default_factory=list)

    # Local measurement schedule: list of (t_local, LOCAL node set)
    schedule_tuples: List[Tuple[int, Set[int]]] = field(default_factory=list)

    # Flow (LOCAL ids): minimal X-flow mapping (node -> correction target nodes)
    flow_local: Dict[int, Set[int]] = field(default_factory=dict)

    # Unified parity directives that pair previous GLOBAL centers with current LOCAL nodes
    parity_x_prev_global_curr_local: List[Tuple[int, List[int]]] = field(
        default_factory=list
    )
    parity_z_prev_global_curr_local: List[Tuple[int, List[int]]] = field(
        default_factory=list
    )

    # Last ancilla layers (LOCAL) keyed by (x, y) -> LOCAL node id, for seam stitching
    seam_last_x: Dict[Tuple[int, int], int] = field(default_factory=dict)
    seam_last_z: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def shift_ids(self, by: int) -> None:
        # change index of every elements
        raise NotImplementedError()

    def shift_coords(self, patch_coord: Tuple[int, int]) -> None:
        # change the coordinates of every element
        raise NotImplementedError()


# ---------------------------------------------------------------------
# Block protocol
# ---------------------------------------------------------------------
class RHGBlock(Protocol):
    """Protocol for an RHG block (structural typing)."""

    logical: int
    d: int

    def emit(self, canvas: "RHGCanvas") -> BlockDelta: ...


# ---------------------------------------------------------------------
# Small utility
# ---------------------------------------------------------------------
def choose_port_node(nodes: Set[int]) -> int:
    """Pick a representative port node from a non-empty set (deterministic)."""
    if not nodes:
        raise ValueError("Port node set is empty.")
    return min(nodes)
