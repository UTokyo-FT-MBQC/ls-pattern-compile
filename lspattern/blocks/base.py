from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Import only for type checking to avoid circular imports at runtime.
    from ..canvas import RHGCanvas


# ---------------------------------------------------------------------
# Minimal GraphState protocol (to keep blocks/canvas decoupled from graphix_zx)
# ---------------------------------------------------------------------
class GraphStateLike(Protocol):
    """Subset of `graphix_zx.graphstate.BaseGraphState` used by blocks/canvas."""

    # --- node/edge book-keeping ---
    @property
    def physical_nodes(self) -> Set[int]: ...
    @property
    def physical_edges(self) -> Set[Tuple[int, int]]: ...
    def neighbors(self, node: int) -> Set[int]: ...
    def add_physical_node(self) -> int: ...
    def add_physical_edge(self, u: int, v: int) -> None: ...

    # --- I/O port labeling for composition ---
    @property
    def input_node_indices(self) -> Dict[int, int]: ...
    @property
    def output_node_indices(self) -> Dict[int, int]: ...
    def register_input(self, node: int) -> int: ...
    def register_output(self, node: int, q_index: int) -> None: ...

    # --- measurement basis assignment (plane & angle live elsewhere) ---
    def assign_meas_basis(self, node: int, meas_basis: Any) -> None: ...

    # --- validation ---
    def is_canonical_form(self) -> bool: ...


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
    local_graph: GraphStateLike

    # MBQC interface (LOCAL ids)
    in_ports: Dict[int, Set[int]] = field(default_factory=dict)   # logical -> set of input-side boundary nodes
    out_ports: Dict[int, Set[int]] = field(default_factory=dict)  # logical -> set of output-side boundary nodes
    out_qmap: Dict[int, Dict[int, int]] = field(default_factory=dict)  # logical -> {LOCAL node -> q_index}

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
    parity_x_prev_global_curr_local: List[Tuple[int, List[int]]] = field(default_factory=list)
    parity_z_prev_global_curr_local: List[Tuple[int, List[int]]] = field(default_factory=list)

    # Last ancilla layers (LOCAL) keyed by (x, y) -> LOCAL node id, for seam stitching
    seam_last_x: Dict[Tuple[int, int], int] = field(default_factory=dict)
    seam_last_z: Dict[Tuple[int, int], int] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Block protocol
# ---------------------------------------------------------------------
class RHGBlock(Protocol):
    """Protocol for an RHG block (structural typing)."""

    logical: int

    def emit(self, canvas: "RHGCanvas") -> BlockDelta: ...


# ---------------------------------------------------------------------
# Small utility
# ---------------------------------------------------------------------
def choose_port_node(nodes: Set[int]) -> int:
    """Pick a representative port node from a non-empty set (deterministic)."""
    if not nodes:
        raise ValueError("Port node set is empty.")
    return min(nodes)
