
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Mapping, Protocol, Optional, Any, Tuple, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from canvas import RHGCanvas

# NOTE:
# We avoid importing heavy graphix_zx types at import time to keep this module lightweight.
# Canvas code will import concrete classes. Here we define minimal contracts.

class GraphStateLike(Protocol):
    """Minimal interface of graphix_zx.graphstate.BaseGraphState used by blocks/canvas."""
    # --- node/edge book-keeping ---
    @property
    def physical_nodes(self) -> set[int]: ...
    @property
    def physical_edges(self) -> set[tuple[int, int]]: ...
    def neighbors(self, node: int) -> set[int]: ...
    def add_physical_node(self) -> int: ...
    def add_physical_edge(self, u: int, v: int) -> None: ...
    # --- I/O port labeling for composition ---
    @property
    def input_node_indices(self) -> dict[int, int]: ...
    @property
    def output_node_indices(self) -> dict[int, int]: ...
    def register_input(self, node: int) -> int: ...
    def register_output(self, node: int, q_index: int) -> None: ...
    # --- measurement basis assignment (plane & angle live elsewhere) ---
    def assign_meas_basis(self, node: int, meas_basis: Any) -> None: ...
    # --- validation ---
    def is_canonical_form(self) -> bool: ...


@dataclass
class BlockDelta:
    """Delta object returned by a block.

    All node indices in this object are LOCAL to `local_graph`.
    The canvas will remap them to GLOBAL node indices when merging.
    """
    # The piece of graph to be appended
    local_graph: GraphStateLike

    # Interface (LOCAL ids)
    in_ports: Dict[int, Set[int]] = field(default_factory=dict)   # logical index -> set of input-side boundary nodes
    out_ports: Dict[int, Set[int]] = field(default_factory=dict)  # logical index -> set of output-side boundary nodes
    out_qmap: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # Geometry annotations (LOCAL -> coord)
    node_coords: Dict[int, tuple[int, int, int]] = field(default_factory=dict)

    # Parity checks (LOCAL ids)
    x_checks: list[Set[int]] = field(default_factory=list)
    z_checks: list[Set[int]] = field(default_factory=list)

    # Optional local measurement grouping (LOCAL ids)
    schedule_tuples: List[Tuple[int, Set[int]]] = field(default_factory=list)

    # Flow (LOCAL ids): minimal x-flow mapping (node -> correction target nodes)
    # If you maintain separate X/Z flows, put X here and let the compiler derive Z (or extend later).
    flow_local: Dict[int, Set[int]] = field(default_factory=dict)
    
    parity_x_prev_global_curr_local: List[Tuple[int, int]] = field(default_factory=list)
    parity_z_prev_global_curr_local: List[Tuple[int, int]] = field(default_factory=list)
    
    seam_last_x:  Dict[Tuple[int,int], int] = field(default_factory=dict)
    seam_last_z:  Dict[Tuple[int,int], int] = field(default_factory=dict)
    
    


class RHGBlock(Protocol):
    """Protocol for an RHG block (structural typing)."""
    logical: int
    def emit(self, canvas: RHGCanvas) -> BlockDelta: ...


def choose_port_node(nodes: Set[int]) -> int:
    """Pick a representative port node from a set (deterministic)."""
    if not nodes:
        raise ValueError("Port node set is empty.")
    return min(nodes)
