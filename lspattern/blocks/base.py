"""RHG blocks and skeletons for lattice-surgery templates."""

from __future__ import annotations

from contextlib import suppress

# imports layout is special due to optional dependency fallback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from graphix_zx.graphstate import BaseGraphState

from lspattern.accumulator import (
    FlowAccumulator,
    ParityAccumulator,
    ScheduleAccumulator,
)
from lspattern.consts.consts import DIRECTIONS3D
from lspattern.tiling.template import (
    RotatedPlanarCubeTemplate,
    ScalableTemplate,
)

if TYPE_CHECKING:
    from lspattern.mytype import (
        NodeIdLocal,
        PatchCoordGlobal3D,
        PatchCoordLocal2D,
        PhysCoordGlobal3D,
        QubitIndexLocal,
        SpatialEdgeSpec,
    )


@dataclass
class RHGBlock:
    """Self-contained RHG slab used as a lattice-surgery building block.

    This class wraps a scalable tiling template together with block-level
    metadata (code distance, spatial edge spec, anchor coordinates and ports),
    and provides materialization into a local 3D RHG graph with roles tagged
    for data and X/Z ancillas.

    Attributes
    ----------
    d : int
        Code distance (also controls temporal height ``2*d`` when materialized).
    edge_spec : SpatialEdgeSpec | None
        Spatial edge specification; alias available via the ``edgespec`` property.
    source : PatchCoordGlobal3D
        3D anchor of this block; ``z`` controls the slab's temporal offset.
    sink : PatchCoordGlobal3D | None
        Optional secondary anchor (unused in the base block implementation).
    template : ScalableTemplate
        Backing scalable tiling; evaluated during init/materialization.
    in_ports, out_ports : set[QubitIndexLocal]
        Logical boundary port sets for this block.
    cout_ports : list[set[QubitIndexLocal]]
        Grouped classical output ports (one group per logical result).
    local_graph : BaseGraphState
        Local RHG graph constructed by ``materialize()``.
    node2coord, coord2node : dict
        Bidirectional maps between node ids and 3D coordinates.
    node2role : dict[int, str]
        Role of each node: ``'data'``, ``'ancilla_x'`` or ``'ancilla_z'``.
    """

    name: ClassVar[str] = __qualname__
    d: int = 3
    edge_spec: SpatialEdgeSpec = field(default_factory=dict)
    # source
    source: PatchCoordGlobal3D = field(default_factory=lambda: (0, 0, 0))
    sink: PatchCoordGlobal3D | None = None
    # When it is Pipe, we have sink and direction (Not implemented here)
    template: ScalableTemplate = field(default_factory=lambda: ScalableTemplate(d=3, edgespec={}))  # evaluated

    # Ports for this block's current logical patch boundary (qubit index sets)
    # classical output ports. One group represents one logical result (to be XORed)
    in_ports: set[QubitIndexLocal] = field(default_factory=set)
    out_ports: set[QubitIndexLocal] = field(default_factory=set)
    cout_ports: list[set[QubitIndexLocal]] = field(default_factory=list)

    schedule: ScheduleAccumulator = field(init=False, default_factory=ScheduleAccumulator)
    flow: FlowAccumulator = field(init=False, default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(init=False, default_factory=ParityAccumulator)

    local_graph: BaseGraphState = field(init=False, default_factory=BaseGraphState)
    node2coord: dict[NodeIdLocal, PhysCoordGlobal3D] = field(init=False, default_factory=dict)
    coord2node: dict[PhysCoordGlobal3D, NodeIdLocal] = field(init=False, default_factory=dict)
    node2role: dict[NodeIdLocal, str] = field(init=False, default_factory=dict)

    final_layer: str = None  # "M", "MX", "MZ", "MY" or "O" (open, no measurement)

    def __post_init__(self) -> None:
        # Sync template parameters (d, edgespec)
        edgespec = self.edge_spec
        if self.template is None:
            self.template = RotatedPlanarCubeTemplate(d=int(self.d), edgespec=edgespec or {})
        else:
            # Ensure d matches
            self.template.d = int(self.d)
            # Prefer explicit edge_spec if provided; otherwise adopt from template
            if edgespec is None:
                edgespec = self.template.edgespec
                self.edge_spec = edgespec  # keep alias in sync

        # Trim spatial boundaries for explicitly open sides and precompute tiling
        es = edgespec or {}
        for side in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
            if str(es.get(side, "")).upper() == "O":
                self.template.trim_spatial_boundary(side)

        self.template.to_tiling()

    # Child class will handle them without any input arguments
    def set_in_ports(self) -> None:
        """Set the input ports for the block."""

    def set_out_ports(self) -> None:
        """Set the output ports for the block."""

    def set_cout_ports(self) -> None:
        """Set the classical output ports for the block."""

    def shift_ids(self, by: int) -> None:
        """
        Shift the qubit indices in the template by a specified integer offset.

        Parameters
        ----------
        by : int
            The amount by which to shift all qubit indices.
        """
        self.template.shift_qindex(by)

    def shift_coords(self, by: PatchCoordGlobal3D) -> None:
        """Shift the patch anchor and the template by a 2D offset.

        The block's anchor (`source`) is 3D; the underlying scalable template
        only needs the XY offset. This mirrors the behavior used by pipes.
        """
        osx, osy, osz = self.source
        dx, dy, dz = by
        self.source = (osx + dx, osy + dy, osz + dz)

        by_template: PatchCoordLocal2D = (dx, dy)
        self.template.shift_coords(by_template)

    def materialize(self) -> RHGBlock:
        """Finalize this block's template and initialize ports.

        - Sync `template.d` with `self.d`.
        - Ensure template edgespec mirrors `self.edge_spec` if provided.
        - Build tiling via `template.to_tiling()`.
        - Invoke port initialization hooks (`set_in_ports`, `set_out_ports`, `set_cout_ports`).
        """
        # Keep core parameters in sync
        self.template.d = int(self.d)
        if self.edge_spec is not None:
            # Align template's edgespec with the block-side alias
            try:
                self.template.edgespec = dict(self.edge_spec)
            except (TypeError, ValueError):
                self.template.edgespec = self.edge_spec  # type: ignore[assignment]

        # Evaluate tiling coordinates (data/X/Z) only when not yet populated.
        # This preserves any absolute XY shifts applied upstream (e.g.,
        # to_temporal_layer() shifts templates before materialization).
        if not (
            getattr(self.template, "data_coords", None)
            or getattr(self.template, "x_coords", None)
            or getattr(self.template, "z_coords", None)
        ):
            self.template.to_tiling()

        # Initialize logical port sets (child classes may override these hooks)
        self.set_in_ports()
        self.set_out_ports()
        self.set_cout_ports()
        # Build the local RHG graph (nodes/edges and coordinate maps) here.
        # This mirrors the per-block logic used in TemporalLayer.compile,
        # but scoped to this block's tiling only.

        # Collect 2D coordinates from the evaluated template
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Construct a 3D RHG slab of height ~2*d (T12 policy)
        d_val = int(self.d)
        max_t = 2 * d_val
        z0 = int(self.source[2]) * (2 * d_val)  # base z-offset per block

        g = BaseGraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes at each time slice
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}
        for t_local in range(max_t + 1):
            t = z0 + t_local
            cur: dict[tuple[int, int], int] = {}
            if self.final_layer is None:
                msg = "final_layer must be set"
                raise AssertionError(msg)
            if t_local == max_t and self.final_layer == "O":
                # add data node only if it is not measurement node
                for x, y in data2d:
                    n = g.add_physical_node()
                    node2coord[n] = (int(x), int(y), int(t))
                    coord2node[int(x), int(y), int(t)] = n
                    node2role[n] = "data"
                    cur[int(x), int(y)] = n
            else:
                # Data nodes every slice except the final sentinel layer
                for x, y in data2d:
                    n = g.add_physical_node()
                    node2coord[n] = (int(x), int(y), int(t))
                    coord2node[int(x), int(y), int(t)] = n
                    node2role[n] = "data"
                    cur[int(x), int(y)] = n
                # Interleave ancillas X/Z by time parity
                if (t_local % 2) == 0:
                    for x, y in x2d:
                        n = g.add_physical_node()
                        node2coord[n] = (int(x), int(y), int(t))
                        coord2node[int(x), int(y), int(t)] = n
                        node2role[n] = "ancilla_x"
                        cur[int(x), int(y)] = n
                else:
                    for x, y in z2d:
                        n = g.add_physical_node()
                        node2coord[n] = (int(x), int(y), int(t))
                        coord2node[int(x), int(y), int(t)] = n
                        node2role[n] = "ancilla_z"
                        cur[int(x), int(y)] = n
            nodes_by_z[t] = cur

        # Intra-slice spatial edges (diagonal neighbors in rotated layout)
        for cur in nodes_by_z.values():
            for (x, y), u in cur.items():
                for dx, dy, dz in DIRECTIONS3D:
                    if dz != 0:
                        continue
                    xy2 = (x + dx, y + dy)
                    v = cur.get(xy2)
                    if v is not None and v > u:
                        with suppress(Exception):
                            g.add_physical_edge(u, v)

        # Inter-slice temporal edges (same XY across consecutive t)
        t_keys = sorted(nodes_by_z.keys())
        for i in range(1, len(t_keys)):
            cur = nodes_by_z[t_keys[i]]
            prev = nodes_by_z[t_keys[i - 1]]
            for xy, u in cur.items():
                v = prev.get(xy)
                if v is not None:
                    with suppress(Exception):
                        g.add_physical_edge(u, v)

        # Register BaseGraphState input/output nodes when ports are defined, so that
        # visualizers relying on BaseGraphState registries can highlight them
        try:
            # Determine z- (min) and z+ (max) among DATA nodes only
            data_coords_all = [c for n, c in node2coord.items() if node2role.get(n) == "data"]
            if data_coords_all:
                zmin = min(c[2] for c in data_coords_all)
                zmax = max(c[2] for c in data_coords_all)

                # XY -> local qubit index based on evaluated template
                xy_to_q = self.template.get_data_indices()

                # Optional: map XY to node ids at z- / z+
                xy_to_innode: dict[tuple[int, int], int] = {}
                xy_to_outnode: dict[tuple[int, int], int] = {}
                for x, y in xy_to_q:
                    n_in = coord2node.get((int(x), int(y), int(zmin)))
                    n_out = coord2node.get((int(x), int(y), int(zmax)))
                    if n_in is not None:
                        xy_to_innode[int(x), int(y)] = n_in
                    if n_out is not None:
                        xy_to_outnode[int(x), int(y)] = n_out

                # Register inputs
                xy_to_lidx: dict[tuple[int, int], int] = {}
                if self.in_ports:
                    inv_q_to_xy = {q: xy for xy, q in xy_to_q.items()}
                    for qidx in self.in_ports:
                        xy = inv_q_to_xy.get(qidx)
                        if xy is None:
                            continue
                        n_in = xy_to_innode.get((int(xy[0]), int(xy[1])))
                        if n_in is not None:
                            lidx = g.register_input(n_in)
                            xy_to_lidx[int(xy[0]), int(xy[1])] = lidx

                # Register outputs (reuse input logical index when possible)
                if self.out_ports:
                    inv_q_to_xy = {q: xy for xy, q in xy_to_q.items()}
                    for qidx in self.out_ports:
                        xy = inv_q_to_xy.get(qidx)
                        if xy is None:
                            continue
                        n_out = xy_to_outnode.get((int(xy[0]), int(xy[1])))
                        if n_out is not None:
                            lidx = xy_to_lidx.get((int(xy[0]), int(xy[1])))
                            if lidx is None:
                                # If there was no corresponding input, use template's
                                # qubit index as logical index for output registration.
                                lidx = int(qidx)
                            g.register_output(n_out, int(lidx))
        except (ValueError, KeyError, AttributeError) as e:
            # Visualization aid only; avoid breaking materialization pipelines
            print(f"Warning: failed to register I/O nodes on RHGBlock: {e}")

        # Store results on the block
        self.local_graph = g
        self.node2coord = node2coord
        self.coord2node = coord2node
        self.node2role = node2role
        self.coord2gid = dict.fromkeys(node2coord.values(), self.template.id_)
        return self

    # --- Compatibility aliases -------------------------------------------------
    # Some parts of the codebase use `edgespec` while this class had `edge_spec`.
    # Provide a property alias for smoother unification with pipes/templates.
    @property
    def edgespec(self) -> SpatialEdgeSpec | None:  # type: ignore[override]
        """Get or set the spatial edge specification (alias for edge_spec).

        Returns
        -------
        SpatialEdgeSpec | None
            The spatial edge specification for this block.
        """
        return self.edge_spec

    @edgespec.setter
    def edgespec(self, v: SpatialEdgeSpec | None) -> None:
        self.edge_spec = v

    def get_tiling_id(self) -> int:
        """Get the base qubit index of the underlying template."""
        tid = self.template.id_
        gids = set(self.coord2gid.values())
        if not all(gid == tid for gid in gids):
            msg = "coord2gid mismatch with template id"
            raise AssertionError(msg)

        # OVERWRITE
        self.set_tiling_id(tid)
        return self.template.id_

    def set_tiling_id(self, new_id: int) -> None:
        """Update the qubit indices in the underlying template by a specified id.

        Parameters
        ----------
        new_id : int
            The new base qubit index for the template.
        """
        self.template.id_ = new_id
        self.coord2gid = dict.fromkeys(self.coord2gid, new_id)

    @staticmethod
    def _boundary_nodes_from_coordmap(
        coord2node: dict[PhysCoordGlobal3D, int],
        node2role: dict[int, str] | None,
        *,
        face: str,
        depth: list[int] | None = None,
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Fast boundary selection utility shared by RHG structures.

        Parameters
        ----------
        coord2node : dict[PhysCoordGlobal3D, int]
            Mapping from physical coordinates to node ids.
        node2role : dict[int, str] | None
            Optional mapping from node id to role string
            (e.g., 'ancilla_x', 'ancilla_z'). If absent/empty, all
            selected nodes are returned under 'data'.
        face : {'x+','x-','y+','y-','z+','z-'}
            Boundary face to query.
        depth : list[int] | None
            Non-negative offsets inward from the selected face. Negative
            inputs are clamped to 0.

        Returns
        -------
        dict[str, list[PhysCoordGlobal3D]]
            Selected coordinates grouped by role: keys 'data', 'xcheck', 'zcheck'.
        """
        if not coord2node:
            return {"data": [], "xcheck": [], "zcheck": []}

        # Normalize and validate inputs
        f = face.strip().lower()
        if f not in {"x+", "x-", "y+", "y-", "z+", "z-"}:
            msg = "face must be one of: x+/x-/y+/y-/z+/z-"
            raise ValueError(msg)
        depths = [d if (isinstance(d, int) and d >= 0) else 0 for d in (depth or [0])]

        # Compute bounds once
        # Avoid repeated attribute lookups for speed
        coords = list(coord2node.keys())
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        xmin, xmax = (min(xs), max(xs))
        ymin, ymax = (min(ys), max(ys))
        zmin, zmax = (min(zs), max(zs))

        # Determine axis and target levels for quick membership tests
        if f[0] == "x":
            axis = 0
            targets = {xmax - d for d in depths} if f[1] == "+" else {xmin + d for d in depths}
        elif f[0] == "y":
            axis = 1
            targets = {ymax - d for d in depths} if f[1] == "+" else {ymin + d for d in depths}
        else:  # f[0] == 'z'
            axis = 2
            targets = {zmax - d for d in depths} if f[1] == "+" else {zmin + d for d in depths}

        # Group results by role (if available)
        has_roles = bool(node2role)
        data: list[PhysCoordGlobal3D] = []
        xcheck: list[PhysCoordGlobal3D] = []
        zcheck: list[PhysCoordGlobal3D] = []

        if has_roles:
            roles = node2role or {}
            for coord in coords:
                if coord[axis] not in targets:
                    continue
                role = (roles.get(coord2node[coord]) or "").lower()
                if role == "ancilla_x":
                    xcheck.append(coord)
                elif role == "ancilla_z":
                    zcheck.append(coord)
                else:
                    data.append(coord)
        else:
            # No role information: return all as data for the selected face
            append = data.append
            for coord in coords:
                if coord[axis] in targets:
                    append(coord)

        return {"data": data, "xcheck": xcheck, "zcheck": zcheck}

    def get_boundary_nodes(
        self,
        *,
        face: str,
        depth: list[int] | None = None,
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Boundary query after temporal composition on the compiled canvas.

        Operates on the global coord2node map using the same semantics as
        TemporalLayer.get_boundary_nodes.
        """
        return self._boundary_nodes_from_coordmap(self.coord2node, self.node2role, face=face, depth=depth)


@dataclass
class RHGBlockSkeleton:
    """A lightweight representation of a block before materialization."""

    name: ClassVar[str] = __qualname__
    d: int
    edgespec: SpatialEdgeSpec
    template: ScalableTemplate = field(init=False)

    def __post_init__(self) -> None:
        self.template = ScalableTemplate(d=self.d, edgespec=self.edgespec)

    def to_block(self) -> RHGBlock:
        """Convert this skeleton into a fully materialized RHGBlock instance."""
        msg = "to_block() must be implemented in subclasses."
        raise NotImplementedError(msg)

    def trim_spatial_boundary(self, direction: str) -> None:
        """Trim the spatial boundaries of the tiling."""
        self.template.trim_spatial_boundary(direction)
