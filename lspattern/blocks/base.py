"""RHG blocks and skeletons for lattice-surgery templates."""

from __future__ import annotations

from contextlib import suppress

# imports layout is special due to optional dependency fallback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from graphix_zx.common import Axis, AxisMeasBasis, MeasBasis, Sign
from graphix_zx.graphstate import GraphState

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
    from collections.abc import Mapping, MutableMapping, Sequence
    from collections.abc import Set as AbstractSet

    from lspattern.mytype import (
        NodeIdGlobal,
        NodeIdLocal,
        PatchCoordGlobal3D,
        PatchCoordLocal2D,
        PhysCoordGlobal3D,
        QubitIndexLocal,
        SpatialEdgeSpec,
    )
else:
    # Import NewType factories for runtime use
    from lspattern.mytype import (
        NodeIdGlobal,
        NodeIdLocal,
        PatchCoordGlobal3D,
        PatchCoordLocal2D,
        PhysCoordGlobal3D,
    )

NUM_EDGE_SPEC_BOUDARY = 4


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
    cout_ports : list[set[NodeIdLocal]]
        Grouped classical output ports, expressed as node-id groups once the
        local graph has been materialized.
    local_graph : GraphState
        Local RHG graph constructed by ``materialize()``.
    meas_basis : MeasBasis
        Measurement basis for non-output nodes in the local graph.
        Currently, the measurement basis is uniform across all non-output nodes.
    node2coord, coord2node : dict
        Bidirectional maps between node ids and 3D coordinates.
    node2role : dict[int, str]
        Role of each node: ``'data'``, ``'ancilla_x'`` or ``'ancilla_z'``.
    node_map_global : dict[NodeIdLocal, NodeIdLocal]
        Mapping from local node IDs to global node IDs (set during graph composition).
    """

    name: ClassVar[str] = __qualname__
    d: int = 3
    edge_spec: SpatialEdgeSpec | None = field(default_factory=dict)
    # source
    source: PatchCoordGlobal3D = field(
        default_factory=lambda: PatchCoordGlobal3D((0, 0, 0))
    )
    sink: PatchCoordGlobal3D | None = None
    # When it is Pipe, we have sink and direction (Not implemented here)
    template: ScalableTemplate = field(
        default_factory=lambda: ScalableTemplate(d=3, edgespec={})
    )  # evaluated

    # Ports for this block's current logical patch boundary (qubit index sets)
    # classical output ports. One group represents one logical result (to be XORed)
    in_ports: set[QubitIndexLocal] = field(default_factory=set)
    out_ports: set[QubitIndexLocal] = field(default_factory=set)
    cout_ports: list[set[NodeIdLocal]] = field(default_factory=list)

    schedule: ScheduleAccumulator = field(
        init=False, default_factory=ScheduleAccumulator
    )
    flow: FlowAccumulator = field(init=False, default_factory=FlowAccumulator)
    parity: ParityAccumulator = field(init=False, default_factory=ParityAccumulator)

    local_graph: GraphState = field(init=False, default_factory=GraphState)
    meas_basis: MeasBasis = field(
        init=False, default_factory=lambda: AxisMeasBasis(Axis.X, Sign.PLUS)
    )
    node2coord: dict[NodeIdLocal, PhysCoordGlobal3D] = field(
        init=False, default_factory=dict
    )
    coord2node: dict[PhysCoordGlobal3D, NodeIdLocal] = field(
        init=False, default_factory=dict
    )
    node2role: dict[NodeIdLocal, str] = field(init=False, default_factory=dict)

    # Node mapping from local to global space (set during graph composition)
    node_map_global: dict[NodeIdLocal, NodeIdLocal] = field(
        init=False, default_factory=dict
    )

    final_layer: str | None = (
        None  # "M", "MX", "MZ", "MY" or "O" (open, no measurement)
    )

    def __post_init__(self) -> None:
        # Sync template parameters (d, edgespec)
        edgespec = self.edge_spec
        if self.template is None:
            self.template = RotatedPlanarCubeTemplate(
                d=int(self.d), edgespec=edgespec or {}
            )
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
    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Set the input ports for the block."""

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Set the output ports for the block."""

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Set the classical output ports for the block."""

    def shift_coords(self, by: PatchCoordGlobal3D) -> None:
        """Shift the patch anchor and the template by a 2D offset.

        The block's anchor (`source`) is 3D; the underlying scalable template
        only needs the XY offset. This mirrors the behavior used by pipes.
        """
        osx, osy, osz = self.source
        dx, dy, dz = by
        self.source = PatchCoordGlobal3D((osx + dx, osy + dy, osz + dz))

        by_template: PatchCoordLocal2D = PatchCoordLocal2D((dx, dy))
        self.template.shift_coords(by_template)

    def _sync_template_parameters(self) -> None:
        """Sync template parameters with block settings."""
        self.template.d = int(self.d)
        if self.edge_spec is not None:
            self.template.edgespec = dict(self.edge_spec)

        # Evaluate tiling coordinates (data/X/Z) only when not yet populated.
        # This preserves any absolute XY shifts applied upstream (e.g.,
        # to_temporal_layer() shifts templates before materialization).
        if not (
            getattr(self.template, "data_coords", None)
            or getattr(self.template, "x_coords", None)
            or getattr(self.template, "z_coords", None)
        ):
            self.template.to_tiling()

    def materialize(self) -> RHGBlock:
        """Finalize this block's template and initialize ports.

        - Sync `template.d` with `self.d`.
        - Ensure template edgespec mirrors `self.edge_spec` if provided.
        - Build tiling via `template.to_tiling()`.
        - Invoke port initialization hooks for quantum ports; classical groups are
          populated after node ids are assigned.
        """
        self._sync_template_parameters()

        # Initialize logical port sets (child classes may override these hooks)
        patch_coord = (self.source[0], self.source[1]) if self.source else None
        self.set_in_ports(patch_coord)
        self.set_out_ports(patch_coord)

        # Build the local RHG graph (nodes/edges and coordinate maps)
        g, node2coord, coord2node, node2role = self._build_3d_graph()

        # Register GraphState input/output nodes for visualization
        self._register_io_nodes(g, node2coord, coord2node, node2role)
        # Assign measurement basis for non-output nodes
        self._assign_meas_bases(g)

        self.local_graph = g
        # Convert to proper NewType dictionaries
        self.node2coord = {
            NodeIdLocal(k): PhysCoordGlobal3D(v) for k, v in node2coord.items()
        }
        self.coord2node = {
            PhysCoordGlobal3D(k): NodeIdLocal(v) for k, v in coord2node.items()
        }
        self.node2role = {NodeIdLocal(k): v for k, v in node2role.items()}

        # Populate classical output groups once node ids are known
        self.set_cout_ports(patch_coord)

        self._construct_detectors()
        self.coord2gid = dict.fromkeys(node2coord.values(), self.template.id_)
        return self

    def _build_3d_graph(
        self,
    ) -> tuple[
        GraphState,
        dict[int, tuple[int, int, int]],
        dict[tuple[int, int, int], int],
        dict[int, str],
    ]:
        """Build 3D RHG graph structure."""
        # Collect 2D coordinates from the evaluated template
        data2d = list(self.template.data_coords or [])
        x2d = list(self.template.x_coords or [])
        z2d = list(self.template.z_coords or [])

        # Construct a 3D RHG slab of height ~2*d
        max_t = 2 * self.d
        z0 = int(self.source[2]) * (2 * self.d)  # base z-offset per block

        g = GraphState()
        node2coord: dict[int, tuple[int, int, int]] = {}
        coord2node: dict[tuple[int, int, int], int] = {}
        node2role: dict[int, str] = {}

        # Assign nodes for each time slice
        nodes_by_z = self._assign_nodes_by_timeslice(
            g, data2d, x2d, z2d, max_t, z0, node2coord, coord2node, node2role
        )

        self._construct_schedule(nodes_by_z, node2role)

        # Add spatial and temporal edges
        RHGBlock._add_spatial_edges(g, nodes_by_z)
        self._add_temporal_edges(g, nodes_by_z)

        return g, node2coord, coord2node, node2role

    # TODO: avoid MutableMapping since it's not safe
    def _assign_nodes_by_timeslice(
        self,
        g: GraphState,
        data2d: Sequence[tuple[int, int]],
        x2d: Sequence[tuple[int, int]],
        z2d: Sequence[tuple[int, int]],
        max_t: int,
        z0: int,
        node2coord: MutableMapping[int, tuple[int, int, int]],
        coord2node: MutableMapping[tuple[int, int, int], int],
        node2role: MutableMapping[int, str],
    ) -> dict[int, dict[tuple[int, int], int]]:
        """Assign nodes for each time slice."""
        nodes_by_z: dict[int, dict[tuple[int, int], int]] = {}
        for t_local in range(max_t + 1):
            t = z0 + t_local
            cur: dict[tuple[int, int], int] = {}
            if self.final_layer is None:
                msg = "final_layer must be set"
                raise AssertionError(msg)
            if t_local == max_t and self.final_layer == "O":
                # add data node
                for x, y in data2d:
                    n = g.add_physical_node()
                    node2coord[n] = (x, y, t)
                    coord2node[x, y, t] = n
                    node2role[n] = "data"
                    cur[x, y] = n
            else:
                # Data nodes every slice except the final sentinel layer
                for x, y in data2d:
                    n = g.add_physical_node()
                    node2coord[n] = (x, y, t)
                    coord2node[x, y, t] = n
                    node2role[n] = "data"
                    cur[x, y] = n
                # Interleave ancillas X/Z by time parity
                if (t % 2) == 0:
                    for x, y in z2d:
                        n = g.add_physical_node()
                        node2coord[n] = (x, y, t)
                        coord2node[x, y, t] = n
                        node2role[n] = "ancilla_z"
                        cur[x, y] = n
                else:
                    for x, y in x2d:
                        n = g.add_physical_node()
                        node2coord[n] = (x, y, t)
                        coord2node[x, y, t] = n
                        node2role[n] = "ancilla_x"
                        cur[x, y] = n
            nodes_by_z[t] = cur
        return nodes_by_z

    @staticmethod
    def _add_spatial_edges(
        g: GraphState, nodes_by_z: Mapping[int, Mapping[tuple[int, int], int]]
    ) -> None:
        """Add intra-slice spatial edges."""
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

    def _add_temporal_edges(
        self, g: GraphState, nodes_by_z: Mapping[int, Mapping[tuple[int, int], int]]
    ) -> None:
        """Add inter-slice temporal edges."""
        t_keys = sorted(nodes_by_z.keys())
        for i in range(1, len(t_keys)):
            cur = nodes_by_z[t_keys[i]]
            prev = nodes_by_z[t_keys[i - 1]]
            for xy, u in cur.items():
                v = prev.get(xy)
                if v is not None:
                    with suppress(Exception):
                        g.add_physical_edge(u, v)
                    self.flow.flow.setdefault(NodeIdLocal(v), set()).add(NodeIdLocal(u))

    def _register_io_nodes(
        self,
        g: GraphState,
        node2coord: Mapping[int, tuple[int, int, int]],
        coord2node: Mapping[tuple[int, int, int], int],
        node2role: Mapping[int, str],
    ) -> None:
        """Register input/output nodes for visualization."""
        try:
            # Determine z- (min) and z+ (max) among DATA nodes only
            data_coords_all = [
                c for n, c in node2coord.items() if node2role.get(n) == "data"
            ]
            if not data_coords_all:
                print(
                    "Warning: no data nodes found in RHGBlock; skipping I/O registration"
                )
                return

            zmin = min(c[2] for c in data_coords_all)
            zmax = max(c[2] for c in data_coords_all)

            # Build coordinate mappings
            xy_to_innode, xy_to_outnode = self._build_coordinate_mappings(
                coord2node, zmin, zmax
            )

            # Register input and output ports
            xy_to_lidx = self._register_input_ports(g, xy_to_innode)
            self._register_output_ports(g, xy_to_outnode, xy_to_lidx)

        except (ValueError, KeyError, AttributeError) as e:
            # Visualization aid only; avoid breaking materialization pipelines
            print(f"Warning: failed to register I/O nodes on RHGBlock: {e}")

    def _assign_meas_bases(self, g: GraphState) -> None:
        """Assign measurement basis for non-output nodes."""
        for node in g.physical_nodes - g.output_node_indices.keys():
            g.assign_meas_basis(node, self.meas_basis)

    def _construct_detectors(self) -> None:
        raise NotImplementedError

    def _construct_schedule(
        self,
        nodes_by_z: Mapping[int, Mapping[tuple[int, int], int]],
        node2role: Mapping[int, str],
    ) -> None:
        for t, nodes in nodes_by_z.items():
            # Separate nodes by role: ancillas first, then data
            ancilla_nodes = []
            data_nodes = []

            for node_id in nodes.values():
                role = node2role.get(NodeIdLocal(node_id), "")
                if role in {"ancilla_x", "ancilla_z"}:
                    ancilla_nodes.append(node_id)
                else:  # data or other roles
                    data_nodes.append(node_id)

            # Schedule ancillas and data at different time slots
            # ancillas at 2*t, data at 2*t+1 to ensure temporal separation
            if ancilla_nodes:
                ancilla_global_nodes = {
                    NodeIdGlobal(node_id) for node_id in ancilla_nodes
                }
                self.schedule.schedule[2 * t] = ancilla_global_nodes

            if data_nodes:
                data_global_nodes = {NodeIdGlobal(node_id) for node_id in data_nodes}
                self.schedule.schedule[2 * t + 1] = data_global_nodes

    def _build_coordinate_mappings(
        self, coord2node: Mapping[tuple[int, int, int], int], zmin: int, zmax: int
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        """Build XY to input/output node mappings."""
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

        return xy_to_innode, xy_to_outnode

    def _register_input_ports(
        self, g: GraphState, xy_to_innode: Mapping[tuple[int, int], int]
    ) -> dict[tuple[int, int], int]:
        """Register input ports and return logical index mapping."""
        xy_to_lidx: dict[tuple[int, int], int] = {}
        if not self.in_ports:  # is this branch necessary?
            return xy_to_lidx

        # Use patch coordinate from source for consistent q_index calculation
        patch_coord = (self.source[0], self.source[1]) if self.source else None

        # For pipes, need to use the same parameters as in set_in_ports
        # TODO: this branch should be refactored without hasattr
        if hasattr(self, "sink") and self.sink is not None:
            # This is a pipe - use pipe-specific parameters
            sink_2d = (self.sink[0], self.sink[1])
            xy_to_q = self.template.get_data_indices(
                patch_coord, patch_type="pipe", sink_patch=sink_2d
            )
        else:
            # This is a cube - use standard parameters
            xy_to_q = self.template.get_data_indices(patch_coord)
        inv_q_to_xy = {q: xy for xy, q in xy_to_q.items()}

        for qidx in self.in_ports:
            xy_raw = inv_q_to_xy.get(qidx)
            if xy_raw is None:
                continue
            # Cast TilingCoord2D to tuple for type compatibility
            xy_tuple = (int(xy_raw[0]), int(xy_raw[1]))
            n_in = xy_to_innode.get(xy_tuple)
            if n_in is not None:
                g.register_input(n_in, qidx)
                xy_to_lidx[xy_tuple] = qidx

        return xy_to_lidx

    def _register_output_ports(
        self,
        g: GraphState,
        xy_to_outnode: Mapping[tuple[int, int], int],
        xy_to_lidx: Mapping[tuple[int, int], int],
    ) -> None:
        """Register output ports."""
        if not self.out_ports:
            return

        # Use patch coordinate from source for consistent q_index calculation
        patch_coord = (self.source[0], self.source[1]) if self.source else None
        # For pipes, need to use the same parameters as in set_out_ports
        if hasattr(self, "sink") and self.sink is not None:
            # This is a pipe - use pipe-specific parameters
            sink_2d = (self.sink[0], self.sink[1])
            xy_to_q = self.template.get_data_indices(
                patch_coord, patch_type="pipe", sink_patch=sink_2d
            )
        else:
            # This is a cube - use standard parameters
            xy_to_q = self.template.get_data_indices(patch_coord)
        inv_q_to_xy = {q: xy for xy, q in xy_to_q.items()}

        for qidx in self.out_ports:
            xy_raw = inv_q_to_xy.get(qidx)
            if xy_raw is None:
                continue
            # Cast TilingCoord2D to tuple for type compatibility
            xy_tuple = (int(xy_raw[0]), int(xy_raw[1]))
            n_out = xy_to_outnode.get(xy_tuple)
            if n_out is not None:
                lidx = xy_to_lidx.get(xy_tuple)
                if lidx is None:
                    # If there was no corresponding input, use template's
                    # qubit index as logical index for output registration.
                    lidx = int(qidx)
                g.register_output(n_out, int(lidx))

    # --- Compatibility aliases -------------------------------------------------
    # Some parts of the codebase use `edgespec` while this class had `edge_spec`.
    # Provide a property alias for smoother unification with pipes/templates.
    @property
    def edgespec(self) -> SpatialEdgeSpec | None:
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
        self.template.id_ = new_id  # type: ignore[assignment]
        self.coord2gid = dict.fromkeys(self.coord2gid, new_id)  # type: ignore[arg-type]

    @staticmethod
    def _validate_boundary_inputs(
        face: str, depth: Sequence[int] | None
    ) -> tuple[str, list[int]]:
        """Validate and normalize boundary query inputs."""
        f = face.strip().lower()
        if f not in {"x+", "x-", "y+", "y-", "z+", "z-"}:
            msg = "face must be one of: x+/x-/y+/y-/z+/z-"
            raise ValueError(msg)
        depths = [d if (isinstance(d, int) and d >= 0) else 0 for d in (depth or [0])]
        return f, depths

    @staticmethod
    def _compute_boundary_targets(
        coords: list[PhysCoordGlobal3D], face: str, depths: Sequence[int]
    ) -> tuple[int, set[int]]:
        """Compute target coordinates for boundary selection."""
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        xmin, xmax = (min(xs), max(xs))
        ymin, ymax = (min(ys), max(ys))
        zmin, zmax = (min(zs), max(zs))

        if face[0] == "x":
            axis = 0
            targets = (
                {xmax - d for d in depths}
                if face[1] == "+"
                else {xmin + d for d in depths}
            )
        elif face[0] == "y":
            axis = 1
            targets = (
                {ymax - d for d in depths}
                if face[1] == "+"
                else {ymin + d for d in depths}
            )
        else:  # face[0] == 'z'
            axis = 2
            targets = (
                {zmax - d for d in depths}
                if face[1] == "+"
                else {zmin + d for d in depths}
            )

        return axis, targets

    @staticmethod
    def _classify_nodes_by_role(
        coords: list[PhysCoordGlobal3D],
        coord2node: Mapping[PhysCoordGlobal3D, int],
        node2role: Mapping[int, str] | None,
        axis: int,
        targets: AbstractSet[int],
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Classify boundary nodes by their roles."""
        data: list[PhysCoordGlobal3D] = []
        xcheck: list[PhysCoordGlobal3D] = []
        zcheck: list[PhysCoordGlobal3D] = []

        has_roles = bool(node2role)
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

    @staticmethod
    def _boundary_nodes_from_coordmap(
        coord2node: Mapping[PhysCoordGlobal3D, int],
        node2role: Mapping[int, str] | None,
        *,
        face: str,
        depth: Sequence[int] | None = None,
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

        # Validate and normalize inputs
        f, depths = RHGBlock._validate_boundary_inputs(face, depth)

        # Compute boundary targets
        coords = list(coord2node.keys())
        axis, targets = RHGBlock._compute_boundary_targets(coords, f, depths)

        # Classify nodes by role
        return RHGBlock._classify_nodes_by_role(
            coords, coord2node, node2role, axis, targets
        )

    def get_boundary_nodes(
        self,
        *,
        face: str,
        depth: Sequence[int] | None = None,
    ) -> dict[str, list[PhysCoordGlobal3D]]:
        """Boundary query after temporal composition on the compiled canvas.

        Operates on the global coord2node map using the same semantics as
        TemporalLayer.get_boundary_nodes.
        """
        # Convert types for compatibility with static method signature
        coord2node_compat: dict[PhysCoordGlobal3D, int] = {
            k: int(v) for k, v in self.coord2node.items()
        }
        node2role_compat: dict[int, str] | None = (
            {int(k): v for k, v in self.node2role.items()} if self.node2role else None
        )
        return self._boundary_nodes_from_coordmap(
            coord2node_compat, node2role_compat, face=face, depth=depth
        )


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


def compute_logical_op_direction(edgespec: SpatialEdgeSpec, obs: str) -> str:
    """Compute the logical operation direction from edge specification and observable.

    Parameters
    ----------
    edgespec : SpatialEdgeSpec
        Spatial edge specification with keys 'LEFT', 'RIGHT', 'TOP', 'BOTTOM'.
    obs : {'X','Z'}
        Logical observable type.

    Returns
    -------
    str
        Logical operation direction: 'H' (horizontal) or 'V' (vertical).

    Raises
    ------
    ValueError
        If the edgespec is invalid or does not support the specified observable.
    """
    es = {
        k: str(v).upper()
        for k, v in edgespec.items()
        if k in {"LEFT", "RIGHT", "TOP", "BOTTOM"}
    }
    if len(es) != NUM_EDGE_SPEC_BOUDARY:
        msg = "edgespec must contain exactly the keys: LEFT, RIGHT, TOP, BOTTOM"
        raise ValueError(msg)

    if obs.upper() == "X":  # TODO: should be Z?
        # X logical operator runs between Z boundaries
        if es["LEFT"] == "Z" and es["RIGHT"] == "Z":
            return "H"
        if es["TOP"] == "Z" and es["BOTTOM"] == "Z":
            return "V"

        msg = "edgespec does not support X logical operator"
        raise ValueError(msg)
    if obs.upper() == "Z":
        # Z logical operator runs between X boundaries
        if es["LEFT"] == "X" and es["RIGHT"] == "X":
            return "H"
        if es["TOP"] == "X" and es["BOTTOM"] == "X":
            return "V"
        msg = "edgespec does not support Z logical operator"
        raise ValueError(msg)
    msg = "obs must be one of: X, Z"
    raise ValueError(msg)
