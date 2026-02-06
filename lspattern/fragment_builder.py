"""Fragment builders for patch-based cube and pipe blocks.

This module provides functions to construct GraphSpec fragments from
patch-based layouts, separating graph generation from Canvas integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphqomb.common import Axis

from lspattern.accumulator import CoordFlowAccumulator, CoordParityAccumulator, CoordScheduleAccumulator
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.fragment import BlockFragment, Boundary, BoundaryFragment, GraphSpec
from lspattern.init_flow_analysis import InitFlowLayerKey
from lspattern.layout import (
    ANCILLA_EDGE_X,
    ANCILLA_EDGE_Z,
    RotatedSurfaceCodeLayoutBuilder,
)
from lspattern.mytype import Coord2D, Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from lspattern.init_flow_analysis import AdjacentPipeData
    from lspattern.loader import BlockConfig, PatchLayoutConfig

_PHYSICAL_CLOCK = 2
ANCILLA_LENGTH = len(ANCILLA_EDGE_X)


def _validate_flow_target(
    tgt_2d: Coord2D,
    tgt_coord: Coord3D,
    data2d: AbstractSet[Coord2D],
    adjacent_pipe_data: AdjacentPipeData | None,
) -> None:
    """Validate that flow target exists in cube data or adjacent pipe data.

    Parameters
    ----------
    tgt_2d : Coord2D
        Target 2D coordinate.
    tgt_coord : Coord3D
        Target 3D coordinate (for error message).
    data2d : AbstractSet[Coord2D]
        Cube data qubit coordinates.
    adjacent_pipe_data : AdjacentPipeData | None
        Optional per-boundary-side data qubit coordinates from adjacent pipes.

    Raises
    ------
    ValueError
        If target is not found in cube or adjacent pipe data.
    """
    tgt_in_cube = tgt_2d in data2d
    tgt_in_adjacent = adjacent_pipe_data is not None and any(
        tgt_2d in pipe_data for pipe_data in adjacent_pipe_data.values()
    )
    if not (tgt_in_cube or tgt_in_adjacent):
        msg = f"Flow target {tgt_coord} not found in cube or adjacent pipe data."
        raise ValueError(msg)


def _get_syndrome_params(
    basis: Axis,
    *,
    is_layer2: bool,
    invert_ancilla_order: bool,
    ancilla_x2d: AbstractSet[Coord2D],
    ancilla_z2d: AbstractSet[Coord2D],
) -> tuple[int, AbstractSet[Coord2D], tuple[tuple[int, int], ...]]:
    """Get syndrome measurement parameters based on layer and inversion settings.

    The pattern follows XOR logic: is_layer2 XOR invert_ancilla_order determines
    whether X-type or Z-type ancilla edges are used.

    - is_x_layer=False: uses same type as basis (X→X, Z→Z)
    - is_x_layer=True: uses opposite type to basis (X→Z, Z→X)
    """
    is_x_layer = is_layer2 ^ invert_ancilla_order
    parity_offset = 0 if basis == Axis.Z else 1
    use_x_type = (basis == Axis.Z) == is_x_layer

    if use_x_type:
        ancilla_2d = ancilla_x2d
        ancilla_edges = ANCILLA_EDGE_X
    else:
        ancilla_2d = ancilla_z2d
        ancilla_edges = ANCILLA_EDGE_Z

    return parity_offset, ancilla_2d, ancilla_edges


def _build_layers(
    block_config: BlockConfig,
    data2d: AbstractSet[Coord2D],
    ancilla_x2d: AbstractSet[Coord2D],
    ancilla_z2d: AbstractSet[Coord2D],
    offset_z: int,
    base_time: int,
    code_distance: int,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
    *,
    init_flow_directions: Mapping[InitFlowLayerKey, Coord2D] | None = None,
    adjacent_pipe_data: AdjacentPipeData | None = None,
    is_pipe: bool = False,
) -> GraphSpec:
    """Build graph layers from 2D coordinates and block configuration.

    This is the core layer-building logic shared by cube and pipe fragments.

    Parameters
    ----------
    block_config : BlockConfig
        Block configuration containing layer definitions.
    data2d : AbstractSet[Coord2D]
        2D data qubit coordinates.
    ancilla_x2d : AbstractSet[Coord2D]
        2D X-ancilla coordinates.
    ancilla_z2d : AbstractSet[Coord2D]
        2D Z-ancilla coordinates.
    offset_z : int
        Z-coordinate offset (in local coordinates, typically 0).
    base_time : int
        Base time offset (in local coordinates, typically 0).
    code_distance : int
        Code distance of the surface code.
    boundary : Mapping[BoundarySide, EdgeSpecValue]
        Boundary specifications for the patch.
    init_flow_directions : Mapping[InitFlowLayerKey, Coord2D] | None
        Optional per-layer flow directions for init layers (local coordinates).
    adjacent_pipe_data : AdjacentPipeData | None
        Optional per-boundary-side data qubit coordinates from adjacent pipes.
        Used for cubes with O (open) boundaries to find flow targets in pipes.
    is_pipe : bool
        Whether this is a pipe fragment. When True, remaining parity and
        initial ancilla flow are not added.

    Returns
    -------
    GraphSpec
        Graph fragment with nodes, edges, schedule, parity, and flow.
    """
    nodes: set[Coord3D] = set()
    edges: set[tuple[Coord3D, Coord3D]] = set()
    pauli_axes: dict[Coord3D, Axis] = {}
    coord2role: dict[Coord3D, NodeRole] = {}

    flow = CoordFlowAccumulator()
    scheduler = CoordScheduleAccumulator()
    parity = CoordParityAccumulator()
    if init_flow_directions is None:
        init_flow_directions = block_config.init_flow_directions

    for layer_idx, layer_cfg in enumerate(block_config):
        z = offset_z + layer_idx * 2
        layer_time = base_time + layer_idx * 2 * (_PHYSICAL_CLOCK + ANCILLA_LENGTH)

        _build_layer1(
            layer_cfg,
            data2d,
            ancilla_x2d,
            ancilla_z2d,
            z,
            layer_time,
            nodes,
            edges,
            pauli_axes,
            coord2role,
            flow,
            scheduler,
            parity,
            code_distance,
            boundary,
            layer_idx=layer_idx,
            init_flow_directions=init_flow_directions,
            adjacent_pipe_data=adjacent_pipe_data,
            is_pipe=is_pipe,
            invert_ancilla_order=block_config.invert_ancilla_order,
        )

        _build_layer2(
            layer_cfg,
            data2d,
            ancilla_x2d,
            ancilla_z2d,
            z,
            layer_time,
            nodes,
            edges,
            pauli_axes,
            coord2role,
            flow,
            scheduler,
            parity,
            code_distance,
            boundary,
            layer_idx=layer_idx,
            init_flow_directions=init_flow_directions,
            adjacent_pipe_data=adjacent_pipe_data,
            is_pipe=is_pipe,
            invert_ancilla_order=block_config.invert_ancilla_order,
        )

    return GraphSpec(
        coord_mode="local",
        time_mode="local",
        nodes=nodes,
        edges=edges,
        pauli_axes=pauli_axes,
        coord2role=coord2role,
        flow=flow,
        scheduler=scheduler,
        parity=parity,
    )


def _build_layer1(  # noqa: C901
    layer_cfg: PatchLayoutConfig,
    data2d: AbstractSet[Coord2D],
    ancilla_x2d: AbstractSet[Coord2D],
    ancilla_z2d: AbstractSet[Coord2D],
    z: int,
    layer_time: int,
    nodes: set[Coord3D],
    edges: set[tuple[Coord3D, Coord3D]],
    pauli_axes: dict[Coord3D, Axis],
    coord2role: dict[Coord3D, NodeRole],
    flow: CoordFlowAccumulator,
    scheduler: CoordScheduleAccumulator,
    parity: CoordParityAccumulator,
    code_distance: int,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
    *,
    layer_idx: int,
    init_flow_directions: Mapping[InitFlowLayerKey, Coord2D] | None = None,
    adjacent_pipe_data: AdjacentPipeData | None = None,
    is_pipe: bool,
    invert_ancilla_order: bool = False,
) -> None:
    """Build layer1 (even z) of a layer pair.

    When invert_ancilla_order is True, X-ancilla is placed instead of Z-ancilla.
    """
    if layer_cfg.layer1.basis is not None:
        layer1_coords: list[Coord3D] = []
        for x, y in data2d:
            coord = Coord3D(x, y, z)
            nodes.add(coord)
            coord2role[coord] = NodeRole.DATA
            pauli_axes[coord] = layer_cfg.layer1.basis
            layer1_coords.append(coord)

            # Temporal edge to previous layer
            prev_coord = Coord3D(x, y, z - 1)
            if prev_coord in nodes:
                edges.add((prev_coord, coord))
                flow.add_flow(prev_coord, coord)
                scheduler.add_entangle_at_time(layer_time, {(prev_coord, coord)})

        scheduler.add_prep_at_time(layer_time, layer1_coords)
        scheduler.add_meas_at_time(layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH + 1, layer1_coords)

        # Parity check with data qubits (when no ancilla in this layer)
        if not layer_cfg.layer1.ancilla and not layer_cfg.layer1.skip_syndrome:
            parity_offset, ancilla_2d, ancilla_edges = _get_syndrome_params(
                layer_cfg.layer1.basis,
                is_layer2=False,
                invert_ancilla_order=invert_ancilla_order,
                ancilla_x2d=ancilla_x2d,
                ancilla_z2d=ancilla_z2d,
            )
            for x, y in ancilla_2d:
                data_collection: set[Coord3D] = set()
                for dx, dy in ancilla_edges:
                    if Coord2D(x + dx, y + dy) in data2d:
                        data_collection.add(Coord3D(x + dx, y + dy, z))
                if data_collection:
                    parity.add_syndrome_measurement(Coord2D(x, y), z + parity_offset, data_collection)

    # Ancilla in layer1 (Z-ancilla by default, X-ancilla if inverted)
    if layer_cfg.layer1.ancilla:
        if invert_ancilla_order:
            ancilla_role, ancilla_2d_source, ancilla_edges, edge_spec = (
                NodeRole.ANCILLA_X,
                ancilla_x2d,
                ANCILLA_EDGE_X,
                EdgeSpecValue.X,
            )
        else:
            ancilla_role, ancilla_2d_source, ancilla_edges, edge_spec = (
                NodeRole.ANCILLA_Z,
                ancilla_z2d,
                ANCILLA_EDGE_Z,
                EdgeSpecValue.Z,
            )

        ancilla_coords: list[Coord3D] = []
        for x, y in ancilla_2d_source:
            coord = Coord3D(x, y, z)
            nodes.add(coord)
            coord2role[coord] = ancilla_role
            pauli_axes[coord] = Axis.X
            ancilla_coords.append(coord)

            for i, (dx, dy) in enumerate(ancilla_edges):
                neighbor = Coord3D(x + dx, y + dy, z)
                if neighbor in nodes:
                    edges.add((coord, neighbor))
                    scheduler.add_entangle_at_time(layer_time + 1 + i, {(coord, neighbor)})

            if not layer_cfg.layer1.init:
                parity.add_syndrome_measurement(Coord2D(x, y), z, {coord})
                if not is_pipe:
                    parity.add_remaining_parity(Coord2D(x, y), z, {coord})
            elif is_pipe:
                # Init layer for pipe: register remaining_parity only (no syndrome_meas)
                # Cube's init layer does not register remaining_parity
                parity.add_remaining_parity(Coord2D(x, y), z, {coord})

        scheduler.add_prep_at_time(layer_time, ancilla_coords)
        scheduler.add_meas_at_time(layer_time + ANCILLA_LENGTH + 1, ancilla_coords)

        # Add flow for initialization layer ancilla qubits
        if layer_cfg.layer1.init and not is_pipe:
            move_vec = None
            if init_flow_directions is not None:
                move_vec = init_flow_directions.get(InitFlowLayerKey(layer_idx, 1))
            if move_vec is None:
                msg = f"Missing init flow direction for layer1 (unit={layer_idx})."
                raise ValueError(msg)
            ancilla_flow = RotatedSurfaceCodeLayoutBuilder.construct_initial_ancilla_flow(
                code_distance,
                Coord2D(0, 0),
                boundary,
                edge_spec,
                move_vec,
                adjacent_data=adjacent_pipe_data,
            )
            for src_2d, tgt_2d_set in ancilla_flow.items():
                src_coord = Coord3D(src_2d.x, src_2d.y, z)
                for tgt_2d in tgt_2d_set:
                    tgt_coord = Coord3D(tgt_2d.x, tgt_2d.y, z)
                    # Only check src_coord in nodes; tgt_coord may be in adjacent pipe
                    # for cubes with O (open) boundaries
                    if src_coord in nodes:
                        _validate_flow_target(tgt_2d, tgt_coord, data2d, adjacent_pipe_data)
                        flow.add_flow(src_coord, tgt_coord)


def _build_layer2(  # noqa: C901
    layer_cfg: PatchLayoutConfig,
    data2d: AbstractSet[Coord2D],
    ancilla_x2d: AbstractSet[Coord2D],
    ancilla_z2d: AbstractSet[Coord2D],
    z: int,
    layer_time: int,
    nodes: set[Coord3D],
    edges: set[tuple[Coord3D, Coord3D]],
    pauli_axes: dict[Coord3D, Axis],
    coord2role: dict[Coord3D, NodeRole],
    flow: CoordFlowAccumulator,
    scheduler: CoordScheduleAccumulator,
    parity: CoordParityAccumulator,
    code_distance: int,
    boundary: Mapping[BoundarySide, EdgeSpecValue],
    *,
    layer_idx: int,
    init_flow_directions: Mapping[InitFlowLayerKey, Coord2D] | None = None,
    adjacent_pipe_data: AdjacentPipeData | None = None,
    is_pipe: bool,
    invert_ancilla_order: bool = False,
) -> None:
    """Build layer2 (odd z) of a layer pair.

    When invert_ancilla_order is True, Z-ancilla is placed instead of X-ancilla.
    """
    if layer_cfg.layer2.basis is not None:
        layer2_coords: list[Coord3D] = []
        for x, y in data2d:
            coord = Coord3D(x, y, z + 1)
            nodes.add(coord)
            coord2role[coord] = NodeRole.DATA
            pauli_axes[coord] = layer_cfg.layer2.basis
            layer2_coords.append(coord)

            # Temporal edge to previous layer
            prev_coord = Coord3D(x, y, z)
            if prev_coord in nodes:
                edges.add((prev_coord, coord))
                flow.add_flow(prev_coord, coord)
                scheduler.add_entangle_at_time(
                    layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH,
                    {(prev_coord, coord)},
                )

        scheduler.add_prep_at_time(layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH, layer2_coords)
        scheduler.add_meas_at_time(layer_time + 2 * (_PHYSICAL_CLOCK + ANCILLA_LENGTH) + 1, layer2_coords)

        # Parity check with data qubits (when no ancilla in this layer)
        if not layer_cfg.layer2.ancilla and not layer_cfg.layer2.skip_syndrome:
            parity_offset, ancilla_2d, ancilla_edges = _get_syndrome_params(
                layer_cfg.layer2.basis,
                is_layer2=True,
                invert_ancilla_order=invert_ancilla_order,
                ancilla_x2d=ancilla_x2d,
                ancilla_z2d=ancilla_z2d,
            )
            for x, y in ancilla_2d:
                data_collection: set[Coord3D] = set()
                for dx, dy in ancilla_edges:
                    if Coord2D(x + dx, y + dy) in data2d:
                        data_collection.add(Coord3D(x + dx, y + dy, z + 1))
                if data_collection:
                    parity.add_syndrome_measurement(Coord2D(x, y), z + 1 + parity_offset, data_collection)

    # Ancilla in layer2 (X-ancilla by default, Z-ancilla if inverted)
    if layer_cfg.layer2.ancilla:
        if invert_ancilla_order:
            ancilla_role, ancilla_2d_source, ancilla_edges, edge_spec = (
                NodeRole.ANCILLA_Z,
                ancilla_z2d,
                ANCILLA_EDGE_Z,
                EdgeSpecValue.Z,
            )
        else:
            ancilla_role, ancilla_2d_source, ancilla_edges, edge_spec = (
                NodeRole.ANCILLA_X,
                ancilla_x2d,
                ANCILLA_EDGE_X,
                EdgeSpecValue.X,
            )

        ancilla_coords: list[Coord3D] = []
        for x, y in ancilla_2d_source:
            coord = Coord3D(x, y, z + 1)
            nodes.add(coord)
            coord2role[coord] = ancilla_role
            pauli_axes[coord] = Axis.X
            ancilla_coords.append(coord)

            for i, (dx, dy) in enumerate(ancilla_edges):
                neighbor = Coord3D(x + dx, y + dy, z + 1)
                if neighbor in nodes:
                    edges.add((coord, neighbor))
                    scheduler.add_entangle_at_time(
                        layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH + 1 + i,
                        {(coord, neighbor)},
                    )

            if not layer_cfg.layer2.init:
                parity.add_syndrome_measurement(Coord2D(x, y), z + 1, {coord})
                if not is_pipe:
                    parity.add_remaining_parity(Coord2D(x, y), z + 1, {coord})
            elif is_pipe:
                # Init layer for pipe: register remaining_parity only (no syndrome_meas)
                # Cube's init layer does not register remaining_parity
                parity.add_remaining_parity(Coord2D(x, y), z + 1, {coord})

        scheduler.add_prep_at_time(layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH, ancilla_coords)
        scheduler.add_meas_at_time(layer_time + _PHYSICAL_CLOCK + 2 * ANCILLA_LENGTH + 1, ancilla_coords)

        # Add flow for initialization layer ancilla qubits
        if layer_cfg.layer2.init and not is_pipe:
            move_vec = None
            if init_flow_directions is not None:
                move_vec = init_flow_directions.get(InitFlowLayerKey(layer_idx, 2))
            if move_vec is None:
                msg = f"Missing init flow direction for layer2 (unit={layer_idx})."
                raise ValueError(msg)
            ancilla_flow = RotatedSurfaceCodeLayoutBuilder.construct_initial_ancilla_flow(
                code_distance,
                Coord2D(0, 0),
                boundary,
                edge_spec,
                move_vec,
                adjacent_data=adjacent_pipe_data,
            )
            for src_2d, tgt_2d_set in ancilla_flow.items():
                src_coord = Coord3D(src_2d.x, src_2d.y, z + 1)
                for tgt_2d in tgt_2d_set:
                    tgt_coord = Coord3D(tgt_2d.x, tgt_2d.y, z + 1)
                    # Only check src_coord in nodes; tgt_coord may be in adjacent pipe
                    # for cubes with O (open) boundaries
                    if src_coord in nodes:
                        _validate_flow_target(tgt_2d, tgt_coord, data2d, adjacent_pipe_data)
                        flow.add_flow(src_coord, tgt_coord)


def build_patch_cube_fragment(
    code_distance: int,
    block_config: BlockConfig,
) -> BlockFragment:
    """Build a BlockFragment for a patch-based cube.

    This function generates a complete graph fragment from a cube's 2D layout
    and block configuration, ready for merging into a Canvas.

    The fragment is generated in local coordinates (origin at (0, 0, 0)).
    Coordinate translation to global position happens during Canvas merge.

    Parameters
    ----------
    code_distance : int
        Code distance of the surface code.
    block_config : BlockConfig
        Block configuration containing boundary and layer definitions.

    Returns
    -------
    BlockFragment
        Complete fragment with graph, boundary, and optional cout.
    """
    # Get 2D coordinates from layout builder using origin position
    # This gives us local coordinates starting from (0, 0)
    data2d, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.cube(
        code_distance,
        Coord2D(0, 0),
        block_config.boundary,
        corner_decisions=block_config.corner_decisions,
    ).to_mutable_sets()

    # Build graph layers (using local coordinates: offset_z=0, base_time=0)
    graph_spec = _build_layers(
        block_config,
        data2d,
        ancilla_x2d,
        ancilla_z2d,
        offset_z=0,
        base_time=0,
        code_distance=code_distance,
        boundary=block_config.boundary,
        adjacent_pipe_data=block_config.adjacent_pipe_data,
    )

    # Build boundary fragment
    boundary_fragment = BoundaryFragment()
    boundary = Boundary(
        top=block_config.boundary[BoundarySide.TOP],
        bottom=block_config.boundary[BoundarySide.BOTTOM],
        left=block_config.boundary[BoundarySide.LEFT],
        right=block_config.boundary[BoundarySide.RIGHT],
    )
    # Store at local position (0, 0, 0) - will be translated when merged
    boundary_fragment.add_boundary(Coord3D(0, 0, 0), boundary)

    return BlockFragment(graph=graph_spec, boundary=boundary_fragment, cout=None)


def _direction_to_local_edge(direction: BoundarySide) -> tuple[Coord3D, Coord3D]:
    """Convert pipe direction to local edge coordinates.

    Parameters
    ----------
    direction : BoundarySide
        Pipe direction (RIGHT, LEFT, TOP, BOTTOM).

    Returns
    -------
    tuple[Coord3D, Coord3D]
        Local edge (start, end) for the pipe.
    """
    start = Coord3D(0, 0, 0)
    if direction == BoundarySide.RIGHT:
        end = Coord3D(1, 0, 0)
    elif direction == BoundarySide.LEFT:
        end = Coord3D(-1, 0, 0)
    elif direction == BoundarySide.TOP:
        end = Coord3D(0, -1, 0)
    else:  # BoundarySide.BOTTOM
        end = Coord3D(0, 1, 0)
    return start, end


def build_patch_pipe_fragment(
    code_distance: int,
    direction: BoundarySide,
    block_config: BlockConfig,
) -> BlockFragment:
    """Build a BlockFragment for a patch-based pipe.

    This function generates a complete graph fragment from a pipe's 2D layout
    and block configuration, ready for merging into a Canvas.

    The fragment is generated in local coordinates. The pipe direction
    determines the shape of the pipe (horizontal vs vertical).

    Parameters
    ----------
    code_distance : int
        Code distance of the surface code.
    direction : BoundarySide
        Direction of the pipe (RIGHT, LEFT, TOP, BOTTOM).
    block_config : BlockConfig
        Block configuration containing boundary and layer definitions.

    Returns
    -------
    BlockFragment
        Complete fragment with graph, boundary, and optional cout.
    """
    # Create local edge based on direction
    local_start, local_end = _direction_to_local_edge(direction)

    # Get 2D coordinates from layout builder using local edge
    data2d, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.pipe(
        code_distance,
        local_start,
        local_end,
        block_config.boundary,
        corner_decisions=block_config.corner_decisions,
    ).to_mutable_sets()

    # Remove the offset that pipe() already computed, since canvas.add_pipe will add it again
    # pipe() computes: offset_x = 2*d for RIGHT, offset_y = -2 for TOP, etc.
    if direction == BoundarySide.RIGHT:
        offset = Coord2D(2 * code_distance, 0)
    elif direction == BoundarySide.LEFT:
        offset = Coord2D(-2, 0)
    elif direction == BoundarySide.TOP:
        offset = Coord2D(0, -2)
    else:  # BOTTOM
        offset = Coord2D(0, 2 * code_distance)

    data2d = {Coord2D(c.x - offset.x, c.y - offset.y) for c in data2d}
    ancilla_x2d = {Coord2D(c.x - offset.x, c.y - offset.y) for c in ancilla_x2d}
    ancilla_z2d = {Coord2D(c.x - offset.x, c.y - offset.y) for c in ancilla_z2d}

    # Build graph layers (using local coordinates: offset_z=0, base_time=0)
    # Note: is_pipe=True skips remaining parity and initial ancilla flow
    graph_spec = _build_layers(
        block_config,
        data2d,
        ancilla_x2d,
        ancilla_z2d,
        offset_z=0,
        base_time=0,
        code_distance=code_distance,
        boundary=block_config.boundary,
        is_pipe=True,
    )

    # Build boundary fragment
    # Pipe registers boundary at both start and end positions
    boundary_fragment = BoundaryFragment()
    boundary = Boundary(
        top=block_config.boundary[BoundarySide.TOP],
        bottom=block_config.boundary[BoundarySide.BOTTOM],
        left=block_config.boundary[BoundarySide.LEFT],
        right=block_config.boundary[BoundarySide.RIGHT],
    )
    # Store boundaries at relative slot positions (0,0,0) and direction offset
    boundary_fragment.add_boundary(Coord3D(0, 0, 0), boundary)
    dx = local_end.x - local_start.x
    dy = local_end.y - local_start.y
    boundary_fragment.add_boundary(Coord3D(dx, dy, 0), boundary)

    return BlockFragment(graph=graph_spec, boundary=boundary_fragment, cout=None)
