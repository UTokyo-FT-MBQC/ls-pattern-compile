"""Fragment builders for patch-based cube and pipe blocks.

This module provides functions to construct GraphSpec fragments from
patch-based layouts, separating graph generation from Canvas integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphqomb.common import Axis

from lspattern.accumulator import CoordFlowAccumulator, CoordParityAccumulator, CoordScheduleAccumulator
from lspattern.consts import BoundarySide
from lspattern.fragment import BlockFragment, Boundary, BoundaryFragment, GraphSpec
from lspattern.layout import (
    ANCILLA_EDGE_X,
    ANCILLA_EDGE_Z,
    RotatedSurfaceCodeLayoutBuilder,
)
from lspattern.mytype import Coord2D, Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from lspattern.loader import BlockConfig, PatchLayoutConfig

_PHYSICAL_CLOCK = 2
ANCILLA_LENGTH = len(ANCILLA_EDGE_X)


def _build_layers(
    block_config: BlockConfig,
    data2d: AbstractSet[Coord2D],
    ancilla_x2d: AbstractSet[Coord2D],
    ancilla_z2d: AbstractSet[Coord2D],
    offset_z: int,
    base_time: int,
    *,
    include_remaining_parity: bool = True,
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
    include_remaining_parity : bool
        Whether to add remaining parity for ancilla qubits. True for cube,
        False for pipe (to match original behavior).

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
            include_remaining_parity=include_remaining_parity,
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
            include_remaining_parity=include_remaining_parity,
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


def _build_layer1(
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
    *,
    include_remaining_parity: bool,
) -> None:
    """Build layer1 (even z) of a layer pair."""
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
            # parity_offset aligns data qubit parity with the corresponding ancilla layer:
            # X-basis data contributes to Z-stabilizers (registered at z+1 where X-ancilla operates)
            # Z-basis data contributes to X-stabilizers (registered at z where Z-ancilla operates)
            parity_offset = 1 if layer_cfg.layer1.basis == Axis.X else 0
            ancilla_2d = ancilla_z2d if layer_cfg.layer1.basis == Axis.Z else ancilla_x2d
            for x, y in ancilla_2d:
                data_collection: set[Coord3D] = set()
                for dx, dy in ANCILLA_EDGE_Z:
                    if Coord2D(x + dx, y + dy) in data2d:
                        data_collection.add(Coord3D(x + dx, y + dy, z))
                if data_collection:
                    parity.add_syndrome_measurement(Coord2D(x, y), z + parity_offset, data_collection)

    # Ancilla Z in layer1
    if layer_cfg.layer1.ancilla:
        ancilla_z_coords: list[Coord3D] = []
        for x, y in ancilla_z2d:
            coord = Coord3D(x, y, z)
            nodes.add(coord)
            coord2role[coord] = NodeRole.ANCILLA_Z
            pauli_axes[coord] = Axis.X
            ancilla_z_coords.append(coord)

            for i, (dx, dy) in enumerate(ANCILLA_EDGE_Z):
                neighbor = Coord3D(x + dx, y + dy, z)
                if neighbor in nodes:
                    edges.add((coord, neighbor))
                    scheduler.add_entangle_at_time(layer_time + 1 + i, {(coord, neighbor)})

            parity.add_syndrome_measurement(Coord2D(x, y), z, {coord})
            if include_remaining_parity:
                parity.add_remaining_parity(Coord2D(x, y), z, {coord})

        scheduler.add_prep_at_time(layer_time, ancilla_z_coords)
        scheduler.add_meas_at_time(layer_time + ANCILLA_LENGTH + 1, ancilla_z_coords)


def _build_layer2(
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
    *,
    include_remaining_parity: bool,
) -> None:
    """Build layer2 (odd z) of a layer pair."""
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
            # parity_offset aligns data qubit parity with the corresponding ancilla layer:
            # X-basis data contributes to Z-stabilizers (registered at z+1 where Z-ancilla operates)
            # Z-basis data contributes to X-stabilizers (registered at z+2 where X-ancilla operates)
            parity_offset = 0 if layer_cfg.layer2.basis == Axis.X else 1
            ancilla_2d = ancilla_z2d if layer_cfg.layer2.basis == Axis.Z else ancilla_x2d
            for x, y in ancilla_2d:
                data_collection: set[Coord3D] = set()
                for dx, dy in ANCILLA_EDGE_X:
                    if Coord2D(x + dx, y + dy) in data2d:
                        data_collection.add(Coord3D(x + dx, y + dy, z + 1))
                if data_collection:
                    parity.add_syndrome_measurement(Coord2D(x, y), z + 1 + parity_offset, data_collection)

    # Ancilla X in layer2
    if layer_cfg.layer2.ancilla:
        ancilla_x_coords: list[Coord3D] = []
        for x, y in ancilla_x2d:
            coord = Coord3D(x, y, z + 1)
            nodes.add(coord)
            coord2role[coord] = NodeRole.ANCILLA_X
            pauli_axes[coord] = Axis.X
            ancilla_x_coords.append(coord)

            for i, (dx, dy) in enumerate(ANCILLA_EDGE_X):
                neighbor = Coord3D(x + dx, y + dy, z + 1)
                if neighbor in nodes:
                    edges.add((coord, neighbor))
                    scheduler.add_entangle_at_time(
                        layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH + 1 + i,
                        {(coord, neighbor)},
                    )

            parity.add_syndrome_measurement(Coord2D(x, y), z + 1, {coord})
            if include_remaining_parity:
                parity.add_remaining_parity(Coord2D(x, y), z + 1, {coord})

        scheduler.add_prep_at_time(layer_time + _PHYSICAL_CLOCK + ANCILLA_LENGTH, ancilla_x_coords)
        scheduler.add_meas_at_time(layer_time + _PHYSICAL_CLOCK + 2 * ANCILLA_LENGTH + 1, ancilla_x_coords)


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
        code_distance, Coord2D(0, 0), block_config.boundary
    ).to_mutable_sets()

    # Build graph layers (using local coordinates: offset_z=0, base_time=0)
    graph_spec = _build_layers(
        block_config,
        data2d,
        ancilla_x2d,
        ancilla_z2d,
        offset_z=0,
        base_time=0,
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
        code_distance, local_start, local_end, block_config.boundary
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
    # Note: include_remaining_parity=False to match original add_pipe behavior
    graph_spec = _build_layers(
        block_config,
        data2d,
        ancilla_x2d,
        ancilla_z2d,
        offset_z=0,
        base_time=0,
        include_remaining_parity=False,
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
