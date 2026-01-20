"""Detector Constructor"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from graphqomb.common import Axis

from lspattern.layout import RotatedSurfaceCodeLayoutBuilder
from lspattern.mytype import Coord2D, Coord3D

if TYPE_CHECKING:
    from lspattern.accumulator import CoordParityAccumulator
    from lspattern.canvas import BoundaryGraph, Canvas
    from lspattern.consts import BoundarySide


def analyze_non_deterministic_regions(
    bgraph: BoundaryGraph,
) -> tuple[set[Coord3D], dict[Coord3D, dict[BoundarySide, frozenset[Axis]]]]:
    """Analyze the boundary graph to find non-deterministic regions.

    Parameters
    ----------
    bgraph : BoundaryGraph
        The boundary graph of the canvas.

    Returns
    -------
    tuple[set[Coord3D], dict[Coord3D, dict[BoundarySide, frozenset[Axis]]]]
        A tuple containing:
        - A set of coordinates that require bulk initialization.
        - A dictionary mapping coordinates to their boundary initialization info,
          where each entry maps a direction to the set of Pauli axes to be initialized.
    """
    bulk_init_coords: set[Coord3D] = set()
    boundary_init_info: dict[Coord3D, dict[BoundarySide, frozenset[Axis]]] = {}

    for coord in bgraph.boundary_map:
        if bgraph.check_bulk_init(coord):
            bulk_init_coords.add(coord)
        else:
            boundary_changes = bgraph.check_boundary_init(coord)
            if boundary_changes:
                boundary_init_info[coord] = boundary_changes

    return bulk_init_coords, boundary_init_info


def remove_non_deterministic_det(canvas: Canvas) -> CoordParityAccumulator:  # noqa: C901
    non_deterministic_coords, boundary_init_info = analyze_non_deterministic_regions(canvas.bgraph)
    new_parity_accumulator = copy.deepcopy(canvas.parity_accumulator)  # NOTE: should refactor

    # process bulk init coords
    for target_coord in non_deterministic_coords:
        block_config = canvas.cube_config[target_coord]
        _, ancilla_x2d, ancilla_z2d = RotatedSurfaceCodeLayoutBuilder.cube(
            canvas.config.d,
            Coord2D(target_coord.x, target_coord.y),
            canvas.cube_config[target_coord].boundary,
        ).to_mutable_sets()
        for layer_idx, layer_cfg in enumerate(block_config):
            if layer_cfg.layer1.basis is not None:
                z = target_coord.z * 2 * canvas.config.d + layer_idx * 2
                # Default: Z-ancilla in layer1, X-ancilla if inverted
                ancilla_coords = ancilla_x2d if block_config.invert_ancilla_order else ancilla_z2d
                for xy in ancilla_coords:
                    new_parity_accumulator.add_non_deterministic_coord(Coord3D(xy.x, xy.y, z))
                break
            if layer_cfg.layer2.basis is not None:
                z = target_coord.z * 2 * canvas.config.d + layer_idx * 2 + 1
                # Default: X-ancilla in layer2, Z-ancilla if inverted
                ancilla_coords = ancilla_z2d if block_config.invert_ancilla_order else ancilla_x2d
                for xy in ancilla_coords:
                    new_parity_accumulator.add_non_deterministic_coord(Coord3D(xy.x, xy.y, z))
                break

    # process boundary init coords
    for target_coord, direction_map in boundary_init_info.items():  # noqa: PLR1702
        block_config = canvas.cube_config[target_coord]
        invert = block_config.invert_ancilla_order
        for direction, axes in direction_map.items():
            x_ancilla2d, z_ancilla2d = RotatedSurfaceCodeLayoutBuilder.cube_boundary_ancillas_for_side(
                canvas.config.d,
                Coord2D(target_coord.x, target_coord.y),
                block_config.boundary,
                direction,
            )
            for axis in axes:
                for layer_idx, layer_cfg in enumerate(block_config):
                    # Determine target sublayer and ancilla coords based on axis and inversion
                    # Z-axis: default layer1, inverted layer2
                    # X-axis: default layer2, inverted layer1
                    if axis == Axis.Z:
                        use_layer1 = not invert
                        ancilla_2d = z_ancilla2d
                    else:  # Axis.X
                        use_layer1 = invert
                        ancilla_2d = x_ancilla2d

                    if use_layer1 and layer_cfg.layer1.ancilla:
                        z = target_coord.z * 2 * canvas.config.d + layer_idx * 2
                        for xy in ancilla_2d:
                            new_parity_accumulator.add_non_deterministic_coord(Coord3D(xy.x, xy.y, z))
                        break
                    if not use_layer1 and layer_cfg.layer2.ancilla:
                        z = target_coord.z * 2 * canvas.config.d + layer_idx * 2 + 1
                        for xy in ancilla_2d:
                            new_parity_accumulator.add_non_deterministic_coord(Coord3D(xy.x, xy.y, z))
                        break

    return new_parity_accumulator


def construct_detector(parity_accumulator: CoordParityAccumulator) -> dict[Coord3D, set[Coord3D]]:
    """Construct detectors from the parity accumulator.

    Parameters
    ----------
    parity_accumulator : CoordParityAccumulator
        The parity accumulator containing syndrome measurements.

    Returns
    -------
    dict[Coord3D, set[Coord3D]]
        A mapping from detector coordinates to sets of involved qubit coordinates.

    Notes
    -----
    An empty syndrome measurement at a given z-coordinate signals a parity reset.
    This is used when data qubits are removed between layers.
    """
    detectors: dict[Coord3D, set[Coord3D]] = {}
    for xy, z_map in parity_accumulator.syndrome_meas.items():
        # reorder z_map keys
        sorted_z_keys = sorted(z_map.keys())
        previous_meas: set[Coord3D] = set()
        for z in sorted_z_keys:
            current_meas = z_map[z]

            # Empty syndrome_meas signals a parity reset
            if not current_meas:
                previous_meas = set()
                continue

            if Coord3D(xy.x, xy.y, z) in parity_accumulator.non_deterministic_coords:
                previous_meas = parity_accumulator.remaining_parity.get(xy, {}).get(z, set())
                continue

            detectors[Coord3D(xy.x, xy.y, z)] = previous_meas.symmetric_difference(current_meas)
            previous_meas = parity_accumulator.remaining_parity.get(xy, {}).get(z, set())
    return detectors
