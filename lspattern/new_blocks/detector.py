"""Detector Constructor"""

from __future__ import annotations

import copy

from lspattern.new_blocks.accumulator import CoordParityAccumulator
from lspattern.new_blocks.canvas import BoundaryGraph, Canvas
from lspattern.new_blocks.layout import RotatedSurfaceCodeLayoutBuilder
from lspattern.new_blocks.mytype import Coord2D, Coord3D


def analyze_non_deterministic_regions(bgraph: BoundaryGraph) -> set[Coord3D]:
    """Analyze the boundary graph to find non-deterministic regions.

    Parameters
    ----------
    bgraph : BoundaryGraph
        The boundary graph of the canvas.

    Returns
    -------
    set[Coord3D]
        A set of coordinates that are in non-deterministic regions.
    """
    non_deterministic_coords = set()
    for coord in bgraph.boundary_map:
        if bgraph.check_bulk_init(coord):
            non_deterministic_coords.add(coord)

        # TODO: Add boundary checks later

    # convert to qubit coordinates
    return non_deterministic_coords


def remove_non_deterministic_det(canvas: Canvas) -> CoordParityAccumulator:
    non_deterministic_coords = analyze_non_deterministic_regions(canvas.bgraph)
    new_parity_accumulator = copy.deepcopy(canvas.parity_accumulator)  # NOTE: should refactor

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
                for xy in ancilla_z2d:
                    new_parity_accumulator.add_non_deterministic_coord(Coord3D(xy.x, xy.y, z))
                break
            if layer_cfg.layer2.basis is not None:
                z = target_coord.z * 2 * canvas.config.d + layer_idx * 2 + 1
                for xy in ancilla_x2d:
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
    """
    detectors: dict[Coord3D, set[Coord3D]] = {}
    for xy, z_map in parity_accumulator.syndrome_meas.items():
        # reorder z_map keys
        sorted_z_keys = sorted(z_map.keys())
        previous_meas = set()
        for z in sorted_z_keys:
            if Coord3D(xy.x, xy.y, z) in parity_accumulator.non_deterministic_coords:
                previous_meas = z_map[z]
                continue

            detectors[Coord3D(xy.x, xy.y, z)] = previous_meas.symmetric_difference(z_map[z])
            previous_meas = z_map[z]
    return detectors
