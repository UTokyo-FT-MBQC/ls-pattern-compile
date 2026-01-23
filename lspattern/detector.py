"""Detector Constructor"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lspattern.mytype import Coord3D

if TYPE_CHECKING:
    from lspattern.accumulator import CoordParityAccumulator


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
        # Initialize previous_meas from remaining_parity at z-1 if available
        # This handles init layers that register remaining_parity without syndrome_meas
        first_z = sorted_z_keys[0] if sorted_z_keys else 0
        previous_meas = set(parity_accumulator.remaining_parity.get(xy, {}).get(first_z - 1, set()))
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
