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
        # Initialize previous_meas from the most recent remaining_parity before first syndrome_meas
        # This handles init layers that may be several z-layers before the first syndrome_meas
        first_z = sorted_z_keys[0] if sorted_z_keys else 0
        remaining_at_xy = parity_accumulator.remaining_parity.get(xy, {})
        # Find the largest z < first_z that has remaining_parity
        prior_z_candidates = [z for z in remaining_at_xy if z < first_z]
        if prior_z_candidates:
            prior_z = max(prior_z_candidates)
            previous_meas = set(remaining_at_xy[prior_z])
        else:
            previous_meas = set()
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
