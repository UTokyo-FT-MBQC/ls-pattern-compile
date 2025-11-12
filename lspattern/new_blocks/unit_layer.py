"""The base definition for RHG unit layers"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from lspattern.new_blocks.accumulator import CoordFlowAccumulator, CoordScheduleAccumulator
from lspattern.new_blocks.layer_data import CoordBasedLayerData
from lspattern.new_blocks.mytype import Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Sequence


class UnitLayer(ABC):
    """Abstract base class for RHG unit layers (2 physical layers)."""

    @property
    @abstractmethod
    def global_pos(self) -> Coord3D:
        """Get the global position of the unit layer.

        Returns
        -------
        Coord3D
            The global (x, y, z) position of the unit layer.
        """
        ...

    @abstractmethod
    def build_metadata(
        self,
        z_offset: int,
    ) -> CoordBasedLayerData:
        """Build coordinate-based metadata for this unit layer.

        Parameters
        ----------
        z_offset : int
            Starting z-coordinate for this layer.

        Returns
        -------
        CoordBasedLayerData
            Layer metadata including coordinates, roles, edges, schedule, flow.
        """
        ...


class CustomUnitLayer(UnitLayer):
    """Custom unit layer defined by user-provided metadata."""

    def __init__(self, global_pos: Coord3D, layer_data: CoordBasedLayerData) -> None:
        """Initialize the custom unit layer with its global offset and metadata."""
        self._global_pos = global_pos
        self._layer_data = layer_data

    @property
    def global_pos(self) -> Coord3D:
        """Return the global (x, y, z) position of the unit layer."""
        return self._global_pos

    def build_metadata(
        self,
        z_offset: int,
    ) -> CoordBasedLayerData:
        """Return the pre-defined coordinate-based metadata for this custom unit layer."""
        return self._layer_data
