"""The base definition for RHG unit layers"""

from __future__ import annotations

from lspattern.new_blocks.layer_data import CoordBasedLayerData
from lspattern.new_blocks.mytype import Coord3D


class CustomUnitLayer:
    """Custom unit layer defined by user-provided metadata."""

    def __init__(self, global_pos: Coord3D) -> None:
        """Initialize the custom unit layer with its global offset and metadata."""
        self._global_pos = global_pos

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
