"""Canvas module for RHG compilation.

This module provides the main compilation framework for converting RHG blocks
into executable quantum patterns with proper temporal layering and flow management.
"""

from __future__ import annotations

# Import main classes from _canvas_impl
from lspattern.canvas._canvas_impl import (
    CompiledRHGCanvas,
    RHGCanvas,
    RHGCanvasSkeleton,
    TemporalLayer,
    add_temporal_layer,
    to_temporal_layer,
)

# Import exceptions
from lspattern.canvas.exceptions import MixedCodeDistanceError

# Import PortManager
from lspattern.canvas.ports import PortManager

__all__ = [
    "CompiledRHGCanvas",
    "MixedCodeDistanceError",
    "PortManager",
    "RHGCanvas",
    "RHGCanvasSkeleton",
    "TemporalLayer",
    "add_temporal_layer",
    "to_temporal_layer",
]
