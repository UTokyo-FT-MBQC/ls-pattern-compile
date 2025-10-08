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

# Import graph composition
from lspattern.canvas.composition import GraphComposer

# Import coordinate mapping
from lspattern.canvas.coordinates import CoordinateMapper

# Import exceptions
from lspattern.canvas.exceptions import MixedCodeDistanceError

# Import PortManager
from lspattern.canvas.ports import PortManager

# Import SeamGenerator
from lspattern.canvas.seams import SeamGenerator

__all__ = [
    "CompiledRHGCanvas",
    "CoordinateMapper",
    "GraphComposer",
    "MixedCodeDistanceError",
    "PortManager",
    "RHGCanvas",
    "RHGCanvasSkeleton",
    "SeamGenerator",
    "TemporalLayer",
    "add_temporal_layer",
    "to_temporal_layer",
]
