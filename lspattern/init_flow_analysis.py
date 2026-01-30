"""Interfaces for init-layer ancilla flow analysis.

This module defines data structures and a stub analysis hook for computing
initial ancilla flow overrides based on canvas-wide boundary analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from lspattern.mytype import Coord2D, Coord3D

if TYPE_CHECKING:
    from lspattern.canvas_loader import CanvasSpec

InitFlowMap = dict[Coord2D, set[Coord2D]]


class InitFlowLayerKey(NamedTuple):
    """Identify a specific unit layer and sublayer within a block."""

    unit_layer_index: int
    sublayer: int  # 1=layer1, 2=layer2


@dataclass(slots=True)
class InitFlowOverrides:
    """Per-block init-layer flow overrides in local (block) coordinates."""

    cube: dict[Coord3D, dict[InitFlowLayerKey, InitFlowMap]] = field(default_factory=dict)
    pipe: dict[tuple[Coord3D, Coord3D], dict[InitFlowLayerKey, InitFlowMap]] = field(default_factory=dict)

    def cube_overrides(self, position: Coord3D) -> dict[InitFlowLayerKey, InitFlowMap]:
        return self.cube.get(position, {})

    def pipe_overrides(self, start: Coord3D, end: Coord3D) -> dict[InitFlowLayerKey, InitFlowMap]:
        return self.pipe.get((start, end), {})


def analyze_init_flow_overrides(spec: CanvasSpec, *, code_distance: int) -> InitFlowOverrides:
    """Analyze boundary relationships and return init-layer flow overrides.

    This is a stub; implement project-specific logic here.
    The returned flow maps must use local block coordinates (origin at 0,0),
    matching fragment_builder expectations.
    """
    _ = (spec, code_distance)
    return InitFlowOverrides()
