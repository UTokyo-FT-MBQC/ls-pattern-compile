"""The base definition for RHG unit layers"""

from __future__ import annotations

from dataclasses import dataclass

from lspattern.consts import BoundarySide, EdgeSpecValue


@dataclass
class UnitLayer:
    """Generic unit layer built from YAML configuration.

    This class represents a unit layer in the RHG lattice, constructed from
    YAML configuration files. It contains metadata about the layer structure
    and provides methods to build coordinate-based layer data.

    Attributes
    ----------
    name : str
        Name of the unit layer (e.g., "MemoryUnit", "InitZero").
    description : str
        Human-readable description of the layer's purpose.
    layout_type : str
        Type of layout (e.g., "rotated_surface_code").
    boundary : dict[BoundarySide, EdgeSpecValue]
        Boundary specifications for each side of the patch.
    layers_config : list[dict]
        Configuration for each physical layer (layer1, layer2).
        Each dict contains: {"basis": str, "ancilla": bool, "init": bool}
    """

    name: str
    description: str
    layout_type: str
    boundary: dict[BoundarySide, EdgeSpecValue]
