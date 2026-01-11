"""
YAML loader for patch layout layer configurations.

This module provides utilities to safely load and validate YAML configuration files
that define patch layout layers for MBQC surface code patterns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, overload

import yaml
from graphqomb.common import Axis
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

    from lspattern.consts import BoundarySide, EdgeSpecValue
    from lspattern.fragment import GraphSpec


class LayerConfig(NamedTuple):
    """
    Configuration for a single physical layer in a patch layout.

    Attributes
    ----------
    basis : Axis | None
        Measurement or initialization basis for qubits in this layer.
        Common values: Axis.X, Axis.Y, Axis.Z, or None (no qubits present).
    ancilla : bool
        Whether ancilla qubits are prepared in this layer.
    syndrome_meas_without_ancilla : bool
        Whether to register syndrome measurements (detector candidates) even when
        ``ancilla`` is ``false`` by inferring parity from neighboring data qubits.
    """

    basis: Axis | None
    ancilla: bool
    syndrome_meas_without_ancilla: bool


class PatchLayoutConfig(NamedTuple):
    """
    Complete configuration for a patch layout unit layer.

    This represents a unit layer that can be used to construct MBQC patterns
    for surface code quantum computing.

    Attributes
    ----------
    name : str
        Identifier for this unit layer (e.g., "MemoryUnit", "InitPlusUnit").
    description : str
        Human-readable description of the layer's purpose.
    layer1 : LayerConfig
        Configuration for layer 1 (typically Z-check layer).
    layer2 : LayerConfig
        Configuration for layer 2 (typically X-check layer).
    """

    name: str
    description: str
    layer1: LayerConfig
    layer2: LayerConfig


class BlockConfig(Sequence[PatchLayoutConfig]):
    """Sequence of PatchLayoutConfig representing a block configuration."""

    boundary: dict[BoundarySide, EdgeSpecValue]
    graph_spec: GraphSpec | None

    def __init__(self, configs: Sequence[PatchLayoutConfig]) -> None:
        self._configs = list(configs)
        self.graph_spec = None

    @overload
    def __getitem__(self, index: int) -> PatchLayoutConfig: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[PatchLayoutConfig]: ...

    def __getitem__(self, index: int | slice) -> PatchLayoutConfig | Sequence[PatchLayoutConfig]:
        return self._configs[index]

    def __len__(self) -> int:
        return len(self._configs)


# Pydantic models for YAML validation


class LayerConfigValidator(BaseModel):
    """Pydantic validator for LayerConfig YAML data."""

    basis: Axis | None = Field(default=Axis.X, description="Measurement basis for the layer")
    ancilla: bool = Field(default=True, description="Whether ancilla qubits are prepared")
    syndrome_meas_without_ancilla: bool = Field(
        default=True,
        description="Whether to register syndrome measurements when ancilla=false",
    )

    @field_validator("basis", mode="before")
    @classmethod
    def convert_basis_to_axis(cls, v: str | Axis | None) -> Axis | None:
        """Convert string basis to Axis enum."""
        if v is None:
            return None
        if isinstance(v, Axis):
            return v
        # Convert string to Axis
        basis_map = {"X": Axis.X, "Y": Axis.Y, "Z": Axis.Z}
        if v not in basis_map:
            msg = f"basis must be 'X', 'Y', 'Z', or None, got: {v}"
            raise ValueError(msg)
        return basis_map[v]


class PatchLayoutConfigValidator(BaseModel):
    """Pydantic validator for PatchLayoutConfig YAML data."""

    name: str = Field(description="Unit layer name")
    description: str = Field(description="Layer description")
    layer1: LayerConfigValidator = Field(description="Layer 1 configuration")
    layer2: LayerConfigValidator = Field(description="Layer 2 configuration")


def load_patch_layout_from_yaml(yaml_path: Path) -> PatchLayoutConfig:
    """
    Load and validate a patch layout configuration from a YAML file.

    This function safely loads YAML configuration files and validates them
    using Pydantic before converting to a strongly-typed NamedTuple.

    Parameters
    ----------
    yaml_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    PatchLayoutConfig
        Validated patch layout configuration as a NamedTuple.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    ValueError
        If the configuration fails validation (missing required fields,
        invalid values, etc.).

    Examples
    --------
    >>> from pathlib import Path
    >>> config_path = Path("lspattern/patch_layout/layers/memory_unit.yml")
    >>> config = load_patch_layout_from_yaml(config_path)
    >>> config.name
    'MemoryUnit'
    >>> config.layer1.basis
    Axis.X
    """
    if not yaml_path.exists():
        msg = f"YAML file not found: {yaml_path}"
        raise FileNotFoundError(msg)

    with yaml_path.open(encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        msg = f"Invalid YAML structure in {yaml_path}: expected dict, got {type(raw_config)}"
        raise TypeError(msg)

    # Validate with Pydantic
    validated = PatchLayoutConfigValidator(**raw_config)

    # Convert to NamedTuple
    return PatchLayoutConfig(
        name=validated.name,
        description=validated.description,
        layer1=LayerConfig(
            basis=validated.layer1.basis,
            ancilla=validated.layer1.ancilla,
            syndrome_meas_without_ancilla=validated.layer1.syndrome_meas_without_ancilla,
        ),
        layer2=LayerConfig(
            basis=validated.layer2.basis,
            ancilla=validated.layer2.ancilla,
            syndrome_meas_without_ancilla=validated.layer2.syndrome_meas_without_ancilla,
        ),
    )
