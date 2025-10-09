"""Tests for enum constants defined in lspattern.consts.

This module tests the type-safe string enums to ensure:
- String compatibility through str mixin
- Dictionary key functionality
- Enum completeness and member counts
"""

from __future__ import annotations

from lspattern.consts import (
    PIPEDIRECTION,
    BoundarySide,
    CoordinateSystem,
    EdgeSpecValue,
    InitializationState,
    NodeRole,
    Observable,
    VisualizationKind,
    VisualizationMode,
)


def test_enum_string_compatibility() -> None:
    """Verify str mixin allows string comparisons."""
    # EdgeSpecValue
    assert EdgeSpecValue.X == "X"
    assert EdgeSpecValue.Z == "Z"
    assert EdgeSpecValue.O == "O"

    # BoundarySide
    assert BoundarySide.LEFT == "LEFT"
    assert BoundarySide.RIGHT == "RIGHT"
    assert BoundarySide.TOP == "TOP"
    assert BoundarySide.BOTTOM == "BOTTOM"
    assert BoundarySide.UP == "UP"
    assert BoundarySide.DOWN == "DOWN"

    # NodeRole
    assert NodeRole.DATA == "data"
    assert NodeRole.ANCILLA_X == "ancilla_x"
    assert NodeRole.ANCILLA_Z == "ancilla_z"

    # PIPEDIRECTION
    assert PIPEDIRECTION.LEFT == "LEFT"
    assert PIPEDIRECTION.RIGHT == "RIGHT"
    assert PIPEDIRECTION.TOP == "TOP"
    assert PIPEDIRECTION.BOTTOM == "BOTTOM"
    assert PIPEDIRECTION.UP == "UP"
    assert PIPEDIRECTION.DOWN == "DOWN"

    # CoordinateSystem
    assert CoordinateSystem.TILING_2D == "tiling2d"
    assert CoordinateSystem.PHYS_3D == "phys3d"
    assert CoordinateSystem.PATCH_3D == "patch3d"

    # VisualizationKind
    assert VisualizationKind.BOTH == "both"
    assert VisualizationKind.X == "x"
    assert VisualizationKind.Z == "z"

    # VisualizationMode
    assert VisualizationMode.HIST == "hist"
    assert VisualizationMode.SLICES == "slices"

    # InitializationState
    assert InitializationState.PLUS == "plus"
    assert InitializationState.ZERO == "zero"

    # Observable
    assert Observable.X == "X"
    assert Observable.Z == "Z"


def test_enum_in_dict_keys() -> None:
    """Verify enums work as dictionary keys."""
    # EdgeSpecValue and BoundarySide
    spec = {BoundarySide.LEFT: EdgeSpecValue.X}
    assert spec[BoundarySide.LEFT] == EdgeSpecValue.X

    # Multiple keys
    full_spec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.Z,
        BoundarySide.TOP: EdgeSpecValue.O,
        BoundarySide.BOTTOM: EdgeSpecValue.X,
    }
    assert full_spec[BoundarySide.LEFT] == EdgeSpecValue.X
    assert full_spec[BoundarySide.RIGHT] == EdgeSpecValue.Z

    # PIPEDIRECTION as dict key
    pipe_config = {
        PIPEDIRECTION.LEFT: "config_left",
        PIPEDIRECTION.RIGHT: "config_right",
    }
    assert pipe_config[PIPEDIRECTION.LEFT] == "config_left"

    # NodeRole as dict key
    node_counts = {
        NodeRole.DATA: 9,
        NodeRole.ANCILLA_X: 6,
        NodeRole.ANCILLA_Z: 6,
    }
    assert node_counts[NodeRole.DATA] == 9


def test_enum_completeness() -> None:
    """Verify all expected enum members exist."""
    assert len(BoundarySide) == 6
    assert len(EdgeSpecValue) == 3
    assert len(NodeRole) == 3
    assert len(PIPEDIRECTION) == 6
    assert len(CoordinateSystem) == 3
    assert len(VisualizationKind) == 3
    assert len(VisualizationMode) == 2
    assert len(InitializationState) == 2
    assert len(Observable) == 2


def test_enum_member_names() -> None:
    """Verify enum member names are correct."""
    # EdgeSpecValue
    assert set(EdgeSpecValue.__members__.keys()) == {"X", "Z", "O"}

    # BoundarySide
    assert set(BoundarySide.__members__.keys()) == {"LEFT", "RIGHT", "TOP", "BOTTOM", "UP", "DOWN"}

    # NodeRole
    assert set(NodeRole.__members__.keys()) == {"DATA", "ANCILLA_X", "ANCILLA_Z"}

    # PIPEDIRECTION
    assert set(PIPEDIRECTION.__members__.keys()) == {"LEFT", "RIGHT", "TOP", "BOTTOM", "UP", "DOWN"}

    # CoordinateSystem
    assert set(CoordinateSystem.__members__.keys()) == {"TILING_2D", "PHYS_3D", "PATCH_3D"}

    # VisualizationKind
    assert set(VisualizationKind.__members__.keys()) == {"BOTH", "X", "Z"}

    # VisualizationMode
    assert set(VisualizationMode.__members__.keys()) == {"HIST", "SLICES"}

    # InitializationState
    assert set(InitializationState.__members__.keys()) == {"PLUS", "ZERO"}

    # Observable
    assert set(Observable.__members__.keys()) == {"X", "Z"}


def test_enum_iteration() -> None:
    """Verify enums can be iterated."""
    # Test that we can iterate over enum values
    boundary_values = list(BoundarySide)
    assert len(boundary_values) == 6
    assert BoundarySide.LEFT in boundary_values
    assert BoundarySide.UP in boundary_values

    # Test EdgeSpecValue iteration
    edge_values = list(EdgeSpecValue)
    assert len(edge_values) == 3
    assert EdgeSpecValue.X in edge_values


def test_enum_type_safety() -> None:
    """Verify enum type safety and isinstance checks."""
    # Test that enum members are instances of their enum class
    assert isinstance(BoundarySide.LEFT, BoundarySide)
    assert isinstance(EdgeSpecValue.X, EdgeSpecValue)
    assert isinstance(NodeRole.DATA, NodeRole)
    assert isinstance(PIPEDIRECTION.UP, PIPEDIRECTION)
    assert isinstance(InitializationState.PLUS, InitializationState)

    # Test that they are also str instances (due to str mixin)
    assert isinstance(BoundarySide.LEFT, str)
    assert isinstance(EdgeSpecValue.X, str)
    assert isinstance(NodeRole.DATA, str)


def test_boundary_side_spatial_temporal_distinction() -> None:
    """Verify distinction between spatial and temporal boundary sides."""
    spatial_boundaries = {BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM}
    temporal_boundaries = {BoundarySide.UP, BoundarySide.DOWN}

    # Ensure all boundary sides are accounted for
    all_boundaries = spatial_boundaries | temporal_boundaries
    assert len(all_boundaries) == 6
    assert all_boundaries == set(BoundarySide)


def test_pipedirection_consistency_with_boundary_side() -> None:
    """Verify PIPEDIRECTION values align with BoundarySide values."""
    # PIPEDIRECTION and BoundarySide should have the same directional values
    common_directions = {"LEFT", "RIGHT", "TOP", "BOTTOM", "UP", "DOWN"}
    assert set(PIPEDIRECTION.__members__.keys()) == common_directions
    assert set(BoundarySide.__members__.keys()) == common_directions

    # Values should also match
    assert PIPEDIRECTION.LEFT.value == BoundarySide.LEFT.value
    assert PIPEDIRECTION.UP.value == BoundarySide.UP.value
