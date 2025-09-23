from __future__ import annotations

import pytest

from lspattern.blocks.base import compute_logical_op_direction


def test_compute_logical_op_direction_x_operator_horizontal() -> None:
    """Test X logical operator with horizontal Z boundaries."""
    edgespec = {"LEFT": "Z", "RIGHT": "Z", "TOP": "X", "BOTTOM": "X"}
    result = compute_logical_op_direction(edgespec, "Z")
    assert result == "H"


def test_compute_logical_op_direction_x_operator_vertical() -> None:
    """Test X logical operator with vertical Z boundaries."""
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    result = compute_logical_op_direction(edgespec, "Z")
    assert result == "V"


def test_compute_logical_op_direction_z_operator_horizontal() -> None:
    """Test Z logical operator with horizontal X boundaries."""
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    result = compute_logical_op_direction(edgespec, "X")
    assert result == "H"


def test_compute_logical_op_direction_z_operator_vertical() -> None:
    """Test Z logical operator with vertical X boundaries."""
    edgespec = {"LEFT": "Z", "RIGHT": "Z", "TOP": "X", "BOTTOM": "X"}
    result = compute_logical_op_direction(edgespec, "X")
    assert result == "V"


def test_compute_logical_op_direction_case_insensitive() -> None:
    """Test that the function handles lowercase inputs correctly."""
    edgespec = {"LEFT": "z", "RIGHT": "z", "TOP": "x", "BOTTOM": "x"}
    result = compute_logical_op_direction(edgespec, "x")
    assert result == "V"

    result = compute_logical_op_direction(edgespec, "z")
    assert result == "H"


def test_compute_logical_op_direction_invalid_observable() -> None:
    """Test that invalid observable raises ValueError."""
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

    with pytest.raises(ValueError, match="obs must be one of: X, Z"):
        compute_logical_op_direction(edgespec, "Y")

    with pytest.raises(ValueError, match="obs must be one of: X, Z"):
        compute_logical_op_direction(edgespec, "invalid")


def test_compute_logical_op_direction_incomplete_edgespec() -> None:
    """Test that incomplete edgespec raises ValueError."""
    incomplete_edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z"}

    with pytest.raises(ValueError, match="edgespec must contain exactly the keys: LEFT, RIGHT, TOP, BOTTOM"):
        compute_logical_op_direction(incomplete_edgespec, "X")


def test_compute_logical_op_direction_extra_keys_in_edgespec() -> None:
    """Test that extra keys in edgespec are ignored."""
    edgespec = {
        "LEFT": "X",
        "RIGHT": "X",
        "TOP": "Z",
        "BOTTOM": "Z",
        "FRONT": "O",  # Extra key should be ignored
        "BACK": "O"    # Extra key should be ignored
    }
    result = compute_logical_op_direction(edgespec, "X")
    assert result == "H"


def test_compute_logical_op_direction_unsupported_x_operator() -> None:
    """Test that unsupported X operator configuration raises ValueError."""
    # X operator requires opposite boundaries to be X type
    edgespec = {"LEFT": "Z", "RIGHT": "X", "TOP": "X", "BOTTOM": "Z"}

    with pytest.raises(ValueError, match="edgespec does not support Z logical operator"):
        compute_logical_op_direction(edgespec, "X")


def test_compute_logical_op_direction_unsupported_z_operator() -> None:
    """Test that unsupported Z operator configuration raises ValueError."""
    # Z operator requires opposite boundaries to be Z type
    edgespec = {"LEFT": "X", "RIGHT": "Z", "TOP": "Z", "BOTTOM": "X"}

    with pytest.raises(ValueError, match="edgespec does not support X logical operator"):
        compute_logical_op_direction(edgespec, "Z")


def test_compute_logical_op_direction_open_boundaries() -> None:
    """Test with open boundaries (O) in some positions."""
    # Valid configuration: X operator with vertical X boundaries
    edgespec = {"LEFT": "O", "RIGHT": "O", "TOP": "X", "BOTTOM": "X"}
    result = compute_logical_op_direction(edgespec, "X")
    assert result == "V"

    # Invalid configuration: Z operator with no Z boundaries
    with pytest.raises(ValueError, match="edgespec does not support X logical operator"):
        compute_logical_op_direction(edgespec, "Z")


def test_compute_logical_op_direction_mixed_boundary_types() -> None:
    """Test various mixed boundary configurations."""
    # Valid configuration for X operator (horizontal)
    edgespec1 = {"LEFT": "X", "RIGHT": "X", "TOP": "O", "BOTTOM": "O"}
    result1 = compute_logical_op_direction(edgespec1, "X")
    assert result1 == "H"

    # Valid configuration for Z operator (vertical)
    edgespec2 = {"LEFT": "O", "RIGHT": "O", "TOP": "Z", "BOTTOM": "Z"}
    result2 = compute_logical_op_direction(edgespec2, "Z")
    assert result2 == "V"
