"""Tests for ParityAccumulator ignore_dangling functionality."""

import pytest

from lspattern.accumulator import ParityAccumulator
from lspattern.mytype import NodeIdLocal, PhysCoordLocal2D


class TestParityAccumulatorIgnoreDangling:
    """Test ignore_dangling functionality in ParityAccumulator."""

    def test_ignore_dangling_attribute(self) -> None:
        """Test that ignore_dangling attribute is properly initialized."""
        accumulator = ParityAccumulator()
        assert accumulator.ignore_dangling == {}

        # Test with initial ignore_dangling data
        ignore_data = {PhysCoordLocal2D((0, 0)): True}
        accumulator = ParityAccumulator(ignore_dangling=ignore_data)
        assert accumulator.ignore_dangling == ignore_data

    def test_remap_nodes_preserves_ignore_dangling(self) -> None:
        """Test that remap_nodes preserves ignore_dangling information."""
        coord = PhysCoordLocal2D((0, 0))
        accumulator = ParityAccumulator(
            checks={coord: {0: {NodeIdLocal(1), NodeIdLocal(2)}}},
            dangling_parity={coord: {NodeIdLocal(3)}},
            ignore_dangling={coord: True},
        )

        node_map = {NodeIdLocal(1): NodeIdLocal(10), NodeIdLocal(2): NodeIdLocal(20), NodeIdLocal(3): NodeIdLocal(30)}
        remapped = accumulator.remap_nodes(node_map)

        assert remapped.ignore_dangling == {coord: True}
        assert remapped.checks == {coord: {0: {NodeIdLocal(10), NodeIdLocal(20)}}}
        assert remapped.dangling_parity == {coord: {NodeIdLocal(30)}}

    def test_merge_with_ignore_dangling_true(self) -> None:
        """Test merge_with when ignore_dangling is True (no connection between dangling and checks)."""
        coord = PhysCoordLocal2D((0, 0))

        # First accumulator with dangling
        first = ParityAccumulator(
            checks={coord: {0: {NodeIdLocal(1)}}},
            dangling_parity={coord: {NodeIdLocal(2)}},
        )

        # Second accumulator with checks and ignore_dangling=True
        second = ParityAccumulator(
            checks={coord: {5: {NodeIdLocal(3)}}},
            dangling_parity={coord: {NodeIdLocal(4)}},
            ignore_dangling={coord: True},
        )

        merged = first.merge_with(second)

        # When ignore_dangling=True, dangling should become separate detector
        # and not merge with second's first check
        expected_checks = {
            coord: {
                0: {NodeIdLocal(1)},  # Original check from first
                5: {NodeIdLocal(3)},  # Check from second (unchanged)
            }
        }

        assert merged.checks == expected_checks
        assert merged.dangling_parity == {coord: {NodeIdLocal(4)}}  # From second
        assert merged.ignore_dangling == {coord: True}  # Inherited from first

    def test_merge_with_ignore_dangling_false(self) -> None:
        """Test merge_with when ignore_dangling is False (normal connection behavior)."""
        coord = PhysCoordLocal2D((0, 0))

        # First accumulator with dangling and ignore_dangling=False (or not set)
        first = ParityAccumulator(
            checks={coord: {0: {NodeIdLocal(1)}}},
            dangling_parity={coord: {NodeIdLocal(2)}},
        )

        # Second accumulator with checks
        second = ParityAccumulator(
            checks={coord: {5: {NodeIdLocal(3)}}},
            dangling_parity={coord: {NodeIdLocal(4)}},
            ignore_dangling={coord: False},
        )

        merged = first.merge_with(second)

        # When ignore_dangling=False, dangling should merge with second's first check
        expected_checks = {
            coord: {
                0: {NodeIdLocal(1)},  # Original check from first
                5: {NodeIdLocal(2), NodeIdLocal(3)},  # Merged: first's dangling + second's check
            }
        }

        assert merged.checks == expected_checks
        assert merged.dangling_parity == {coord: {NodeIdLocal(4)}}  # From second
        assert merged.ignore_dangling == {coord: False}  # Inherited from first

    def test_merge_with_no_ignore_dangling_default_behavior(self) -> None:
        """Test merge_with with default behavior when ignore_dangling is not set."""
        coord = PhysCoordLocal2D((0, 0))

        # First accumulator with dangling (no ignore_dangling set)
        first = ParityAccumulator(
            checks={coord: {0: {NodeIdLocal(1)}}},
            dangling_parity={coord: {NodeIdLocal(2)}},
        )

        # Second accumulator with checks
        second = ParityAccumulator(
            checks={coord: {5: {NodeIdLocal(3)}}},
            dangling_parity={coord: {NodeIdLocal(4)}},
        )

        merged = first.merge_with(second)

        # Default behavior: dangling should merge with second's first check
        expected_checks = {
            coord: {
                0: {NodeIdLocal(1)},  # Original check from first
                5: {NodeIdLocal(2), NodeIdLocal(3)},  # Merged: first's dangling + second's check
            }
        }

        assert merged.checks == expected_checks
        assert merged.dangling_parity == {coord: {NodeIdLocal(4)}}  # From second
        assert merged.ignore_dangling == {}  # No ignore_dangling settings

    def test_merge_parallel_ignore_dangling_inheritance(self) -> None:
        """Test merge_parallel properly inherits ignore_dangling from either accumulator."""
        coord1 = PhysCoordLocal2D((0, 0))
        coord2 = PhysCoordLocal2D((1, 1))

        # First accumulator with ignore_dangling at coord1
        first = ParityAccumulator(
            checks={coord1: {0: {NodeIdLocal(1)}}},
            dangling_parity={coord1: {NodeIdLocal(2)}},
            ignore_dangling={coord1: True},
        )

        # Second accumulator with ignore_dangling at coord2
        second = ParityAccumulator(
            checks={coord2: {0: {NodeIdLocal(3)}}},
            dangling_parity={coord2: {NodeIdLocal(4)}},
            ignore_dangling={coord2: True},
        )

        merged = first.merge_parallel(second)

        # Both coordinates should have ignore_dangling=True in result
        expected_ignore = {coord1: True, coord2: True}
        assert merged.ignore_dangling == expected_ignore

    def test_merge_parallel_ignore_dangling_or_logic(self) -> None:
        """Test merge_parallel uses OR logic for ignore_dangling (if either is True, result is True)."""
        coord = PhysCoordLocal2D((0, 0))

        # First accumulator with ignore_dangling=True
        first = ParityAccumulator(
            checks={coord: {0: {NodeIdLocal(1)}}},
            ignore_dangling={coord: True},
        )

        # Second accumulator with ignore_dangling=False
        second = ParityAccumulator(
            checks={coord: {0: {NodeIdLocal(2)}}},
            ignore_dangling={coord: False},
        )

        merged = first.merge_parallel(second)

        # Result should be True (OR logic)
        assert merged.ignore_dangling == {coord: True}

        # Test reverse case
        merged_reverse = second.merge_parallel(first)
        assert merged_reverse.ignore_dangling == {coord: True}

    def test_merge_with_inherit_ignore_dangling_from_other(self) -> None:
        """Test that merge_with inherits ignore_dangling from other accumulator."""
        coord = PhysCoordLocal2D((0, 0))

        # First accumulator without ignore_dangling
        first = ParityAccumulator(
            dangling_parity={coord: {NodeIdLocal(1)}},
        )

        # Second accumulator with ignore_dangling
        second = ParityAccumulator(
            checks={coord: {0: {NodeIdLocal(2)}}},
            ignore_dangling={coord: True},
        )

        merged = first.merge_with(second)

        # Should inherit ignore_dangling from second
        assert merged.ignore_dangling == {coord: True}
