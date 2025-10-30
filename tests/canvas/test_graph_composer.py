"""Unit tests for GraphComposer class."""

from __future__ import annotations

from lspattern.canvas.composition import GraphComposer
from lspattern.canvas.coordinates import CoordinateMapper
from lspattern.canvas.ports import PortManager


class TestGraphComposerBasic:
    """Test basic GraphComposer functionality."""

    def test_initialization(self) -> None:
        """Test GraphComposer initializes correctly with dependencies."""
        coord_mapper = CoordinateMapper()
        port_manager = PortManager()
        composer = GraphComposer(coord_mapper, port_manager)

        assert composer.coord_mapper is coord_mapper
        assert composer.port_manager is port_manager
