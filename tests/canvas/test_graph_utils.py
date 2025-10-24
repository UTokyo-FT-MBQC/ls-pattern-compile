"""Unit tests for graph remapping utilities."""

from __future__ import annotations

import pytest
from graphqomb.graphstate import GraphState

from lspattern.canvas.graph_utils import (
    create_remapped_graphstate,
    remap_graph_edges,
    remap_graph_nodes,
    remap_measurement_bases,
)
from lspattern.mytype import NodeIdLocal


class TestRemapGraphNodes:
    """Test remap_graph_nodes functionality."""

    def test_remap_basic(self) -> None:
        """Test basic node remapping."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()

        nmap = {NodeIdLocal(n1): NodeIdLocal(10), NodeIdLocal(n2): NodeIdLocal(20)}
        created, gdst = remap_graph_nodes(gsrc, nmap)

        # Should create nodes with new IDs
        assert 10 in created
        assert 20 in created
        assert len(created) == 2
        assert len(gdst.physical_nodes) == 2

    def test_remap_empty_graph(self) -> None:
        """Test remapping empty graph."""
        gsrc = GraphState()
        nmap: dict[NodeIdLocal, NodeIdLocal] = {}

        created, gdst = remap_graph_nodes(gsrc, nmap)

        assert created == {}
        assert len(gdst.physical_nodes) == 0

    def test_remap_identity_mapping(self) -> None:
        """Test remapping with identity mapping."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()

        # Identity mapping
        nmap = {NodeIdLocal(n1): NodeIdLocal(n1), NodeIdLocal(n2): NodeIdLocal(n2)}
        created, gdst = remap_graph_nodes(gsrc, nmap)

        assert n1 in created
        assert n2 in created
        assert len(created) == 2

    def test_remap_many_to_one_mapping(self) -> None:
        """Test that many-to-one mapping raises an error."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        n3 = gsrc.add_physical_node()

        # Map n1 and n2 to same new ID - this should raise KeyError
        nmap = {
            NodeIdLocal(n1): NodeIdLocal(100),
            NodeIdLocal(n2): NodeIdLocal(100),
            NodeIdLocal(n3): NodeIdLocal(200),
        }

        import pytest
        with pytest.raises(KeyError, match="Node 100 is already created"):
            remap_graph_nodes(gsrc, nmap)

    def test_remap_partial_mapping(self) -> None:
        """Test remapping where some nodes are not in the mapping."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()

        # Only map n1, n2 uses default (self)
        nmap = {NodeIdLocal(n1): NodeIdLocal(10)}
        created, gdst = remap_graph_nodes(gsrc, nmap)

        assert 10 in created
        assert n2 in created  # n2 maps to itself
        assert len(created) == 2


class TestRemapMeasurementBases:
    """Test remap_measurement_bases functionality."""

    def test_remap_measurement_bases_basic(self) -> None:
        """Test basic measurement basis remapping."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        gsrc.assign_meas_basis(n1, (0, 0, 1))  # Z basis
        gsrc.assign_meas_basis(n2, (1, 0, 0))  # X basis

        gdst = GraphState()
        created = {10: gdst.add_physical_node(), 20: gdst.add_physical_node()}
        nmap = {NodeIdLocal(n1): NodeIdLocal(10), NodeIdLocal(n2): NodeIdLocal(20)}

        remap_measurement_bases(gsrc, gdst, nmap, created)

        assert gdst.meas_bases[created[10]] == (0, 0, 1)
        assert gdst.meas_bases[created[20]] == (1, 0, 0)

    def test_remap_measurement_bases_partial(self) -> None:
        """Test remapping when only some nodes have measurement bases."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        gsrc.assign_meas_basis(n1, (0, 0, 1))  # Only n1 has measurement basis

        gdst = GraphState()
        created = {10: gdst.add_physical_node(), 20: gdst.add_physical_node()}
        nmap = {NodeIdLocal(n1): NodeIdLocal(10), NodeIdLocal(n2): NodeIdLocal(20)}

        remap_measurement_bases(gsrc, gdst, nmap, created)

        assert created[10] in gdst.meas_bases
        assert created[20] not in gdst.meas_bases

    def test_remap_measurement_bases_empty(self) -> None:
        """Test remapping with no measurement bases."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()

        gdst = GraphState()
        created = {10: gdst.add_physical_node()}
        nmap = {NodeIdLocal(n1): NodeIdLocal(10)}

        remap_measurement_bases(gsrc, gdst, nmap, created)

        # No measurement bases should be set
        assert len(gdst.meas_bases) == 0


class TestRemapGraphEdges:
    """Test remap_graph_edges functionality."""

    def test_remap_edges_basic(self) -> None:
        """Test basic edge remapping."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        gsrc.add_physical_edge(n1, n2)

        gdst = GraphState()
        new_n1 = gdst.add_physical_node()
        new_n2 = gdst.add_physical_node()
        created = {10: new_n1, 20: new_n2}
        nmap = {NodeIdLocal(n1): NodeIdLocal(10), NodeIdLocal(n2): NodeIdLocal(20)}

        remap_graph_edges(gsrc, gdst, nmap, created)

        # Check edge exists between remapped nodes
        assert (new_n1, new_n2) in gdst.physical_edges or (new_n2, new_n1) in gdst.physical_edges

    def test_remap_edges_multiple(self) -> None:
        """Test remapping multiple edges."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        n3 = gsrc.add_physical_node()
        gsrc.add_physical_edge(n1, n2)
        gsrc.add_physical_edge(n2, n3)
        gsrc.add_physical_edge(n1, n3)

        gdst = GraphState()
        new_n1 = gdst.add_physical_node()
        new_n2 = gdst.add_physical_node()
        new_n3 = gdst.add_physical_node()
        created = {10: new_n1, 20: new_n2, 30: new_n3}
        nmap = {
            NodeIdLocal(n1): NodeIdLocal(10),
            NodeIdLocal(n2): NodeIdLocal(20),
            NodeIdLocal(n3): NodeIdLocal(30),
        }

        remap_graph_edges(gsrc, gdst, nmap, created)

        assert len(gdst.physical_edges) == 3

    def test_remap_edges_empty(self) -> None:
        """Test remapping graph with no edges."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()

        gdst = GraphState()
        new_n1 = gdst.add_physical_node()
        created = {10: new_n1}
        nmap = {NodeIdLocal(n1): NodeIdLocal(10)}

        remap_graph_edges(gsrc, gdst, nmap, created)

        assert len(gdst.physical_edges) == 0

    def test_remap_edges_with_partial_mapping(self) -> None:
        """Test edge remapping where some nodes use default mapping."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        gsrc.add_physical_edge(n1, n2)

        gdst = GraphState()
        new_n1 = gdst.add_physical_node()
        new_n2 = gdst.add_physical_node()
        created = {10: new_n1, n2: new_n2}  # n2 maps to itself
        nmap = {NodeIdLocal(n1): NodeIdLocal(10)}  # Only n1 is explicitly mapped

        remap_graph_edges(gsrc, gdst, nmap, created)

        # Edge should exist between new_n1 and new_n2
        assert (new_n1, new_n2) in gdst.physical_edges or (new_n2, new_n1) in gdst.physical_edges


class TestCreateRemappedGraphState:
    """Test create_remapped_graphstate functionality."""

    def test_create_remapped_graphstate_empty(self) -> None:
        """Test remapping empty graph."""
        gsrc = GraphState()
        nmap: dict[NodeIdLocal, NodeIdLocal] = {}

        result = create_remapped_graphstate(gsrc, nmap)

        assert result is not None
        assert len(result.physical_nodes) == 0
        assert len(result.physical_edges) == 0

    def test_create_remapped_graphstate_basic(self) -> None:
        """Test basic graph remapping with nodes and edges."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        gsrc.add_physical_edge(n1, n2)

        nmap = {NodeIdLocal(n1): NodeIdLocal(10), NodeIdLocal(n2): NodeIdLocal(20)}
        result = create_remapped_graphstate(gsrc, nmap)

        assert result is not None
        assert len(result.physical_nodes) == 2
        assert len(result.physical_edges) == 1

    def test_create_remapped_graphstate_with_measurement_bases(self) -> None:
        """Test graph remapping with measurement bases."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        gsrc.assign_meas_basis(n1, (0, 0, 1))
        gsrc.assign_meas_basis(n2, (1, 0, 0))

        nmap = {NodeIdLocal(n1): NodeIdLocal(10), NodeIdLocal(n2): NodeIdLocal(20)}
        result = create_remapped_graphstate(gsrc, nmap)

        assert result is not None
        assert len(result.meas_bases) == 2

    def test_create_remapped_graphstate_complete(self) -> None:
        """Test complete graph remapping with nodes, edges, and measurement bases."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        n3 = gsrc.add_physical_node()
        gsrc.add_physical_edge(n1, n2)
        gsrc.add_physical_edge(n2, n3)
        gsrc.assign_meas_basis(n1, (0, 0, 1))
        gsrc.assign_meas_basis(n3, (1, 0, 0))

        nmap = {
            NodeIdLocal(n1): NodeIdLocal(100),
            NodeIdLocal(n2): NodeIdLocal(200),
            NodeIdLocal(n3): NodeIdLocal(300),
        }
        result = create_remapped_graphstate(gsrc, nmap)

        assert result is not None
        assert len(result.physical_nodes) == 3
        assert len(result.physical_edges) == 2
        assert len(result.meas_bases) == 2

    def test_create_remapped_graphstate_identity(self) -> None:
        """Test graph remapping with identity mapping."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        gsrc.add_physical_edge(n1, n2)

        # Identity mapping
        nmap = {NodeIdLocal(n1): NodeIdLocal(n1), NodeIdLocal(n2): NodeIdLocal(n2)}
        result = create_remapped_graphstate(gsrc, nmap)

        assert result is not None
        assert len(result.physical_nodes) == 2
        assert len(result.physical_edges) == 1


class TestGraphUtilsEdgeCases:
    """Test edge cases and error conditions."""

    def test_remap_with_large_node_ids(self) -> None:
        """Test remapping with large node IDs."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()

        nmap = {NodeIdLocal(n1): NodeIdLocal(999999)}
        result = create_remapped_graphstate(gsrc, nmap)

        assert result is not None
        assert len(result.physical_nodes) == 1

    def test_remap_preserves_graph_structure(self) -> None:
        """Test that remapping preserves graph structure."""
        gsrc = GraphState()
        nodes = [gsrc.add_physical_node() for _ in range(5)]
        # Create a simple path graph
        for i in range(4):
            gsrc.add_physical_edge(nodes[i], nodes[i + 1])

        # Remap to different IDs
        nmap = {NodeIdLocal(nodes[i]): NodeIdLocal(100 + i) for i in range(5)}
        result = create_remapped_graphstate(gsrc, nmap)

        assert result is not None
        assert len(result.physical_nodes) == 5
        assert len(result.physical_edges) == 4

    def test_remap_many_to_one_reduces_graph_size(self) -> None:
        """Test that many-to-one mapping raises an error."""
        gsrc = GraphState()
        n1 = gsrc.add_physical_node()
        n2 = gsrc.add_physical_node()
        n3 = gsrc.add_physical_node()

        # Map all to same node - this should raise KeyError
        nmap = {
            NodeIdLocal(n1): NodeIdLocal(42),
            NodeIdLocal(n2): NodeIdLocal(42),
            NodeIdLocal(n3): NodeIdLocal(42),
        }

        import pytest
        with pytest.raises(KeyError, match="Node 42 is already created"):
            create_remapped_graphstate(gsrc, nmap)
