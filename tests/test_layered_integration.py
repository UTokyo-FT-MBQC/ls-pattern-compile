"""Integration tests for layered RHG blocks in canvas compilation.

This module tests:
- Full canvas compilation with layered blocks
- Port setting correctness across layers
- Detector/parity accumulation correctness
- Flow accumulator correctness
- Integration with canvas temporal layers
"""

from __future__ import annotations

from typing import cast

from graphqomb.graphstate import GraphState
from lspattern.blocks.cubes.layered import (
    LayeredInitPlusCubeSkeleton,
    LayeredInitZeroCubeSkeleton,
    LayeredMemoryCubeSkeleton,
    LayeredRHGCube,
)
from lspattern.blocks.cubes.measure import MeasureXSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D


def test_layered_blocks_full_canvas_compilation() -> None:
    """Test full canvas compilation with layered blocks."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    # Create canvas with layered init + measure
    skeleton = RHGCanvasSkeleton(name="Layered Test Canvas")

    init_skeleton = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

    measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), measure_skeleton)

    # Materialize and compile
    canvas = skeleton.to_canvas()
    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    # Check that compilation succeeded
    global_graph = compiled_canvas.global_graph
    assert global_graph is not None
    assert len(global_graph.physical_nodes) > 0

    # Check that temporal layers are created
    assert len(compiled_canvas.layers) > 0

    # Check that schedule is populated
    assert len(compiled_canvas.schedule.schedule) > 0

    # Check that flow is populated
    assert len(compiled_canvas.flow.flow) > 0


def test_layered_blocks_port_setting() -> None:
    """Test that port setting is correct for layered blocks."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    # Create layered init cube
    init_skeleton = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    init_cube = init_skeleton.to_block()
    init_cube.materialize()

    # Init cube should have no input ports
    assert len(init_cube.in_ports) == 0
    # But should have output ports
    assert len(init_cube.out_ports) > 0

    # Create layered memory cube
    memory_skeleton = LayeredMemoryCubeSkeleton(d=d, edgespec=edgespec)
    memory_cube = memory_skeleton.to_block()
    memory_cube.materialize()

    # Memory cube should have both input and output ports
    assert len(memory_cube.in_ports) > 0
    assert len(memory_cube.out_ports) > 0

    # Input and output ports should be the same for memory
    assert memory_cube.in_ports == memory_cube.out_ports


def test_layered_blocks_parity_accumulation() -> None:
    """Test that parity accumulation works correctly across layers."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    skeleton = RHGCanvasSkeleton(name="Parity Test Canvas")

    init_skeleton = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

    measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), measure_skeleton)

    canvas = skeleton.to_canvas()
    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    # Check that parity checks are populated
    assert len(compiled_canvas.parity.checks) > 0

    # Verify that parity checks contain valid node IDs
    global_graph = compiled_canvas.global_graph
    assert global_graph is not None
    for z_dict in compiled_canvas.parity.checks.values():
        for node_set in z_dict.values():
            assert len(node_set) > 0
            for node_id in node_set:
                assert int(node_id) in global_graph.physical_nodes


def test_layered_blocks_flow_accumulation() -> None:
    """Test that flow accumulation works correctly."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    skeleton = RHGCanvasSkeleton(name="Flow Test Canvas")

    init_skeleton = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

    measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), measure_skeleton)

    canvas = skeleton.to_canvas()
    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    # Check that flow is populated
    assert len(compiled_canvas.flow.flow) > 0

    # Verify that flow entries reference valid nodes
    global_graph = compiled_canvas.global_graph
    assert global_graph is not None
    for src, dsts in compiled_canvas.flow.flow.items():
        assert int(src) in global_graph.physical_nodes
        for dst in dsts:
            assert int(dst) in global_graph.physical_nodes


def test_layered_blocks_temporal_layer_integration() -> None:
    """Test that layered blocks integrate correctly with temporal layers."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    skeleton = RHGCanvasSkeleton(name="Temporal Layer Test")

    # Add multiple cubes at different temporal layers
    init_skeleton = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

    memory_skeleton = LayeredMemoryCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), memory_skeleton)

    measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 2)), measure_skeleton)

    canvas = skeleton.to_canvas()
    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    # Check that multiple temporal layers are created
    assert len(compiled_canvas.layers) >= 3

    # Check that each layer has data
    for layer in compiled_canvas.layers:
        # Each layer should have at least one cube or pipe
        assert len(layer.cubes_) > 0 or len(layer.pipes_) > 0


def test_layered_blocks_multiple_patches() -> None:
    """Test canvas with multiple layered blocks in spatial arrangement."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    skeleton = RHGCanvasSkeleton(name="Multi-Patch Test")

    # Add two init cubes at different spatial positions
    init_skeleton1 = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton1)

    init_skeleton2 = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((1, 0, 0)), init_skeleton2)

    # Add measure cubes
    measure_skeleton1 = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), measure_skeleton1)

    measure_skeleton2 = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((1, 0, 1)), measure_skeleton2)

    canvas = skeleton.to_canvas()
    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    # Check that all cubes are included
    global_graph = compiled_canvas.global_graph
    assert global_graph is not None
    assert len(global_graph.physical_nodes) > 0

    # Schedule should cover all nodes
    assert len(compiled_canvas.schedule.schedule) > 0


def test_layered_blocks_detector_correctness() -> None:
    """Test that detectors are correctly constructed for layered blocks."""
    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    skeleton = RHGCanvasSkeleton(name="Detector Test")

    init_skeleton = LayeredInitPlusCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

    memory_skeleton = LayeredMemoryCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), memory_skeleton)

    measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 2)), measure_skeleton)

    canvas = skeleton.to_canvas()
    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    # Check that parity checks are present
    assert len(compiled_canvas.parity.checks) > 0

    # Verify structure of parity checks
    for z_dict in compiled_canvas.parity.checks.values():
        assert isinstance(z_dict, dict)
        for node_set in z_dict.values():
            assert isinstance(node_set, set)
            assert len(node_set) > 0


def test_layered_init_zero_integration() -> None:
    """Test LayeredInitZeroCube integration in canvas."""

    d = 3
    edgespec = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    skeleton = RHGCanvasSkeleton(name="InitZero Test")

    init_zero_skeleton = LayeredInitZeroCubeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_zero_skeleton)

    measure_skeleton = MeasureXSkeleton(d=d, edgespec=edgespec)
    skeleton.add_cube(PatchCoordGlobal3D((0, 0, 1)), measure_skeleton)

    canvas = skeleton.to_canvas()
    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    # Check compilation succeeds
    global_graph = compiled_canvas.global_graph
    assert global_graph is not None
    assert len(global_graph.physical_nodes) > 0

    # InitZero should have thinner first layer
    init_zero_cube = cast(LayeredRHGCube, canvas.cubes_[PatchCoordGlobal3D((0, 0, 0))])
    assert init_zero_cube.unit_layers[0].__class__.__name__ == "InitZeroUnitLayer"
