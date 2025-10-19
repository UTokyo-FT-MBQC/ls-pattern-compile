"""RHG canvas module for skeleton and materialized canvas.

This module provides the RHGCanvasSkeleton and RHGCanvas classes for
representing both the skeleton (unmaterialized) and materialized versions
of the RHG canvas.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from operator import itemgetter
from typing import TYPE_CHECKING

from lspattern.accumulator import FlowAccumulator, ParityAccumulator, ScheduleAccumulator
from lspattern.blocks.cubes.base import RHGCube
from lspattern.blocks.pipes.base import RHGPipe
from lspattern.canvas.compiled import CompiledRHGCanvas, add_temporal_layer
from lspattern.canvas.layer import TemporalLayer, to_temporal_layer
from lspattern.canvas.ports import PortManager
from lspattern.consts import EDGE_TUPLE_SIZE, BoundarySide
from lspattern.mytype import (
    PatchCoordGlobal3D,
    PipeCoordGlobal3D,
)
from lspattern.utils import get_direction

if TYPE_CHECKING:
    from lspattern.blocks.cubes.base import RHGCubeSkeleton
    from lspattern.blocks.pipes.base import RHGPipeSkeleton
    from lspattern.tiling.template import ScalableTemplate


@dataclass
class RHGCanvasSkeleton:  # BlockGraph in tqec
    """Skeleton representation of an RHG canvas before materialization.

    Contains unmaterialized cube and pipe skeletons, which can be trimmed
    and then materialized into a concrete RHGCanvas.

    Attributes
    ----------
    name : str
        Name of the canvas skeleton.
    template : ScalableTemplate | None
        Optional template for the canvas.
    cubes_ : dict[PatchCoordGlobal3D, RHGCubeSkeleton]
        Dictionary of cube positions to cube skeletons.
    pipes_ : dict[PipeCoordGlobal3D, RHGPipeSkeleton]
        Dictionary of pipe coordinates to pipe skeletons.
    """

    name: str = "Blank Canvas Skeleton"
    # Optional template placeholder for future use
    template: ScalableTemplate | None = None
    cubes_: dict[PatchCoordGlobal3D, RHGCubeSkeleton] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipeSkeleton] = field(default_factory=dict)

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCubeSkeleton) -> None:
        """Add a cube skeleton at the specified position."""
        self.cubes_[position] = cube

    def add_pipe(self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipeSkeleton) -> None:
        """Add a pipe skeleton between start and end positions."""
        pipe_coord = PipeCoordGlobal3D((start, end))
        self.pipes_[pipe_coord] = pipe

    @staticmethod
    def _get_spatial_direction(dx: int, dy: int) -> tuple[BoundarySide, BoundarySide] | None:
        """Get trim directions for spatial pipe."""
        if dx == 1 and dy == 0:
            return BoundarySide.RIGHT, BoundarySide.LEFT  # X+ direction
        if dx == -1 and dy == 0:
            return BoundarySide.LEFT, BoundarySide.RIGHT  # X- direction
        if dy == 1 and dx == 0:
            return BoundarySide.TOP, BoundarySide.BOTTOM  # Y+ direction
        if dy == -1 and dx == 0:
            return BoundarySide.BOTTOM, BoundarySide.TOP  # Y- direction
        return None

    def _trim_adjacent_cubes(
        self, u: PatchCoordGlobal3D, v: PatchCoordGlobal3D, left_dir: BoundarySide, right_dir: BoundarySide
    ) -> None:
        """Trim boundaries of adjacent cubes."""
        left = self.cubes_.get(u)
        right = self.cubes_.get(v)

        if left is not None:
            left.trim_spatial_boundary(left_dir)
        if right is not None:
            right.trim_spatial_boundary(right_dir)

    def trim_spatial_boundaries(self) -> None:
        """Trim spatial boundaries of adjacent cubes."""
        for pipe_coord in list(self.pipes_.keys()):
            coord_tuple = tuple(pipe_coord)
            if len(coord_tuple) != EDGE_TUPLE_SIZE:
                msg = f"Expected pipe coordinate tuple of size {EDGE_TUPLE_SIZE}, got {len(coord_tuple)}"
                raise ValueError(msg)
            u, v = coord_tuple
            ux, uy, uz = u
            vx, vy, vz = v

            # Skip temporal pipes
            if uz != vz:
                continue

            dx, dy = vx - ux, vy - uy
            directions = self._get_spatial_direction(dx, dy)

            if directions is not None:
                left_dir, right_dir = directions
                self._trim_adjacent_cubes(u, v, left_dir, right_dir)

    def to_canvas(self) -> RHGCanvas:
        """Materialize the skeleton into a concrete RHGCanvas.

        Returns
        -------
        RHGCanvas
            The materialized canvas with concrete cubes and pipes.
        """
        self.trim_spatial_boundaries()

        trimmed_cubes_skeleton = self.cubes_.copy()
        trimmed_pipes_skeleton = self.pipes_.copy()

        cubes_: dict[PatchCoordGlobal3D, RHGCube] = {}
        for pos, c in trimmed_cubes_skeleton.items():
            # Materialize block and attach its 3D anchor so z-offset is correct
            blk = c.to_block()
            blk.source = pos
            if not isinstance(blk, RHGCube):
                msg = f"Expected RHGCube, got {type(blk)}"
                raise TypeError(msg)
            cubes_[pos] = blk
        pipes_: dict[PipeCoordGlobal3D, RHGPipe] = {}
        for pipe_coord, p in trimmed_pipes_skeleton.items():
            source, sink = pipe_coord
            # Type: ignore because pipe skeletons override to_block with source/sink args
            block = p.to_block(source, sink)  # type: ignore[call-arg]
            if not isinstance(block, RHGPipe):
                msg = f"Expected RHGPipe, got {type(block)}"
                raise TypeError(msg)
            pipes_[pipe_coord] = block

        return RHGCanvas(
            name=self.name,
            cubes_=cubes_,
            pipes_=pipes_,
        )


@dataclass
class RHGCanvas:  # TopologicalComputationGraph in tqec
    """Materialized RHG canvas containing concrete cubes and pipes.

    This class represents the fully materialized RHG canvas that can be
    split into temporal layers and compiled into a CompiledRHGCanvas.

    Attributes
    ----------
    name : str
        Name of the canvas.
    cubes_ : dict[PatchCoordGlobal3D, RHGCube]
        Dictionary of cube positions to concrete cubes.
    pipes_ : dict[PipeCoordGlobal3D, RHGPipe]
        Dictionary of pipe coordinates to concrete pipes.
    layers : list[TemporalLayer]
        List of temporal layers (populated after splitting).
    """

    name: str = "Blank Canvas"

    cubes_: dict[PatchCoordGlobal3D, RHGCube] = field(default_factory=dict)
    pipes_: dict[PipeCoordGlobal3D, RHGPipe] = field(default_factory=dict)
    layers: list[TemporalLayer] = field(default_factory=list)

    def add_cube(self, position: PatchCoordGlobal3D, cube: RHGCube) -> None:
        """Add a concrete cube at the specified position."""
        self.cubes_[position] = cube
        # Reset one-shot guard so layers can be rebuilt after topology changes
        with suppress(AttributeError):
            self._to_temporal_layers_called = False

    def add_pipe(self, start: PatchCoordGlobal3D, end: PatchCoordGlobal3D, pipe: RHGPipe) -> None:
        """Add a concrete pipe between start and end positions."""
        pipe_coord = PipeCoordGlobal3D((start, end))
        self.pipes_[pipe_coord] = pipe
        # Reset one-shot guard so layers can be rebuilt after topology changes
        with suppress(AttributeError):
            self._to_temporal_layers_called = False

    def to_temporal_layers(self) -> dict[int, TemporalLayer]:
        """Split the canvas into temporal layers by z-coordinate.

        Returns
        -------
        dict[int, TemporalLayer]
            Dictionary mapping z-coordinates to temporal layers.

        Raises
        ------
        RuntimeError
            If this method is called more than once per canvas instance.
        """
        # Disallow multiple calls to prevent duplicate XY shifts on templates.
        if getattr(self, "_to_temporal_layers_called", False):
            msg = (
                "RHGCanvas.to_temporal_layers() can be called at most once per canvas. "
                "Rebuild the canvas (or use RHGCanvasSkeleton.to_canvas()) before calling again."
            )
            raise RuntimeError(msg)
        temporal_layers: dict[int, TemporalLayer] = {}

        for z in range(max(self.cubes_.keys(), key=itemgetter(2))[2] + 1):
            cubes = {pos: c for pos, c in self.cubes_.items() if pos[2] == z}
            pipes = {}
            for pipe_coord, p in self.pipes_.items():
                coord_tuple = tuple(pipe_coord)
                if len(coord_tuple) == EDGE_TUPLE_SIZE:
                    start, end = coord_tuple
                    if start[2] == z and end[2] == z:
                        pipes[pipe_coord] = p

            layer = to_temporal_layer(z, cubes, pipes)
            temporal_layers[z] = layer

        with suppress(AttributeError):
            self._to_temporal_layers_called = True
        return temporal_layers

    def compile(self) -> CompiledRHGCanvas:
        """Compile the canvas into a CompiledRHGCanvas.

        Returns
        -------
        CompiledRHGCanvas
            The compiled canvas with all temporal layers composed.
        """
        temporal_layers = self.to_temporal_layers()
        # Initialize an empty compiled canvas with required accumulators
        initial_parity = ParityAccumulator()

        cgraph = CompiledRHGCanvas(
            layers=[],
            global_graph=None,
            coord2node={},
            port_manager=PortManager(),
            schedule=ScheduleAccumulator(),
            flow=FlowAccumulator(),
            parity=initial_parity,
            zlist=[],
        )

        # Note: q_index consistency is now automatically ensured by patch coordinate-based calculation

        # Compose layers in increasing temporal order, wiring any cross-layer pipes
        for z in sorted(temporal_layers.keys()):
            layer = temporal_layers[z]
            cgraph = add_temporal_layer(cgraph, layer)
        return cgraph
