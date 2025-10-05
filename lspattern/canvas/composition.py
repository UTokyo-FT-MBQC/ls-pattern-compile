"""Graph composition logic for RHG canvas temporal layers.

This module handles the composition of graph states from RHG blocks (cubes and pipes),
including node mapping, coordinate processing, and port management during graph merging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from graphix_zx.graphstate import BaseGraphState, GraphState, compose

from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D, PhysCoordGlobal3D, PipeCoordGlobal3D

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lspattern.blocks.cubes.base import RHGCube
    from lspattern.blocks.pipes.base import RHGPipe
    from lspattern.canvas.coordinates import CoordinateMapper
    from lspattern.canvas.ports import PortManager


class GraphComposer:
    """Handles graph composition logic for temporal layers.

    This class encapsulates the logic for composing graph states from RHG blocks,
    managing node remapping, coordinate processing, and port management during
    the composition process.
    """

    def __init__(
        self,
        coord_mapper: CoordinateMapper,
        port_manager: PortManager,
    ) -> None:
        """Initialize the graph composer.

        Parameters
        ----------
        coord_mapper : CoordinateMapper
            Coordinate mapper for managing node-coordinate mappings.
        port_manager : PortManager
            Port manager for managing input/output/cout ports.
        """
        self.coord_mapper = coord_mapper
        self.port_manager = port_manager

    @staticmethod
    def compose_single_cube(
        pos: PatchCoordGlobal3D,  # noqa: ARG004
        blk: RHGCube,
        g: BaseGraphState,
    ) -> tuple[GraphState, Mapping[int, int], Mapping[int, int]]:
        """Compose a single cube into the graph.

        Parameters
        ----------
        pos : PatchCoordGlobal3D
            Position of the cube. This parameter is currently unused but retained
            for API consistency with the overall graph composition interface. It may
            be used in future extensions for position-dependent composition logic.
        blk : RHGCube
            The cube block to compose.
        g : BaseGraphState
            The current graph state.

        Returns
        -------
        tuple[GraphState, Mapping[int, int], Mapping[int, int]]
            New graph state and node mappings (node_map1, node_map2).
        """
        g2 = blk.local_graph

        g_new, node_map1, node_map2 = compose(g, g2)
        return g_new, node_map1, node_map2

    def process_cube_coordinates(self, blk: RHGCube, node_map2: Mapping[int, int]) -> None:
        """Process cube coordinates and roles with node mapping.

        Parameters
        ----------
        blk : RHGCube
            The cube block whose coordinates to process.
        node_map2 : Mapping[int, int]
            Node mapping from local to global node IDs.
        """
        # All blocks now use absolute coordinates - no z_shift needed
        # Ingest coords/roles
        for old_n, coord in blk.node2coord.items():
            new_n = node_map2.get(old_n)
            if new_n is None:
                continue
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            c_new = PhysCoordGlobal3D((x, y, z))
            role = blk.node2role.get(old_n)
            self.coord_mapper.add_node(NodeIdLocal(new_n), c_new, role)

    def process_cube_coordinates_direct(self, blk: RHGCube) -> None:
        """Process cube coordinates directly without node mapping.

        Used when there's only a single cube in the layer.

        Parameters
        ----------
        blk : RHGCube
            The cube block whose coordinates to process.
        """
        # All blocks now use absolute coordinates - no z_shift needed
        # Directly use node coordinates
        for node, coord in blk.node2coord.items():
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            c_new = PhysCoordGlobal3D((x, y, z))
            role = blk.node2role.get(node)
            self.coord_mapper.add_node(node, c_new, role)

    def process_cube_ports(self, pos: PatchCoordGlobal3D, blk: RHGCube, node_map2: Mapping[int, int]) -> None:
        """Process cube ports with node mapping.

        Parameters
        ----------
        pos : PatchCoordGlobal3D
            Position of the cube.
        blk : RHGCube
            The cube block whose ports to process.
        node_map2 : Mapping[int, int]
            Node mapping from local to global node IDs.
        """
        if blk.in_ports:
            mapped_nodes = [NodeIdLocal(node_map2[n]) for n in blk.in_ports if n in node_map2]
            self.port_manager.add_in_ports(pos, mapped_nodes)
        if blk.out_ports:
            mapped_nodes = [NodeIdLocal(node_map2[n]) for n in blk.out_ports if n in node_map2]
            self.port_manager.add_out_ports(pos, mapped_nodes)
        if blk.cout_ports:
            for group in blk.cout_ports:
                mapped_group: list[NodeIdLocal] = []
                for node in group:
                    new_id = node_map2.get(int(node))
                    if new_id is None:
                        continue
                    mapped_group.append(NodeIdLocal(new_id))
                self.port_manager.register_cout_group(pos, mapped_group)

    def process_cube_ports_direct(self, pos: PatchCoordGlobal3D, blk: RHGCube) -> None:
        """Process cube ports directly without node mapping.

        Used when there's only a single cube in the layer.

        Parameters
        ----------
        pos : PatchCoordGlobal3D
            Position of the cube.
        blk : RHGCube
            The cube block whose ports to process.
        """
        # Process input ports
        input_port_nodes = [NodeIdLocal(node) for node, _ in blk.local_graph.input_node_indices.items()]
        self.port_manager.add_in_ports(pos, input_port_nodes)

        # Process output ports
        output_port_nodes = [NodeIdLocal(node) for node, _ in blk.local_graph.output_node_indices.items()]
        self.port_manager.add_out_ports(pos, output_port_nodes)

        if blk.cout_ports:
            for group in blk.cout_ports:
                mapped_group = [NodeIdLocal(int(node)) for node in group]
                self.port_manager.register_cout_group(pos, mapped_group)

    def process_pipe_ports(self, pipe_coord: PipeCoordGlobal3D, pipe: RHGPipe, node_map2: Mapping[int, int]) -> None:
        """Process pipe ports with node mapping.

        Parameters
        ----------
        pipe_coord : PipeCoordGlobal3D
            Coordinate of the pipe (source, sink).
        pipe : RHGPipe
            The pipe block whose ports to process.
        node_map2 : Mapping[int, int]
            Node mapping from local to global node IDs.
        """
        source, sink = pipe_coord
        if pipe.in_ports:
            patch_pos = PatchCoordGlobal3D(source)
            mapped_nodes = [NodeIdLocal(node_map2[n]) for n in pipe.in_ports if n in node_map2]
            self.port_manager.add_in_ports(patch_pos, mapped_nodes)
        if pipe.out_ports:
            patch_pos = PatchCoordGlobal3D(sink)
            mapped_nodes = [NodeIdLocal(node_map2[n]) for n in pipe.out_ports if n in node_map2]
            self.port_manager.add_out_ports(patch_pos, mapped_nodes)
        if pipe.cout_ports:
            patch_pos = PatchCoordGlobal3D(sink)
            for group in pipe.cout_ports:
                mapped_group: list[NodeIdLocal] = []
                for node in group:
                    new_id = node_map2.get(int(node))
                    if new_id is None:
                        continue
                    mapped_group.append(NodeIdLocal(new_id))
                self.port_manager.register_cout_group(patch_pos, mapped_group)

    def compose_pipe_graphs(
        self,
        g: BaseGraphState,
        pipes: dict[PipeCoordGlobal3D, RHGPipe],
    ) -> GraphState:
        """Compose pipe graphs into the main graph state.

        Parameters
        ----------
        g : BaseGraphState
            The current graph state.
        pipes : dict[PipeCoordGlobal3D, RHGPipe]
            Dictionary of pipe coordinates to pipe blocks.

        Returns
        -------
        GraphState
            The updated graph state with pipes composed.

        Notes
        -----
        This method has side effects on the input pipe objects:
        - Sets the `node_map_global` attribute on each pipe for accumulator merging.
        """
        for pipe_coord, pipe in pipes.items():
            g2 = pipe.local_graph

            g_new, node_map1, node_map2 = compose(g, g2)
            # Store node mapping for later use in accumulator merging
            pipe.node_map_global = {NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map2.items()}
            self.coord_mapper.remap_nodes(node_map1)
            self.port_manager.remap_ports(node_map1)
            g = g_new

            # All blocks now use absolute coordinates - no z_shift needed
            for old_n, coord in pipe.node2coord.items():
                new_n = node_map2.get(old_n)
                if new_n is None:
                    continue
                # XY and Z are already in absolute coordinates (shifted in to_temporal_layer)
                x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                c_new = PhysCoordGlobal3D((x, y, z))
                role = pipe.node2role.get(old_n)
                self.coord_mapper.add_node(NodeIdLocal(new_n), c_new, role)

            # Process pipe ports with node mapping
            self.process_pipe_ports(pipe_coord, pipe, node_map2)

        return cast("GraphState", g)

    def build_graph_from_blocks(
        self,
        cubes: dict[PatchCoordGlobal3D, RHGCube],
        pipes: dict[PipeCoordGlobal3D, RHGPipe],
    ) -> GraphState:
        """Build the quantum graph state from cubes and pipes.

        Parameters
        ----------
        cubes : dict[PatchCoordGlobal3D, RHGCube]
            Dictionary of cube coordinates to cube blocks.
        pipes : dict[PipeCoordGlobal3D, RHGPipe]
            Dictionary of pipe coordinates to pipe blocks.

        Returns
        -------
        GraphState
            The composed graph state.

        Notes
        -----
        This method has side effects on the input cube and pipe objects:
        - Sets the `node_map_global` attribute on each cube and pipe for
          accumulator merging during temporal layer compilation.
        """
        # Special case: single block - use its GraphState directly to preserve q_indices
        if len(cubes) == 1 and len(pipes) == 0:
            pos, blk = next(iter(cubes.items()))
            g = blk.local_graph
            # Process coordinates and ports without composition
            self.process_cube_coordinates_direct(blk)
            self.process_cube_ports_direct(pos, blk)
            return g

        # Multiple blocks case - compose as before
        g_state: GraphState = GraphState()

        # Compose cube graphs
        for pos, blk in cubes.items():
            g_state, node_map1, node_map2 = self.compose_single_cube(pos, blk, g_state)
            # Store node mapping for later use in accumulator merging
            blk.node_map_global = {NodeIdLocal(k): NodeIdLocal(v) for k, v in node_map2.items()}
            self.coord_mapper.remap_nodes(node_map1)
            self.port_manager.remap_ports(node_map1)
            self.process_cube_coordinates(blk, node_map2)
            self.process_cube_ports(pos, blk, node_map2)

        # Compose pipe graphs (spatial pipes in this layer)
        return self.compose_pipe_graphs(g_state, pipes)
