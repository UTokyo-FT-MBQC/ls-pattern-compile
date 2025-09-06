  # Accept either a pre-materialized block or a skeleton.
        if isinstance(block, RHGBlockSkeleton):
            block = block.to_block()
        # Require a template and a materialized local graph
        if block.template is None:
            raise ValueError(
                "Block has no template; set block.template before add_block()."
            )
        # TODO: Materialize 方針について再考する
        block.materialize()
        if block.graph_local is None:
            raise ValueError("Block.materialize() did not produce graph_local.")
        if not block.node2coord:
            raise ValueError("Block has empty node2coord after materialize().")

        # shift coordinates for placement (ids remain local to block graph)
        block.shift_coords(pos)

        # ConnectedTiling 用に保持
        self.blocks_[pos] = block

        # Update patch registry (ports will be set after composition)
        self.patches.append(pos)
        # T12: Do not compose block graphs here; layer graph is built in materialize()
        return

        # Compose this block's graph in parallel with the existing layer graph
        g2: GraphState = block.graph_local
        node_map1: dict[int, int] = {}
        node_map2: dict[int, int] = {}
        if self.local_graph is None:
            # First block in the layer: adopt directly; identity node_map2
            self.local_graph = g2
            node_map2 = {n: n for n in g2.physical_nodes}
        else:
            # Compose in parallel and adopt the new graph
            g1: GraphState = self.local_graph
            g_new, node_map1, node_map2 = compose_in_parallel(g1, g2)
            self.local_graph = g_new

            # Remap existing registries by node_map1
            if node_map1:
                self.node2coord = {
                    node_map1.get(n, n): c for n, c in self.node2coord.items()
                }
                self.coord2node = {
                    c: node_map1.get(n, n) for c, n in self.coord2node.items()
                }
                self.node2role = {
                    node_map1.get(n, n): r for n, r in self.node2role.items()
                }

                # Remap existing port sets and flat lists
                for p, nodes in list(self.in_portset.items()):
                    self.in_portset[p] = [node_map1.get(n, n) for n in nodes]
                for p, nodes in list(self.out_portset.items()):
                    self.out_portset[p] = [node_map1.get(n, n) for n in nodes]
                if hasattr(self, "cout_portset") and isinstance(
                    self.cout_portset, dict
                ):
                    for p, nodes in list(self.cout_portset.items()):
                        self.cout_portset[p] = [node_map1.get(n, n) for n in nodes]
                self.in_ports = [node_map1.get(n, n) for n in self.in_ports]
                self.out_ports = [node_map1.get(n, n) for n in self.out_ports]

        # Set in/out ports for this block using node_map2
        self.in_portset[pos] = [node_map2[n] for n in block.in_ports if n in node_map2]
        self.out_portset[pos] = [node_map2[n] for n in block.out_ports if n in node_map2]
        self.in_ports.extend(self.in_portset.get(pos, []))
        self.out_ports.extend(self.out_portset.get(pos, []))

        # Add the new block geometry via node_map2
        for old_n, coord in block.node2coord.items():
            nn = node_map2.get(old_n)
            if nn is None:
                continue
            # Detect coordinate collisions with already-placed blocks/nodes
            if coord in self.coord2node:
                existing_nn = self.coord2node[coord]
                raise ValueError(
                    f"Coordinate collision: {coord} already occupied by node {existing_nn} "
                    f"when adding block at {pos}."
                )
            self.node2coord[nn] = coord
            self.coord2node[coord] = nn

        # Record roles if provided by the block (for visualization)
        for old_n, role in block.node2role.items():
            nn = node_map2.get(old_n)
            if nn is not None:
                self.node2role[nn] = role

        self.qubit_count = len(self.local_graph.physical_nodes)
