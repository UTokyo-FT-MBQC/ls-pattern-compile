# Pipe Initialize

class InitPlusPipe(RHGPipe):
    def __init__(
        self,
        d: int,
        edgespec: SpatialEdgeSpec,
        direction: PIPEDIRECTION,
        **kwargs,
    ):
        super().__init__(d=d, edgespec=edgespec, direction=direction, **kwargs)
        self.template = RotatedPlanarPipetemplate(d=d, edgespec=edgespec)
        self.materialize()

    def materialize(self) -> None:
        if self.graph_local and self.node2coord:
            return  # Already materialized

        # Generate tiling coordinates from the template
        tiling_data = self.template.to_tiling()
        data_coords_2d = tiling_data["data"]
        x_coords_2d = tiling_data["X"]
        z_coords_2d = tiling_data["Z"]

        # Map 2D tiling coordinates to 3D physical coordinates (z=0 for now)
        # and assign roles
        node_id_counter = 0
        node2coord_local: dict[NodeIdLocal, PhysCoordLocal3D] = {}
        coord2node_local: dict[PhysCoordLocal3D, NodeIdLocal] = {}
        node2role_local: dict[NodeIdLocal, str] = {}

        # Data qubits
        for x, y in data_coords_2d:
            coord_3d = (x, y, 0)
            node2coord_local[node_id_counter] = coord_3d
            coord2node_local[coord_3d] = node_id_counter
            node2role_local[node_id_counter] = "data"
            node_id_counter += 1

        # X-type stabilizers
        for x, y in x_coords_2d:
            coord_3d = (x, y, 0)
            node2coord_local[node_id_counter] = coord_3d
            coord2node_local[coord_3d] = node_id_counter
            node2role_local[node_id_counter] = "ancilla_x"
            node_id_counter += 1
