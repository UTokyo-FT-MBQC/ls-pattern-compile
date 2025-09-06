"""
Debug script for Task T26: Remove unnecessary getattr usages under lspattern/*.

This script builds a minimal TemporalLayer with two adjacent cubes connected by a
pipe, compiles the layer, and prints basic summaries to verify code paths that
previously relied on getattr now work via direct attribute access.
"""

from __future__ import annotations

from lspattern.blocks.cubes.base import RHGCube
from lspattern.blocks.pipes.base import RHGPipe
from lspattern.canvas import TemporalLayer
from lspattern.mytype import SpatialEdgeSpec
from lspattern.tiling.template import (
    RotatedPlanarCubeTemplate,
    RotatedPlanarPipetemplate,
)


def main() -> None:
    # Simple all-open boundaries for cubes
    spec: SpatialEdgeSpec = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}

    # Two cubes at (0,0,0) and (1,0,0)
    c1 = RHGCube(d=3, edge_spec=spec, template=RotatedPlanarCubeTemplate(d=3, edgespec=spec))
    c2 = RHGCube(d=3, edge_spec=spec, template=RotatedPlanarCubeTemplate(d=3, edgespec=spec))

    # A spatial pipe connecting them (source->sink on +X)
    p = RHGPipe(
        d=3,
        edge_spec={"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"},
        source=(0, 0, 0),
        sink=(1, 0, 0),
        template=RotatedPlanarPipetemplate(d=3, edgespec={"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}),
    )

    # Build and compile single layer z=0
    layer = TemporalLayer(z=0)
    layer.add_cube((0, 0, 0), c1.materialize())
    layer.add_cube((1, 0, 0), c2.materialize())
    layer.add_pipe((0, 0, 0), (1, 0, 0), p.materialize())
    layer.compile()

    # Summaries
    print("=== TemporalLayer Compile Summary ===")
    print(f"z: {layer.z}")
    print(f"qubit_count: {layer.qubit_count}")
    print(f"node2coord size: {len(layer.node2coord)}")
    print(f"graph nodes: {len(layer.local_graph.physical_nodes) if layer.local_graph else 0}")
    print(f"graph edges: {len(layer.local_graph.physical_edges) if layer.local_graph else 0}")

    # Quick access to ensure direct-attribute fields are present
    assert isinstance(layer.tiling_node_maps, dict)
    assert isinstance(layer.node2coord, dict)
    assert layer.local_graph is not None


if __name__ == "__main__":
    main()

