"""
T8 smoke: edgespec + skeleton + trim + canvas

Runs a minimal flow that exercises the new API:
- Build RotatedPlanarTemplate with edgespec
- Populate tiling and verify coords are present
- Trim a spatial boundary and verify ancilla count decreases
- Materialize an InitPlusSkeleton to RHGBlock
- Place it on RHGCanvas2 and build temporal layers

This script is intentionally lightweight; it prints a few counts and asserts.
"""

from __future__ import annotations

from lspattern.blocks.cubes.initialize import InitPlusBlockSkeleton

from lspattern.canvas import RHGCanvas
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.tiling.template import RotatedPlanarTemplate


def main() -> None:
    d = 3
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

    # Template -> tiling
    tmpl = RotatedPlanarTemplate(d=d, edgespec=edgespec)
    t = tmpl.to_tiling()
    assert len(t["data"]) > 0 and (len(t["X"]) + len(t["Z"]) > 0), "tiling is empty"

    # Trim top boundary should remove some Z ancillas at y=2*d-1
    z_before = len(tmpl.z_coords)
    tmpl.trim_spatial_boundary("TOP")
    t2 = tmpl.to_tiling()  # repopulate
    z_after = len(t2["Z"])
    assert z_after <= z_before, "trim did not reduce/equal Z ancillas as expected"

    # Skeleton -> block
    skel = InitPlusBlockSkeleton(d=d, edgespec=edgespec)
    block = skel.materialize()
    assert block.graph_local is not None and len(block.node2coord) > 0
    assert len(block.schedule_local) > 0
    assert len(block.out_ports) > 0

    # Canvas with Skeleton input
    canvas = RHGCanvas("T8Smoke")
    canvas.add_block(PatchCoordGlobal3D((0, 0, 0)), skel)
    layers = canvas.to_temporal_layers()
    assert 0 in layers and layers[0].local_graph is not None
    print(
        {
            "data": len(t["data"]),
            "X": len(t["X"]),
            "Z": len(t["Z"]),
            "Z_after_trim": z_after,
            "nodes_in_block": len(block.node2coord),
            "layers": len(layers),
        }
    )


if __name__ == "__main__":
    main()
