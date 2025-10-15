"""Demonstration of user customization of LayeredRHGCube to add a measure and init in a single cube."""

# %%
import pathlib
from typing import ClassVar

from lspattern.blocks.cubes.base import RHGCubeSkeleton
from lspattern.blocks.cubes.layered import LayeredRHGCube
from lspattern.blocks.layers import EmptyUnitLayer, InitPlusUnitLayer, MeasureXUnitLayer
from lspattern.blocks.unit_layer import UnitLayer
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue, TimeBoundarySpecValue
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly


class MeasureAndInitCube(LayeredRHGCube):
    name: ClassVar[str] = "MeasureAndInitCube"

    def set_in_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory: assign all data qubits as input ports."""
        idx_map = self.template.get_data_indices_cube(patch_coord)
        indices = set(idx_map.values())
        if len(indices) == 0:
            msg = "MeasureAndInitCube: in_ports should not be empty."
            raise AssertionError(msg)
        self.in_ports = indices

    def set_out_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory: assign all data qubits as output ports."""
        idx_map = self.template.get_data_indices_cube(patch_coord)
        self.out_ports = set(idx_map.values())

    def set_cout_ports(self, patch_coord: tuple[int, int] | None = None) -> None:
        """Memory does not have classical output ports."""
        return super().set_cout_ports(patch_coord)

    def _construct_detectors(self) -> None:
        """Detectors are already constructed by unit layers via parity accumulator."""
        # The parity accumulator is already populated during layer construction
        # No additional detector construction needed


class MeasureAndInitCubeSkeleton(RHGCubeSkeleton):
    name: ClassVar[str] = "MeasureAndInitCubeSkeleton"

    def to_block(self) -> MeasureAndInitCube:
        # Apply spatial open-boundary trimming if specified
        for direction in (BoundarySide.LEFT, BoundarySide.RIGHT, BoundarySide.TOP, BoundarySide.BOTTOM):
            if self.edgespec.get(direction, EdgeSpecValue.O) == EdgeSpecValue.O:
                self.trim_spatial_boundary(direction)

        # Evaluate template coordinates
        self.template.to_tiling()

        # Create sequence of unit layers
        unit_layers: list[UnitLayer] = []
        unit_layers.append(MeasureXUnitLayer())
        for _ in range(self.d - 2):
            unit_layers.append(EmptyUnitLayer())
        unit_layers.append(InitPlusUnitLayer())

        # Validate unit_layers length
        if len(unit_layers) > self.d:
            msg = f"Unit layers length ({len(unit_layers)}) cannot exceed code distance d ({self.d})"
            raise ValueError(msg)

        block = MeasureAndInitCube(
            d=self.d,
            edge_spec=self.edgespec,
            template=self.template,
            unit_layers=unit_layers,
        )
        block.final_layer = TimeBoundarySpecValue.O
        return block


# %%
d = 3

# Create canvas skeleton
skeleton = RHGCanvasSkeleton(name=f"MeasureAndInitCube Demo (d={d})")

# Define edge specification
edgespec: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
}

# Add MeasureAndInitCube at position (0, 0, 0)
measure_init_skeleton = MeasureAndInitCubeSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), measure_init_skeleton)

# Convert skeleton to canvas
canvas = skeleton.to_canvas()
print(f"Created canvas with {len(canvas.cubes_)} cubes and {len(canvas.pipes_)} pipes")

# Compile the canvas
compiled_canvas = canvas.compile()
print(f"\nCompiled canvas has {len(compiled_canvas.layers)} temporal layers")
if compiled_canvas.global_graph:
    num_qubits = len(compiled_canvas.global_graph.physical_nodes)
    print(f"Global graph has {num_qubits} qubits")

# Print schedule information
schedule = compiled_canvas.schedule.compact()
print(f"\nSchedule has {len(schedule.schedule)} time slots")
for t, nodes in schedule.schedule.items():
    print(f"Time {t}: {len(nodes)} nodes")

# Print flow information
print(f"\nFlow has {len(compiled_canvas.flow.flow)} entries")

# Print parity information
num_parity_checks = sum(len(group_dict) for group_dict in compiled_canvas.parity.checks.values())
print(f"Parity checks: {num_parity_checks} checks at {len(compiled_canvas.parity.checks)} coordinates")

# Visualize the compiled canvas
fig = visualize_compiled_canvas_plotly(compiled_canvas, width=1000, height=800)
title = f"MeasureAndInitCube Visualization (d={d})"
fig.update_layout(title=title)

# Save to HTML file
pathlib.Path("figures").mkdir(exist_ok=True)
output_path = "figures/measure_and_init_cube_plotly.html"
fig.write_html(output_path)
fig.show()
print(f"\nVisualization saved to {output_path}")

print("\n" + "=" * 80)
print(f"MeasureAndInitCube Demo completed (d={d})")
print(f"Structure: [MeasureX *1, Empty *{d - 2}, InitPlus *1]")
print(f"Total temporal layers: {len(compiled_canvas.layers)}")
print(f"Total qubits: {num_qubits if compiled_canvas.global_graph else 'unknown'}")
print("=" * 80)
