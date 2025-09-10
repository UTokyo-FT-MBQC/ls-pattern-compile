#!/usr/bin/env python3
"""New-API demo: RHG memory experiment (InitPlus -> Memory -> MeasureX).

This example builds a single-logical memory line using the current codebase structure
with RHGCanvas, TemporalLayer composition, and compilation.

Usage:
  python examples/new_rhg_memory.py
"""

# %%
import traceback

# TODO: Implement missing imports when available
# import pymatching
# import stim
# from graphix_zx.stim_compiler import stim_compile
# from graphix_zx.pattern import Pattern, print_pattern
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.mytype import PatchCoordGlobal3D, SpatialEdgeSpec
from lspattern.visualizers import visualize_compiled_canvas

# %%
# Demo 1: Create simple RHG memory canvas (InitPlus -> Memory)
d = 2

# Create canvas skeleton
skeleton = RHGCanvasSkeleton(name="Simple RHG Memory Canvas")

# Add InitPlus cube at position (0,0,0)
edgespec: SpatialEdgeSpec = {"TOP": "X", "BOTTOM": "Z", "LEFT": "X", "RIGHT": "Z"}
init_skeleton = InitPlusCubeSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

# Add memory pipe from (0,0,0) to (0,0,1) - temporal connection
memory_skeleton = MemoryPipeSkeleton(d=d, edgespec=edgespec)
skeleton.add_pipe(PatchCoordGlobal3D((0, 0, 0)), PatchCoordGlobal3D((0, 0, 1)), memory_skeleton)

# Convert skeleton to canvas
simple_canvas = skeleton.to_canvas()
print(f"Created simple canvas with {len(simple_canvas.cubes_)} cubes and {len(simple_canvas.pipes_)} pipes")

# %%
# Demo 2: Visualize the simple canvas
try:
    compiled_simple = simple_canvas.compile()
    visualize_compiled_canvas(
        compiled_simple,
        save_path="figures/simple_rhg_lattice.png",
        show=True,
    )
except (AttributeError, ImportError, NotImplementedError) as e:
    print(f"Visualization not available: {e}")

# %%
# Demo 3: Create extended memory canvas with multiple rounds
d = 5
r = 5  # number of memory rounds

skeleton = RHGCanvasSkeleton(name="Extended RHG Memory Canvas")

# Define edge specification
edgespec: SpatialEdgeSpec = {"TOP": "X", "BOTTOM": "Z", "LEFT": "X", "RIGHT": "Z"}

# Add InitPlus cube at the beginning
init_skeleton = InitPlusCubeSkeleton(d=d, edgespec=edgespec)
skeleton.add_cube(PatchCoordGlobal3D((0, 0, 0)), init_skeleton)

# Add memory rounds
for i in range(r):
    z_current = i
    z_next = i + 1

    memory_skeleton = MemoryPipeSkeleton(d=d, edgespec=edgespec)
    skeleton.add_pipe(PatchCoordGlobal3D((0, 0, z_current)), PatchCoordGlobal3D((0, 0, z_next)), memory_skeleton)

extended_canvas = skeleton.to_canvas()
print(f"Created extended canvas with {len(extended_canvas.cubes_)} cubes and {len(extended_canvas.pipes_)} pipes")

# %%
# Demo 4: Compile the extended canvas and generate pattern
print("Compiling extended canvas...")
try:
    compiled_canvas = extended_canvas.compile()

    print(f"Compiled canvas has {len(compiled_canvas.layers)} temporal layers")
    print(f"Global graph has {getattr(compiled_canvas.global_graph, 'num_qubits', 'unknown')} qubits")

    # Check schedule information
    print(f"Schedule has {len(compiled_canvas.schedule.schedule)} time slots")
    for t, nodes in compiled_canvas.schedule.schedule.items():
        print(f"Time {t}: {len(nodes)} nodes")

    # Generate pattern from compiled canvas
    if compiled_canvas.global_graph is not None:
        # Convert flow accumulator to proper format
        xflow = {}
        for src, dsts in compiled_canvas.flow.xflow.items():
            xflow[int(src)] = {int(dst) for dst in dsts}

        pattern = compile_canvas(
            compiled_canvas.global_graph,
            xflow=xflow,
            x_parity=[{int(node) for node in group} for group in compiled_canvas.parity.x_checks],
            z_parity=[{int(node) for node in group} for group in compiled_canvas.parity.z_checks],
        )
        print("Pattern compilation successful")
        # print_pattern(pattern)  # TODO: Implement when available

        # Define logical observables
        logical = set(range(d))
        print(f"Logical X: {logical}")
        logical_observables = {0: logical}

    else:
        print("No global graph available for pattern compilation")

except (ValueError, AttributeError, NotImplementedError) as e:
    print(f"Compilation failed: {e}")
    traceback.print_exc()

# %%
# Demo 5: Circuit creation placeholder (TODO: implement when stim is available)
print("\n=== Circuit Creation (Stub) ===")
print(f"Circuit creation for d={d} not implemented yet")
print("TODO: Implement when stim and pymatching are available")

# TODO: Implement when dependencies are available
# def create_circuit(pattern: Pattern, noise: float) -> stim.Circuit:
#     logical_observables = {0: {i for i in range(d)}}
#     stim_str = stim_compile(
#         pattern,
#         logical_observables,
#         after_clifford_depolarization=noise,
#         before_measure_flip_probability=noise,
#     )
#     return stim.Circuit(stim_str)

# noise = 0.001
# circuit = create_circuit(pattern, noise)
# print(f"num_qubits: {circuit.num_qubits}")

# %%
# Demo 6: Error correction simulation placeholder (TODO: implement when pymatching is available)
print("\n=== Error Correction Simulation (Stub) ===")
print("Error correction simulation not implemented yet")
print("TODO: Implement when pymatching is available")

# TODO: Implement when dependencies are available
# dem = circuit.detector_error_model(decompose_errors=True)
# print(dem)

# matching = pymatching.Matching.from_detector_error_model(dem)
# print(matching)

# err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
# print(len(err))
# print(err)

# %%
# Demo 7: Visualization export placeholder (TODO: implement when dependencies are available)
print("\n=== Visualization Export (Stub) ===")
print("Visualization export not implemented yet")
print("TODO: Implement SVG export functionality")

# TODO: Implement when dependencies are available
# svg = dem.diagram(type="match-graph-svg")
# import pathlib
# pathlib.Path("figures/new_rhg_memory_dem.svg").write_text(str(svg), encoding="utf-8")

# %%
