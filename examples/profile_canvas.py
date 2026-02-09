"""Profile canvas YAML: load, visualize, and compile to MBQC pattern.

This script loads a canvas YAML file and performs:
1. Load canvas and display statistics
2. 3D Plotly visualization (optional)
3. 2D slice visualization (optional)
4. Flow cycle check (debug)
5. Compile to Pattern and show profile
"""

# %%
from __future__ import annotations

from pathlib import Path
from statistics import mean

from lspattern.canvas_loader import load_canvas
from lspattern.compiler import compile_canvas_to_pattern
from lspattern.debug_utils import check_coord_flow_cycle
from lspattern.visualizer import visualize_canvas_plotly
from lspattern.visualizer_2d import visualize_canvas_matplotlib_2d

# =============================================================================
# Configuration - Edit these values
# =============================================================================
input_yaml = Path(__file__).parent / "design" / "distillation_canvas.yml"
code_distance = 3
output_dir = Path(__file__).parent / "output"

# Visualization options
enable_3d_viz = True  # Set to False to skip 3D visualization
slice_z: int | None = 60  # Z coordinate for 2D slice (None to skip)
aspect_ratio: tuple[int, int, int] | None  = None  # Plotly 3D aspect ratio

# =============================================================================

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input YAML: {input_yaml}")
print(f"Code distance: {code_distance}")
print(f"Output directory: {output_dir}")
print()

# %%
# Step 1: Load canvas from YAML
print("Step 1: Loading canvas from YAML...")
canvas, spec = load_canvas(input_yaml, code_distance=code_distance)

num_nodes = len(canvas.nodes)
num_edges = len(canvas.edges)
print(f"  Canvas name: {spec.name}")
print(f"  Nodes: {num_nodes:,}")
print(f"  Edges: {num_edges:,}")
print()

# %%
# Step 2: Visualizations (optional)
if enable_3d_viz:
    print("Step 2: Generating 3D Plotly visualization...")
    fig_3d = visualize_canvas_plotly(canvas, aspect_ratio=aspect_ratio)
    fig_3d.update_layout(title=f"3D Graph: {spec.name} (d={code_distance})")

    # Save as HTML for later viewing
    html_path = output_dir / f"{input_yaml.stem}_3d.html"
    fig_3d.write_html(str(html_path))
    print(f"  Saved 3D view: {html_path}")

    # Show interactive 3D view
    fig_3d.show()

# %%
if slice_z is not None:
    print(f"Step 2b: Creating 2D slice visualization at Z={slice_z}...")
    available_z = sorted({node.z for node in canvas.nodes})
    if slice_z not in available_z:
        msg = f"Requested slice_z={slice_z} is not available. Available z values: {available_z}"
        raise ValueError(msg)

    fig_slice = visualize_canvas_matplotlib_2d(
        canvas,
        target_z=slice_z,
        reverse_axes=True,
    )
    fig_slice.suptitle(f"2D Slice: {spec.name} (d={code_distance}, z={slice_z})")
    slice_path = output_dir / f"{input_yaml.stem}_z{slice_z}_slice.png"
    fig_slice.savefig(slice_path, dpi=200, bbox_inches="tight")
    print(f"  Saved 2D slice: {slice_path}")
    fig_slice.show()

# %%
# Step 3: Check for flow cycles (debug)
print()
print("Step 3: Checking flow graph for cycles...")
try:
    check_coord_flow_cycle(canvas.flow.flow, canvas.edges)
    print("  No cycles detected in flow graph.")
except ValueError as e:
    print(f"  ERROR: {e}")
    print()
    print("Debug info:")
    print(f"  Total flow entries: {len(canvas.flow.flow)}")
    print(f"  Total edges: {len(canvas.edges)}")
    raise

# %%
# Step 4: Compile to Pattern
print()
print("Step 4: Compiling canvas to MBQC pattern...")
pattern, graph, node_map = compile_canvas_to_pattern(canvas)

num_commands = len(pattern)
print(f"  Pattern commands: {num_commands:,}")
print(f"  Graph nodes: {len(graph.physical_nodes):,}")
print(f"  Node map entries: {len(node_map):,}")
print()

# %%
# Step 5: Profile pattern
idle_values = list(pattern.idle_times.values())
try:
    throughput = pattern.throughput
except ValueError:
    throughput = None

pattern_profile = {
    "commands": len(pattern),
    "max_space": pattern.max_space,
    "depth": pattern.depth,
    "active_volume": pattern.active_volume,
    "volume": pattern.volume,
    "throughput": throughput,
    "idle_qubits": len(idle_values),
    "idle_min": min(idle_values) if idle_values else 0,
    "idle_mean": mean(idle_values) if idle_values else 0.0,
    "idle_max": max(idle_values) if idle_values else 0,
}

print("Pattern profile:")
for key, value in pattern_profile.items():
    print(f"  {key}: {value}")
print()

# %%
print()
print("Pipeline completed successfully!")
print(f"  Output directory: {output_dir}")
