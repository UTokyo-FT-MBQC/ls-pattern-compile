"""Profile canvas YAML: load, visualize, and compile to MBQC pattern.

This script loads a canvas YAML file and performs:
1. Load canvas and display statistics
2. 3D Plotly visualization (optional)
3. 2D slice visualization (optional)
4. Flow cycle check (debug)
5. Compile to Pattern and show profile
6. Export pattern to .ptn
"""

# %%
from __future__ import annotations

from pathlib import Path
from statistics import mean

import yaml
from graphqomb.command import M
from graphqomb.ptn_format import dump as dump_ptn

from lspattern.canvas_loader import load_canvas
from lspattern.compiler import compile_canvas_to_pattern
from lspattern.debug_utils import check_coord_flow_cycle
from lspattern.visualizer import visualize_canvas_plotly
from lspattern.visualizer_2d import visualize_canvas_matplotlib_2d

# =============================================================================
# Configuration - Edit these values
# =============================================================================
input_yaml = Path(__file__).parent / "design" / "distillation_canvas.yml"
code_distance = 5
output_dir = Path(__file__).parent / "output"
profile_table_config = Path(__file__).parent / "profile_table_config.yml"

# Visualization options
enable_3d_viz = False  # Set to False to skip 3D visualization
slice_z: int | None = 60  # Z coordinate for 2D slice (None to skip)
aspect_ratio: tuple[int, int, int] | None  = None  # Plotly 3D aspect ratio

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

measurement_depth = pattern.depth_of((M,))
pattern_profile = {
    "commands": len(pattern),
    "max_space": pattern.max_space,
    "depth": pattern.depth,
    "depth_m": measurement_depth,
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
# Step 5b: Generate profile summary table (Matplotlib)
import matplotlib.pyplot as plt

commands_no_tick = len(pattern) - pattern.depth
all_metrics = {
    "commands_no_tick": commands_no_tick,
    "max_space": pattern.max_space,
    "depth": pattern.depth,
    "depth_m": measurement_depth,
    "active_volume": pattern.active_volume,
    "volume": pattern.volume,
    "throughput": throughput,
    "idle_qubits": len(idle_values),
    "idle_min": min(idle_values) if idle_values else 0,
    "idle_mean": mean(idle_values) if idle_values else 0.0,
    "idle_max": max(idle_values) if idle_values else 0,
}

with open(profile_table_config) as f:
    table_cfg = yaml.safe_load(f)

metric_entries = table_cfg["metrics"]
style = table_cfg.get("style", {})

col_labels = [m["label"] for m in metric_entries]
cell_values = [f"{all_metrics[m['key']]:,}" for m in metric_entries]

figsize = style.get("figsize", [5.0, 1.5])
fig_table, ax = plt.subplots(figsize=figsize)
ax.axis("off")

tbl = ax.table(
    cellText=[cell_values],
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(style.get("fontsize", 11))
tbl.scale(1.0, style.get("row_height", 1.6))

header_bg = style.get("header_bg", "#4472C4")
header_fg = style.get("header_fg", "white")
cell_bg = style.get("cell_bg", "#D9E2F3")
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor(header_bg)
    tbl[0, j].set_text_props(color=header_fg, fontweight="bold")
    tbl[1, j].set_facecolor(cell_bg)

fig_table.suptitle(
    f"{spec.name} (d={code_distance})",
    fontsize=style.get("title_fontsize", 12),
    fontweight="bold",
)
fig_table.tight_layout()

table_path = output_dir / f"{input_yaml.stem}_profile_table.png"
fig_table.savefig(table_path, dpi=style.get("dpi", 200), bbox_inches="tight")
print(f"  Saved profile table: {table_path}")
fig_table.show()

# %%
print("Step 6: Exporting pattern to .ptn...")
ptn_path = output_dir / f"{input_yaml.stem}.ptn"
dump_ptn(pattern, ptn_path)
print(f"  M-command depth: {measurement_depth}")
print(f"  Saved pattern: {ptn_path}")
print()

# %%
print()
print("Pipeline completed successfully!")
print(f"  Output directory: {output_dir}")
