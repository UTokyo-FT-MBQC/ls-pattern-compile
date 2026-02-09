"""Export Canvas to GraphQOMB Studio JSON format from YAML spec."""

# %%
from __future__ import annotations

import json
from pathlib import Path

from lspattern.canvas_loader import load_canvas
from lspattern.export import canvas_to_graphqomb_studio_dict, export_canvas_to_graphqomb_studio

# =============================================================================
# Configuration - modify these parameters as needed
# =============================================================================
spec_name = "design/cnot.yml"
code_distance = 3
output_dir = Path("output")

# Optional coordinate range filter (closed intervals). Use None for unbounded.
x_min: int | None = None
x_max: int | None = None
y_min: int | None = None
y_max: int | None = None
z_min: int | None = None
z_max: int | None = None

# %%
# Load canvas from YAML spec
print(f"Loading canvas from '{spec_name}' (d={code_distance})...")
canvas, spec = load_canvas(spec_name, code_distance=code_distance)

# %%
# Display summary
print("\n=== Canvas Summary ===")
print(f"Name: {spec.name}")
print(f"Description: {spec.description}")
print(f"Code distance: {code_distance}")
print(f"Cubes: {len(spec.cubes)}")
print(f"Pipes: {len(spec.pipes)}")
print(f"Nodes: {len(canvas.nodes)}")
print(f"Edges: {len(canvas.edges)}")
print(f"Logical observables: {len(canvas.logical_observables)}")

# %%
# Preview JSON structure (first N lines)
PREVIEW_LINES = 100
print("\n=== GraphQOMB Studio JSON Preview ===")
result = canvas_to_graphqomb_studio_dict(
    canvas,
    name=spec.name,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    z_min=z_min,
    z_max=z_max,
)
json_str = json.dumps(result, indent=2)
all_lines = json_str.split("\n")
preview_lines = all_lines[:PREVIEW_LINES]
print("\n".join(preview_lines))
if len(all_lines) > PREVIEW_LINES:
    print(f"... ({len(all_lines)} total lines)")

# %%
# Export to JSON file
output_dir.mkdir(exist_ok=True)
spec_stem = Path(spec_name).stem
output_path = output_dir / f"{spec_stem}_d{code_distance}.json"

export_canvas_to_graphqomb_studio(
    canvas,
    output_path,
    name=spec.name,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    z_min=z_min,
    z_max=z_max,
)
print("\n=== Exported ===")
print(f"Output: {output_path}")
print(f"File size: {output_path.stat().st_size:,} bytes")

# %%
