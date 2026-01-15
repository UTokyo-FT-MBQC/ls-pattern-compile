"""Export Canvas to graphqomb-studio JSON format demo.

This example shows how to:
1. Load a canvas from YAML specification
2. Export it to graphqomb-studio/v1 JSON format
3. Customize export with input/output node settings
"""

# %%
from __future__ import annotations

from pathlib import Path

from lspattern.canvas_loader import load_canvas
from lspattern.exporter import ExportConfig, export_to_studio, save_to_studio_json

# %%
# Load canvas from YAML specification
code_distance = 3

# Use absolute path for YAML file
examples_dir = Path(__file__).parent
spec_path = examples_dir / "design" / "ancilla_free_memory.yml"
canvas, spec = load_canvas(spec_path, code_distance=code_distance, extra_paths=[examples_dir])

print(f"Loaded canvas '{spec.name}' (d={code_distance})")
print(f"  Nodes: {len(canvas.nodes)}")
print(f"  Edges: {len(canvas.edges)}")
print(f"  Cubes: {len(spec.cubes)}")

# %%
# Export to graphqomb-studio JSON format (using defaults)
# - All nodes become intermediate nodes by default
# - To specify input/output nodes, use ExportConfig
# - zflow is set to "auto" for graphqomb-studio to compute

data = export_to_studio(canvas, spec.name)

print("\n=== Export Summary ===")
print(f"Schema: {data['$schema']}")
print(f"Name: {data['name']}")
print(f"Nodes: {len(data['nodes'])}")
print(f"Edges: {len(data['edges'])}")
print(f"Flow xflow entries: {len(data['flow']['xflow'])}")
print(f"Flow zflow: {data['flow']['zflow']}")
print(f"Schedule timeline steps: {len(data['schedule']['timeline'])}")

# %%
# Show node role distribution
roles = {"input": 0, "output": 0, "intermediate": 0}
for node in data["nodes"]:
    roles[node["role"]] += 1

print("\n=== Node Roles ===")
for role, count in roles.items():
    print(f"  {role}: {count}")

# %%
# Show output nodes (none by default, since we didn't specify any)
print("\n=== Output Nodes ===")
output_nodes = [n for n in data["nodes"] if n["role"] == "output"]
if output_nodes:
    for node in output_nodes[:5]:  # Show first 5
        coord = node["coordinate"]
        print(f"  {node['id']}: coordinate=({coord['x']}, {coord['y']}, {coord['z']})")
    if len(output_nodes) > 5:  # noqa: PLR2004
        print(f"  ... and {len(output_nodes) - 5} more")
else:
    print("  (None - all nodes are intermediate by default)")

# %%
# Show sample intermediate nodes with measurement bases
print("\n=== Sample Intermediate Nodes ===")
intermediate_nodes = [n for n in data["nodes"] if n["role"] == "intermediate"]
for node in intermediate_nodes[:3]:  # Show first 3
    basis = node["measBasis"]
    print(f"  {node['id']}: axis={basis['axis']}, sign={basis['sign']}")

# %%
# Show schedule timeline (first few steps)
print("\n=== Schedule Timeline (first 3 steps) ===")
for slice_data in data["schedule"]["timeline"][:3]:
    print(f"  Time {slice_data['time']}:")
    print(f"    Prepare: {len(slice_data['prepareNodes'])} nodes")
    print(f"    Entangle: {len(slice_data['entangleEdges'])} edges")
    print(f"    Measure: {len(slice_data['measureNodes'])} nodes")

# %%
# Save to JSON file
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"{spec.name.replace(' ', '_')}_d{code_distance}.json"

save_to_studio_json(canvas, spec.name, output_path)
print(f"\n=== Saved to {output_path} ===")

# %%
# Example with custom ExportConfig
# You can specify explicit input/output nodes if needed

print("\n=== Custom Export Example ===")

# Get first few nodes as example input/output nodes
sample_input_nodes = set(list(canvas.nodes)[:2])
sample_output_nodes = set(list(canvas.nodes)[2:4])  # Next 2 nodes as output

config = ExportConfig(
    input_nodes=sample_input_nodes,
    output_nodes=sample_output_nodes,
)

data_custom = export_to_studio(canvas, f"{spec.name}_custom", config=config)

roles_custom = {"input": 0, "output": 0, "intermediate": 0}
for node in data_custom["nodes"]:
    roles_custom[node["role"]] += 1

print("With custom config (2 input nodes, 2 output nodes):")
for role, count in roles_custom.items():
    print(f"  {role}: {count}")

# %%
