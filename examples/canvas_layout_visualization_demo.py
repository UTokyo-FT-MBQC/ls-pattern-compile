"""Demo script for visualize_canvas_layout function.

This script demonstrates how to visualize YAML canvas layouts at specific z-layers.
Can be run as a script or interactively with Jupyter/VS Code cell execution.
"""

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from lspattern.canvas_loader import load_canvas_spec
from lspattern.visualizer_2d import (
    # NOTE: These private functions (_*) are used for debugging purposes only.
    # They are internal APIs and may change without notice.
    _collect_blocks_at_z,
    _generate_cube_coordinates,
    _generate_pipe_coordinates,
    _merge_patch_coordinates,
    visualize_canvas_layout,
)

# =============================================================================
# Configuration - Edit these values
# =============================================================================
spec_name = "design/L_patch.yml"
code_distance = 3
target_z = 1

# =============================================================================

EXAMPLES_DIR = Path(__file__).parent
yaml_path = EXAMPLES_DIR / spec_name

# %%
# Count qubits by role
spec = load_canvas_spec(yaml_path)
cubes_at_z, pipes_at_z = _collect_blocks_at_z(spec.cubes, spec.pipes, target_z)
all_coords = []
for cube in cubes_at_z:
    all_coords.append(_generate_cube_coordinates(cube, code_distance))
for pipe in pipes_at_z:
    all_coords.append(_generate_pipe_coordinates(pipe, code_distance))
merged = _merge_patch_coordinates(all_coords)

print(f"Canvas: {spec_name}, d={code_distance}, z={target_z}")
print(f"  Data qubits:  {len(merged.data)}")
print(f"  X ancillas:   {len(merged.ancilla_x)}")
print(f"  Z ancillas:   {len(merged.ancilla_z)}")
print(f"  Total:        {len(merged.data) + len(merged.ancilla_x) + len(merged.ancilla_z)}")

# %%
# Visualize the specified z-layer
fig = visualize_canvas_layout(yaml_path, code_distance=code_distance, target_z=target_z)
plt.show()

# %%
