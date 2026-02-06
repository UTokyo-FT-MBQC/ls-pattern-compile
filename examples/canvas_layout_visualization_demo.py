"""Demo script for visualize_canvas_layout function.

This script demonstrates how to visualize YAML canvas layouts at specific z-layers.
Can be run as a script or interactively with Jupyter/VS Code cell execution.
"""

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from lspattern.visualizer_2d import visualize_canvas_layout

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
# Visualize the specified z-layer
fig = visualize_canvas_layout(yaml_path, code_distance=code_distance, target_z=target_z)
print(f"Canvas: {spec_name}, d={code_distance}, z={target_z}")
plt.show()

# %%
