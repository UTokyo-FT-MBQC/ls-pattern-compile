"""Export a 3D z-sweep MP4 movie from a canvas YAML spec."""

from __future__ import annotations

from pathlib import Path

from lspattern.canvas_loader import load_canvas
from lspattern.video_3d import export_canvas_z_sweep_3d_mp4

spec_name = "design/distillation_canvas.yml"
code_distance = 5
z_window = 60
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

canvas, spec = load_canvas(spec_name, code_distance=code_distance)
out_path = output_dir / f"{Path(spec_name).stem}_z_sweep_3d.mp4"

export_canvas_z_sweep_3d_mp4(
    canvas,
    out_path,
    fps=15,
    z_window=z_window,
    width=640,
    height=640,
    node_size_scale=0.2,
    edge_width_scale=0.2,
    highlight_size_scale=1.0,
    highlight_current_layer=True,
    non_current_alpha=0.1,
    camera_eye=(1.5, 1, 1),
    lock_view=True,
    aspect_ratio=(1.5, 0.9, 0.6),
    show_progress_bar=True,
)

print(f"Exported 3D z-sweep for '{spec.name}' to {out_path}")
