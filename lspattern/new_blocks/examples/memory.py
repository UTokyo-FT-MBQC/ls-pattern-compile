"""Build and visualize a memory canvas using packaged YAML specs."""

from __future__ import annotations

from pathlib import Path

from lspattern.new_blocks.canvas_loader import load_canvas
from lspattern.new_blocks.visualizer import visualize_canvas_plotly


def main(spec_name: str | Path = "memory_canvas.yml") -> None:
    canvas, spec = load_canvas(spec_name)
    fig = visualize_canvas_plotly(canvas)
    print(f"Loaded canvas '{spec.name}' (d={spec.code_distance}) with {len(spec.cubes)} cubes")
    fig.show()


if __name__ == "__main__":
    main()
