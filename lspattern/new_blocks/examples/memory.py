"""Build and visualize a memory canvas from YAML specs.

This script manually stitches block/layer YAMLs together because the
new_blocks build helpers are not finished yet. Once a dedicated builder
lands, this loader logic should migrate there.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.canvas import Canvas, CanvasConfig
from lspattern.new_blocks.loader import BlockConfig, load_patch_layout_from_yaml
from lspattern.new_blocks.mytype import Coord3D
from lspattern.new_blocks.visualizer import visualize_canvas_plotly


REPO_ROOT = Path(__file__).resolve().parents[2]
_BLOCKS_DIR = REPO_ROOT / "lspattern" / "new_blocks" / "patch_layout" / "blocks"
_LAYERS_DIR = REPO_ROOT / "lspattern" / "new_blocks" / "patch_layout" / "layers"
_DEFAULT_CANVAS_SPEC = REPO_ROOT / "lspattern" / "new_blocks" / "examples" / "memory_canvas.yml"


def _snakeify(name: str) -> str:
    """Convert CamelCase identifiers to snake_case."""
    snake = ""
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0:
            snake += "_"
        snake += ch.lower()
    return snake


def _to_block_filename(block_name: str) -> str:
    base = _snakeify(block_name)
    if base.endswith("_block"):
        base = base[: -len("_block")]
    return f"{base}.yml"


def _to_layer_filename(layer_name: str) -> str:
    base = _snakeify(layer_name)
    if base.endswith("_unit"):
        base = base[: -len("_unit")]
    return f"{base}.yml"


def _parse_boundary(raw: dict) -> dict[BoundarySide, EdgeSpecValue]:
    boundary: dict[BoundarySide, EdgeSpecValue] = {}
    for side_str, spec_str in raw.items():
        boundary[BoundarySide(side_str.upper())] = EdgeSpecValue(spec_str.upper())
    return boundary


def _resolve_num_layers(layer_cfg: dict, distance: int, consumed_layers: int) -> int:
    """Resolve how many times to repeat a unit layer.

    Supports either a fixed count (`num_layers`) or a distance-aware count
    (`num_layers_from_distance`). The latter adds an offset to the canvas
    code distance, letting YAML stay declarative while remaining covariant
    with `d`.
    """
    if "num_layers" in layer_cfg:
        return int(layer_cfg["num_layers"])

    if "num_layers_from_distance" in layer_cfg:
        spec = layer_cfg["num_layers_from_distance"]
        if isinstance(spec, str) and spec.lower() in {"rest", "remaining", "fill"}:
            return max(distance - consumed_layers, 0)

        if isinstance(spec, dict):
            scale = int(spec.get("scale", 1))
            offset = int(spec.get("offset", 0))
        else:
            scale = 1
            offset = int(spec)
        return max(scale * distance + offset, 0)

    return int(layer_cfg.get("params", {}).get("num_layers", 1))


def _load_block_config(block_name: str, distance: int) -> BlockConfig:
    block_path = _BLOCKS_DIR / _to_block_filename(block_name)
    with block_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    boundary = _parse_boundary(cfg.get("boundary", {}))

    patch_configs = []
    consumed_layers = 0
    for layer_cfg in cfg.get("layers", []):
        layer_name = layer_cfg["type"]
        num_layers = _resolve_num_layers(layer_cfg, distance, consumed_layers)
        layer_path = _LAYERS_DIR / _to_layer_filename(layer_name)
        patch_layout = load_patch_layout_from_yaml(layer_path)
        patch_configs.extend([patch_layout] * num_layers)
        consumed_layers += num_layers

    block_config = BlockConfig(patch_configs)
    block_config.boundary = boundary  # type: ignore[attr-defined]
    return block_config


def _load_canvas(path: Path) -> tuple[Canvas, list[str]]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    canvas_cfg = CanvasConfig(
        name=cfg["name"],
        description=cfg.get("description", ""),
        d=cfg["code_distance"],
        tiling=cfg.get("layout", "rotated_surface_code"),
    )
    canvas = Canvas(canvas_cfg)

    added: list[str] = []
    for cube in cfg.get("cube", []):
        pos = cube["position"]
        block_name = cube["block"]
        block_cfg = _load_block_config(block_name, canvas.config.d)
        canvas.add_cube(Coord3D(*pos), block_cfg)
        added.append(f"{block_name}@{tuple(pos)}")

    return canvas, added


def main(yaml_path: Path = _DEFAULT_CANVAS_SPEC) -> None:
    canvas, added = _load_canvas(yaml_path)
    fig = visualize_canvas_plotly(canvas)
    print(f"Loaded canvas from {yaml_path.name}: {', '.join(added)}")
    fig.show()


if __name__ == "__main__":
    main()
