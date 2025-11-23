"""YAML loaders for canvas/block/layer specs (new_blocks).

This module keeps lookup details (package resources vs. user dirs) inside
the library so callers can build canvases by referring to logical names
only, e.g., ``load_canvas("memory_canvas")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

import yaml

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.canvas import Canvas, CanvasConfig
from lspattern.new_blocks.loader import BlockConfig, PatchLayoutConfig, load_patch_layout_from_yaml
from lspattern.new_blocks.mytype import Coord3D


_DEFAULT_BOUNDARY: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
}

_RESOURCE_PACKAGES = {
    "canvas": ("lspattern.new_blocks.examples", "lspattern.new_blocks.patch_layout.canvas"),
    "blocks": ("lspattern.new_blocks.patch_layout.blocks",),
    "layers": ("lspattern.new_blocks.patch_layout.layers",),
}


@dataclass(frozen=True, slots=True)
class LogicalObservableSpec:
    token: str


@dataclass
class CanvasCubeSpec:
    position: Coord3D
    block: str
    boundary: dict[BoundarySide, EdgeSpecValue]
    logical_observable: LogicalObservableSpec | None


@dataclass
class CanvasSpec:
    name: str
    description: str
    code_distance: int
    layout: str
    cubes: list[CanvasCubeSpec]


def _snakeify(name: str) -> str:
    snake = ""
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0:
            snake += "_"
        snake += ch.lower()
    return snake


def _candidate_filenames(name: str) -> list[str]:
    path = Path(name)
    stem = path.stem
    ext = path.suffix
    base_candidates = {stem, _snakeify(stem)}
    lower_stem = stem.lower()
    if lower_stem.endswith("_block"):
        base_candidates.add(lower_stem[: -len("_block")])
    if lower_stem.endswith("_unit"):
        base_candidates.add(lower_stem[: -len("_unit")])

    exts = [ext] if ext else [".yml", ".yaml"]

    candidates: list[str] = []
    for base in base_candidates:
        for suffix in exts:
            candidate = f"{base}{suffix}"
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _iter_search_paths(paths: Iterable[Path | str]) -> Iterable[Path]:
    for p in paths:
        yield Path(p)


def _resolve_yaml(kind: str, name: str | Path, extra_paths: Sequence[Path | str]) -> Traversable | Path:
    """Resolve YAML by name from user dirs first, then packaged resources."""

    # Explicit path
    candidate_path = Path(name)
    if candidate_path.is_file():
        return candidate_path

    candidates = _candidate_filenames(str(name))

    # User-provided search paths
    for root in _iter_search_paths(extra_paths):
        for candidate in candidates:
            path = root / candidate
            if path.is_file():
                return path

    # Packaged resources
    for pkg in _RESOURCE_PACKAGES.get(kind, ()):
        for candidate in candidates:
            traversable = resources.files(pkg).joinpath(candidate)
            if traversable.is_file():
                return traversable

    msg = f"YAML '{name}' not found in {list(_iter_search_paths(extra_paths))} or packaged {_RESOURCE_PACKAGES.get(kind, ())}"
    raise FileNotFoundError(msg)


def _parse_edge_spec(value: object) -> EdgeSpecValue:
    if isinstance(value, EdgeSpecValue):
        return value
    return EdgeSpecValue(str(value).upper())


def _parse_boundary(
    spec: object | None, fallback: Mapping[BoundarySide, EdgeSpecValue]
) -> dict[BoundarySide, EdgeSpecValue]:
    if spec is None:
        return dict(fallback)

    # String form e.g., "XXZZ" (T, B, L, R)
    if isinstance(spec, str):
        cleaned = spec.strip().replace(" ", "")
        if len(cleaned) != 4:
            msg = f"Boundary string must have 4 chars (T,B,L,R order), got: {spec}"
            raise ValueError(msg)
        return {
            BoundarySide.TOP: _parse_edge_spec(cleaned[0]),
            BoundarySide.BOTTOM: _parse_edge_spec(cleaned[1]),
            BoundarySide.LEFT: _parse_edge_spec(cleaned[2]),
            BoundarySide.RIGHT: _parse_edge_spec(cleaned[3]),
        }

    # Mapping form
    if isinstance(spec, Mapping):
        result: dict[BoundarySide, EdgeSpecValue] = dict(fallback)
        for side, val in spec.items():
            side_enum = BoundarySide(side.upper()) if not isinstance(side, BoundarySide) else side
            result[side_enum] = _parse_edge_spec(val)
        return result

    msg = f"Unsupported boundary spec: {spec!r}"
    raise TypeError(msg)


def _parse_logical_observable(value: object | None) -> LogicalObservableSpec | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "none", "null"}:
            return None
        token = cleaned.upper().replace("-", "").replace("_", "").replace(" ", "")
        if len(token) == 1 and token in {"X", "Z"}:
            return LogicalObservableSpec(token=token)
        if len(token) == 2 and all(ch in {"T", "B", "L", "R"} for ch in token):
            if token[0] == token[1]:
                msg = f"Logical observable sides must be distinct; got duplicate '{token[0]}'."
                raise ValueError(msg)
            return LogicalObservableSpec(token=token)
        msg = f"Logical observable must be 'X', 'Z', or two distinct chars from T/B/L/R. Got: {value}"
        raise ValueError(msg)
    msg = f"Unsupported logical_observables spec: {value!r}"
    raise TypeError(msg)


def _resolve_num_layers(layer_cfg: Mapping[str, object], distance: int, consumed_layers: int) -> int:
    if "num_layers" in layer_cfg:
        return int(layer_cfg["num_layers"])

    if "num_layers_from_distance" in layer_cfg:
        spec = layer_cfg["num_layers_from_distance"]
        if isinstance(spec, str) and spec.lower() in {"rest", "remaining", "fill"}:
            return max(distance - consumed_layers, 0)
        if isinstance(spec, Mapping):
            scale = int(spec.get("scale", 1))
            offset = int(spec.get("offset", 0))
            return max(scale * distance + offset, 0)
        return max(int(spec), 0)

    params = layer_cfg.get("params", {})
    if isinstance(params, Mapping) and "num_layers" in params:
        return int(params["num_layers"])

    return 1


def load_layer_config_from_name(name: str | Path, *, extra_paths: Sequence[Path | str] = ()) -> PatchLayoutConfig:
    traversable = _resolve_yaml("layers", name, extra_paths)
    if isinstance(traversable, Path):
        return load_patch_layout_from_yaml(traversable)
    with resources.as_file(traversable) as tmp_path:
        return load_patch_layout_from_yaml(tmp_path)


def _load_layer_config(name: str | Path, *, extra_paths: Sequence[Path | str]) -> PatchLayoutConfig:
    return load_layer_config_from_name(name, extra_paths=extra_paths)


def load_block_config_from_name(
    name: str | Path,
    *,
    code_distance: int,
    extra_paths: Sequence[Path | str] = (),
    boundary_override: Mapping[BoundarySide, EdgeSpecValue] | None = None,
) -> BlockConfig:
    traversable = _resolve_yaml("blocks", name, extra_paths)
    with traversable.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    boundary = _parse_boundary(cfg.get("boundary"), _DEFAULT_BOUNDARY)
    if boundary_override is not None:
        boundary = dict(boundary_override)

    patch_configs = []
    consumed = 0
    for layer_cfg in cfg.get("layers", []):
        layer_name = layer_cfg["type"]
        num_layers = _resolve_num_layers(layer_cfg, code_distance, consumed)
        patch_layout = _load_layer_config(layer_name, extra_paths=extra_paths)
        patch_configs.extend([patch_layout] * num_layers)
        consumed += num_layers

    block_config = BlockConfig(patch_configs)
    block_config.boundary = boundary  # type: ignore[attr-defined]
    return block_config


def load_canvas_spec(name: str | Path, *, extra_paths: Sequence[Path | str] = ()) -> CanvasSpec:
    traversable = _resolve_yaml("canvas", name, extra_paths)
    with traversable.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cubes: list[CanvasCubeSpec] = []
    for cube_cfg in cfg.get("cube", []):
        pos = Coord3D(*cube_cfg["position"])
        block = cube_cfg["block"]
        boundary = _parse_boundary(cube_cfg.get("boundary"), _DEFAULT_BOUNDARY)
        logical = _parse_logical_observable(cube_cfg.get("logical_observables"))
        cubes.append(
            CanvasCubeSpec(
                position=pos,
                block=block,
                boundary=boundary,
                logical_observable=logical,
            )
        )

    return CanvasSpec(
        name=cfg["name"],
        description=cfg.get("description", ""),
        code_distance=int(cfg["code_distance"]),
        layout=cfg.get("layout", "rotated_surface_code"),
        cubes=cubes,
    )


def build_canvas(spec: CanvasSpec, *, extra_paths: Sequence[Path | str] = ()) -> Canvas:
    canvas = Canvas(CanvasConfig(spec.name, spec.description, spec.code_distance, spec.layout))

    for cube in spec.cubes:
        block_config = load_block_config_from_name(
            cube.block,
            code_distance=spec.code_distance,
            extra_paths=extra_paths,
            boundary_override=cube.boundary,
        )
        canvas.add_cube(cube.position, block_config)

    return canvas


def load_canvas(name: str | Path, *, extra_paths: Sequence[Path | str] = ()) -> tuple[Canvas, CanvasSpec]:
    spec = load_canvas_spec(name, extra_paths=extra_paths)
    canvas = build_canvas(spec, extra_paths=extra_paths)
    return canvas, spec
