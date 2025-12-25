"""YAML loaders for canvas/block/layer specs (new_blocks).

This module keeps lookup details (package resources vs. user dirs) inside
the library so callers can build canvases by referring to logical names
only, e.g., ``load_canvas("memory_canvas")``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

import yaml

if TYPE_CHECKING:
    from importlib.abc import Traversable

from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.new_blocks.canvas import Canvas, CanvasConfig
from lspattern.new_blocks.layout.rotated_surface_code import RotatedSurfaceCodeLayoutBuilder
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


@dataclass(frozen=True, slots=True)
class CompositeLogicalObservableSpec:
    """Represents a logical observable spanning multiple cubes/pipes."""

    cubes: tuple[Coord3D, ...]
    pipes: tuple[tuple[Coord3D, Coord3D], ...]


def _normalize_paths(paths: Sequence[Path | str]) -> tuple[Path, ...]:
    normalized: list[Path] = []
    for path in paths:
        path_obj = Path(path)
        if path_obj not in normalized:
            normalized.append(path_obj)
    return tuple(normalized)


def _augment_search_paths(source: Traversable | Path, extra_paths: Sequence[Path | str]) -> tuple[Path, ...]:
    """Return search paths with the source directory (if any) prepended."""

    search_paths = list(_normalize_paths(extra_paths))
    if isinstance(source, Path):
        parent = source.parent
        if parent not in search_paths:
            search_paths.insert(0, parent)
    return tuple(search_paths)


def _merge_search_paths(*groups: Sequence[Path | str]) -> tuple[Path, ...]:
    merged: list[Path] = []
    for group in groups:
        for path in group:
            path_obj = Path(path)
            if path_obj not in merged:
                merged.append(path_obj)
    return tuple(merged)


@dataclass
class CanvasCubeSpec:
    position: Coord3D
    block: str
    boundary: dict[BoundarySide, EdgeSpecValue]
    logical_observable: LogicalObservableSpec | None


@dataclass
class CanvasPipeSpec:
    start: Coord3D
    end: Coord3D
    block: str
    boundary: dict[BoundarySide, EdgeSpecValue]
    logical_observable: LogicalObservableSpec | None


@dataclass
class CanvasSpec:
    name: str
    description: str
    layout: str
    cubes: list[CanvasCubeSpec]
    pipes: list[CanvasPipeSpec]
    search_paths: tuple[Path, ...] = ()
    logical_observables: tuple[CompositeLogicalObservableSpec, ...] = ()


_SNAKE_CAMEL_RE_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_SNAKE_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def _snakeify(name: str) -> str:
    snake = _SNAKE_CAMEL_RE_1.sub(r"\1_\2", name)
    snake = _SNAKE_CAMEL_RE_2.sub(r"\1_\2", snake)
    return snake.replace("-", "_").lower()


def _candidate_filenames(name: str) -> list[str]:  # noqa: C901
    path = Path(name)
    stem = path.stem
    ext = path.suffix
    base_candidates: set[str] = set()

    def _add_candidate(value: str) -> None:
        if value and value not in base_candidates:
            base_candidates.add(value)

    _add_candidate(stem)
    _add_candidate(_snakeify(stem))

    suffixes = ("_block", "block", "_unit", "unit")
    queue = list(base_candidates)
    while queue:
        candidate = queue.pop()
        snake = _snakeify(candidate)
        if snake not in base_candidates:
            base_candidates.add(snake)
            queue.append(snake)

        lower_candidate = candidate.lower()
        for suffix in suffixes:
            if lower_candidate.endswith(suffix) and len(candidate) > len(suffix):
                trimmed = candidate[: -len(suffix)]
                if trimmed and trimmed not in base_candidates:
                    base_candidates.add(trimmed)
                    queue.append(trimmed)

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

    msg = (
        f"YAML '{name}' not found in {list(_iter_search_paths(extra_paths))} "
        f"or packaged {_RESOURCE_PACKAGES.get(kind, ())}"
    )
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
        if len(cleaned) != 4:  # noqa: PLR2004
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
        if len(token) == 2 and all(ch in {"T", "B", "L", "R"} for ch in token):  # noqa: PLR2004
            if token[0] == token[1]:
                msg = f"Logical observable sides must be distinct; got duplicate '{token[0]}'."
                raise ValueError(msg)
            return LogicalObservableSpec(token=token)
        msg = f"Logical observable must be 'X', 'Z', or two distinct chars from T/B/L/R. Got: {value}"
        raise ValueError(msg)
    msg = f"Unsupported logical_observables spec: {value!r}"
    raise TypeError(msg)


def _parse_composite_logical_observables(
    spec: object | None,
) -> tuple[CompositeLogicalObservableSpec, ...]:
    """Parse top-level logical_observables section.

    Parameters
    ----------
    spec : object | None
        The raw YAML value for the logical_observables section.

    Returns
    -------
    tuple[CompositeLogicalObservableSpec, ...]
        Parsed composite logical observable specifications.
    """
    if spec is None:
        return ()
    if not isinstance(spec, Sequence) or isinstance(spec, (str, bytes)):
        msg = f"logical_observables must be a list, got: {type(spec)}"
        raise TypeError(msg)

    result: list[CompositeLogicalObservableSpec] = []
    for entry in spec:
        if not isinstance(entry, Mapping):
            msg = f"Each logical_observables entry must be a mapping, got: {type(entry)}"
            raise TypeError(msg)

        raw_cubes = entry.get("cube", [])
        raw_pipes = entry.get("pipe", [])

        cubes = tuple(starmap(Coord3D, raw_cubes))
        pipes = tuple((Coord3D(*p[0]), Coord3D(*p[1])) for p in raw_pipes)

        result.append(CompositeLogicalObservableSpec(cubes=cubes, pipes=pipes))

    return tuple(result)


def _resolve_num_layers(layer_cfg: Mapping[str, object], distance: int, consumed_layers: int) -> int:
    if "num_layers" in layer_cfg:
        return int(cast("SupportsInt", layer_cfg["num_layers"]))

    if "num_layers_from_distance" in layer_cfg:
        spec = layer_cfg["num_layers_from_distance"]
        if isinstance(spec, str) and spec.lower() in {"rest", "remaining", "fill"}:
            return max(distance - consumed_layers, 0)
        if isinstance(spec, Mapping):
            scale = int(spec.get("scale", 1))
            offset = int(spec.get("offset", 0))
            return max(scale * distance + offset, 0)
        return max(int(cast("SupportsInt", spec)), 0)

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
    search_paths = _augment_search_paths(traversable, extra_paths)
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
        patch_layout = _load_layer_config(layer_name, extra_paths=search_paths)
        patch_configs.extend([patch_layout] * num_layers)
        consumed += num_layers

    block_config = BlockConfig(patch_configs)
    block_config.boundary = boundary
    return block_config


def load_canvas_spec(name: str | Path, *, extra_paths: Sequence[Path | str] = ()) -> CanvasSpec:
    traversable = _resolve_yaml("canvas", name, extra_paths)
    search_paths = _augment_search_paths(traversable, extra_paths)
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

    raw_pipes = cfg.get("pipe")
    if raw_pipes is None:
        pipe_cfgs: list[dict[str, object]] = []
    elif isinstance(raw_pipes, Sequence) and not isinstance(raw_pipes, (str, bytes)):
        pipe_cfgs = list(raw_pipes)
    else:
        msg = f"Pipe spec must be a sequence, got: {type(raw_pipes)}"
        raise TypeError(msg)

    pipes: list[CanvasPipeSpec] = []
    for pipe_cfg in pipe_cfgs:
        if not isinstance(pipe_cfg, Mapping):
            msg = f"Each pipe entry must be a mapping, got: {type(pipe_cfg)}"
            raise TypeError(msg)
        start = Coord3D(*cast("Sequence[int]", pipe_cfg["start"]))
        end = Coord3D(*cast("Sequence[int]", pipe_cfg["end"]))
        block = cast("str", pipe_cfg["block"])
        boundary = _parse_boundary(pipe_cfg.get("boundary"), _DEFAULT_BOUNDARY)
        logical = _parse_logical_observable(pipe_cfg.get("logical_observables"))
        pipes.append(
            CanvasPipeSpec(
                start=start,
                end=end,
                block=block,
                boundary=boundary,
                logical_observable=logical,
            )
        )

    logical_obs = _parse_composite_logical_observables(cfg.get("logical_observables"))

    return CanvasSpec(
        name=cfg["name"],
        description=cfg.get("description", ""),
        layout=cfg.get("layout", "rotated_surface_code"),
        cubes=cubes,
        pipes=pipes,
        search_paths=search_paths,
        logical_observables=logical_obs,
    )


def build_canvas(spec: CanvasSpec, *, code_distance: int, extra_paths: Sequence[Path | str] = ()) -> Canvas:
    canvas = Canvas(CanvasConfig(spec.name, spec.description, code_distance, spec.layout))
    search_paths = _merge_search_paths(spec.search_paths, extra_paths)

    for cube in spec.cubes:
        block_config = load_block_config_from_name(
            cube.block,
            code_distance=code_distance,
            extra_paths=search_paths,
            boundary_override=cube.boundary,
        )
        canvas.add_cube(cube.position, block_config, cube.logical_observable)

    for pipe in spec.pipes:
        block_config = load_block_config_from_name(
            pipe.block,
            code_distance=code_distance,
            extra_paths=search_paths,
            boundary_override=pipe.boundary,
        )
        canvas.add_pipe((pipe.start, pipe.end), block_config)

        if pipe.logical_observable is not None:
            pipe_coords = RotatedSurfaceCodeLayoutBuilder.pipe(code_distance, pipe.start, pipe.end, pipe.boundary)
            canvas.compute_pipe_cout_from_logical_observable(
                (pipe.start, pipe.end),
                block_config,
                pipe.logical_observable,
                pipe_coords.ancilla_x,
                pipe_coords.ancilla_z,
            )

    canvas.logical_observables = spec.logical_observables
    return canvas


def load_canvas(
    name: str | Path, *, code_distance: int, extra_paths: Sequence[Path | str] = ()
) -> tuple[Canvas, CanvasSpec]:
    spec = load_canvas_spec(name, extra_paths=extra_paths)
    canvas = build_canvas(spec, code_distance=code_distance, extra_paths=extra_paths)
    return canvas, spec
