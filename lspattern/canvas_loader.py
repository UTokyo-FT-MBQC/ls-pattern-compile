"""YAML loaders for canvas/block/layer specs.

This module keeps lookup details (package resources vs. user dirs) inside
the library so callers can build canvases by referring to logical names
only, e.g., ``load_canvas("memory_canvas")``.
"""

from __future__ import annotations

import re
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

import yaml
from graphqomb.common import Axis

if TYPE_CHECKING:
    from importlib.abc import Traversable

from lspattern.canvas import Canvas, CanvasConfig
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.fragment import GraphSpec
from lspattern.loader import BlockConfig, PatchLayoutConfig, load_patch_layout_from_yaml
from lspattern.mytype import Coord2D, Coord3D, NodeRole

_DEFAULT_BOUNDARY: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
}

_RESOURCE_PACKAGES = {
    "canvas": ("lspattern.examples", "lspattern.patch_layout.canvas"),
    "blocks": ("lspattern.patch_layout.blocks",),
    "layers": ("lspattern.patch_layout.layers",),
}


@dataclass(frozen=True, slots=True)
class LogicalObservableSpec:
    token: str | None = None
    nodes: tuple[Coord3D, ...] = ()


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


def _resolve_json(name: str | Path, extra_paths: Sequence[Path | str]) -> Path:
    """Resolve JSON by path or search paths (filesystem only)."""

    candidate = Path(name)
    candidates = [candidate]
    if candidate.suffix == "":
        candidates.append(candidate.with_suffix(".json"))

    # Explicit / relative path
    for path in candidates:
        if path.is_file():
            return path

    # Search paths
    for root in _iter_search_paths(extra_paths):
        for cand in candidates:
            path = root / cand
            if path.is_file():
                return path

    msg = f"JSON '{name}' not found in {list(_iter_search_paths(extra_paths))}"
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
    if isinstance(value, Mapping):
        if "token" in value:
            token_value = value["token"]
            if not isinstance(token_value, str):
                msg = f"logical_observables.token must be a string, got: {type(token_value)}"
                raise TypeError(msg)
            return _parse_logical_observable(token_value)
        raw_nodes = value.get("nodes", value.get("coords"))
        if raw_nodes is None:
            msg = "logical_observables mapping must provide 'token' or 'nodes'."
            raise ValueError(msg)
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
            msg = f"logical_observables.nodes must be a list, got: {type(raw_nodes)}"
            raise TypeError(msg)
        coords = tuple(_parse_coord3d(coord) for coord in raw_nodes)
        if not coords:
            msg = "logical_observables.nodes must contain at least one coordinate."
            raise ValueError(msg)
        return LogicalObservableSpec(nodes=coords)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        coords = tuple(_parse_coord3d(coord) for coord in value)
        if not coords:
            return None
        return LogicalObservableSpec(nodes=coords)
    msg = f"Unsupported logical_observables spec: {value!r}"
    raise TypeError(msg)


def _parse_coord3d(value: object) -> Coord3D:
    if isinstance(value, Coord3D):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:  # noqa: PLR2004
        x, y, z = value
        return Coord3D(int(cast("SupportsInt", x)), int(cast("SupportsInt", y)), int(cast("SupportsInt", z)))
    msg = f"Expected Coord3D as [x, y, z], got: {value!r}"
    raise TypeError(msg)


def _parse_coord2d(value: object) -> Coord2D:
    if isinstance(value, Coord2D):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:  # noqa: PLR2004
        x, y = value
        return Coord2D(int(cast("SupportsInt", x)), int(cast("SupportsInt", y)))
    msg = f"Expected Coord2D as [x, y], got: {value!r}"
    raise TypeError(msg)


def _parse_axis(value: object) -> Axis:
    if isinstance(value, Axis):
        return value
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned in {"X", "Y", "Z"}:
            return Axis[cleaned]
    msg = f"Expected basis 'X'/'Y'/'Z', got: {value!r}"
    raise ValueError(msg)


def _parse_node_role(value: object | None) -> NodeRole | None:
    if value is None:
        return None
    if isinstance(value, NodeRole):
        return value
    if isinstance(value, str):
        cleaned = value.strip().upper()
        try:
            return NodeRole[cleaned]
        except KeyError as exc:
            msg = f"Unknown node role: {value!r}"
            raise ValueError(msg) from exc
    msg = f"Unsupported node role: {value!r}"
    raise TypeError(msg)


def _parse_edge_entry(entry: object) -> tuple[Coord3D, Coord3D]:
    if isinstance(entry, Mapping):
        for key_a, key_b in (("a", "b"), ("u", "v"), ("src", "dst"), ("start", "end")):
            if key_a in entry and key_b in entry:
                return _parse_coord3d(entry[key_a]), _parse_coord3d(entry[key_b])
        msg = f"Edge mapping must have a/b (or u/v), got keys: {sorted(entry.keys())}"
        raise ValueError(msg)
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:  # noqa: PLR2004
        a, b = entry
        return _parse_coord3d(a), _parse_coord3d(b)
    msg = f"Edge entry must be [[x,y,z],[x,y,z]] or mapping, got: {entry!r}"
    raise TypeError(msg)


def _parse_graph_spec(value: object) -> GraphSpec:  # noqa: C901, PLR0912
    if not isinstance(value, Mapping):
        msg = f"graph must be a mapping, got: {type(value)}"
        raise TypeError(msg)

    coord_mode = str(value.get("coord_mode", "local")).strip().lower()
    if coord_mode not in {"local", "global"}:
        msg = f"graph.coord_mode must be 'local' or 'global', got: {coord_mode!r}"
        raise ValueError(msg)

    time_mode = str(value.get("time_mode", "local")).strip().lower()
    if time_mode not in {"local", "global"}:
        msg = f"graph.time_mode must be 'local' or 'global', got: {time_mode!r}"
        raise ValueError(msg)

    spec = GraphSpec(coord_mode=coord_mode, time_mode=time_mode)

    # Nodes
    raw_nodes = value.get("nodes", [])
    if raw_nodes is None:
        raw_nodes = []
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)):
        msg = f"graph.nodes must be a list, got: {type(raw_nodes)}"
        raise TypeError(msg)

    for node_entry in raw_nodes:
        if not isinstance(node_entry, Mapping):
            msg = f"Each graph.nodes entry must be a mapping, got: {type(node_entry)}"
            raise TypeError(msg)
        coord = _parse_coord3d(node_entry.get("coord"))
        basis = _parse_axis(node_entry.get("basis"))
        role = _parse_node_role(node_entry.get("role"))
        spec.nodes.add(coord)
        spec.pauli_axes[coord] = basis
        if role is not None:
            spec.coord2role[coord] = role

    # Edges
    raw_edges = value.get("edges", [])
    if raw_edges is None:
        raw_edges = []
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        msg = f"graph.edges must be a list, got: {type(raw_edges)}"
        raise TypeError(msg)
    for entry in raw_edges:
        spec.edges.add(_parse_edge_entry(entry))

    # xflow
    raw_xflow = value.get("xflow", [])
    if raw_xflow is None:
        raw_xflow = []
    if not isinstance(raw_xflow, Sequence) or isinstance(raw_xflow, (str, bytes)):
        msg = f"graph.xflow must be a list, got: {type(raw_xflow)}"
        raise TypeError(msg)
    for entry in raw_xflow:
        if isinstance(entry, Mapping):
            from_coord = _parse_coord3d(entry.get("from"))
            raw_to = entry.get("to", [])
            if raw_to is None:
                raw_to = []
            if isinstance(raw_to, Sequence) and not isinstance(raw_to, (str, bytes)):
                for to_coord in raw_to:
                    spec.flow.add_flow(from_coord, _parse_coord3d(to_coord))
            else:
                spec.flow.add_flow(from_coord, _parse_coord3d(raw_to))
            continue
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:  # noqa: PLR2004
            from_coord, to_coord = entry
            spec.flow.add_flow(_parse_coord3d(from_coord), _parse_coord3d(to_coord))
            continue
        msg = f"Each graph.xflow entry must be a mapping or [from,to] pair, got: {entry!r}"
        raise TypeError(msg)

    # schedule
    schedule_cfg = value.get("schedule", None)
    if schedule_cfg is not None:
        if not isinstance(schedule_cfg, Mapping):
            msg = f"graph.schedule must be a mapping, got: {type(schedule_cfg)}"
            raise TypeError(msg)

        def _load_time_groups(raw: object, *, kind: str) -> list[Mapping[str, object]]:
            if raw is None:
                return []
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                msg = f"graph.schedule.{kind} must be a list, got: {type(raw)}"
                raise TypeError(msg)
            groups: list[Mapping[str, object]] = []
            for item in raw:
                if not isinstance(item, Mapping):
                    msg = f"graph.schedule.{kind} entries must be mappings, got: {type(item)}"
                    raise TypeError(msg)
                groups.append(item)
            return groups

        for group in _load_time_groups(schedule_cfg.get("prep"), kind="prep"):
            time = int(cast("SupportsInt", group.get("time")))
            raw_coords = group.get("nodes", [])
            if raw_coords is None:
                raw_coords = []
            if not isinstance(raw_coords, Sequence) or isinstance(raw_coords, (str, bytes)):
                msg = f"graph.schedule.prep.nodes must be a list, got: {type(raw_coords)}"
                raise TypeError(msg)
            coords = [_parse_coord3d(coord) for coord in raw_coords]
            spec.scheduler.add_prep_at_time(time, coords)

        for group in _load_time_groups(schedule_cfg.get("meas"), kind="meas"):
            time = int(cast("SupportsInt", group.get("time")))
            raw_coords = group.get("nodes", [])
            if raw_coords is None:
                raw_coords = []
            if not isinstance(raw_coords, Sequence) or isinstance(raw_coords, (str, bytes)):
                msg = f"graph.schedule.meas.nodes must be a list, got: {type(raw_coords)}"
                raise TypeError(msg)
            coords = [_parse_coord3d(coord) for coord in raw_coords]
            spec.scheduler.add_meas_at_time(time, coords)

        for group in _load_time_groups(schedule_cfg.get("entangle"), kind="entangle"):
            time = int(cast("SupportsInt", group.get("time")))
            raw_edges = group.get("edges", [])
            if raw_edges is None:
                raw_edges = []
            if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
                msg = f"graph.schedule.entangle.edges must be a list, got: {type(raw_edges)}"
                raise TypeError(msg)
            edges = [_parse_edge_entry(edge) for edge in raw_edges]
            spec.scheduler.add_entangle_at_time(time, edges)
            spec.edges.update(edges)

    # detector candidates (parity accumulator)
    det_cfg = value.get("detector_candidates", None)
    if det_cfg is not None:
        if not isinstance(det_cfg, Mapping):
            msg = f"graph.detector_candidates must be a mapping, got: {type(det_cfg)}"
            raise TypeError(msg)

        def _parse_parity_groups(raw: object, *, kind: str) -> None:
            if raw is None:
                return
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                msg = f"graph.detector_candidates.{kind} must be a list, got: {type(raw)}"
                raise TypeError(msg)
            for group in raw:
                if not isinstance(group, Mapping):
                    msg = f"graph.detector_candidates.{kind} entries must be mappings, got: {type(group)}"
                    raise TypeError(msg)
                xy = _parse_coord2d(group.get("id"))
                rounds = group.get("rounds", [])
                if rounds is None:
                    rounds = []
                if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)):
                    msg = f"graph.detector_candidates.{kind}.rounds must be a list, got: {type(rounds)}"
                    raise TypeError(msg)
                for round_entry in rounds:
                    if not isinstance(round_entry, Mapping):
                        msg = (
                            f"graph.detector_candidates.{kind}.rounds entries must be mappings, "
                            f"got: {type(round_entry)}"
                        )
                        raise TypeError(msg)
                    z = int(cast("SupportsInt", round_entry.get("z")))
                    nodes = round_entry.get("nodes", [])
                    if nodes is None:
                        nodes = []
                    coords = [_parse_coord3d(coord) for coord in cast("Sequence[object]", nodes)]
                    if kind == "syndrome_meas":
                        spec.parity.add_syndrome_measurement(xy, z, coords)
                    else:
                        spec.parity.add_remaining_parity(xy, z, coords)

        _parse_parity_groups(det_cfg.get("syndrome_meas"), kind="syndrome_meas")
        _parse_parity_groups(det_cfg.get("remaining_parity"), kind="remaining_parity")

        raw_non_det = det_cfg.get("non_deterministic", [])
        if raw_non_det is None:
            raw_non_det = []
        if not isinstance(raw_non_det, Sequence) or isinstance(raw_non_det, (str, bytes)):
            msg = f"graph.detector_candidates.non_deterministic must be a list, got: {type(raw_non_det)}"
            raise TypeError(msg)
        for coord in raw_non_det:
            spec.parity.add_non_deterministic_coord(_parse_coord3d(coord))

    # Validate cross-references (basis, edges, schedule, parity must reference nodes)
    if not spec.nodes:
        msg = "graph.nodes must contain at least one node"
        raise ValueError(msg)
    if set(spec.pauli_axes) != spec.nodes:
        msg = "graph.nodes and graph.nodes[].basis must be provided for every node"
        raise ValueError(msg)

    def _assert_known(coord: Coord3D, *, context: str) -> None:
        if coord not in spec.nodes:
            msg = f"{context} references undefined node: {coord}"
            raise ValueError(msg)

    for a, b in spec.edges:
        _assert_known(a, context="graph.edges")
        _assert_known(b, context="graph.edges")

    for from_coord, to_coords in spec.flow.flow.items():
        _assert_known(from_coord, context="graph.xflow")
        for to_coord in to_coords:
            _assert_known(to_coord, context="graph.xflow")

    for coords in spec.scheduler.prep_time.values():
        for coord in coords:
            _assert_known(coord, context="graph.schedule.prep")

    for coords in spec.scheduler.meas_time.values():
        for coord in coords:
            _assert_known(coord, context="graph.schedule.meas")

    for edges in spec.scheduler.entangle_time.values():
        for a, b in edges:
            _assert_known(a, context="graph.schedule.entangle")
            _assert_known(b, context="graph.schedule.entangle")

    for z_map in spec.parity.syndrome_meas.values():
        for coords in z_map.values():
            for coord in coords:
                _assert_known(coord, context="graph.detector_candidates.syndrome_meas")

    for z_map in spec.parity.remaining_parity.values():
        for coords in z_map.values():
            for coord in coords:
                _assert_known(coord, context="graph.detector_candidates.remaining_parity")

    return spec


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

    if cfg.get("graph") is not None:
        if cfg.get("layers"):
            msg = "Block YAML cannot define both 'layers' and 'graph'."
            raise ValueError(msg)
        graph_ref = cfg["graph"]
        if not isinstance(graph_ref, (str, Path)):
            msg = "Block YAML 'graph' must be a JSON filename/path (string)."
            raise TypeError(msg)
        if isinstance(graph_ref, str) and not graph_ref.strip():
            msg = "Block YAML 'graph' must be a non-empty JSON filename/path."
            raise ValueError(msg)

        graph_path = _resolve_json(graph_ref, search_paths)
        with graph_path.open("r", encoding="utf-8") as f:
            try:
                graph_cfg = json.load(f)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in graph file '{graph_path}': {exc}"
                raise ValueError(msg) from exc

        graph_spec = _parse_graph_spec(graph_cfg)
        block_config = BlockConfig([])
        block_config.boundary = boundary
        block_config.graph_spec = graph_spec
        return block_config

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
        canvas.add_pipe((pipe.start, pipe.end), block_config, pipe.logical_observable)

    canvas.logical_observables = spec.logical_observables
    return canvas


def load_canvas(
    name: str | Path, *, code_distance: int, extra_paths: Sequence[Path | str] = ()
) -> tuple[Canvas, CanvasSpec]:
    spec = load_canvas_spec(name, extra_paths=extra_paths)
    canvas = build_canvas(spec, code_distance=code_distance, extra_paths=extra_paths)
    return canvas, spec
