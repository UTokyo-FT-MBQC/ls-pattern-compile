"""Convert liblsqecc slices JSON into lspattern canvas YAML."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import Any

import yaml

_QUBIT_PATCH_TYPE = "Qubit"
_ANCILLA_PATCH_TYPE = "Ancilla"
_MEASUREMENT_ACTIVITY = "Measurement"
_BOUNDARY_KEYS = ("Top", "Bottom", "Left", "Right")
_QUBIT_ID_RE = re.compile(r"^\s*Id:\s*(-?\d+)\s*$")
_ANCILLA_JOIN = "ancillajoin"
_SOLID_STITCHED_VALUES = {"solidstiched", "solidstitched"}
_DASHED_STITCHED_VALUES = {"dashedstiched", "dashedstitched"}
_SIDE_TO_OFFSET = {
    "Top": (0, -1),
    "Bottom": (0, 1),
    "Left": (-1, 0),
    "Right": (1, 0),
}
_OPPOSITE_SIDE = {
    "Top": "Bottom",
    "Bottom": "Top",
    "Left": "Right",
    "Right": "Left",
}
_SIDE_INDEX = {side: idx for idx, side in enumerate(_BOUNDARY_KEYS)}
_ROTATION_TEMPLATE_FILES = {
    "left": "_patch_rotation2_left.yml",
    "right": "_patch_rotation2_right.yml",
    "top": "_patch_rotation2_top.yml",
    "bottom": "_patch_rotation2_bottom.yml",
}
_ROTATION_INPUT_REL = {
    "left": (0, 0),
    "right": (1, 0),
    "top": (0, 0),
    "bottom": (0, 1),
}
_ROTATION_HELPER_REL = {
    "left": (1, 0),
    "right": (0, 0),
    "top": (0, 1),
    "bottom": (0, 0),
}

Coord2D = tuple[int, int]
Coord3D = tuple[int, int, int]

_DISTILLATION_PATCH_TYPE = "DistillationQubit"
_MAGIC_STATE_RE = re.compile(r"Time to next magic state:(\d+)")


class LibLsQeccImportError(RuntimeError):
    """Raised when liblsqecc slices cannot be converted into canvas YAML."""


@dataclass(frozen=True)
class DistillationFactory:
    """A rectangular distillation factory region detected from slices."""

    origin: Coord2D
    width: int
    height: int
    z_period: int
    outer_ring: frozenset[Coord2D]
    inner_cells: frozenset[Coord2D]


DistillationTemplateFn = Callable[
    ["DistillationFactory", int],
    tuple[list[dict[str, object]], list[dict[str, object]]],
]


@dataclass(frozen=True)
class _AncillaComponent:
    z: int
    basis: str  # "ZZ" or "XX"
    cells: frozenset[Coord2D]
    ancilla_links: frozenset[tuple[Coord2D, Coord2D]]
    qubit_links: frozenset[tuple[Coord2D, Coord2D]]


@dataclass(frozen=True)
class _AncillaComponentExtracted:
    z: int
    cells: frozenset[Coord2D]
    ancilla_links: frozenset[tuple[Coord2D, Coord2D]]
    qubit_links: frozenset[tuple[Coord2D, Coord2D]]
    endpoint_stitch_kinds: tuple[tuple[Coord2D, frozenset[str]], ...]


@dataclass(frozen=True)
class _PipeInternal:
    start: Coord3D
    end: Coord3D
    block: str
    basis: str  # "ZZ" or "XX"


@dataclass(frozen=True)
class _RotationTemplate:
    direction: str
    input_rel: Coord2D
    helper_rel: Coord2D
    input_boundary_at_z0: str
    cubes: tuple[tuple[Coord3D, str, str], ...]
    pipes: tuple[tuple[Coord3D, Coord3D, str, str], ...]


def _normalize_edge_value(edge: object, *, context: str) -> str:
    if not isinstance(edge, str):
        msg = f"{context}: edge value must be a string, got {type(edge)}"
        raise LibLsQeccImportError(msg)
    return edge.strip().lower()


def _edge_to_boundary_char(edge: object, *, context: str) -> str:
    normalized = _normalize_edge_value(edge, context=context)
    if normalized == "solid":
        return "Z"
    if normalized == "dashed":
        return "X"
    if normalized in {"none", *_SOLID_STITCHED_VALUES, *_DASHED_STITCHED_VALUES}:
        return "O"

    msg = f"{context}: unsupported edge value {edge!r}"
    raise LibLsQeccImportError(msg)


def _is_ancilla_join(edge: object) -> bool:
    if not isinstance(edge, str):
        return False
    return edge.strip().lower() == _ANCILLA_JOIN


def _stitched_kind(edge: object) -> str | None:
    if not isinstance(edge, str):
        return None
    normalized = edge.strip().lower()
    if normalized in _SOLID_STITCHED_VALUES:
        return "solid"
    if normalized in _DASHED_STITCHED_VALUES:
        return "dashed"
    return None


def _extract_qubit_id(cell: Mapping[str, Any]) -> int | None:
    text = cell.get("text")
    if not isinstance(text, str):
        return None
    match = _QUBIT_ID_RE.match(text)
    if match is None:
        return None
    return int(match.group(1))


def _is_qubit_cell(cell: Mapping[str, Any] | None) -> bool:
    return cell is not None and cell.get("patch_type") == _QUBIT_PATCH_TYPE


def _is_distillation_cell(cell: Mapping[str, Any] | None) -> bool:
    return cell is not None and cell.get("patch_type") == _DISTILLATION_PATCH_TYPE


def _is_ancilla_cell(cell: Mapping[str, Any] | None) -> bool:
    return cell is not None and cell.get("patch_type") == _ANCILLA_PATCH_TYPE


def _is_measurement_cell(cell: Mapping[str, Any]) -> bool:
    activity = cell.get("activity")
    if not isinstance(activity, Mapping):
        msg = "Internal error: validated cell is missing activity mapping."
        raise LibLsQeccImportError(msg)
    return activity.get("activity_type") == _MEASUREMENT_ACTIVITY


def _is_init_cell(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> bool:
    if not _is_qubit_cell(previous):
        return True

    current_id = _extract_qubit_id(current)
    previous_id = _extract_qubit_id(previous)
    return current_id is not None and previous_id is not None and current_id != previous_id


def _cell_context(z: int, y: int, x: int) -> str:
    return f"slice={z}, row={y}, col={x}"


def _validate_cell(cell: object, *, z: int, y: int, x: int) -> Mapping[str, Any]:
    context = _cell_context(z, y, x)
    if not isinstance(cell, Mapping):
        msg = f"{context}: cell must be mapping or null, got {type(cell)}"
        raise LibLsQeccImportError(msg)

    if "patch_type" not in cell:
        msg = f"{context}: missing required key 'patch_type'"
        raise LibLsQeccImportError(msg)
    if not isinstance(cell["patch_type"], str):
        msg = f"{context}: 'patch_type' must be a string"
        raise LibLsQeccImportError(msg)

    edges = cell.get("edges")
    if not isinstance(edges, Mapping):
        msg = f"{context}: missing or invalid 'edges' mapping"
        raise LibLsQeccImportError(msg)
    for key in _BOUNDARY_KEYS:
        if key not in edges:
            msg = f"{context}: missing edges.{key}"
            raise LibLsQeccImportError(msg)
        if not isinstance(edges[key], str):
            msg = f"{context}: edges.{key} must be a string"
            raise LibLsQeccImportError(msg)

    activity = cell.get("activity")
    if not isinstance(activity, Mapping):
        msg = f"{context}: missing or invalid 'activity' mapping"
        raise LibLsQeccImportError(msg)
    if "activity_type" not in activity:
        msg = f"{context}: missing activity.activity_type"
        raise LibLsQeccImportError(msg)

    return cell


def _validate_and_normalize_slices(raw_slices: object) -> list[list[list[Mapping[str, Any] | None]]]:  # noqa: C901
    if not isinstance(raw_slices, list):
        msg = f"Top-level slices must be a list, got {type(raw_slices)}"
        raise LibLsQeccImportError(msg)

    normalized: list[list[list[Mapping[str, Any] | None]]] = []
    expected_rows: int | None = None
    expected_cols: int | None = None

    for z, slice_obj in enumerate(raw_slices):
        if not isinstance(slice_obj, list):
            msg = f"slice={z}: each slice must be a list of rows, got {type(slice_obj)}"
            raise LibLsQeccImportError(msg)

        if expected_rows is None:
            expected_rows = len(slice_obj)
        elif len(slice_obj) != expected_rows:
            msg = f"slice={z}: inconsistent row count (expected {expected_rows}, got {len(slice_obj)})"
            raise LibLsQeccImportError(msg)

        normalized_rows: list[list[Mapping[str, Any] | None]] = []
        for y, row_obj in enumerate(slice_obj):
            if not isinstance(row_obj, list):
                msg = f"slice={z}, row={y}: row must be a list, got {type(row_obj)}"
                raise LibLsQeccImportError(msg)

            if expected_cols is None:
                expected_cols = len(row_obj)
            elif len(row_obj) != expected_cols:
                msg = (
                    f"slice={z}, row={y}: inconsistent column count "
                    f"(expected {expected_cols}, got {len(row_obj)})"
                )
                raise LibLsQeccImportError(msg)

            normalized_cells: list[Mapping[str, Any] | None] = []
            for x, cell in enumerate(row_obj):
                if cell is None:
                    normalized_cells.append(None)
                    continue
                normalized_cells.append(_validate_cell(cell, z=z, y=y, x=x))
            normalized_rows.append(normalized_cells)
        normalized.append(normalized_rows)

    return normalized


def _cell_boundary_string(cell: Mapping[str, Any], *, z: int, y: int, x: int) -> str:
    context = _cell_context(z, y, x)
    edges = cell.get("edges")
    if not isinstance(edges, Mapping):
        msg = f"{context}: missing or invalid 'edges' mapping"
        raise LibLsQeccImportError(msg)
    top = _edge_to_boundary_char(edges["Top"], context=f"{context} edges.Top")
    bottom = _edge_to_boundary_char(edges["Bottom"], context=f"{context} edges.Bottom")
    left = _edge_to_boundary_char(edges["Left"], context=f"{context} edges.Left")
    right = _edge_to_boundary_char(edges["Right"], context=f"{context} edges.Right")
    return f"{top}{bottom}{left}{right}"


def _cube_sort_key(entry: Mapping[str, object]) -> tuple[int, int, int]:
    position = entry.get("position")
    if not isinstance(position, list) or len(position) != 3:  # noqa: PLR2004
        msg = f"Internal error: invalid cube position entry {position!r}"
        raise LibLsQeccImportError(msg)

    x, y, z = position
    if not isinstance(x, int) or not isinstance(y, int) or not isinstance(z, int):
        msg = f"Internal error: cube position must be integer triplet, got {position!r}"
        raise LibLsQeccImportError(msg)
    return z, y, x


def _pipe_sort_key(entry: Mapping[str, object]) -> tuple[int, int, int, int, int, int]:
    start = entry.get("start")
    end = entry.get("end")
    if not isinstance(start, list) or len(start) != 3 or not isinstance(end, list) or len(end) != 3:  # noqa: PLR2004
        msg = f"Internal error: invalid pipe entry: {entry!r}"
        raise LibLsQeccImportError(msg)
    if not all(isinstance(v, int) for v in [*start, *end]):
        msg = f"Internal error: pipe coordinates must be integers: {entry!r}"
        raise LibLsQeccImportError(msg)
    return start[2], start[1], start[0], end[2], end[1], end[0]


def _coord_side_to_neighbor(coord: Coord2D, side: str) -> Coord2D:
    dx, dy = _SIDE_TO_OFFSET[side]
    return coord[0] + dx, coord[1] + dy


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _basis_from_stitch_kinds(kinds: set[str], *, context: str) -> str:
    if kinds == {"solid"}:
        return "ZZ"
    if kinds == {"dashed"}:
        return "XX"
    msg = f"{context}: unsupported or mixed stitched basis kinds: {sorted(kinds)}"
    raise LibLsQeccImportError(msg)


def _detect_distillation_factories(  # noqa: C901
    normalized: list[list[list[Mapping[str, Any] | None]]],
) -> list[DistillationFactory]:
    """Detect distillation factory regions from the first slice via connected-component analysis."""
    if not normalized:
        return []
    first_slice = normalized[0]

    # Collect distillation cell coordinates from z=0.
    distill_coords: set[Coord2D] = set()
    distill_cells: dict[Coord2D, Mapping[str, Any]] = {}
    for y, row in enumerate(first_slice):
        for x, cell in enumerate(row):
            if _is_distillation_cell(cell):
                distill_coords.add((x, y))
                if cell is not None:
                    distill_cells[(x, y)] = cell

    if not distill_coords:
        return []

    # 4-connected component analysis.
    visited: set[Coord2D] = set()
    components: list[set[Coord2D]] = []
    for start in sorted(distill_coords):
        if start in visited:
            continue
        stack = [start]
        component: set[Coord2D] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            cx, cy = current
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cx + dx, cy + dy)
                if neighbor in distill_coords and neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)

    factories: list[DistillationFactory] = []
    for comp in components:
        xs = [c[0] for c in comp]
        ys = [c[1] for c in comp]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        origin: Coord2D = (min_x, min_y)
        width = max_x - min_x + 1
        height = max_y - min_y + 1

        # Classify cells: active (any non-None edge) vs inactive (all None edges).
        outer_ring: set[Coord2D] = set()
        inner_cells: set[Coord2D] = set()
        for coord in comp:
            cell = distill_cells.get(coord)
            if cell is None:
                inner_cells.add((coord[0] - min_x, coord[1] - min_y))
                continue
            edges = cell.get("edges")
            if isinstance(edges, Mapping):
                has_active = any(
                    isinstance(edges.get(k), str) and edges.get(k, "").strip().lower() != "none"
                    for k in _BOUNDARY_KEYS
                )
            else:
                has_active = False
            rel = (coord[0] - min_x, coord[1] - min_y)
            if has_active:
                outer_ring.add(rel)
            else:
                inner_cells.add(rel)

        # Detect z_period from "Time to next magic state:N" text.
        z_period = 0
        for coord in comp:
            cell = distill_cells.get(coord)
            if cell is None:
                continue
            text = cell.get("text")
            if not isinstance(text, str):
                continue
            match = _MAGIC_STATE_RE.search(text)
            if match is not None:
                n = int(match.group(1))
                if n > z_period:
                    z_period = n
        if z_period == 0:
            # Scan later slices for this factory's origin cell.
            for z in range(1, len(normalized)):
                slice_rows = normalized[z]
                ox, oy = origin
                if oy < len(slice_rows) and ox < len(slice_rows[oy]):
                    cell = slice_rows[oy][ox]
                    if cell is not None and isinstance(cell.get("text"), str):
                        m = _MAGIC_STATE_RE.search(cell["text"])
                        if m is not None:
                            n = int(m.group(1))
                            if n > z_period:
                                z_period = n
            if z_period == 0:
                z_period = 1  # Fallback: treat as period 1

        factories.append(
            DistillationFactory(
                origin=origin,
                width=width,
                height=height,
                z_period=z_period,
                outer_ring=frozenset(outer_ring),
                inner_cells=frozenset(inner_cells),
            )
        )

    return factories


def default_distillation_template(
    factory: DistillationFactory,
    total_slices: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Placeholder distillation template. Edit to customize."""
    cubes: list[dict[str, object]] = []
    pipes: list[dict[str, object]] = []
    ox, oy = factory.origin
    transposed = factory.width > factory.height  # 5x3

    for round_start in range(0, total_slices, factory.z_period):
        for dz in range(factory.z_period):
            z = round_start + dz
            if z >= total_slices:
                break
            if dz == 0:
                block = "InitZeroBlock"
            elif dz == factory.z_period - 1:
                block = "MeasureZBlock"
            else:
                block = "MemoryBlock"
            for dx, dy in sorted(factory.outer_ring):
                rx, ry = (dy, dx) if transposed else (dx, dy)
                cubes.append({
                    "position": [ox + rx, oy + ry, z],
                    "block": block,
                    "boundary": "XXZZ",
                })
    return cubes, pipes


def _extract_ancilla_components(  # noqa: C901
    slice_rows: list[list[Mapping[str, Any] | None]],
    *,
    z: int,
) -> list[_AncillaComponentExtracted]:
    height = len(slice_rows)
    width = len(slice_rows[0]) if height > 0 else 0

    ancilla_cells: dict[Coord2D, Mapping[str, Any]] = {}
    for y, row in enumerate(slice_rows):
        for x, cell in enumerate(row):
            if _is_ancilla_cell(cell):
                if cell is None:
                    msg = f"Internal error: ancilla cell unexpectedly None at slice={z}, row={y}, col={x}"
                    raise LibLsQeccImportError(msg)
                ancilla_cells[x, y] = cell

    adjacency: dict[Coord2D, set[Coord2D]] = {coord: set() for coord in ancilla_cells}
    for coord, cell in ancilla_cells.items():
        edges = cell["edges"]
        if not isinstance(edges, Mapping):
            msg = f"slice={z}: invalid ancilla edges at coord={coord}"
            raise LibLsQeccImportError(msg)

        for side in _BOUNDARY_KEYS:
            if not _is_ancilla_join(edges[side]):
                continue
            neighbor = _coord_side_to_neighbor(coord, side)
            neighbor_cell = ancilla_cells.get(neighbor)
            if neighbor_cell is None:
                continue

            neighbor_edges = neighbor_cell["edges"]
            if not isinstance(neighbor_edges, Mapping):
                msg = f"slice={z}: invalid ancilla edges at neighbor coord={neighbor}"
                raise LibLsQeccImportError(msg)
            if _is_ancilla_join(neighbor_edges[_OPPOSITE_SIDE[side]]):
                adjacency[coord].add(neighbor)

    components: list[_AncillaComponentExtracted] = []
    visited: set[Coord2D] = set()

    for start in sorted(ancilla_cells):
        if start in visited:
            continue

        stack = [start]
        cells: set[Coord2D] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cells.add(current)
            stack.extend(neighbor for neighbor in adjacency[current] if neighbor not in visited)

        ancilla_links: set[tuple[Coord2D, Coord2D]] = set()
        for coord in cells:
            for neighbor in adjacency[coord]:
                if neighbor not in cells:
                    continue
                if coord < neighbor:
                    ancilla_links.add((coord, neighbor))

        endpoint_stitch_kinds: dict[Coord2D, set[str]] = defaultdict(set)
        qubit_links: set[tuple[Coord2D, Coord2D]] = set()

        for coord in cells:
            x, y = coord
            for side in _BOUNDARY_KEYS:
                nx, ny = _coord_side_to_neighbor(coord, side)
                if not _in_bounds(nx, ny, width, height):
                    continue
                neighbor_cell = slice_rows[ny][nx]
                if not _is_qubit_cell(neighbor_cell):
                    continue
                if neighbor_cell is None:
                    msg = f"Internal error: qubit cell unexpectedly None at slice={z}, row={ny}, col={nx}"
                    raise LibLsQeccImportError(msg)

                neighbor_edges = neighbor_cell.get("edges")
                if not isinstance(neighbor_edges, Mapping):
                    msg = f"slice={z}, row={ny}, col={nx}: invalid qubit edges"
                    raise LibLsQeccImportError(msg)

                stitched_kind = _stitched_kind(neighbor_edges[_OPPOSITE_SIDE[side]])
                if stitched_kind is None:
                    continue

                qubit_coord = (nx, ny)
                endpoint_stitch_kinds[qubit_coord].add(stitched_kind)
                qubit_links.add((coord, qubit_coord))

        frozen_endpoint_kinds = tuple(
            sorted((coord, frozenset(kinds)) for coord, kinds in endpoint_stitch_kinds.items()),
        )
        components.append(
            _AncillaComponentExtracted(
                z=z,
                cells=frozenset(cells),
                ancilla_links=frozenset(ancilla_links),
                qubit_links=frozenset(qubit_links),
                endpoint_stitch_kinds=frozen_endpoint_kinds,
            ),
        )

    return components


def _extract_valid_ancilla_components(
    slice_rows: list[list[Mapping[str, Any] | None]],
    *,
    z: int,
) -> list[_AncillaComponent]:
    extracted = _extract_ancilla_components(slice_rows, z=z)
    components: list[_AncillaComponent] = []
    for component in extracted:
        endpoint_stitch_kinds = dict(component.endpoint_stitch_kinds)

        # Ignore non-measurement ancilla regions (e.g., distillation-side routing artifacts).
        if len(endpoint_stitch_kinds) != 2:  # noqa: PLR2004
            continue

        stitched_kinds: set[str] = set()
        for kinds in endpoint_stitch_kinds.values():
            stitched_kinds.update(kinds)

        basis = _basis_from_stitch_kinds(
            stitched_kinds,
            context=f"slice={z}, ancilla_component_start={next(iter(component.cells), None)}",
        )
        components.append(
            _AncillaComponent(
                z=component.z,
                basis=basis,
                cells=component.cells,
                ancilla_links=component.ancilla_links,
                qubit_links=component.qubit_links,
            ),
        )
    return components


def _swap_xz_boundary(boundary: str) -> str:
    trans = {"X": "Z", "Z": "X"}
    return "".join(trans.get(ch, ch) for ch in boundary)


def _side_from_to(a: Coord2D, b: Coord2D) -> str | None:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 1 and dy == 0:
        return "Right"
    if dx == -1 and dy == 0:
        return "Left"
    if dx == 0 and dy == 1:
        return "Bottom"
    if dx == 0 and dy == -1:
        return "Top"
    return None


def _rotation_direction_from_input(input_coord: Coord2D, helper_coord: Coord2D) -> str | None:
    if helper_coord[0] == input_coord[0] - 1 and helper_coord[1] == input_coord[1]:
        return "right"
    if helper_coord[0] == input_coord[0] + 1 and helper_coord[1] == input_coord[1]:
        return "left"
    if helper_coord[0] == input_coord[0] and helper_coord[1] == input_coord[1] - 1:
        return "bottom"
    if helper_coord[0] == input_coord[0] and helper_coord[1] == input_coord[1] + 1:
        return "top"
    return None


def _can_match_rotation_boundary(
    source_boundary: str,
    template_boundary: str,
    *,
    join_side: str,
    swap_xz: bool,
) -> bool:
    join_idx = _SIDE_INDEX[join_side]
    candidate = _swap_xz_boundary(template_boundary) if swap_xz else template_boundary
    for idx, (source_ch, cand_ch) in enumerate(zip(source_boundary, candidate, strict=True)):
        if idx == join_idx:
            continue
        if source_ch in {"X", "Z"} and cand_ch in {"X", "Z"} and source_ch != cand_ch:
            return False
    return True


@lru_cache(maxsize=1)
def _load_rotation_templates() -> dict[str, _RotationTemplate]:
    base_dir = Path(__file__).resolve().parents[2] / "examples" / "design" / "utils"
    templates: dict[str, _RotationTemplate] = {}

    for direction, filename in _ROTATION_TEMPLATE_FILES.items():
        path = base_dir / filename
        if not path.exists():
            return {}

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            msg = f"Rotation template must be a mapping: {path}"
            raise LibLsQeccImportError(msg)

        cube_raw = loaded.get("cube")
        pipe_raw = loaded.get("pipe")
        if not isinstance(cube_raw, list) or not isinstance(pipe_raw, list):
            msg = f"Rotation template must define list-valued cube/pipe sections: {path}"
            raise LibLsQeccImportError(msg)

        cubes: list[tuple[Coord3D, str, str]] = []
        pipes: list[tuple[Coord3D, Coord3D, str, str]] = []
        for cube in cube_raw:
            if not isinstance(cube, Mapping):
                msg = f"Rotation template cube entry must be mapping: {path}"
                raise LibLsQeccImportError(msg)
            pos = cube.get("position")
            block = cube.get("block")
            boundary = cube.get("boundary")
            if (
                not isinstance(pos, list)
                or len(pos) != 3  # noqa: PLR2004
                or not all(isinstance(v, int) for v in pos)
                or not isinstance(block, str)
                or not isinstance(boundary, str)
                or len(boundary) != 4  # noqa: PLR2004
            ):
                msg = f"Invalid cube entry in rotation template: {path}"
                raise LibLsQeccImportError(msg)
            cubes.append(((pos[0], pos[1], pos[2]), block, boundary))

        for pipe in pipe_raw:
            if not isinstance(pipe, Mapping):
                msg = f"Rotation template pipe entry must be mapping: {path}"
                raise LibLsQeccImportError(msg)
            start = pipe.get("start")
            end = pipe.get("end")
            block = pipe.get("block")
            boundary = pipe.get("boundary")
            if (
                not isinstance(start, list)
                or len(start) != 3  # noqa: PLR2004
                or not all(isinstance(v, int) for v in start)
                or not isinstance(end, list)
                or len(end) != 3  # noqa: PLR2004
                or not all(isinstance(v, int) for v in end)
                or not isinstance(block, str)
                or not isinstance(boundary, str)
                or len(boundary) != 4  # noqa: PLR2004
            ):
                msg = f"Invalid pipe entry in rotation template: {path}"
                raise LibLsQeccImportError(msg)
            pipes.append(((start[0], start[1], start[2]), (end[0], end[1], end[2]), block, boundary))

        input_rel = _ROTATION_INPUT_REL[direction]
        helper_rel = _ROTATION_HELPER_REL[direction]
        input_boundary = None
        for rel, _block, boundary in cubes:
            if rel[2] == 0 and (rel[0], rel[1]) == input_rel:
                input_boundary = boundary
                break

        if input_boundary is None:
            msg = f"Rotation template missing input boundary cube at z=0: {path}"
            raise LibLsQeccImportError(msg)

        templates[direction] = _RotationTemplate(
            direction=direction,
            input_rel=input_rel,
            helper_rel=helper_rel,
            input_boundary_at_z0=input_boundary,
            cubes=tuple(cubes),
            pipes=tuple(pipes),
        )

    return templates


def _detect_rotation_overrides(
    normalized: list[list[list[Mapping[str, Any] | None]]],
) -> tuple[dict[Coord3D, dict[str, object]], dict[tuple[Coord3D, Coord3D], dict[str, object]]]:
    templates = _load_rotation_templates()
    if not templates:
        return {}, {}

    components_per_slice: list[list[_AncillaComponentExtracted]] = []
    for z, slice_rows in enumerate(normalized):
        components_per_slice.append(_extract_ancilla_components(slice_rows, z=z))

    candidate_zs_by_cells: dict[frozenset[Coord2D], list[int]] = defaultdict(list)
    for z, components in enumerate(components_per_slice):
        for component in components:
            if len(component.endpoint_stitch_kinds) != 0:
                continue
            if len(component.cells) != 2:  # noqa: PLR2004
                continue
            if len(component.ancilla_links) != 1:
                continue
            candidate_zs_by_cells[component.cells].append(z)

    cube_overrides: dict[Coord3D, dict[str, object]] = {}
    pipe_overrides: dict[tuple[Coord3D, Coord3D], dict[str, object]] = {}

    for cells, zs in candidate_zs_by_cells.items():
        zs_sorted = sorted(zs)
        if not zs_sorted:
            continue

        run_start = zs_sorted[0]
        run_end = zs_sorted[0]

        def process_run(start: int, end: int) -> None:
            run_len = end - start + 1
            if run_len != 3:  # noqa: PLR2004
                return
            if start <= 0:
                return
            z_after = end + 1
            if z_after >= len(normalized):
                return

            coords = sorted(cells)
            input_candidates: list[tuple[Coord2D, str]] = []
            for coord in coords:
                x, y = coord
                prev_cell = normalized[start - 1][y][x]
                after_cell = normalized[z_after][y][x]
                if not _is_qubit_cell(prev_cell) or not _is_qubit_cell(after_cell):
                    continue
                if prev_cell is None or after_cell is None:
                    continue

                prev_id = _extract_qubit_id(prev_cell)
                after_id = _extract_qubit_id(after_cell)
                same_id = prev_id is not None and after_id is not None and prev_id == after_id
                if not same_id:
                    continue
                source_boundary = _cell_boundary_string(prev_cell, z=start - 1, y=y, x=x)
                input_candidates.append((coord, source_boundary))

            if len(input_candidates) != 1:
                return

            input_coord, source_boundary = input_candidates[0]
            helper_coord = coords[0] if coords[1] == input_coord else coords[1]
            direction = _rotation_direction_from_input(input_coord, helper_coord)
            if direction is None:
                return

            template = templates.get(direction)
            if template is None:
                return

            join_side = _side_from_to(input_coord, helper_coord)
            if join_side is None:
                return

            swap_xz: bool | None = None
            for swap_candidate in (False, True):
                if _can_match_rotation_boundary(
                    source_boundary,
                    template.input_boundary_at_z0,
                    join_side=join_side,
                    swap_xz=swap_candidate,
                ):
                    swap_xz = swap_candidate
                    break

            if swap_xz is None:
                return

            anchor_x = input_coord[0] - template.input_rel[0]
            anchor_y = input_coord[1] - template.input_rel[1]
            anchor_z = start

            for rel_coord, block, boundary in template.cubes:
                x = anchor_x + rel_coord[0]
                y = anchor_y + rel_coord[1]
                z = anchor_z + rel_coord[2]
                if z < 0 or z >= len(normalized):
                    return
                if y < 0 or y >= len(normalized[z]):
                    return
                if x < 0 or x >= len(normalized[z][y]):
                    return
                adjusted_boundary = _swap_xz_boundary(boundary) if swap_xz else boundary
                cube_overrides[(x, y, z)] = {
                    "position": [x, y, z],
                    "block": block,
                    "boundary": adjusted_boundary,
                }

            for rel_start, rel_end, block, boundary in template.pipes:
                start_abs = (anchor_x + rel_start[0], anchor_y + rel_start[1], anchor_z + rel_start[2])
                end_abs = (anchor_x + rel_end[0], anchor_y + rel_end[1], anchor_z + rel_end[2])
                key = _pipe_key(start_abs, end_abs)
                adjusted_boundary = _swap_xz_boundary(boundary) if swap_xz else boundary
                pipe_overrides[key] = {
                    "start": [start_abs[0], start_abs[1], start_abs[2]],
                    "end": [end_abs[0], end_abs[1], end_abs[2]],
                    "block": block,
                    "boundary": adjusted_boundary,
                }

        for z in zs_sorted[1:]:
            if z == run_end + 1:
                run_end = z
                continue
            process_run(run_start, run_end)
            run_start = z
            run_end = z
        process_run(run_start, run_end)

    return cube_overrides, pipe_overrides


def _short_block_for_basis(basis: str) -> str:
    if basis == "ZZ":
        return "ShortXMemoryBlock"
    if basis == "XX":
        return "ShortZMemoryBlock"
    msg = f"Internal error: unsupported basis for short block: {basis}"
    raise LibLsQeccImportError(msg)


def _init_block_for_basis(basis: str) -> str:
    if basis == "ZZ":
        return "InitPlusBlock"
    if basis == "XX":
        return "InitZeroBlock"
    msg = f"Internal error: unsupported basis for init block: {basis}"
    raise LibLsQeccImportError(msg)


def _measure_block_for_basis(basis: str) -> str:
    if basis == "ZZ":
        return "MeasureXBlock"
    if basis == "XX":
        return "MeasureZBlock"
    msg = f"Internal error: unsupported basis for measure block: {basis}"
    raise LibLsQeccImportError(msg)


def _fill_char_for_basis(basis: str) -> str:
    if basis == "ZZ":
        return "X"
    if basis == "XX":
        return "Z"
    msg = f"Internal error: unsupported basis for fill boundary: {basis}"
    raise LibLsQeccImportError(msg)


def _build_ancilla_blocks(  # noqa: C901
    components: Sequence[_AncillaComponent],
) -> tuple[dict[Coord3D, str], dict[Coord3D, str]]:
    basis_by_coord3: dict[Coord3D, str] = {}
    timeline_by_coord2: dict[Coord2D, list[tuple[int, str]]] = defaultdict(list)

    for component in components:
        for x, y in component.cells:
            coord3 = (x, y, component.z)
            existing_basis = basis_by_coord3.get(coord3)
            if existing_basis is not None and existing_basis != component.basis:
                msg = (
                    "Conflicting ancilla basis at the same coordinate/time: "
                    f"coord={coord3}, {existing_basis} vs {component.basis}"
                )
                raise LibLsQeccImportError(msg)
            basis_by_coord3[coord3] = component.basis
            timeline_by_coord2[x, y].append((component.z, component.basis))

    block_by_coord3: dict[Coord3D, str] = {}

    for coord2, timeline in timeline_by_coord2.items():
        timeline_sorted = sorted(timeline, key=itemgetter(0))
        segment_start = 0
        for idx in range(1, len(timeline_sorted) + 1):
            end_segment = idx == len(timeline_sorted)
            if not end_segment:
                prev_z, prev_basis = timeline_sorted[idx - 1]
                cur_z, cur_basis = timeline_sorted[idx]
                if cur_z == prev_z + 1 and cur_basis == prev_basis:
                    continue

            segment = timeline_sorted[segment_start:idx]
            basis = segment[0][1]
            zs = [z for z, _basis in segment]
            if len(segment) == 1:
                block = _short_block_for_basis(basis)
                block_by_coord3[coord2[0], coord2[1], zs[0]] = block
            else:
                z_first = zs[0]
                z_last = zs[-1]
                for z in zs:
                    if z == z_first:
                        block = _init_block_for_basis(basis)
                    elif z == z_last:
                        block = _measure_block_for_basis(basis)
                    else:
                        block = "MemoryBlock"
                    block_by_coord3[coord2[0], coord2[1], z] = block

            segment_start = idx

    return block_by_coord3, basis_by_coord3


def _direction_side(from_coord: Coord3D, to_coord: Coord3D) -> str:
    fx, fy, fz = from_coord
    tx, ty, tz = to_coord
    if fz != tz:
        msg = f"Only in-slice pipes are supported, got z mismatch: {from_coord} -> {to_coord}"
        raise LibLsQeccImportError(msg)

    dx = tx - fx
    dy = ty - fy
    if abs(dx) + abs(dy) != 1:
        msg = f"Pipe endpoints must be 4-neighbors: {from_coord} -> {to_coord}"
        raise LibLsQeccImportError(msg)

    if dx == 1:
        return "Right"
    if dx == -1:
        return "Left"
    if dy == 1:
        return "Bottom"
    return "Top"


def _pipe_key(a: Coord3D, b: Coord3D) -> tuple[Coord3D, Coord3D]:
    return (a, b) if a <= b else (b, a)


def _add_pipe(
    pipes_by_key: dict[tuple[Coord3D, Coord3D], _PipeInternal],
    *,
    a: Coord3D,
    b: Coord3D,
    block: str,
    basis: str,
) -> None:
    if a == b:
        msg = f"Internal error: zero-length pipe at {a}"
        raise LibLsQeccImportError(msg)

    key = _pipe_key(a, b)
    existing = pipes_by_key.get(key)
    if existing is not None:
        if existing.basis != basis:
            msg = f"Conflicting pipe basis for edge {key}: {existing.basis} vs {basis}"
            raise LibLsQeccImportError(msg)
        if existing.block != block:
            msg = f"Conflicting pipe block for edge {key}: {existing.block} vs {block}"
            raise LibLsQeccImportError(msg)
        return

    pipes_by_key[key] = _PipeInternal(start=key[0], end=key[1], block=block, basis=basis)


def _boundary_chars_to_string(chars: Mapping[str, str]) -> str:
    return "".join(chars[side] for side in _BOUNDARY_KEYS)


def _parse_boundary_string(boundary: object, *, context: str) -> dict[str, str]:
    if not isinstance(boundary, str) or len(boundary) != 4:  # noqa: PLR2004
        msg = f"{context}: boundary must be 4-char string, got {boundary!r}"
        raise LibLsQeccImportError(msg)
    return {side: boundary[idx] for idx, side in enumerate(_BOUNDARY_KEYS)}


def convert_slices_to_canvas_yaml(  # noqa: C901
    slices: Sequence[object],
    *,
    name: str,
    description: str | None = None,
    distillation_template: DistillationTemplateFn | None = None,
) -> str:
    """Convert liblsqecc slices JSON content to lspattern canvas YAML text."""
    if not name.strip():
        msg = "Canvas name must not be empty."
        raise LibLsQeccImportError(msg)

    normalized = _validate_and_normalize_slices(list(slices))
    rotation_cube_overrides, rotation_pipe_overrides = _detect_rotation_overrides(normalized)

    cube_entries_by_coord: dict[Coord3D, dict[str, object]] = {}
    qubit_coords: set[Coord3D] = set()

    previous_slice: list[list[Mapping[str, Any] | None]] | None = None
    for z, slice_rows in enumerate(normalized):
        for y, row in enumerate(slice_rows):
            for x, cell in enumerate(row):
                if not _is_qubit_cell(cell):
                    continue
                if cell is None:
                    continue

                previous_cell: Mapping[str, Any] | None = None
                if previous_slice is not None:
                    previous_cell = previous_slice[y][x]

                is_init = _is_init_cell(cell, previous_cell)
                is_measure = _is_measurement_cell(cell)

                if is_init:
                    block = "InitZeroBlock"
                elif is_measure:
                    block = "MeasureZBlock"
                else:
                    block = "MemoryBlock"

                coord3 = (x, y, z)
                cube_entries_by_coord[coord3] = {
                    "position": [x, y, z],
                    "block": block,
                    "boundary": _cell_boundary_string(cell, z=z, y=y, x=x),
                }
                qubit_coords.add(coord3)
        previous_slice = slice_rows

    ancilla_components: list[_AncillaComponent] = []
    for z, slice_rows in enumerate(normalized):
        ancilla_components.extend(_extract_valid_ancilla_components(slice_rows, z=z))

    ancilla_block_by_coord, ancilla_basis_by_coord = _build_ancilla_blocks(ancilla_components)

    ancilla_coords: set[Coord3D] = set()
    for coord3, block in ancilla_block_by_coord.items():
        if coord3 in cube_entries_by_coord:
            msg = f"Conflicting cube assignment at {coord3}: qubit and ancilla overlap"
            raise LibLsQeccImportError(msg)
        x, y, z = coord3
        cube_entries_by_coord[coord3] = {
            "position": [x, y, z],
            "block": block,
            "boundary": "",  # Filled after pipe analysis.
        }
        ancilla_coords.add(coord3)

    pipes_by_key: dict[tuple[Coord3D, Coord3D], _PipeInternal] = {}

    for component in ancilla_components:
        z = component.z

        for anc_a_2d, anc_b_2d in component.ancilla_links:
            anc_a_3d = (anc_a_2d[0], anc_a_2d[1], z)
            anc_b_3d = (anc_b_2d[0], anc_b_2d[1], z)
            block_a = ancilla_block_by_coord.get(anc_a_3d)
            block_b = ancilla_block_by_coord.get(anc_b_3d)
            if block_a is None or block_b is None:
                msg = f"Missing ancilla cube for internal pipe at z={z}: {anc_a_2d} - {anc_b_2d}"
                raise LibLsQeccImportError(msg)

            # Adjacent ancilla cells in the same bus may have different
            # block kinds when their timelines differ (e.g., one cell
            # lives for a single clock while a neighbor spans multiple).
            # Use the block of the lexicographically smaller coordinate
            # for deterministic pipe assignment.
            pipe_block = block_a if anc_a_3d <= anc_b_3d else block_b

            _add_pipe(
                pipes_by_key,
                a=anc_a_3d,
                b=anc_b_3d,
                block=pipe_block,
                basis=component.basis,
            )

        for anc_2d, qubit_2d in component.qubit_links:
            anc_3d = (anc_2d[0], anc_2d[1], z)
            qubit_3d = (qubit_2d[0], qubit_2d[1], z)
            if qubit_3d not in qubit_coords:
                msg = f"Missing qubit cube at {qubit_3d} for ancilla connection"
                raise LibLsQeccImportError(msg)
            block = ancilla_block_by_coord.get(anc_3d)
            if block is None:
                msg = f"Missing ancilla cube at {anc_3d} for qubit connection"
                raise LibLsQeccImportError(msg)

            _add_pipe(
                pipes_by_key,
                a=anc_3d,
                b=qubit_3d,
                block=block,
                basis=component.basis,
            )

    # Detect direct qubit-to-qubit stitched connections (lattice surgery
    # merges without an intermediate ancilla bus) and create pipes for
    # them so that O boundaries are properly backed by pipe entries.
    for z, slice_rows in enumerate(normalized):
        height = len(slice_rows)
        width = len(slice_rows[0]) if height > 0 else 0
        for y, row in enumerate(slice_rows):
            for x, cell in enumerate(row):
                if not _is_qubit_cell(cell):
                    continue
                if cell is None:
                    continue
                edges = cell["edges"]
                for side in ("Bottom", "Right"):
                    kind = _stitched_kind(edges[side])
                    if kind is None:
                        continue
                    nx, ny = _coord_side_to_neighbor((x, y), side)
                    if not _in_bounds(nx, ny, width, height):
                        continue
                    neighbor = slice_rows[ny][nx]
                    if not _is_qubit_cell(neighbor):
                        continue
                    if neighbor is None:
                        continue
                    opp_kind = _stitched_kind(neighbor["edges"][_OPPOSITE_SIDE[side]])
                    if opp_kind is None or opp_kind != kind:
                        continue
                    basis = "ZZ" if kind == "solid" else "XX"
                    block = _short_block_for_basis(basis)
                    a_3d = (x, y, z)
                    b_3d = (nx, ny, z)
                    _add_pipe(pipes_by_key, a=a_3d, b=b_3d, block=block, basis=basis)

    connected_sides: dict[Coord3D, set[str]] = defaultdict(set)
    pipe_entries: list[dict[str, object]] = []

    pipe_iteration = sorted(
        pipes_by_key.values(),
        key=lambda item: (item.start[2], item.start[1], item.start[0], item.end[2], item.end[1], item.end[0]),
    )
    for pipe in pipe_iteration:
        side_from_start = _direction_side(pipe.start, pipe.end)
        side_from_end = _OPPOSITE_SIDE[side_from_start]

        connected_sides[pipe.start].add(side_from_start)
        connected_sides[pipe.end].add(side_from_end)

        fill = _fill_char_for_basis(pipe.basis)
        boundary_chars = dict.fromkeys(_BOUNDARY_KEYS, fill)
        boundary_chars[side_from_start] = "O"
        boundary_chars[side_from_end] = "O"

        pipe_entries.append(
            {
                "start": [pipe.start[0], pipe.start[1], pipe.start[2]],
                "end": [pipe.end[0], pipe.end[1], pipe.end[2]],
                "block": pipe.block,
                "boundary": _boundary_chars_to_string(boundary_chars),
            }
        )

    for coord3, entry in cube_entries_by_coord.items():
        boundary_chars: dict[str, str]

        if coord3 in ancilla_coords:
            basis = ancilla_basis_by_coord.get(coord3)
            if basis is None:
                msg = f"Internal error: missing ancilla basis for {coord3}"
                raise LibLsQeccImportError(msg)
            fill = _fill_char_for_basis(basis)
            boundary_chars = dict.fromkeys(_BOUNDARY_KEYS, fill)
        else:
            boundary_chars = _parse_boundary_string(entry["boundary"], context=f"cube@{coord3}")

        for side in connected_sides.get(coord3, set()):
            boundary_chars[side] = "O"

        entry["boundary"] = _boundary_chars_to_string(boundary_chars)

    for coord3, entry in rotation_cube_overrides.items():
        cube_entries_by_coord[coord3] = entry

    if rotation_pipe_overrides:
        pipe_entries_by_key = {
            _pipe_key(
                tuple(entry["start"]),  # type: ignore[arg-type]
                tuple(entry["end"]),  # type: ignore[arg-type]
            ): entry
            for entry in pipe_entries
        }
        for key, entry in rotation_pipe_overrides.items():
            pipe_entries_by_key[key] = entry
        pipe_entries = list(pipe_entries_by_key.values())

    cube_entries = list(cube_entries_by_coord.values())
    cube_entries.sort(key=_cube_sort_key)
    pipe_entries.sort(key=_pipe_sort_key)

    if distillation_template is not None:
        factories = _detect_distillation_factories(normalized)
        for factory in factories:
            dist_cubes, dist_pipes = distillation_template(factory, len(normalized))
            cube_entries.extend(dist_cubes)
            pipe_entries.extend(dist_pipes)
        cube_entries.sort(key=_cube_sort_key)
        pipe_entries.sort(key=_pipe_sort_key)

    canvas_dict = {
        "name": name,
        "description": description or "Imported from liblsqecc slices JSON",
        "layout": "rotated_surface_code",
        "cube": cube_entries,
        "pipe": pipe_entries,
    }
    return yaml.safe_dump(canvas_dict, sort_keys=False, width=1000, default_flow_style=False)


def convert_slices_file_to_canvas_yaml(
    input_json: Path | str,
    output_yml: Path | str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    distillation_template: DistillationTemplateFn | None = None,
) -> str:
    """Convert a liblsqecc slices JSON file into lspattern canvas YAML text."""
    input_path = Path(input_json)
    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Failed to read input JSON file {input_path}: {exc}"
        raise LibLsQeccImportError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {input_path}: {exc}"
        raise LibLsQeccImportError(msg) from exc

    canvas_name = name or input_path.stem
    yaml_text = convert_slices_to_canvas_yaml(
        raw, name=canvas_name, description=description, distillation_template=distillation_template,
    )

    if output_yml is not None:
        output_path = Path(output_yml)
        try:
            output_path.write_text(yaml_text, encoding="utf-8")
        except OSError as exc:
            msg = f"Failed to write output YAML file {output_path}: {exc}"
            raise LibLsQeccImportError(msg) from exc

    return yaml_text
