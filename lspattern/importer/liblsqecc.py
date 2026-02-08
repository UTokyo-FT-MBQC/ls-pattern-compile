"""Convert liblsqecc slices JSON into lspattern canvas YAML."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

Coord2D = tuple[int, int]
Coord3D = tuple[int, int, int]


class LibLsQeccImportError(RuntimeError):
    """Raised when liblsqecc slices cannot be converted into canvas YAML."""


@dataclass(frozen=True)
class _AncillaComponent:
    z: int
    basis: str  # "ZZ" or "XX"
    cells: frozenset[Coord2D]
    ancilla_links: frozenset[tuple[Coord2D, Coord2D]]
    qubit_links: frozenset[tuple[Coord2D, Coord2D]]


@dataclass(frozen=True)
class _PipeInternal:
    start: Coord3D
    end: Coord3D
    block: str
    basis: str  # "ZZ" or "XX"


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


def _extract_valid_ancilla_components(  # noqa: C901
    slice_rows: list[list[Mapping[str, Any] | None]],
    *,
    z: int,
) -> list[_AncillaComponent]:
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

    components: list[_AncillaComponent] = []
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

        # Ignore non-measurement ancilla regions (e.g., distillation-side routing artifacts).
        if len(endpoint_stitch_kinds) != 2:  # noqa: PLR2004
            continue

        stitched_kinds: set[str] = set()
        for kinds in endpoint_stitch_kinds.values():
            stitched_kinds.update(kinds)

        basis = _basis_from_stitch_kinds(stitched_kinds, context=f"slice={z}, ancilla_component_start={start}")
        components.append(
            _AncillaComponent(
                z=z,
                basis=basis,
                cells=frozenset(cells),
                ancilla_links=frozenset(ancilla_links),
                qubit_links=frozenset(qubit_links),
            )
        )

    return components


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
) -> str:
    """Convert liblsqecc slices JSON content to lspattern canvas YAML text."""
    if not name.strip():
        msg = "Canvas name must not be empty."
        raise LibLsQeccImportError(msg)

    normalized = _validate_and_normalize_slices(list(slices))

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
            if block_a != block_b:
                msg = (
                    "Adjacent ancilla cubes have different block kinds at same time: "
                    f"{anc_a_3d}={block_a}, {anc_b_3d}={block_b}"
                )
                raise LibLsQeccImportError(msg)

            _add_pipe(
                pipes_by_key,
                a=anc_a_3d,
                b=anc_b_3d,
                block=block_a,
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

    cube_entries = list(cube_entries_by_coord.values())
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
    yaml_text = convert_slices_to_canvas_yaml(raw, name=canvas_name, description=description)

    if output_yml is not None:
        output_path = Path(output_yml)
        try:
            output_path.write_text(yaml_text, encoding="utf-8")
        except OSError as exc:
            msg = f"Failed to write output YAML file {output_path}: {exc}"
            raise LibLsQeccImportError(msg) from exc

    return yaml_text
