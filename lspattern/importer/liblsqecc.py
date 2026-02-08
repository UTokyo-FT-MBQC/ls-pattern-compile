"""Convert liblsqecc slices JSON into lspattern canvas YAML."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_QUBIT_PATCH_TYPE = "Qubit"
_MEASUREMENT_ACTIVITY = "Measurement"
_BOUNDARY_KEYS = ("Top", "Bottom", "Left", "Right")
_QUBIT_ID_RE = re.compile(r"^\s*Id:\s*(-?\d+)\s*$")


class LibLsQeccImportError(RuntimeError):
    """Raised when liblsqecc slices cannot be converted into canvas YAML."""


def _edge_to_boundary_char(edge: object, *, context: str) -> str:
    if not isinstance(edge, str):
        msg = f"{context}: edge value must be a string, got {type(edge)}"
        raise LibLsQeccImportError(msg)

    normalized = edge.strip().lower()
    if normalized == "solid":
        return "Z"
    if normalized == "dashed":
        return "X"
    if normalized in {"none", "solidstiched", "dashedstiched", "solidstitched", "dashedstitched"}:
        return "O"

    msg = f"{context}: unsupported edge value {edge!r}"
    raise LibLsQeccImportError(msg)


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


def convert_slices_to_canvas_yaml(
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
    cube_entries: list[dict[str, object]] = []

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

                cube_entries.append(
                    {
                        "position": [x, y, z],
                        "block": block,
                        "boundary": _cell_boundary_string(cell, z=z, y=y, x=x),
                    }
                )
        previous_slice = slice_rows

    cube_entries.sort(key=_cube_sort_key)

    canvas_dict = {
        "name": name,
        "description": description or "Imported from liblsqecc slices JSON",
        "layout": "rotated_surface_code",
        "cube": cube_entries,
        "pipe": [],
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
