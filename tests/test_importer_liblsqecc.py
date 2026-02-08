"""Tests for liblsqecc slices importer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import yaml

from lspattern.importer.liblsqecc import (
    LibLsQeccImportError,
    convert_slices_file_to_canvas_yaml,
    convert_slices_to_canvas_yaml,
)


def _qubit_cell(
    *,
    qid: int,
    top: str = "Solid",
    bottom: str = "Solid",
    left: str = "Dashed",
    right: str = "Dashed",
    measurement: bool = False,
) -> dict[str, Any]:
    return {
        "patch_type": "Qubit",
        "edges": {
            "Top": top,
            "Bottom": bottom,
            "Left": left,
            "Right": right,
        },
        "activity": {"activity_type": "Measurement" if measurement else None},
        "text": f"Id: {qid}",
    }


def _ancilla_cell(
    *,
    top: str = "None",
    bottom: str = "None",
    left: str = "None",
    right: str = "None",
) -> dict[str, Any]:
    return {
        "patch_type": "Ancilla",
        "edges": {
            "Top": top,
            "Bottom": bottom,
            "Left": left,
            "Right": right,
        },
        "activity": {"activity_type": None},
        "text": "",
    }


def _distillation_cell() -> dict[str, Any]:
    return {
        "patch_type": "DistillationQubit",
        "edges": {
            "Top": "None",
            "Bottom": "None",
            "Left": "None",
            "Right": "None",
        },
        "activity": {"activity_type": None},
        "text": "",
    }


def _load_yaml(yaml_text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(yaml_text)
    assert isinstance(loaded, dict)
    return loaded


def _cube_entry(canvas: dict[str, Any], position: list[int]) -> dict[str, Any]:
    cube = canvas["cube"]
    assert isinstance(cube, list)
    for entry in cube:
        assert isinstance(entry, dict)
        if entry.get("position") == position:
            return entry
    msg = f"Cube entry not found at position {position}"
    raise AssertionError(msg)


def _pipe_entry(canvas: dict[str, Any], start: list[int], end: list[int]) -> dict[str, Any]:
    pipe = canvas["pipe"]
    assert isinstance(pipe, list)
    for entry in pipe:
        assert isinstance(entry, dict)
        if (entry.get("start") == start and entry.get("end") == end) or (
            entry.get("start") == end and entry.get("end") == start
        ):
            return entry
    msg = f"Pipe entry not found for edge {start} <-> {end}"
    raise AssertionError(msg)


def test_basic_cube_generation_from_minimal_slices() -> None:
    slices = [
        [[None, _qubit_cell(qid=7)]],
        [[None, _qubit_cell(qid=7)]],
    ]

    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="minimal"))

    assert canvas["name"] == "minimal"
    assert canvas["layout"] == "rotated_surface_code"
    assert _cube_entry(canvas, [1, 0, 0])["block"] == "InitZeroBlock"
    assert _cube_entry(canvas, [1, 0, 1])["block"] == "MemoryBlock"


def test_measurement_maps_to_measurez() -> None:
    slices = [
        [[_qubit_cell(qid=1)]],
        [[_qubit_cell(qid=1, measurement=True)]],
    ]

    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="meas"))
    assert _cube_entry(canvas, [0, 0, 1])["block"] == "MeasureZBlock"


def test_init_detection_by_appearance_and_id_change() -> None:
    slices = [
        [[_qubit_cell(qid=1)]],
        [[_qubit_cell(qid=1)]],
        [[_qubit_cell(qid=2)]],  # ID change at same coordinate
        [[None]],
        [[_qubit_cell(qid=3)]],  # re-appearance after disappearance
    ]

    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="init-detect"))
    assert _cube_entry(canvas, [0, 0, 0])["block"] == "InitZeroBlock"
    assert _cube_entry(canvas, [0, 0, 1])["block"] == "MemoryBlock"
    assert _cube_entry(canvas, [0, 0, 2])["block"] == "InitZeroBlock"
    assert _cube_entry(canvas, [0, 0, 4])["block"] == "InitZeroBlock"


def test_boundary_stitched_edges_become_open_boundary() -> None:
    slices = [
        [[_qubit_cell(qid=4, top="Solid", bottom="SolidStiched", left="DashedStiched", right="Dashed")]]
    ]
    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="boundary"))
    assert _cube_entry(canvas, [0, 0, 0])["boundary"] == "ZOOX"


def test_non_qubit_cells_ignored() -> None:
    slices = [
        [[_distillation_cell(), _ancilla_cell(), _qubit_cell(qid=9)]],
    ]
    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="ignore"))
    assert len(canvas["cube"]) == 1
    assert _cube_entry(canvas, [2, 0, 0])["block"] == "InitZeroBlock"


def test_output_has_empty_pipe_section() -> None:
    slices = [[[_qubit_cell(qid=0)]]]
    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="pipe-empty"))
    assert canvas["pipe"] == []


def test_short_zz_ancilla_maps_to_shortx_and_generates_pipes() -> None:
    slices = [
        [[
            _qubit_cell(qid=1, right="SolidStiched"),
            _ancilla_cell(),
            _qubit_cell(qid=2, left="SolidStiched"),
        ]]
    ]

    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="ancilla-short-zz"))

    assert _cube_entry(canvas, [1, 0, 0])["block"] == "ShortXMemoryBlock"
    assert _cube_entry(canvas, [1, 0, 0])["boundary"] == "XXOO"
    assert _cube_entry(canvas, [0, 0, 0])["boundary"] == "ZZXO"
    assert _cube_entry(canvas, [2, 0, 0])["boundary"] == "ZZOX"

    left_pipe = _pipe_entry(canvas, [0, 0, 0], [1, 0, 0])
    right_pipe = _pipe_entry(canvas, [1, 0, 0], [2, 0, 0])
    assert left_pipe["block"] == "ShortXMemoryBlock"
    assert right_pipe["block"] == "ShortXMemoryBlock"
    assert left_pipe["boundary"] == "XXOO"
    assert right_pipe["boundary"] == "XXOO"


def test_short_xx_ancilla_maps_to_shortz_and_generates_pipes() -> None:
    slices = [
        [[
            _qubit_cell(qid=1, right="DashedStiched"),
            _ancilla_cell(),
            _qubit_cell(qid=2, left="DashedStiched"),
        ]]
    ]

    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="ancilla-short-xx"))

    assert _cube_entry(canvas, [1, 0, 0])["block"] == "ShortZMemoryBlock"
    assert _cube_entry(canvas, [1, 0, 0])["boundary"] == "ZZOO"
    assert _pipe_entry(canvas, [0, 0, 0], [1, 0, 0])["boundary"] == "ZZOO"
    assert _pipe_entry(canvas, [1, 0, 0], [2, 0, 0])["boundary"] == "ZZOO"


def test_long_lived_zz_ancilla_maps_to_init_memory_measure() -> None:
    slices = [
        [[
            _qubit_cell(qid=1, right="SolidStiched"),
            _ancilla_cell(),
            _qubit_cell(qid=2, left="SolidStiched"),
        ]],
        [[
            _qubit_cell(qid=1, right="SolidStiched"),
            _ancilla_cell(),
            _qubit_cell(qid=2, left="SolidStiched"),
        ]],
        [[
            _qubit_cell(qid=1, right="SolidStiched"),
            _ancilla_cell(),
            _qubit_cell(qid=2, left="SolidStiched"),
        ]],
    ]

    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="ancilla-long-zz"))

    assert _cube_entry(canvas, [1, 0, 0])["block"] == "InitPlusBlock"
    assert _cube_entry(canvas, [1, 0, 1])["block"] == "MemoryBlock"
    assert _cube_entry(canvas, [1, 0, 2])["block"] == "MeasureXBlock"

    pipe = canvas["pipe"]
    assert isinstance(pipe, list)
    assert len(pipe) == 6
    expected_by_z = {0: "InitPlusBlock", 1: "MemoryBlock", 2: "MeasureXBlock"}
    for z, expected_block in expected_by_z.items():
        z_pipes = [entry for entry in pipe if entry["start"][2] == z and entry["end"][2] == z]
        assert len(z_pipes) == 2
        assert {entry["block"] for entry in z_pipes} == {expected_block}
        assert {entry["boundary"] for entry in z_pipes} == {"XXOO"}


def test_adjacent_ancilla_cells_generate_internal_pipe() -> None:
    slices = [
        [[
            _qubit_cell(qid=1, right="SolidStiched"),
            _ancilla_cell(right="AncillaJoin"),
            _ancilla_cell(left="AncillaJoin"),
            _qubit_cell(qid=2, left="SolidStiched"),
        ]]
    ]

    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="ancilla-internal-pipe"))

    assert _cube_entry(canvas, [1, 0, 0])["boundary"] == "XXOO"
    assert _cube_entry(canvas, [2, 0, 0])["boundary"] == "XXOO"
    assert _pipe_entry(canvas, [1, 0, 0], [2, 0, 0])["block"] == "ShortXMemoryBlock"
    assert _pipe_entry(canvas, [1, 0, 0], [2, 0, 0])["boundary"] == "XXOO"
    assert len(canvas["pipe"]) == 3


def test_ancilla_without_two_stitched_endpoints_is_ignored() -> None:
    slices = [[[_ancilla_cell()]]]
    canvas = _load_yaml(convert_slices_to_canvas_yaml(slices, name="ancilla-ignored"))
    assert canvas["cube"] == []
    assert canvas["pipe"] == []


def test_invalid_input_shape_raises() -> None:
    slices = [
        [[None], [None, None]],
    ]
    with pytest.raises(LibLsQeccImportError, match="column count"):
        convert_slices_to_canvas_yaml(slices, name="invalid")


def test_file_conversion_writes_yaml(tmp_path: Path) -> None:
    slices = [[[_qubit_cell(qid=11)]]]
    input_path = tmp_path / "sample_slices.json"
    output_path = tmp_path / "sample_canvas.yml"
    input_path.write_text(json.dumps(slices), encoding="utf-8")

    yaml_text = convert_slices_file_to_canvas_yaml(input_path, output_path)
    loaded = _load_yaml(yaml_text)

    assert output_path.read_text(encoding="utf-8") == yaml_text
    assert loaded["name"] == "sample_slices"
    assert _cube_entry(loaded, [0, 0, 0])["block"] == "InitZeroBlock"
