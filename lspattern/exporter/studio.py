"""Export Canvas to graphqomb-studio JSON format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from graphqomb.common import Axis

    from lspattern.canvas import Canvas
    from lspattern.exporter.types import (
        AxisMeasBasisDict,
        CoordinateDict,
        FlowDefinitionDict,
        GraphEdgeDict,
        GraphNodeDict,
        InputNodeDict,
        IntermediateNodeDict,
        OutputNodeDict,
        ScheduleDict,
        TimeSliceDict,
    )
    from lspattern.mytype import Coord3D


def coord_to_id(coord: Coord3D) -> str:
    """Convert Coord3D to string node ID.

    Parameters
    ----------
    coord : Coord3D
        The 3D coordinate to convert.

    Returns
    -------
    str
        Node ID in format "x_y_z".
    """
    return f"{coord.x}_{coord.y}_{coord.z}"


def coord_to_dict(coord: Coord3D) -> CoordinateDict:
    """Convert Coord3D to coordinate dictionary.

    Parameters
    ----------
    coord : Coord3D
        The 3D coordinate to convert.

    Returns
    -------
    CoordinateDict
        Coordinate dictionary with x, y, z fields.
    """
    return {"x": float(coord.x), "y": float(coord.y), "z": float(coord.z)}


def axis_to_meas_basis(axis: Axis, sign: str = "PLUS") -> AxisMeasBasisDict:
    """Convert Axis enum to measurement basis dictionary.

    Parameters
    ----------
    axis : Axis
        The measurement axis (X, Y, or Z).
    sign : str
        The measurement sign ("PLUS" or "MINUS"). Defaults to "PLUS".

    Returns
    -------
    AxisMeasBasisDict
        Measurement basis dictionary.
    """
    axis_name = axis.name
    if axis_name not in {"X", "Y", "Z"}:
        msg = f"Invalid axis: {axis_name}"
        raise ValueError(msg)
    sign_literal: Literal["PLUS", "MINUS"] = "PLUS" if sign != "MINUS" else "MINUS"
    axis_literal: Literal["X", "Y", "Z"]
    if axis_name == "X":
        axis_literal = "X"
    elif axis_name == "Y":
        axis_literal = "Y"
    else:
        axis_literal = "Z"
    return {"type": "axis", "axis": axis_literal, "sign": sign_literal}


def normalize_edge_id(source_id: str, target_id: str) -> str:
    """Normalize edge ID by sorting endpoints alphabetically.

    Parameters
    ----------
    source_id : str
        Source node ID.
    target_id : str
        Target node ID.

    Returns
    -------
    str
        Normalized edge ID in format "a-b" where a < b alphabetically.
    """
    a, b = sorted([source_id, target_id])
    return f"{a}-{b}"


@dataclass
class ExportConfig:
    """Configuration for studio export.

    Attributes
    ----------
    input_nodes : set[Coord3D]
        Set of coordinates to mark as input nodes.
    output_nodes : set[Coord3D] | None
        Set of coordinates to mark as output nodes.
        If None, uses canvas.couts and canvas.pipe_couts.
    qubit_index_map : dict[Coord3D, int]
        Explicit qubit index assignments for input/output nodes.
    """

    input_nodes: set[Coord3D] = field(default_factory=set)
    output_nodes: set[Coord3D] | None = None
    qubit_index_map: dict[Coord3D, int] = field(default_factory=dict)


def _collect_output_nodes_from_couts(canvas: Canvas) -> set[Coord3D]:
    """Collect output node candidates from canvas couts.

    Parameters
    ----------
    canvas : Canvas
        The canvas to collect output nodes from.

    Returns
    -------
    set[Coord3D]
        Set of output node coordinates.
    """
    output_nodes: set[Coord3D] = set()
    for coords in canvas.couts.values():
        output_nodes.update(coords)
    for coords in canvas.pipe_couts.values():
        output_nodes.update(coords)
    return output_nodes


def _convert_nodes(
    canvas: Canvas,
    config: ExportConfig,
) -> list[GraphNodeDict]:
    """Convert canvas nodes to studio format.

    Parameters
    ----------
    canvas : Canvas
        The canvas containing nodes.
    config : ExportConfig
        Export configuration.

    Returns
    -------
    list[GraphNodeDict]
        List of node dictionaries.

    Raises
    ------
    ValueError
        If a non-output node is missing measurement axis.
    """
    nodes: list[GraphNodeDict] = []

    # Use couts as default output nodes if not explicitly specified
    output_nodes = config.output_nodes
    if output_nodes is None:
        output_nodes = _collect_output_nodes_from_couts(canvas)
    input_nodes = config.input_nodes

    # Auto-assign qubit indices if not provided
    qubit_index_map = dict(config.qubit_index_map)
    input_idx = 0
    output_idx = 0

    for coord in sorted(canvas.nodes):
        node_id = coord_to_id(coord)

        if coord in output_nodes:
            # Output nodes: no measBasis, require qubitIndex
            if coord in qubit_index_map:
                q_idx = qubit_index_map[coord]
            else:
                q_idx = output_idx
                output_idx += 1
            node: OutputNodeDict = {
                "id": node_id,
                "coordinate": coord_to_dict(coord),
                "role": "output",
                "qubitIndex": q_idx,
            }
            nodes.append(node)

        elif coord in input_nodes:
            # Input nodes: require measBasis and qubitIndex
            axis = canvas.pauli_axes.get(coord)
            if axis is None:
                msg = f"Input node {coord} missing measurement axis"
                raise ValueError(msg)
            if coord in qubit_index_map:
                q_idx = qubit_index_map[coord]
            else:
                q_idx = input_idx
                input_idx += 1
            input_node: InputNodeDict = {
                "id": node_id,
                "coordinate": coord_to_dict(coord),
                "role": "input",
                "measBasis": axis_to_meas_basis(axis),
                "qubitIndex": q_idx,
            }
            nodes.append(input_node)

        else:
            # Intermediate nodes: require measBasis, no qubitIndex
            axis = canvas.pauli_axes.get(coord)
            if axis is None:
                msg = f"Intermediate node {coord} missing measurement axis"
                raise ValueError(msg)
            intermediate_node: IntermediateNodeDict = {
                "id": node_id,
                "coordinate": coord_to_dict(coord),
                "role": "intermediate",
                "measBasis": axis_to_meas_basis(axis),
            }
            nodes.append(intermediate_node)

    return nodes


def _convert_edges(canvas: Canvas) -> list[GraphEdgeDict]:
    """Convert canvas edges to studio format.

    Parameters
    ----------
    canvas : Canvas
        The canvas containing edges.

    Returns
    -------
    list[GraphEdgeDict]
        List of edge dictionaries.
    """
    edges: list[GraphEdgeDict] = []
    seen_edge_ids: set[str] = set()

    for coord_a, coord_b in sorted(canvas.edges):
        source_id = coord_to_id(coord_a)
        target_id = coord_to_id(coord_b)
        edge_id = normalize_edge_id(source_id, target_id)

        # Skip duplicate edges
        if edge_id in seen_edge_ids:
            continue
        seen_edge_ids.add(edge_id)

        edges.append({
            "id": edge_id,
            "source": source_id,
            "target": target_id,
        })

    return edges


def _convert_flow(canvas: Canvas) -> FlowDefinitionDict:
    """Convert canvas flow to studio format.

    Canvas only stores xflow (via canvas.flow). zflow is always set to "auto"
    to let graphqomb-studio compute it automatically.

    Parameters
    ----------
    canvas : Canvas
        The canvas containing flow.

    Returns
    -------
    FlowDefinitionDict
        Flow definition dictionary with xflow and zflow="auto".
    """
    xflow: dict[str, list[str]] = {}

    for from_coord, to_coords in canvas.flow.flow.items():
        from_id = coord_to_id(from_coord)
        to_ids = [coord_to_id(c) for c in sorted(to_coords)]
        xflow[from_id] = to_ids

    return {"xflow": xflow, "zflow": "auto"}


def _convert_schedule(canvas: Canvas) -> ScheduleDict:
    """Convert canvas scheduler to studio format.

    Parameters
    ----------
    canvas : Canvas
        The canvas containing scheduler.

    Returns
    -------
    ScheduleDict
        Schedule dictionary.
    """
    prep_time: dict[str, int | None] = {}
    meas_time: dict[str, int | None] = {}
    entangle_time: dict[str, int | None] = {}

    # Build node/edge -> time mappings
    for time, coords in canvas.scheduler.prep_time.items():
        for coord in coords:
            node_id = coord_to_id(coord)
            prep_time[node_id] = time

    for time, coords in canvas.scheduler.meas_time.items():
        for coord in coords:
            node_id = coord_to_id(coord)
            meas_time[node_id] = time

    for time, edge_set in canvas.scheduler.entangle_time.items():
        for coord_a, coord_b in edge_set:
            edge_id = normalize_edge_id(coord_to_id(coord_a), coord_to_id(coord_b))
            entangle_time[edge_id] = time

    # Build timeline
    all_times: set[int] = set(canvas.scheduler.prep_time.keys())
    all_times.update(canvas.scheduler.meas_time.keys())
    all_times.update(canvas.scheduler.entangle_time.keys())

    timeline: list[TimeSliceDict] = []
    for time in sorted(all_times):
        prep_coords = canvas.scheduler.prep_time.get(time, set())
        meas_coords = canvas.scheduler.meas_time.get(time, set())
        entangle_edges = canvas.scheduler.entangle_time.get(time, set())

        slice_dict: TimeSliceDict = {
            "time": time,
            "prepareNodes": [coord_to_id(c) for c in sorted(prep_coords)],
            "entangleEdges": [
                normalize_edge_id(coord_to_id(a), coord_to_id(b)) for a, b in sorted(entangle_edges)
            ],
            "measureNodes": [coord_to_id(c) for c in sorted(meas_coords)],
        }
        timeline.append(slice_dict)

    return {
        "prepareTime": prep_time,
        "measureTime": meas_time,
        "entangleTime": entangle_time,
        "timeline": timeline,
    }


def export_to_studio(
    canvas: Canvas,
    name: str,
    *,
    config: ExportConfig | None = None,
) -> dict[str, Any]:
    """Export Canvas to graphqomb-studio JSON format.

    Parameters
    ----------
    canvas : Canvas
        The canvas to export.
    name : str
        Name for the exported project.
    config : ExportConfig | None
        Optional export configuration. If None, uses defaults:
        - output_nodes derived from canvas.couts/pipe_couts
        - input_nodes empty (all non-output nodes become intermediate)

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary in graphqomb-studio/v1 format.

    Raises
    ------
    ValueError
        If required data is missing (e.g., measurement axis for intermediate node).
    """
    if config is None:
        config = ExportConfig()

    return {
        "$schema": "graphqomb-studio/v1",
        "name": name,
        "nodes": _convert_nodes(canvas, config),
        "edges": _convert_edges(canvas),
        "flow": _convert_flow(canvas),
        "schedule": _convert_schedule(canvas),
    }


def save_to_studio_json(
    canvas: Canvas,
    name: str,
    path: Path,
    *,
    config: ExportConfig | None = None,
    indent: int = 2,
) -> None:
    """Save Canvas to graphqomb-studio JSON file.

    Parameters
    ----------
    canvas : Canvas
        The canvas to export.
    name : str
        Name for the exported project.
    path : Path
        Output file path.
    config : ExportConfig | None
        Optional export configuration.
    indent : int
        JSON indentation level (default 2).
    """
    data = export_to_studio(canvas, name, config=config)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
