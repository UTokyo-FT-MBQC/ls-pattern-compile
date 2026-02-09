"""Export Canvas to GraphQOMB Studio JSON format."""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphqomb.common import Axis

from lspattern.detector import construct_detector

if TYPE_CHECKING:
    from lspattern.accumulator import CoordScheduleAccumulator
    from lspattern.canvas import Canvas
    from lspattern.mytype import Coord3D


def _coord_to_node_id(coord: Coord3D) -> str:
    """Convert a 3D coordinate to a unique node ID string.

    Parameters
    ----------
    coord : Coord3D
        The coordinate to convert.

    Returns
    -------
    str
        A unique node ID string in the format "n_{x}_{y}_{z}".
    """
    return f"n_{coord.x}_{coord.y}_{coord.z}"


def _normalize_edge_id(source_id: str, target_id: str) -> str:
    """Generate a normalized edge ID by sorting source and target alphabetically.

    Parameters
    ----------
    source_id : str
        The source node ID.
    target_id : str
        The target node ID.

    Returns
    -------
    str
        A normalized edge ID in the format "{smaller_id}-{larger_id}".
    """
    if source_id < target_id:
        return f"{source_id}-{target_id}"
    return f"{target_id}-{source_id}"


def _axis_to_string(axis: Axis) -> str:
    """Convert Axis enum to string representation.

    Parameters
    ----------
    axis : Axis
        The axis enum value.

    Returns
    -------
    str
        String representation ("X", "Y", or "Z").
    """
    if axis == Axis.X:
        return "X"
    if axis == Axis.Y:
        return "Y"
    return "Z"


def _validate_coordinate_range(
    *,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
    z_min: int | None = None,
    z_max: int | None = None,
) -> None:
    """Validate axis-aligned coordinate range bounds.

    Raises
    ------
    ValueError
        If any axis has min > max.
    """
    axis_bounds = (
        ("x", x_min, x_max),
        ("y", y_min, y_max),
        ("z", z_min, z_max),
    )
    for axis_name, min_value, max_value in axis_bounds:
        if min_value is not None and max_value is not None and min_value > max_value:
            msg = (
                f"Invalid coordinate range: {axis_name}_min ({min_value}) must be less than or equal to "
                f"{axis_name}_max ({max_value})"
            )
            raise ValueError(msg)


def _coord_in_range(
    coord: Coord3D,
    *,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
    z_min: int | None = None,
    z_max: int | None = None,
) -> bool:
    """Return whether a coordinate is inside the closed axis-aligned range."""
    return (
        (x_min is None or coord.x >= x_min)
        and (x_max is None or coord.x <= x_max)
        and (y_min is None or coord.y >= y_min)
        and (y_max is None or coord.y <= y_max)
        and (z_min is None or coord.z >= z_min)
        and (z_max is None or coord.z <= z_max)
    )


def _build_allowed_nodes(
    canvas: Canvas,
    *,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
    z_min: int | None = None,
    z_max: int | None = None,
) -> set[Coord3D]:
    """Build the coordinate set that is kept by export range filtering."""
    _validate_coordinate_range(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )
    return {
        coord
        for coord in canvas.nodes
        if _coord_in_range(
            coord,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
    }


def _convert_nodes(canvas: Canvas, allowed_nodes: set[Coord3D] | None = None) -> list[dict[str, Any]]:
    """Convert canvas nodes to GraphQOMB Studio format.

    Parameters
    ----------
    canvas : Canvas
        The canvas to convert nodes from.

    Returns
    -------
    list[dict[str, Any]]
        List of node dictionaries in GraphQOMB Studio format.
    """
    nodes = []
    coords = canvas.nodes if allowed_nodes is None else allowed_nodes
    pauli_axes = canvas.pauli_axes
    for coord in sorted(coords):
        node_id = _coord_to_node_id(coord)
        axis = pauli_axes.get(coord, Axis.Z)
        node_dict: dict[str, Any] = {
            "id": node_id,
            "coordinate": {"x": coord.x, "y": coord.y, "z": coord.z},
            "role": "intermediate",
            "measBasis": {
                "type": "axis",
                "axis": _axis_to_string(axis),
                "sign": "PLUS",
            },
        }
        nodes.append(node_dict)
    return nodes


def _convert_edges(canvas: Canvas, allowed_nodes: set[Coord3D] | None = None) -> list[dict[str, str]]:
    """Convert canvas edges to GraphQOMB Studio format.

    Parameters
    ----------
    canvas : Canvas
        The canvas to convert edges from.

    Returns
    -------
    list[dict[str, str]]
        List of edge dictionaries in GraphQOMB Studio format.
    """
    edges = []
    seen_ids: set[str] = set()
    for coord1, coord2 in canvas.edges:
        if allowed_nodes is not None and (coord1 not in allowed_nodes or coord2 not in allowed_nodes):
            continue
        source_id = _coord_to_node_id(coord1)
        target_id = _coord_to_node_id(coord2)
        edge_id = _normalize_edge_id(source_id, target_id)
        if edge_id in seen_ids:
            continue
        seen_ids.add(edge_id)
        # Use smaller ID as source for consistency
        if source_id < target_id:
            edges.append({"id": edge_id, "source": source_id, "target": target_id})
        else:
            edges.append({"id": edge_id, "source": target_id, "target": source_id})
    return sorted(edges, key=operator.itemgetter("id"))


def _convert_xflow(canvas: Canvas, allowed_nodes: set[Coord3D] | None = None) -> dict[str, list[str]]:
    """Convert canvas flow to GraphQOMB Studio xflow format.

    Parameters
    ----------
    canvas : Canvas
        The canvas to convert flow from.

    Returns
    -------
    dict[str, list[str]]
        X-flow dictionary mapping source node IDs to lists of target node IDs.
    """
    xflow: dict[str, list[str]] = {}
    for from_coord, to_coords in canvas.flow.flow.items():
        if allowed_nodes is not None and from_coord not in allowed_nodes:
            continue
        from_id = _coord_to_node_id(from_coord)
        to_ids = sorted(
            _coord_to_node_id(c)
            for c in to_coords
            if allowed_nodes is None or c in allowed_nodes
        )
        if to_ids:
            xflow[from_id] = to_ids
    return xflow


def _build_node_time_map(
    canvas: Canvas,
    time_dict: dict[int, set[Coord3D]],
    allowed_nodes: set[Coord3D] | None = None,
) -> dict[str, int | None]:
    """Build a mapping from node IDs to their scheduled time.

    Parameters
    ----------
    canvas : Canvas
        The canvas containing node coordinates.
    time_dict : dict[int, set[Coord3D]]
        Time to coordinate mapping from scheduler.

    Returns
    -------
    dict[str, int | None]
        Mapping from node ID to scheduled time (or None if not scheduled).
    """
    nodes_for_map = canvas.nodes if allowed_nodes is None else allowed_nodes
    result: dict[str, int | None] = {_coord_to_node_id(c): None for c in nodes_for_map}
    for time, coords in time_dict.items():
        for coord in coords:
            if coord in nodes_for_map:
                result[_coord_to_node_id(coord)] = time
    return result


def _build_entangle_time_map(
    scheduler: CoordScheduleAccumulator,
    allowed_nodes: set[Coord3D] | None = None,
) -> dict[str, int]:
    """Build a mapping from edge IDs to their entanglement time.

    Parameters
    ----------
    scheduler : CoordScheduleAccumulator
        The scheduler containing entanglement time data.

    Returns
    -------
    dict[str, int]
        Mapping from edge ID to entanglement time.
    """
    result: dict[str, int] = {}
    for time, edges in scheduler.entangle_time.items():
        for coord1, coord2 in edges:
            if allowed_nodes is not None and (coord1 not in allowed_nodes or coord2 not in allowed_nodes):
                continue
            source_id = _coord_to_node_id(coord1)
            target_id = _coord_to_node_id(coord2)
            edge_id = _normalize_edge_id(source_id, target_id)
            result[edge_id] = time
    return result


def _build_timeline(
    scheduler: CoordScheduleAccumulator,
    allowed_nodes: set[Coord3D] | None = None,
) -> list[dict[str, Any]]:
    """Build timeline array from scheduler data.

    Parameters
    ----------
    scheduler : CoordScheduleAccumulator
        The scheduler containing time-indexed operations.

    Returns
    -------
    list[dict[str, Any]]
        List of timeline entries, each containing time and operations at that time.
    """
    all_times: set[int] = set()
    all_times.update(scheduler.prep_time.keys())
    all_times.update(scheduler.meas_time.keys())
    all_times.update(scheduler.entangle_time.keys())

    timeline: list[dict[str, Any]] = []
    for time in sorted(all_times):
        prep_nodes = sorted(
            _coord_to_node_id(c)
            for c in scheduler.prep_time.get(time, set())
            if allowed_nodes is None or c in allowed_nodes
        )
        meas_nodes = sorted(
            _coord_to_node_id(c)
            for c in scheduler.meas_time.get(time, set())
            if allowed_nodes is None or c in allowed_nodes
        )
        entangle_edges = sorted(
            _normalize_edge_id(_coord_to_node_id(c1), _coord_to_node_id(c2))
            for c1, c2 in scheduler.entangle_time.get(time, set())
            if allowed_nodes is None or (c1 in allowed_nodes and c2 in allowed_nodes)
        )
        if not prep_nodes and not entangle_edges and not meas_nodes:
            continue
        timeline.append(
            {
                "time": time,
                "prepareNodes": prep_nodes,
                "entangleEdges": entangle_edges,
                "measureNodes": meas_nodes,
            }
        )
    return timeline


def _convert_schedule(canvas: Canvas, allowed_nodes: set[Coord3D] | None = None) -> dict[str, Any]:
    """Convert canvas scheduler to GraphQOMB Studio schedule format.

    Parameters
    ----------
    canvas : Canvas
        The canvas to convert schedule from.

    Returns
    -------
    dict[str, Any]
        Schedule dictionary containing prepareTime, measureTime, entangleTime, and timeline.
    """
    scheduler = canvas.scheduler
    return {
        "prepareTime": _build_node_time_map(canvas, scheduler.prep_time, allowed_nodes=allowed_nodes),
        "measureTime": _build_node_time_map(canvas, scheduler.meas_time, allowed_nodes=allowed_nodes),
        "entangleTime": _build_entangle_time_map(scheduler, allowed_nodes=allowed_nodes),
        "timeline": _build_timeline(scheduler, allowed_nodes=allowed_nodes),
    }


def _convert_detectors(canvas: Canvas, allowed_nodes: set[Coord3D] | None = None) -> list[list[str]]:
    """Convert canvas parity accumulator to detector groups.

    Parameters
    ----------
    canvas : Canvas
        The canvas to construct detectors from.

    Returns
    -------
    list[list[str]]
        List of detector groups, each containing node IDs.
    """
    detectors = construct_detector(canvas.parity_accumulator)
    result: list[list[str]] = []
    for _, coords in sorted(detectors.items()):
        filtered_coords = coords if allowed_nodes is None else {c for c in coords if c in allowed_nodes}
        group = sorted(_coord_to_node_id(c) for c in filtered_coords)
        if group:
            result.append(group)
    return result


def _convert_logical_observables(
    canvas: Canvas, allowed_nodes: set[Coord3D] | None = None
) -> dict[str, list[str]]:
    """Convert canvas couts and pipe_couts to logical observable groups.

    Parameters
    ----------
    canvas : Canvas
        The canvas to convert logical observables from.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping observable labels to lists of node IDs.
    """
    merged: dict[str, set[Coord3D]] = {}

    # Merge cube couts
    for label_map in canvas.couts.values():
        for label, coords in label_map.items():
            if label not in merged:
                merged[label] = set()
            merged[label].update(coords)

    # Merge pipe couts
    for label_map in canvas.pipe_couts.values():
        for label, coords in label_map.items():
            if label not in merged:
                merged[label] = set()
            merged[label].update(coords)

    # Convert to output format
    result: dict[str, list[str]] = {}
    for label in sorted(merged.keys()):
        coords = merged[label]
        if allowed_nodes is not None:
            coords = {coord for coord in coords if coord in allowed_nodes}
        if not coords:
            continue
        result[label] = sorted(_coord_to_node_id(c) for c in coords)

    return result


def export_canvas_to_graphqomb_studio(
    canvas: Canvas,
    output_path: str | Path,
    name: str = "Exported from lspattern",
    *,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
    z_min: int | None = None,
    z_max: int | None = None,
) -> None:
    """Export Canvas to GraphQOMB Studio JSON format.

    Parameters
    ----------
    canvas : Canvas
        The canvas to export.
    output_path : str | Path
        Path to write the JSON file.
    name : str
        Project name in the exported JSON.
    x_min, x_max, y_min, y_max, z_min, z_max : int | None
        Closed-range bounds for node coordinates to export. Unspecified bounds
        are treated as unbounded.

    Examples
    --------
    >>> from lspattern.canvas import Canvas, CanvasConfig
    >>> config = CanvasConfig(name="test", description="test canvas", d=3, tiling="rotated_surface_code")
    >>> canvas = Canvas(config)
    >>> export_canvas_to_graphqomb_studio(canvas, "output.json", name="My Project")
    """
    allowed_nodes = _build_allowed_nodes(
        canvas,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )
    project: dict[str, Any] = {
        "$schema": "graphqomb-studio/v1",
        "name": name,
        "nodes": _convert_nodes(canvas, allowed_nodes=allowed_nodes),
        "edges": _convert_edges(canvas, allowed_nodes=allowed_nodes),
        "flow": {
            "xflow": _convert_xflow(canvas, allowed_nodes=allowed_nodes),
            "zflow": "auto",
        },
        "ftqc": {
            "parityCheckGroup": _convert_detectors(canvas, allowed_nodes=allowed_nodes),
            "logicalObservableGroup": _convert_logical_observables(canvas, allowed_nodes=allowed_nodes),
        },
        "schedule": _convert_schedule(canvas, allowed_nodes=allowed_nodes),
    }

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(project, f, indent=2)


def canvas_to_graphqomb_studio_dict(
    canvas: Canvas,
    name: str = "Exported from lspattern",
    *,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
    z_min: int | None = None,
    z_max: int | None = None,
) -> dict[str, Any]:
    """Convert Canvas to GraphQOMB Studio JSON dictionary.

    This is useful for testing or when you need the dictionary without writing to a file.

    Parameters
    ----------
    canvas : Canvas
        The canvas to convert.
    name : str
        Project name in the output dictionary.
    x_min, x_max, y_min, y_max, z_min, z_max : int | None
        Closed-range bounds for node coordinates to export. Unspecified bounds
        are treated as unbounded.

    Returns
    -------
    dict[str, Any]
        The GraphQOMB Studio JSON dictionary.
    """
    allowed_nodes = _build_allowed_nodes(
        canvas,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )
    return {
        "$schema": "graphqomb-studio/v1",
        "name": name,
        "nodes": _convert_nodes(canvas, allowed_nodes=allowed_nodes),
        "edges": _convert_edges(canvas, allowed_nodes=allowed_nodes),
        "flow": {
            "xflow": _convert_xflow(canvas, allowed_nodes=allowed_nodes),
            "zflow": "auto",
        },
        "ftqc": {
            "parityCheckGroup": _convert_detectors(canvas, allowed_nodes=allowed_nodes),
            "logicalObservableGroup": _convert_logical_observables(canvas, allowed_nodes=allowed_nodes),
        },
        "schedule": _convert_schedule(canvas, allowed_nodes=allowed_nodes),
    }
