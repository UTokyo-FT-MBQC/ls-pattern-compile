"""TypedDict definitions for graphqomb-studio/v1 JSON schema."""

from __future__ import annotations

from typing import Literal, TypedDict


class CoordinateDict(TypedDict):
    """3D coordinate dictionary."""

    x: float
    y: float
    z: float


class AxisMeasBasisDict(TypedDict):
    """Axis-based measurement basis dictionary."""

    type: Literal["axis"]
    axis: Literal["X", "Y", "Z"]
    sign: Literal["PLUS", "MINUS"]


class PlannerMeasBasisDict(TypedDict):
    """Planner-based measurement basis dictionary."""

    type: Literal["planner"]
    plane: Literal["XY", "YZ", "XZ"]
    angleCoeff: float


MeasBasisDict = AxisMeasBasisDict | PlannerMeasBasisDict


class InputNodeDict(TypedDict):
    """Input node dictionary."""

    id: str
    coordinate: CoordinateDict
    role: Literal["input"]
    measBasis: MeasBasisDict
    qubitIndex: int


class OutputNodeDict(TypedDict):
    """Output node dictionary."""

    id: str
    coordinate: CoordinateDict
    role: Literal["output"]
    qubitIndex: int


class IntermediateNodeDict(TypedDict):
    """Intermediate node dictionary."""

    id: str
    coordinate: CoordinateDict
    role: Literal["intermediate"]
    measBasis: MeasBasisDict


GraphNodeDict = InputNodeDict | OutputNodeDict | IntermediateNodeDict


class GraphEdgeDict(TypedDict):
    """Graph edge dictionary."""

    id: str
    source: str
    target: str


class FlowDefinitionDict(TypedDict):
    """Flow definition dictionary."""

    xflow: dict[str, list[str]]
    zflow: dict[str, list[str]] | Literal["auto"]


class TimeSliceDict(TypedDict):
    """Time slice dictionary for schedule timeline."""

    time: int
    prepareNodes: list[str]
    entangleEdges: list[str]
    measureNodes: list[str]


class ScheduleDict(TypedDict):
    """Schedule result dictionary."""

    prepareTime: dict[str, int | None]
    measureTime: dict[str, int | None]
    entangleTime: dict[str, int | None]
    timeline: list[TimeSliceDict]


class StudioProjectDict(TypedDict):
    """graphqomb-studio/v1 project dictionary."""

    name: str
    nodes: list[GraphNodeDict]
    edges: list[GraphEdgeDict]
    flow: FlowDefinitionDict
    schedule: ScheduleDict
