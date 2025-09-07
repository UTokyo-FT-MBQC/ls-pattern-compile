from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D


def _build_compiled_canvas_T43() -> CompiledRHGCanvas:
    """Replicate examples/visualize_T43.ipynb without modification."""
    d = 3
    canvass = RHGCanvasSkeleton("Memory X")

    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_trimmed = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}

    blocks = [
        (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((3, 3, 0)), InitPlusCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((2, 2, 1)), InitPlusCubeSkeleton(d=d, edgespec=edgespec_trimmed)),
        (PatchCoordGlobal3D((4, 4, 0)), InitPlusCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((4, 4, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
    ]
    pipes = [
        (
            PatchCoordGlobal3D((0, 0, 0)),
            PatchCoordGlobal3D((0, 0, 1)),
            MemoryPipeSkeleton(d=d, edgespec=edgespec),
        )
    ]

    for pos, sk in blocks:
        canvass.add_cube(pos, sk)
    for u, v, psk in pipes:
        canvass.add_pipe(u, v, psk)

    canvas = canvass.to_canvas()
    compiled: CompiledRHGCanvas = canvas.compile()
    return compiled


def _coord_key(c: tuple[int, int, int]) -> str:
    return f"{int(c[0])},{int(c[1])},{int(c[2])}"


def _snapshot_compiled_canvas(cg: CompiledRHGCanvas) -> dict[str, Any]:
    # Reverse coord2node for convenience
    coord2node = dict(cg.coord2node or {})
    node2coord = {int(nid): (int(x), int(y), int(z)) for (x, y, z), nid in coord2node.items()}
    g = getattr(cg, "global_graph", None)

    # Nodes as sorted coordinate triples
    coords_sorted = sorted([(int(x), int(y), int(z)) for (x, y, z) in coord2node.keys()])

    # Edges mapped to coordinates, sorted deterministically
    edges_coords: list[list[int]] = []
    if g is not None and hasattr(g, "physical_edges"):
        for u, v in g.physical_edges:
            cu = node2coord.get(int(u))
            cv = node2coord.get(int(v))
            if cu is None or cv is None:
                continue
            a = list(cu)
            b = list(cv)
            # order within an edge for determinism
            edge = a + b if tuple(a) <= tuple(b) else b + a
            edges_coords.append(edge)
    edges_coords.sort()

    # GraphState I/O registries mapped to coords
    inputs = {}
    outputs = {}
    if g is not None:
        if hasattr(g, "input_node_indices") and getattr(g, "input_node_indices"):
            for nid, lidx in g.input_node_indices.items():
                c = node2coord.get(int(nid))
                if c is not None:
                    inputs[_coord_key(c)] = int(lidx)
        if hasattr(g, "output_node_indices") and getattr(g, "output_node_indices"):
            for nid, lidx in g.output_node_indices.items():
                c = node2coord.get(int(nid))
                if c is not None:
                    outputs[_coord_key(c)] = int(lidx)

    # Portsets mapped to coords per patch
    def _ports_to_coords(portset: dict[tuple[int, int, int], list[int]]):
        snap: dict[str, list[str]] = {}
        for pos, nodes in (portset or {}).items():
            key = _coord_key((int(pos[0]), int(pos[1]), int(pos[2])))
            lst: list[str] = []
            for n in nodes:
                c = node2coord.get(int(n))
                if c is not None:
                    lst.append(_coord_key(c))
            snap[key] = sorted(lst)
        return snap

    in_ports = _ports_to_coords(cg.in_portset)
    out_ports = _ports_to_coords(cg.out_portset)
    cout_ports = _ports_to_coords(cg.cout_portset)

    snapshot = {
        "meta": {
            "layers": len(getattr(cg, "layers", []) or []),
            "zlist": list(getattr(cg, "zlist", []) or []),
            "coord_map": len(coord2node),
            "nodes": len(getattr(g, "physical_nodes", []) or []) if g is not None else 0,
            "edges": len(getattr(g, "physical_edges", []) or []) if g is not None else 0,
        },
        "coords": [list(c) for c in coords_sorted],
        "edges_coords": edges_coords,
        "inputs": dict(sorted(inputs.items())),
        "outputs": dict(sorted(outputs.items())),
        "in_ports": dict(sorted(in_ports.items())),
        "out_ports": dict(sorted(out_ports.items())),
        "cout_ports": dict(sorted(cout_ports.items())),
    }
    return snapshot


def _load_expected_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_snapshot(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


@pytest.mark.parametrize("update", [bool(int(os.environ.get("UPDATE_SNAPSHOTS", "0")))])
def test_T43_temporal_and_spatial_snapshot(update: bool) -> None:
    compiled = _build_compiled_canvas_T43()
    got = _snapshot_compiled_canvas(compiled)

    snap_path = Path(__file__).parent / "snapshots" / "T43_compiled_canvas.json"
    expected = _load_expected_snapshot(snap_path)

    if expected is None or update:
        # 初回（または明示更新）にスナップショットを書き出す
        _save_snapshot(snap_path, got)
        expected = got

    assert got == expected, "CompiledRHGCanvas snapshot mismatch for T43"
