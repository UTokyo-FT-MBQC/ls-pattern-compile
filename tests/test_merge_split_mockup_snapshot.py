from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import NodeIdLocal, PatchCoordGlobal3D, PhysCoordGlobal3D


def _build_compiled_canvas_mockup() -> CompiledRHGCanvas:
    """Replicate examples/merge_split_mockup.ipynb without modification."""
    d = 3
    canvass = RHGCanvasSkeleton("Memory X")

    edgespec: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec1: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.X,
        BoundarySide.RIGHT: EdgeSpecValue.O,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec2: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.X,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }
    edgespec_trimmed: dict[BoundarySide, EdgeSpecValue] = {
        BoundarySide.LEFT: EdgeSpecValue.O,
        BoundarySide.RIGHT: EdgeSpecValue.O,
        BoundarySide.TOP: EdgeSpecValue.Z,
        BoundarySide.BOTTOM: EdgeSpecValue.Z,
    }

    blocks = [
        (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((1, 0, 0)), InitPlusCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec1)),
        (PatchCoordGlobal3D((1, 0, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec2)),
        (PatchCoordGlobal3D((0, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
        (PatchCoordGlobal3D((1, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
    ]
    pipes = [
        (
            PatchCoordGlobal3D((0, 0, 1)),
            PatchCoordGlobal3D((1, 0, 1)),
            MemoryPipeSkeleton(d=d, edgespec=edgespec_trimmed),
        ),
        (
            PatchCoordGlobal3D((0, 0, 0)),
            PatchCoordGlobal3D((0, 0, 1)),
            MemoryPipeSkeleton(d=d, edgespec=edgespec_trimmed),
        ),
        # The following are commented in the notebook and must remain inactive:
        # (
        #     PatchCoordGlobal3D((0, 0, 1)),
        #     PatchCoordGlobal3D((0, 0, 2)),
        #     MemoryPipeSkeleton(d=d, edgespec=edgespec_trimmed),
        # ),
        # (
        #     PatchCoordGlobal3D((1, 0, 0)),
        #     PatchCoordGlobal3D((1, 0, 1)),
        #     MemoryPipeSkeleton(d=d, edgespec=edgespec_trimmed),
        # ),
        # (
        #     PatchCoordGlobal3D((1, 0, 1)),
        #     PatchCoordGlobal3D((1, 0, 2)),
        #     MemoryPipeSkeleton(d=d, edgespec=edgespec_trimmed),
        # ),
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


def _snapshot_compiled_canvas(cg: CompiledRHGCanvas) -> dict[str, Any]:  # noqa: C901
    # Reverse coord2node for convenience
    coord2node = dict(cg.coord2node or {})
    node2coord = {int(nid): (int(x), int(y), int(z)) for (x, y, z), nid in coord2node.items()}
    g = getattr(cg, "global_graph", None)

    # Nodes as sorted coordinate triples
    coords_sorted = sorted([(int(x), int(y), int(z)) for (x, y, z) in coord2node])

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
            edge = a + b if tuple(a) <= tuple(b) else b + a
            edges_coords.append(edge)
    edges_coords.sort()

    # GraphState I/O registries mapped to coords
    inputs = {}
    outputs = {}
    if g is not None:
        if hasattr(g, "input_node_indices") and g.input_node_indices:
            for nid, lidx in g.input_node_indices.items():
                c = node2coord.get(int(nid))
                if c is not None:
                    inputs[_coord_key(c)] = int(lidx)
        if hasattr(g, "output_node_indices") and g.output_node_indices:
            for nid, lidx in g.output_node_indices.items():
                c = node2coord.get(int(nid))
                if c is not None:
                    outputs[_coord_key(c)] = int(lidx)

    # Portsets mapped to coords per patch
    def _portset_to_ints(portset: dict[PatchCoordGlobal3D, list[NodeIdLocal]]) -> dict[tuple[int, int, int], list[int]]:
        converted: dict[tuple[int, int, int], list[int]] = {}
        for pos, nodes in (portset or {}).items():
            px, py, pz = cast("tuple[int, int, int]", pos)
            converted[int(px), int(py), int(pz)] = [int(n) for n in nodes]
        return converted

    def _ports_to_coords(portset: dict[tuple[int, int, int], list[int]]) -> dict[str, list[str]]:
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

    in_ports = _ports_to_coords(_portset_to_ints(cg.in_portset))
    out_ports = _ports_to_coords(_portset_to_ints(cg.out_portset))
    cout_ports = _ports_to_coords(_portset_to_ints(cg.cout_portset_cube))

    return {
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


def _load_expected_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def _save_snapshot(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


@pytest.mark.parametrize("update", [bool(int(os.environ.get("UPDATE_SNAPSHOTS", "0")))])
def test_merge_split_mockup_snapshot(update: bool) -> None:
    compiled = _build_compiled_canvas_mockup()
    got = _snapshot_compiled_canvas(compiled)

    snap_path = Path(__file__).parent / "snapshots" / "mockup_compiled_canvas.json"
    expected = _load_expected_snapshot(snap_path)

    if expected is None or update:
        _save_snapshot(snap_path, got)
        expected = got

    assert got == expected, "CompiledRHGCanvas snapshot mismatch for merge_split_mockup"


def test_cout_group_resolution_interface() -> None:
    compiled = _build_compiled_canvas_mockup()
    # Initially no cout groups are registered
    sample_coord, sample_node = next(iter(compiled.coord2node.items()))
    assert compiled.get_cout_group_by_coord(sample_coord) is None

    # Inject a cout group and ensure lookup works
    first_layer = compiled.layers[0]
    patch = first_layer.patches[0]
    compiled.port_manager.cout_port_groups_cube = {patch: [[sample_node]]}
    compiled.port_manager.rebuild_cout_group_cache()

    fetched = compiled.get_cout_group_by_coord(sample_coord)
    assert fetched is not None
    fetched_patch, nodes = fetched
    assert fetched_patch == patch
    assert nodes == [sample_node]

    resolved = compiled.resolve_cout_groups({"logical": [sample_coord]})
    assert resolved == {"logical": [sample_node]}

    with pytest.raises(KeyError):
        compiled.resolve_cout_groups({"missing": [PhysCoordGlobal3D((999, 999, 999))]})
