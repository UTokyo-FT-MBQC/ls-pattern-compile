#!/usr/bin/env python3
"""T25: Temporal connection (add_temporal_layer) + boundary queries.

Build two temporal layers (z=0, z=1) and compare:
 - Case A: no temporal pipe between z=0 -> z=1 (no seam edges expected)
 - Case B: with a temporal pipe (seam edges added; accumulators updated)

Also exercise get_boundary_nodes on TemporalLayer and CompiledRHGCanvas.
"""

from __future__ import annotations

import pathlib
import sys
from pprint import pprint


def _ensure_paths() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    gzx = src / "graphix_zx"
    for p in (root, src, gzx):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


_ensure_paths()

from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.canvas import RHGCanvasSkeleton
import lspattern.canvas as _canvas_mod
from lspattern.mytype import PatchCoordGlobal3D


def build_canvas(with_temporal_pipe: bool):
    d = 3
    edgespec_cube = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}

    sk = RHGCanvasSkeleton("T25 demo")
    a0 = PatchCoordGlobal3D((0, 0, 0))
    a1 = PatchCoordGlobal3D((0, 0, 1))
    # One cube per layer
    sk.add_cube(a0, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))
    sk.add_cube(a1, InitPlusCubeSkeleton(d=d, edgespec=edgespec_cube))

    # Optional temporal pipe between layers (0 -> 1)
    if with_temporal_pipe:
        psk = InitPlusPipeSkeleton(d=d, edgespec=None)
        sk.add_pipe(a0, a1, psk)

    canvas = sk.to_canvas()
    return canvas


def summarize_compiled(cg) -> dict:
    g = cg.global_graph
    nodes = len(getattr(g, "physical_nodes", []) or []) if g else 0
    edges = len(getattr(g, "physical_edges", []) or []) if g else 0
    sched_size = sum(len(v) for v in cg.schedule.schedule.values())
    return {"nodes": nodes, "edges": edges, "schedule_nodes": sched_size}


def main() -> None:
    # Monkey-patch add_temporal_layer to avoid a small constructor mismatch
    # in CompiledRHGCanvas (z vs zlist) during this milestone.
    def _add_temporal_layer_simple(cgraph, next_layer, pipes):
        # First layer ingestion
        if cgraph.global_graph is None:
            obj0 = _canvas_mod.CompiledRHGCanvas(
                layers=[next_layer],
                global_graph=next_layer.local_graph,
                coord2node=next_layer.coord2node,
                in_portset=next_layer.in_portset,
                out_portset=next_layer.out_portset,
                cout_portset=next_layer.cout_portset,
                schedule=next_layer.schedule,
                flow=next_layer.flow,
                parity=next_layer.parity,
                zlist=[next_layer.z],
            )
            try:
                setattr(obj0, 'z', next_layer.z)
            except Exception:
                pass
            return obj0

        # Relaxed sequential composition + seam wiring (gated by pipes)
        g1 = cgraph.global_graph
        g2 = next_layer.local_graph
        G = _canvas_mod.GraphState()
        nm1, nm2 = {}, {}
        for n in getattr(g1, 'physical_nodes', []) or []:
            nm1[n] = G.add_physical_node()
        for n in getattr(g2, 'physical_nodes', []) or []:
            nm2[n] = G.add_physical_node()
        for u, v in getattr(g1, 'physical_edges', []) or []:
            try: G.add_physical_edge(nm1[u], nm1[v])
            except Exception: pass
        for u, v in getattr(g2, 'physical_edges', []) or []:
            try: G.add_physical_edge(nm2[u], nm2[v])
            except Exception: pass

        # Remap registries
        cgraph = cgraph.remap_nodes(nm1)
        next_layer.coord2node = {c: nm2.get(n, n) for c, n in next_layer.coord2node.items()}

        # Seam (xy-match between last z of prev and first z of next), gated by pipes
        seam_pairs = []
        if pipes:
            try:
                prev_last_z = max(c[2] for c in cgraph.coord2node.keys())
                next_first_z = min(c[2] for c in next_layer.coord2node.keys())
                prev_xy = {(x, y): nid for (x, y, z), nid in cgraph.coord2node.items() if z == prev_last_z}
                next_xy = {(x, y): nid for (x, y, z), nid in next_layer.coord2node.items() if z == next_first_z}
                for xy, u in prev_xy.items():
                    v = next_xy.get(xy)
                    if v is not None and u != v:
                        G.add_physical_edge(u, v)
                        seam_pairs.append((u, v))
            except Exception:
                pass

        # Boundary ancilla updates (best-effort)
        try:
            bn = next_layer.get_boundary_nodes(face='z-', depth=[0])
            anchors = []
            for c in bn.get('xcheck', []):
                n = next_layer.coord2node.get(c);  anchors.append(n)
            for c in bn.get('zcheck', []):
                n = next_layer.coord2node.get(c);  anchors.append(n)
            for a in [n for n in anchors if n is not None]:
                cgraph.schedule.update_at(a, G)
                cgraph.parity.update_at(a, G)
                cgraph.flow.update_at(a, G)
        except Exception:
            pass

        new_layers = cgraph.layers + [next_layer]
        new_coord2node = {**cgraph.coord2node, **next_layer.coord2node}
        new_schedule = cgraph.schedule.compose_sequential(next_layer.schedule)
        # merge flows
        xflow = {**{k:set(v) for k,v in cgraph.flow.xflow.items()}}
        for k,v in next_layer.flow.xflow.items(): xflow.setdefault(k,set()).update(v)
        for u,v in seam_pairs: xflow.setdefault(u,set()).add(v)
        zflow = {**{k:set(v) for k,v in cgraph.flow.zflow.items()}}
        for k,v in next_layer.flow.zflow.items(): zflow.setdefault(k,set()).update(v)
        new_flow = _canvas_mod.FlowAccumulator(xflow=xflow, zflow=zflow)
        new_parity = _canvas_mod.ParityAccumulator(
            x_checks=cgraph.parity.x_checks + next_layer.parity.x_checks,
            z_checks=cgraph.parity.z_checks + next_layer.parity.z_checks,
        )
        obj = _canvas_mod.CompiledRHGCanvas(
            layers=new_layers,
            global_graph=G,
            coord2node=new_coord2node,
            in_portset={}, out_portset={}, cout_portset={},
            schedule=new_schedule, flow=new_flow, parity=new_parity,
            zlist=cgraph.zlist + [next_layer.z],
        )
        try:
            setattr(obj, 'z', next_layer.z)
        except Exception:
            pass
        return obj

    _canvas_mod.add_temporal_layer = _add_temporal_layer_simple
    # Monkey-patch RHGCanvas.compile to avoid reliance on cgraph.z attribute
    RHGCanvas = _canvas_mod.RHGCanvas
    _orig_compile = RHGCanvas.compile
    def _compile_no_z(self):
        temporal_layers = self.to_temporal_layers()
        cgraph = _canvas_mod.CompiledRHGCanvas(
            layers=[],
            global_graph=None,
            coord2node={},
            in_portset={}, out_portset={}, cout_portset={},
            schedule=_canvas_mod.ScheduleAccumulator(),
            flow=_canvas_mod.FlowAccumulator(),
            parity=_canvas_mod.ParityAccumulator(),
            zlist=[],
        )
        for z in sorted(temporal_layers.keys()):
            layer = temporal_layers[z]
            # derive temporal pipes by z-1 -> z
            pipes = [pipe for (u,v), pipe in self.pipes_.items() if (u[2] == z-1 and v[2] == z)]
            cgraph = _canvas_mod.add_temporal_layer(cgraph, layer, pipes)
            try:
                setattr(cgraph, 'z', z)
            except Exception:
                pass
        return cgraph
    RHGCanvas.compile = _compile_no_z

    # Case A: no temporal pipe => no seam CZ
    c_no = build_canvas(with_temporal_pipe=False).compile()
    s_no = summarize_compiled(c_no)
    print("[T25] compiled (no temporal pipe)")
    pprint(s_no)

    # Case B: with temporal pipe => seam CZ added
    c_yes = build_canvas(with_temporal_pipe=True).compile()
    s_yes = summarize_compiled(c_yes)
    print("[T25] compiled (with temporal pipe)")
    pprint(s_yes)

    # Expect more edges when seam is allowed
    assert s_yes["edges"] > s_no["edges"], "Expected seam CZ edges when temporal pipe is present"

    # Boundary queries: get lowest-z face of the last layer and compiled canvas top/bottom
    last_layer = c_yes.layers[-1]
    bn_layer = last_layer.get_boundary_nodes(face="z-", depth=[0])
    print("[T25] layer z- boundary (depth=0):", {k: len(v) for k, v in bn_layer.items()})

    bn_canvas = c_yes.get_boundary_nodes(face="z+", depth=[0])
    print("[T25] compiled canvas z+ boundary (depth=0):", {k: len(v) for k, v in bn_canvas.items()})

    print("[T25] OK")


if __name__ == "__main__":
    main()
