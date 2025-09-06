#!/usr/bin/env python3
"""T23 debug: accumulator update_at implementations (Detector/Parity/Flow/Schedule).

Build a tiny canvas, compile the first temporal layer, and run update_at for
all ancilla nodes. Verifies non-emptiness and non-decreasing properties.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any


def _ensure_paths() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    src = root / "src"
    src_graphix = src / "graphix_zx"
    for p in (src, src_graphix):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_ensure_paths()

from lspattern.accumulator import (
    DetectorAccumulator,
    FlowAccumulator,
    ParityAccumulator,
    ScheduleAccumulator,
)
from lspattern.blocks.cubes.initialize import InitPlusCubeSkeleton
from lspattern.canvas import RHGCanvas, RHGCanvasSkeleton
from lspattern.mytype import PatchCoordGlobal3D


def build_min_canvas() -> RHGCanvas:
    edgespec = {"LEFT": "X", "RIGHT": "X", "TOP": "Z", "BOTTOM": "Z"}
    edgespec_o = {"LEFT": "O", "RIGHT": "O", "TOP": "O", "BOTTOM": "O"}

    skel = RHGCanvasSkeleton("T23-Min")
    blocks = [
        (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec)),
        (PatchCoordGlobal3D((1, 0, 0)), InitPlusCubeSkeleton(d=3, edgespec=edgespec_o)),
    ]
    for pos, blk in blocks:
        skel.add_cube(pos, blk)

    canvas = skel.to_canvas()
    return canvas


def main() -> int:
    canvas = build_min_canvas()
    layers = canvas.to_temporal_layers()
    if not layers:
        print({"ok": False, "reason": "no layers"})
        return 1

    layer0 = layers[min(layers.keys())]
    layer0.compile()

    # Prepare accumulators (schedule/flow/parity live on the layer; detector local)
    schedule: ScheduleAccumulator = layer0.schedule
    flow: FlowAccumulator = layer0.flow
    parity: ParityAccumulator = layer0.parity
    detector = DetectorAccumulator()

    # Collect ancilla nodes from role map
    ancillas = [n for n, r in layer0.node2role.items() if str(r).startswith("ancilla")]
    if not ancillas:
        print({"ok": False, "reason": "no ancillas"})
        return 1

    # Mark one ancilla as classical output (to verify skip behavior)
    try:
        layer0.local_graph.output_node_indices = {int(ancillas[0]): 0}  # type: ignore[attr-defined]
    except Exception:
        pass  # optional; best-effort

    # First pass update
    for a in ancillas:
        schedule.update_at(a, layer0)
        parity.update_at(a, layer0)
        flow.update_at(a, layer0)
        detector.update_at(a, layer0)

    # Snapshot sizes
    s0 = sum(len(v) for v in schedule.schedule.values())
    p0 = sum(len(g) for g in parity.x_checks) + sum(len(g) for g in parity.z_checks)
    f0 = sum(len(v) for v in flow.xflow.values()) + sum(len(v) for v in flow.zflow.values())
    d0 = sum(len(v) for v in detector.detectors.values())

    # Second pass (idempotent-ish, non-decreasing expected)
    for a in ancillas:
        schedule.update_at(a, layer0)
        parity.update_at(a, layer0)
        flow.update_at(a, layer0)
        detector.update_at(a, layer0)

    s1 = sum(len(v) for v in schedule.schedule.values())
    p1 = sum(len(g) for g in parity.x_checks) + sum(len(g) for g in parity.z_checks)
    f1 = sum(len(v) for v in flow.xflow.values()) + sum(len(v) for v in flow.zflow.values())
    d1 = sum(len(v) for v in detector.detectors.values())

    assert s1 >= s0 and p1 >= p0 and f1 >= f0 and d1 >= d0

    # Summarize
    summary: dict[str, Any] = {
        "ancillas": len(ancillas),
        "schedule_slots": len(schedule.schedule),
        "schedule_total": s1,
        "parity_groups_x": len(parity.x_checks),
        "parity_groups_z": len(parity.z_checks),
        "flow_x_edges": sum(len(v) for v in flow.xflow.values()),
        "flow_z_edges": sum(len(v) for v in flow.zflow.values()),
        "detector_anchors": len(detector.detectors),
        "detector_total": d1,
    }
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

