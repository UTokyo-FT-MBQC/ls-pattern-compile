"""
CNOT Gate Implementation using RHG Lattice Surgery

This example demonstrates a CNOT (controlled-NOT) gate implementation using
the blocks-and-pipes architecture with lattice surgery operations.

Circuit structure:
- Clock 0: Initialize control |0⟩, target |+⟩, ancilla |+⟩
- Clock 1-2: Memory and ZZ-basis merge (control-target)
- Clock 3: XX-basis split measurement
- Clock 4-5: XX-basis merge (target-ancilla)
- Clock 6: Final measurements (Z for control, X for target and ancilla)
"""

# %%
import pathlib
import stim

import pymatching
from lspattern.blocks.cubes.initialize import (
    InitPlusCubeThinLayerSkeleton,
    InitZeroCubeThinLayerSkeleton,
)
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton, InitZeroPipeSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton, MeasureZPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureXSkeleton, MeasureZSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.compile import compile_to_stim
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.utils import to_edgespec
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
d = 3

canvass = RHGCanvasSkeleton("CNOT")

edgespec: dict[BoundarySide, EdgeSpecValue] = to_edgespec("ZZXX")
blocks = [
    # Clock 0 (init Zero)
    (
        PatchCoordGlobal3D((0, 0, 0)),
        InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 0)),
        InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 1, 0)),
        InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    # Clock 1 (Memory)
    (
        PatchCoordGlobal3D((0, 0, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 1, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    # Clock 2 (Merge ZZ)
    (
        PatchCoordGlobal3D((0, 0, 2)),
        MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZOX")),
    ),
    (
        PatchCoordGlobal3D((0, 1, 2)),
        MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZZXO")),
    ),
    (
        PatchCoordGlobal3D((1, 1, 2)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 3)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 3)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 1, 3)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    # Clock 3 (Split and Merge XX)
    (
        PatchCoordGlobal3D((0, 0, 3)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 3)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 1, 3)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    # Clock 4 (Split and Memory)
    (
        PatchCoordGlobal3D((0, 0, 4)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 4)),
        MemoryCubeSkeleton(d=d, edgespec=to_edgespec("ZOXX")),
    ),
    (
        PatchCoordGlobal3D((1, 1, 4)),
        MemoryCubeSkeleton(d=d, edgespec=to_edgespec("OZXX")),
    ),
    # Clock 5 (Memory)
    (
        PatchCoordGlobal3D((0, 0, 5)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 5)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 1, 5)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    # Clock 6 (Measure Z all)
    (
        PatchCoordGlobal3D((0, 0, 6)),
        MeasureZSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 1, 6)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 1, 6)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
]
pipes = [
    # Clock 2 (Merge ZZ -> Split ZZ)
    (
        PatchCoordGlobal3D((0, 0, 2)),
        PatchCoordGlobal3D((0, 1, 2)),
        InitPlusPipeSkeleton(d=d, edgespec=to_edgespec("ZZOO")),
    ),
    # # Clock 3 (Split XX)
    (
        PatchCoordGlobal3D((0, 0, 3)),
        PatchCoordGlobal3D((0, 1, 3)),
        MeasureXPipeSkeleton(d=d, edgespec=to_edgespec("OOOO")),
    ),
    # Clock 4
    (
        PatchCoordGlobal3D((0, 1, 4)),
        PatchCoordGlobal3D((1, 1, 4)),
        InitZeroPipeSkeleton(d=d, edgespec=to_edgespec("OOXX")),
    ),
    # Clock 5
    (
        PatchCoordGlobal3D((0, 1, 5)),
        PatchCoordGlobal3D((1, 1, 5)),
        MeasureZPipeSkeleton(d=d, edgespec=to_edgespec("OOOO")),
    ),
]

for block in blocks:
    canvass.add_cube(*block)
for pipe in pipes:
    canvass.add_pipe(*pipe)

canvas = canvass.to_canvas()

compiled_canvas: CompiledRHGCanvas = canvas.compile()
nnodes = (
    len(getattr(compiled_canvas.global_graph, "physical_nodes", []) or [])
    if compiled_canvas.global_graph
    else 0
)
nedges = (
    len(getattr(compiled_canvas.global_graph, "physical_edges", []) or [])
    if compiled_canvas.global_graph
    else 0
)
print(
    {
        "layers": len(compiled_canvas.layers),
        "nodes": nnodes,
        "edges": nedges,
        "coord_map": len(compiled_canvas.coord2node),
    }
)
output_indices = compiled_canvas.global_graph.output_node_indices or {}  # type: ignore[union-attr]
print(f"output qubits: {output_indices}")
fig3d = visualize_compiled_canvas_plotly(
    compiled_canvas,
    show_edges=True,
)
fig3d.show()

# %%

# Print flow and parity information
xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}

parity = []
for group_dict in compiled_canvas.parity.checks.values():
    for group in group_dict.values():
        parity.append({int(node) for node in group})

# Print X flow organized by schedule if available
print("X flow:")
compact_schedule = compiled_canvas.schedule.compact()
for t, nodes in compact_schedule.schedule.items():
    flows_at_time = []
    for node in nodes:
        if node in xflow:
            flows_at_time.append(f"{node} -> {xflow[node]}")
    if flows_at_time:
        print(f"  Time {t}: {', '.join(flows_at_time)}")
# Print any remaining flows not in schedule
scheduled_nodes = set()
for nodes in compact_schedule.schedule.values():
    scheduled_nodes.update(nodes)
remaining_flows = {
    src: dsts for src, dsts in xflow.items() if src not in scheduled_nodes
}
if remaining_flows:
    remaining_flow_strs = [f"{src} -> {dsts}" for src, dsts in remaining_flows.items()]
    print(f"  Unscheduled flows: {', '.join(remaining_flow_strs)}")

# %%

print("X parity")
for coord, group_list in compiled_canvas.parity.checks.items():
    print(f"  {coord}: {group_list}")

print("\nDetailed Flow Information:")
print(f"Total flow edges: {sum(len(dsts) for dsts in xflow.values())}")
print(f"Flow sources: {len(xflow)}")
print(
    f"Flow coverage: {len(set().union(*xflow.values()) if xflow else set())} unique destinations"
)

# Print detector/stabilizer information
print("\nDetector Information:")
total_detectors = sum(
    len(group_dict) for group_dict in compiled_canvas.parity.checks.values()
)
print(f"Total detectors: {total_detectors}")
for coord, group_dict in compiled_canvas.parity.checks.items():
    print(f"  Patch {coord}: {len(group_dict)} detector groups")

# classical outs
cout_portmap = compiled_canvas.cout_portset_cube
cout_portmap_pipe = compiled_canvas.cout_portset_pipe
print(f"Classical output ports: {cout_portmap}")
print(f"Classical output ports (pipes): {cout_portmap_pipe}")

# %%
# Circuit creation using compile_to_stim
noise = 0.001
circuit = compile_to_stim(
    compiled_canvas,
    logical_observable_coords={
        0: [
            PatchCoordGlobal3D((0, 0, 6)),
        ],
        1: [
            PatchCoordGlobal3D((1, 1, 6)),
        ],
    },
    p_before_meas_flip=noise,
)
print(f"num_qubits: {circuit.num_qubits}")
# print(circuit)

# %%
# Error correction simulation
dem = circuit.detector_error_model(decompose_errors=True)

matching = pymatching.Matching.from_detector_error_model(dem)
print(matching)

err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
print(f"Shortest error length: {len(err)}")
print(err)

# %%
# Visualization export
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/cnot_0x.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/cnot_0x.svg")


# %%
def get_measured_qubits(circuit: stim.Circuit) -> list[int]:
    """
    Return the qubit indices that are actually measured in a stim circuit.

    Stim emits instructions such as ``MX`` or ``MZ`` that may operate on
    multiple qubits per instruction (e.g. ``MX 0 1 2``). A naive string-based
    search like ``re.findall(r"MX (\\d+)", ...)`` only captures the first
    operand from each instruction and therefore incorrectly reports that the
    remaining qubits are unmeasured. This helper iterates over the parsed stim
    instructions and collects every qubit target touched by a measurement
    command.
    """
    measured: list[int] = []
    for instruction in circuit:
        if instruction.name.startswith("M"):
            for target in instruction.targets_copy():
                if target.is_qubit_target:
                    measured.append(target.value)
    return measured


mq = get_measured_qubits(circuit)

dlines = []
for instr in circuit:
    if instr.name.startswith("DETECTOR"):
        det = []
        for target in instr.targets_copy():
            det.append(target.value)
        dlines.append(det)
# detectors

detectors = []
for det in dlines:
    # now extract the indices
    nodeids = [mq[di] for di in det]
    detectors.append(nodeids)

# detectors

obslines = []
for instr in circuit:
    if instr.name.startswith("OBSERVABLE_INCLUDE"):
        det = []
        for target in instr.targets_copy():
            det.append(target.value)
        obslines.append(det)
# obslines

lobs_indices = obslines[0]
lobs = []
for ind in lobs_indices:
    nodeid = mq[ind]
    lobs.append(nodeid)
print("Logical observable nodes:", sorted(lobs))

fig3d = visualize_compiled_canvas_plotly(
    compiled_canvas,
    show_edges=True,
    show_xparity=False,
    hilight_nodes=sorted(lobs),
)
fig3d.show()
