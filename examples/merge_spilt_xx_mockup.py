"""
Merge and Split
"""

# %%
import pathlib

import pymatching
import stim
from graphix_zx.pattern import Pattern, print_pattern
from graphix_zx.scheduler import Scheduler
from graphix_zx.stim_compiler import stim_compile

from lspattern.blocks.cubes.initialize import (
    InitZeroCubeThinLayerSkeleton,
    InitPlusCubeThinLayerSkeleton,
)
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import (
    InitPlusPipeSkeleton,
    InitZeroPipeSkeleton,
    InitZeroPipeThinLayerSkeleton,
)
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton, MeasureZPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton, MeasureXSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
d = 3


canvass = RHGCanvasSkeleton("Merge and Split")

edgespec: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec1: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.Z,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec2: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.Z,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec_trimmed: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.X,
    BoundarySide.BOTTOM: EdgeSpecValue.X,
}
edgespec_measure_trimmed: dict[BoundarySide, EdgeSpecValue] = {
    BoundarySide.LEFT: EdgeSpecValue.O,
    BoundarySide.RIGHT: EdgeSpecValue.O,
    BoundarySide.TOP: EdgeSpecValue.O,
    BoundarySide.BOTTOM: EdgeSpecValue.O,
}
blocks = [
    (
        PatchCoordGlobal3D((0, 0, 0)),
        InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 0)),
        InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 1)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 2)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec1),
    ),
    (
        PatchCoordGlobal3D((1, 0, 2)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec2),
    ),
    (
        PatchCoordGlobal3D((0, 0, 3)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 3)),
        MemoryCubeSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((0, 0, 4)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
    (
        PatchCoordGlobal3D((1, 0, 4)),
        MeasureXSkeleton(d=d, edgespec=edgespec),
    ),
]
pipes = [
    # (
    #     PatchCoordGlobal3D((0, 0, 1)),
    #     PatchCoordGlobal3D((1, 0, 1)),
    #     InitZeroPipeThinLayerSkeleton(d=d, edgespec=edgespec_measure_trimmed),
    # ),
    (
        PatchCoordGlobal3D((0, 0, 2)),
        PatchCoordGlobal3D((1, 0, 2)),
        InitZeroPipeSkeleton(d=d, edgespec=edgespec_trimmed),
    ),
    (
        PatchCoordGlobal3D((0, 0, 3)),
        PatchCoordGlobal3D((1, 0, 3)),
        MeasureZPipeSkeleton(d=d, edgespec=edgespec_measure_trimmed),
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
    hilight_nodes=[193, 363, 369, 360],
)
fig3d.show()

# %%

# Print flow and parity information
xflow = {}
for src, dsts in compiled_canvas.flow.flow.items():
    xflow[int(src)] = {int(dst) for dst in dsts}
x_parity = []
for group_dict in compiled_canvas.parity.checks.values():
    for group in group_dict.values():
        x_parity.append({int(node) for node in group})

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

print("X parity")

ck = {
    (11, 1): {
        5: {49},
        7: {49, 153},
        9: {153, 179},
        11: {179, 205},
        13: {305, 205},
        15: {305, 330},
        17: {330, 355},
        19: {490, 355},
        21: {490, 516},
        23: {516, 542},
        24: {561, 562, 558, 542, 559},
    },
    (13, 1): {
        6: {140, 36},
        8: {140, 166},
        10: {192, 166},
        12: {192, 293},
        14: {293, 318},
        16: {318, 343},
        18: {477, 343},
        20: {477, 503},
        22: {529, 503},
    },
    (3, 1): {
        5: {23},
        7: {75, 23},
        9: {75, 101},
        11: {101, 127},
        13: {230, 127},
        15: {230, 255},
        17: {280, 255},
        19: {280, 412},
        21: {412, 438},
        23: {464, 438},
        24: {464, 549, 550, 552, 553},
    },
    (5, -1): {13: {361}, 15: {361, 373}, 17: {385, 373}},
    (5, 1): {
        6: {10, 62},
        8: {88, 62},
        10: {88, 114},
        14: {368, 114},
        16: {368, 380},
        18: {545, 546, 380, 399},
        20: {425, 399},
        22: {425, 451},
    },
    (7, 3): {
        6: {37, 141},
        8: {141, 167},
        10: {193, 167},
        14: {193, 369},
        16: {369, 381},
        18: {546, 547, 381, 478},
        20: {504, 478},
        22: {504, 530},
    },
    (-1, 3): {
        6: {11, 63},
        8: {89, 63},
        10: {89, 115},
        12: {218, 115},
        14: {218, 243},
        16: {243, 268},
        18: {400, 268},
        20: {400, 426},
        22: {426, 452},
    },
    (11, 5): {
        5: {51},
        7: {51, 155},
        9: {155, 181},
        11: {181, 207},
        13: {307, 207},
        15: {307, 332},
        17: {332, 357},
        19: {492, 357},
        21: {492, 518},
        23: {544, 518},
        24: {544, 564, 565},
    },
    (1, 3): {
        5: {24},
        7: {24, 76},
        9: {76, 102},
        11: {128, 102},
        13: {128, 231},
        15: {256, 231},
        17: {256, 281},
        19: {281, 413},
        21: {413, 439},
        23: {465, 439},
        24: {465, 551, 552, 554, 555},
    },
    (3, 5): {
        5: {25},
        7: {25, 77},
        9: {77, 103},
        11: {129, 103},
        13: {232, 129},
        15: {232, 257},
        17: {257, 282},
        19: {282, 414},
        21: {440, 414},
        23: {440, 466},
        24: {466, 555, 556},
    },
    (7, 1): {13: {362}, 15: {362, 374}, 17: {386, 374}},
    (9, 3): {
        5: {50},
        7: {50, 154},
        9: {154, 180},
        11: {180, 206},
        13: {306, 206},
        15: {306, 331},
        17: {331, 356},
        19: {491, 356},
        21: {491, 517},
        23: {517, 543},
        24: {560, 561, 563, 564, 543},
    },
    (1, 1): {
        6: {9, 61},
        8: {61, 87},
        10: {113, 87},
        12: {113, 217},
        14: {217, 242},
        16: {242, 267},
        18: {267, 398},
        20: {424, 398},
        22: {424, 450},
    },
    (11, 3): {
        6: {142, 38},
        8: {168, 142},
        10: {168, 194},
        12: {194, 294},
        14: {294, 319},
        16: {344, 319},
        18: {344, 479},
        20: {505, 479},
        22: {505, 531},
    },
    (1, -1): {
        5: {22},
        7: {74, 22},
        9: {74, 100},
        11: {100, 126},
        13: {229, 126},
        15: {229, 254},
        17: {254, 279},
        19: {411, 279},
        21: {411, 437},
        23: {437, 463},
        24: {548, 549, 463},
    },
    (3, 3): {
        6: {64, 12},
        8: {64, 90},
        10: {90, 116},
        12: {219, 116},
        14: {219, 244},
        16: {244, 269},
        18: {401, 269},
        20: {401, 427},
        22: {427, 453},
    },
    (9, -1): {
        5: {48},
        7: {48, 152},
        9: {152, 178},
        11: {178, 204},
        13: {304, 204},
        15: {304, 329},
        17: {329, 354},
        19: {489, 354},
        21: {489, 515},
        23: {515, 541},
        24: {557, 541, 558},
    },
    (5, 3): {15: {363, 375}, 17: {387, 375}},
    (7, 5): {15: {376, 364}, 17: {376, 388}},
    (9, 1): {
        6: {35, 139},
        8: {139, 165},
        10: {165, 191},
        12: {292, 191},
        14: {292, 317},
        16: {317, 342},
        18: {476, 342},
        20: {476, 502},
        22: {528, 502},
    },
}

# compiled_canvas.parity.checks = ck  # type: ignore[assignment]
# print(compiled_canvas.parity.checks)
for coord, group_list in compiled_canvas.parity.checks.items():
    print(f"  {coord}: {group_list}")

# Print detailed flow information for visualization
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
cout_portmap = compiled_canvas.cout_portset
print(f"Classical output ports: {cout_portmap}")


# %%
# Pattern generation
# Create scheduler
scheduler = Scheduler(compiled_canvas.global_graph, xflow=xflow)

# Set up timing based on compiled_canvas.schedule
compact_schedule = compiled_canvas.schedule.compact()
print(f"Schedule has {len(compact_schedule.schedule)} time slots")

# Initialize prepare_time and measure_time dictionaries
prep_time = {}
meas_time = {}

# Set input nodes to have no preparation time (None)
if compiled_canvas.global_graph is not None:
    input_nodes = set(compiled_canvas.global_graph.input_node_indices.keys())
    for node in compiled_canvas.global_graph.physical_nodes:
        if node not in input_nodes:
            prep_time[node] = 0  # Non-input nodes prepared at time 0

    # Set measurement times based on schedule
    output_indices = compiled_canvas.global_graph.output_node_indices or {}
    output_nodes = set(output_indices.keys())
    for node in compiled_canvas.global_graph.physical_nodes:
        if node not in output_nodes:
            # Find when this node is scheduled for measurement
            meas_time[node] = 1  # Default measurement time
            for time_slot, nodes in compact_schedule.schedule.items():
                if node in nodes:
                    meas_time[node] = (
                        time_slot + 1
                    )  # Shift by 1 to account for preparation at time 0
                    break

# Configure scheduler with manual timing
scheduler.manual_schedule(prepare_time=prep_time, measure_time=meas_time)

pattern = compile_canvas(
    compiled_canvas.global_graph,
    xflow=xflow,
    x_parity=x_parity,
    z_parity=[],
    scheduler=scheduler,
)
print("Pattern compilation successful")
print_pattern(pattern)

# set logical observables
coord2logical_group = {
    # 0: {PatchCoordGlobal3D((0, 0, 3)), PatchCoordGlobal3D((1, 0, 3))},  # First output patch
    0: {PatchCoordGlobal3D((0, 0, 4))},
    # 1: {PatchCoordGlobal3D((1, 0, 3))},  # Second output patch
}
logical_observables = {}
for i, group in coord2logical_group.items():
    nodes = []
    for coord in group:
        if coord in cout_portmap:
            nodes.extend(cout_portmap[coord])
    logical_observables[i] = set(nodes)


# %%
# Circuit creation
def create_circuit(pattern: Pattern, noise: float) -> stim.Circuit:
    print(f"Using logical observables: {logical_observables}")
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=0,
        before_measure_flip_probability=noise,
    )
    return stim.Circuit(stim_str)


noise = 0.001
circuit = create_circuit(pattern, noise)
print(f"num_qubits: {circuit.num_qubits}")
# print(circuit)

# %%
# Error correction simulation
try:
    dem = circuit.detector_error_model(decompose_errors=True)
    print(dem)
except ValueError as e:
    print(f"Error creating DEM with decompose_errors=True: {e}")
    print("Retrying without decompose_errors...")
    dem = circuit.detector_error_model(decompose_errors=False)
    print(dem)

matching = pymatching.Matching.from_detector_error_model(dem)
print(matching)

err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(f"Shortest error length: {len(err)}")
print(err)

# %%
# Visualization export
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("figures").mkdir(exist_ok=True)
pathlib.Path("figures/merge_split_dem.svg").write_text(str(svg), encoding="utf-8")
print("SVG diagram saved to figures/merge_split_dem.svg")


# %%
