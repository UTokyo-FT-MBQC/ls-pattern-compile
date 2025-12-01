# %%
# %%
import pathlib
import os

import pymatching
from lspattern.blocks.cubes.initialize import (
    InitZeroCubeThinLayerSkeleton,
    InitPlusCubeThinLayerSkeleton,
    InitPlusCubeSkeleton,
    # InitZeroCubeSkeleton,
)
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton, InitZeroPipeSkeleton, InitPlusPipeThinLayerSkeleton
from lspattern.blocks.pipes.measure import MeasureXPipeSkeleton, MeasureZPipeSkeleton
from lspattern.blocks.pipes.memory import MemoryPipeSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton, MeasureXSkeleton
from lspattern.canvas import CompiledRHGCanvas, RHGCanvasSkeleton
from lspattern.compile import compile_to_stim, stim_compile
from lspattern.consts import BoundarySide, EdgeSpecValue
from lspattern.mytype import PatchCoordGlobal3D, PipeCoordGlobal3D
from lspattern.visualizers import visualize_compiled_canvas_plotly

# %%
def create_circuit(d, noise):

    canvass = RHGCanvasSkeleton("Rotation d=3")

    edgespec: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec_inv: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}

    edgespec_1_1: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}

    edgespec_1_2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.O}
    edgespec_2_2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.X}
    edgespec_3_2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}
    pipespec_1_2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.O}
    pipespec_2_2: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}

    edgespec_1_3: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}
    edgespec_2_3: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.X, BoundarySide.RIGHT: EdgeSpecValue.X, BoundarySide.TOP: EdgeSpecValue.Z, BoundarySide.BOTTOM: EdgeSpecValue.Z}

    edgespec_2_4: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}
    edgespec_3_4: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}
    pipespec_2_4: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}

    pipespec_2_5: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.X}

    edgespec_1_6: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.X, BoundarySide.BOTTOM: EdgeSpecValue.O}
    edgespec_2_6: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.X}
    pipespec_1_6: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.O}

    pipespec_1_7: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.Z, BoundarySide.RIGHT: EdgeSpecValue.Z, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.O}

    pipespec_meas: dict[BoundarySide, EdgeSpecValue] = {BoundarySide.LEFT: EdgeSpecValue.O, BoundarySide.RIGHT: EdgeSpecValue.O, BoundarySide.TOP: EdgeSpecValue.O, BoundarySide.BOTTOM: EdgeSpecValue.O}

    blocks = [

        (PatchCoordGlobal3D((0, 0, 0)), InitPlusCubeThinLayerSkeleton(d=d, edgespec=edgespec)),
        
        (PatchCoordGlobal3D((0, 0, 1)), MemoryCubeSkeleton(d=d, edgespec=edgespec)),
        
        (PatchCoordGlobal3D((0, 0, 2)), MemoryCubeSkeleton(d=d, edgespec=edgespec_1_2)),
        (PatchCoordGlobal3D((0, -1, 2)), InitPlusCubeSkeleton(d=d, edgespec=edgespec_2_2)),
        (PatchCoordGlobal3D((1, -1, 2)), InitPlusCubeSkeleton(d=d, edgespec=edgespec_3_2)),
        
        (PatchCoordGlobal3D((0, 0, 3)), MeasureXSkeleton(d=d, edgespec=edgespec_1_3)),
        (PatchCoordGlobal3D((0, -1, 3)), MeasureXSkeleton(d=d, edgespec=edgespec_2_3)),
        (PatchCoordGlobal3D((1, -1, 3)), MemoryCubeSkeleton(d=d, edgespec=edgespec_inv)),
        
        (PatchCoordGlobal3D((0, -1, 4)), InitPlusCubeSkeleton(d=d, edgespec=edgespec_2_4)),
        (PatchCoordGlobal3D((1, -1, 4)), MemoryCubeSkeleton(d=d, edgespec=edgespec_3_4)),
        
        (PatchCoordGlobal3D((0, -1, 5)), MemoryCubeSkeleton(d=d, edgespec=edgespec_inv)),
        (PatchCoordGlobal3D((1, -1, 5)), MeasureXSkeleton(d=d, edgespec=edgespec_inv)),
        
        (PatchCoordGlobal3D((0, 0, 6)), InitPlusCubeSkeleton(d=d, edgespec=edgespec_1_6)),
        (PatchCoordGlobal3D((0, -1, 6)), MemoryCubeSkeleton(d=d, edgespec=edgespec_2_6)),
        
        (PatchCoordGlobal3D((0, 0, 7)), MemoryCubeSkeleton(d=d, edgespec=edgespec_inv)),
        (PatchCoordGlobal3D((0, -1, 7)), MeasureXSkeleton(d=d, edgespec=edgespec_inv)),
        
        (PatchCoordGlobal3D((0, 0, 8)), MeasureXSkeleton(d=d, edgespec=edgespec_inv)),
    ]

    pipes = [
        (
            PatchCoordGlobal3D((0, 0, 2)),
            PatchCoordGlobal3D((0, -1, 2)),
            InitPlusPipeSkeleton(d=d, edgespec=pipespec_1_2),
        ),
        (
            PatchCoordGlobal3D((0, -1, 2)),
            PatchCoordGlobal3D((1, -1, 2)),
            InitPlusPipeSkeleton(d=d, edgespec=pipespec_2_2),
        ),
        (
            PatchCoordGlobal3D((0, 0, 3)),
            PatchCoordGlobal3D((0, -1, 3)),
            MeasureXPipeSkeleton(d=d, edgespec=pipespec_1_2),
        ),
        (
            PatchCoordGlobal3D((0, -1, 3)),
            PatchCoordGlobal3D((1, -1, 3)),
            MeasureXPipeSkeleton(d=d, edgespec=pipespec_2_2),
        ),
        (
            PatchCoordGlobal3D((0, -1, 4)),
            PatchCoordGlobal3D((1, -1, 4)),
            InitPlusPipeSkeleton(d=d, edgespec=pipespec_2_4),
        ),
        (
            PatchCoordGlobal3D((0, -1, 5)),
            PatchCoordGlobal3D((1, -1, 5)),
            MeasureXPipeSkeleton(d=d, edgespec=pipespec_2_5),
        ),
        (
            PatchCoordGlobal3D((0, 0, 6)),
            PatchCoordGlobal3D((0, -1, 6)),
            InitPlusPipeSkeleton(d=d, edgespec=pipespec_1_6),
        ),
        (
            PatchCoordGlobal3D((0, 0, 7)),
            PatchCoordGlobal3D((0, -1, 7)),
            MeasureXPipeSkeleton(d=d, edgespec=pipespec_1_7),
        ),
    ]

    for block in blocks:
        canvass.add_cube(*block)
    for pipe in pipes:
        canvass.add_pipe(*pipe)

    canvas = canvass.to_canvas()

    compiled_canvas: CompiledRHGCanvas = canvas.compile()

    if d == 3:
        detector_update = {
            frozenset({361, 496}): {361, 496, 400},
            frozenset({360, 494, 495}): {360, 494, 495, 398, 399},
            frozenset({358, 494}): {358, 494, 413},
            frozenset({359, 495, 496}): {359, 495, 496, 414, 415},
            frozenset({394, 497}): {394, 497, 409},
            
            frozenset({685, 775}): {685, 775, 770},
            frozenset({683, 773, 774}): {683, 773, 774, 764, 767},
            
            frozenset({247, 521}): {521}
        }

        detector_remove = {
            frozenset({336}),
            frozenset({337}),
            
            frozenset({395, 497, 498}),
            frozenset({397, 499}),
            
            frozenset({390, 427}),
            
            # frozenset({247, 521}),
            frozenset({236, 509, 534}),
            frozenset({237, 511, 536}),
            
            frozenset({684, 774, 775}),
            frozenset({682, 773}),
            
            frozenset({677, 696}),
            
            frozenset({661}),
            frozenset({659}),
            
            frozenset({785, 810, 163}),
            frozenset({164, 788, 813})
        }
    elif d == 5:
        detector_update = {
            frozenset({1611, 2134, 2135}): {1611, 2134, 2135, 1714, 1715},
            frozenset({1612, 2136, 2137}): {1612, 2136, 2137, 1716, 1717},
            frozenset({1613, 2138}): {1613, 2138, 1718},
            
            frozenset({1608, 2134}): {1608, 2134, 1759},
            frozenset({1609, 2135, 2136}): {1609, 2135, 2136, 1760, 1761},
            frozenset({1610, 2137, 2138}): {1610, 2137, 2138, 1762, 1763},
            
            frozenset({1708, 2139}): {1708, 2139, 1743},
            
            frozenset({2963, 3363}): {2963, 3363, 3354},
            frozenset({2961, 3361, 3362}): {2961, 3361, 3362, 3349, 3344},
            frozenset({2959, 3359, 3360}): {2959, 3359, 3360, 3334, 3339},
            
            frozenset({2204, 1142}): {2204},
            frozenset({2205, 1143}): {2205},
        }

        detector_remove = {
            frozenset({1531}),
            frozenset({1532}),
            frozenset({1533}),
            
            frozenset({1709, 2139, 2140}),
            frozenset({1711, 2141, 2142}),
            frozenset({1713, 2143}),
            
            frozenset({1700, 1792}),
            frozenset({1702, 1798}),
            
            frozenset({3471, 763, 3399}),
            frozenset({3390, 3462, 758}),
            frozenset({760, 3466, 3394}),
            frozenset({762, 3468, 3396}),
            frozenset({3472, 3400, 764}),
            frozenset({3393, 3465, 759}),
            frozenset({761, 3395, 3467}),
            frozenset({3389, 757, 3461}),
            
            frozenset({2250, 1116, 2178}),
            frozenset({2244, 2172, 1111}),
            frozenset({1113, 2246, 2174}),
            frozenset({2249, 2177, 1115}),
            frozenset({2242, 2170, 1110}),
            frozenset({1112, 2245, 2173}),
            frozenset({2169, 2241, 1109}),
            # frozenset({2205, 1143}),
            # frozenset({2204, 1142}),
            frozenset({1114, 2247, 2175}),
            
            frozenset({2962, 3362, 3363}),
            frozenset({2960, 3360, 3361}),
            frozenset({2958, 3359}),
            
            frozenset({2951, 2997}),
            frozenset({2949, 2991}),
            
            frozenset({2879}),
            frozenset({2881}),
            frozenset({2883}),
        }
    else:
        raise ValueError("Unsupported d")
    
    for key1, item in compiled_canvas.parity.checks.items():
        to_be_removed = []
        for key2, check_nodes in item.items():
            for detector_key in detector_update.keys():
                if check_nodes == detector_key:
                    compiled_canvas.parity.checks[key1][key2] = detector_update[detector_key]
            for detector_key in detector_remove:
                if check_nodes == detector_key:
                    to_be_removed.append(key2)
        for key in to_be_removed:
            del compiled_canvas.parity.checks[key1][key]
            
    coord2logical_group = {
        0: {PatchCoordGlobal3D((0, 0, 8)), PipeCoordGlobal3D((PatchCoordGlobal3D((0, 0, 7)), PatchCoordGlobal3D((0, -1, 7)))), PatchCoordGlobal3D((0, -1, 7))}
    }
    
    circuit = compile_to_stim(
        compiled_canvas,
        coord2logical_group,
        p_before_meas_flip=noise,
        p_depol_after_clifford=noise,
    )
    
    return circuit

# %%
import sinter

tasks = [
    sinter.Task(
        circuit=create_circuit(d, noise),
        json_metadata={"d": d, "p": noise},
    )
    for d in [3, 5]
    for noise in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    # for noise in [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 1e-1]
]

# %%
collected_stats: list[sinter.TaskStats] = sinter.collect(
    num_workers=os.cpu_count() or 1,
    tasks=tasks,
    decoders=["pymatching"],
    max_shots=10_000_000,
    max_errors=100_000,
    count_observable_error_combos=True,
    print_progress=True,
)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
sinter.plot_error_rate(
    ax=ax,
    stats=collected_stats,
    x_func=lambda stat: stat.json_metadata["p"],
    # group_func=lambda stat: stat.json_metadata["d"],
    group_func=lambda stat: f"d={stat.json_metadata['d']}",
    failure_units_per_shot_func=lambda stat: stat.json_metadata["d"],
)
# ax.set_ylim(5e-3, 5e-2)
# ax.set_xlim(0.008, 0.012)
ax.loglog()
ax.set_title("Rotation Error Rates under Circuit Noise")
ax.set_xlabel("Physical Error Rate")
ax.set_ylabel("Logical Error Rate")
ax.grid(which="major")
ax.grid(which="minor")
ax.legend()
fig.set_dpi(120)
# fig.savefig("figures/memory_sim.png", bbox_inches="tight")
plt.show()

# %%



