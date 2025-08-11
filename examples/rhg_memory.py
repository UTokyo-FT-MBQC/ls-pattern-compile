"""RHG memory example."""

# %%

import pathlib
import stim
import pymatching
from graphix_zx.pattern import print_pattern
from graphix_zx.stim_compiler import stim_compile
from lspattern.ops import memory
from lspattern.rhg import create_rhg, visualize_rhg

# %%
d = 3
r = 1
rhg_lattice, coord2node, x, z, grouping = create_rhg(d, r)
visualize_rhg(rhg_lattice, coord2node)

for group in grouping:
    print(f"group: {group}")

length = 2 * d - 1
# logical = set(range(d)) # logical Z (This is not deterministic with |+> initialization)
logical = set(length * i for i in range(d))
print(f"logical X: {logical}")
logical_observables = {0: logical}

# %%
pattern = memory(d, r)
print_pattern(pattern)

# %%
# compile to stim
stim_str = stim_compile(
    pattern,
    logical_observables,
    after_clifford_depolarization=0,
    before_measure_flip_probability=0,
)
print(stim_str)

# %%


def create_circuit(d: int, rounds: int, noise: float) -> stim.Circuit:
    pattern = memory(d, rounds)
    length = 2 * d - 1
    logical_observables = {0: {length * i for i in range(d)}}
    stim_str = stim_compile(
        pattern,
        logical_observables,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=0,
    )
    return stim.Circuit(stim_str)


# %%

noise = 0.001
circuit = create_circuit(d, r, noise)
print(f"num_qubits: {circuit.num_qubits}")

dem = circuit.detector_error_model(decompose_errors=True)
print(dem)

# %%

matching = pymatching.Matching.from_detector_error_model(dem)
print(matching)
# matching.draw()


# %%
err = dem.shortest_graphlike_error(ignore_ungraphlike_errors=False)
print(len(err))
print(err)

# %%
svg = dem.diagram(type="match-graph-svg")
pathlib.Path("dem.svg").write_text(str(svg), encoding="utf-8")

# %%
