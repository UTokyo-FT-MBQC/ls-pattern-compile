"""RHG memory simulation example."""

# %%

from operator import le
from graphix_zx.pattern import print_pattern
from graphix_zx.stim_compiler import stim_compile

from lspattern.rhg import create_rhg, visualize_rhg
from lspattern.ops import memory

# %%
d = 3
rhg_lattice, coord2node, x, z = create_rhg(d)
visualize_rhg(rhg_lattice, coord2node)

length = 2 * d - 1
logical_x = set(range(d))
logical_z = set(length * i for i in range(d))
print(f"logical X: {logical_x}")
print(f"logical Z: {logical_z}")
logical_observables = {0: logical_x, 1: logical_z}

# %%
pattern = memory(d)
print_pattern(pattern)

# %%
# compile to stim
stim_str = stim_compile(
    pattern,
    logical_observables,
    after_clifford_depolarization=0.001,
    before_measure_flip_probability=0.01,
)
print(stim_str)

# %%
