"""RHG memory simulation example."""

# %%

from graphix_zx.pattern import print_pattern
from graphix_zx.stim_compiler import stim_compile

from lspattern.rhg import create_rhg, visualize_rhg
from lspattern.ops import memory

# %%
d = 5
rhg_lattice, coord2node, x, z = create_rhg(d, d, d)
visualize_rhg(rhg_lattice, coord2node)

# %%
pattern = memory(d, d, d)
print_pattern(pattern)

# %%
# compile to stim
stim_str = stim_compile(pattern)
print(stim_str)

# %%
