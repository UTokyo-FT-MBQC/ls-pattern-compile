"""RHG memory simulation example."""

# %%
from lspattern.rhg import create_rhg, visualize_rhg
from lspattern.ops import memory

# %%
d = 5
rhg_lattice, coord2node = create_rhg(d, d, d)
visualize_rhg(rhg_lattice, coord2node)

# %%
