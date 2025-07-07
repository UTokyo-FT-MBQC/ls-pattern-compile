from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState
from graphix_zx import qompiler

from rhg import create_rhg

def memory(Lx, Ly, Lz):
    """
    Return a pattern for a memory operation.    
    """
    
    lattice_state, coord2node = create_rhg(Lx, Ly, Lz)
    node2coord: dict[int, tuple[int,int,int]] = {node: coord for coord, node in coord2node.items()}
    f = dict()
    for node1 in lattice_state.physical_nodes:
        x1, y1, z1 = node2coord[node1]
        for node2 in lattice_state.physical_nodes:
            x2, y2, z2 = node2coord[node2]
            if (x1, y1, z1) == (x2, y2, z2 - 1):
                f[node1] = {node2}
            
    pattern = qompiler.qompile(lattice_state, f)
    
    return pattern