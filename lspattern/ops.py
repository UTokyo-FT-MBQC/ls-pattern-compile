from graphix_zx import qompiler

from lspattern.rhg import create_rhg


def memory(d: int):
    """
    Return a pattern for a memory operation.
    """

    lattice_state, coord2node, x_parity_check_groups, z_parity_check_groups = (
        create_rhg(d, d)
    )
    node2coord: dict[int, tuple[int, int, int]] = {
        node: coord for coord, node in coord2node.items()
    }
    f = dict()
    for node1 in lattice_state.physical_nodes:
        x1, y1, z1 = node2coord[node1]
        node2 = coord2node.get((x1, y1, z1 + 1), None)
        if node2 is not None:
            f[node1] = {node2}

    pattern = qompiler.qompile(
        lattice_state,
        f,
        x_parity_check_group=x_parity_check_groups,
        z_parity_check_group=z_parity_check_groups,
    )

    return pattern
