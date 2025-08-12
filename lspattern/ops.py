from graphix_zx.qompiler import qompile
from graphix_zx.scheduler import Scheduler

from lspattern.rhg import create_rhg


def memory(d: int, r: int):
    """Return a pattern for a memory operation."""
    rhg_result = create_rhg(d, d, r)
    lattice_state = rhg_result.graph_state
    coord2node = rhg_result.coord_to_node
    x_parity_check_groups = rhg_result.x_parity_checks
    z_parity_check_groups = rhg_result.z_parity_checks
    grouping = rhg_result.measurement_groups

    node2coord: dict[int, tuple[int, int, int]] = {
        node: coord for coord, node in coord2node.items()
    }
    f = dict()
    for node1 in lattice_state.physical_nodes:
        x1, y1, z1 = node2coord[node1]
        node2 = coord2node.get((x1, y1, z1 + 1), None)
        if node2 is not None:
            f[node1] = {node2}

    # scheduler
    scheduler = Scheduler(lattice_state)

    prep_time = dict.fromkeys(
        lattice_state.physical_nodes - set(lattice_state.input_node_indices),
        0,
    )
    meas_time: dict[int, int] = dict.fromkeys(set(lattice_state.input_node_indices), 1)
    time_index = 2
    for group in grouping:
        for node in group:
            meas_time[node] = time_index
        time_index += 1

    # schedule based on grouping
    scheduler.from_manual_design(
        prepare_time=prep_time,
        measure_time=meas_time,
    )

    pattern = qompile(
        lattice_state,
        f,
        x_parity_check_group=x_parity_check_groups,
        z_parity_check_group=z_parity_check_groups,
        scheduler=scheduler,
    )

    return pattern
