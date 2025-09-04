# RHG extention for two-qubit merge operation
from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState

allowed_parities = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
data_parities = [(0, 0, 0), (1, 1, 0), (0, 0, 1), (1, 1, 1)]
ancilla_x_check_parity = (0, 1, 0)
ancilla_z_check_parity = (1, 0, 1)


def create_rhg_rect(
    dx: int,
    dy: int,
    rounds: int,
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
) -> tuple[
    GraphState,
    dict[tuple[int, int, int], int],
    list[set[int]],  # this should be generalized in the future development
    list[set[int]],
    list[set[int]],
]:
    """Create a Raussendorf lattice (RHG) with the specified distance `d`.

    Parameters
    ----------
    d : int
        The distance of the RHG lattice.
    rounds : int
        The number of rounds for the RHG lattice.
    allowed_parities : list[tuple[int, int, int]], optional
        The allowed parity patterns, by default allowed_parities

    Returns
    -------
    tuple[ GraphState, dict[tuple[int, int, int], int], list[set[int]], list[set[int]], list[set[int]]]
        The created RHG lattice and its associated data.

    """
    length_x = 2 * dx - 1
    length_y = 2 * dy - 1
    length_z = 2 * rounds + 1
    return _create_rhg(
        length_x,
        length_y,
        length_z,
        allowed_parities=allowed_parities,
    )


def create_rhg(
    d: int,
    rounds: int,
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
) -> tuple[
    GraphState,
    dict[tuple[int, int, int], int],
    list[set[int]],  # this should be generalized in the future development
    list[set[int]],
    list[set[int]],
]:
    """Create a Raussendorf lattice (RHG) with the specified distance `d`.

    Parameters
    ----------
    d : int
        The distance of the RHG lattice.
    rounds : int
        The number of rounds for the RHG lattice.
    allowed_parities : list[tuple[int, int, int]], optional
        The allowed parity patterns, by default allowed_parities

    Returns
    -------
    tuple[ GraphState, dict[tuple[int, int, int], int], list[set[int]], list[set[int]], list[set[int]]]
        The created RHG lattice and its associated data.

    """
    length_xy = 2 * d - 1
    length_z = 2 * rounds + 1
    return _create_rhg(
        length_xy,
        length_xy,
        length_z,
        allowed_parities=allowed_parities,
    )


def _create_rhg(
    Lx: int,
    Ly: int,
    Lz: int,
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
    offsets: tuple[int, int, int] = (0, 0, 0),
) -> tuple[
    GraphState,
    dict[tuple[int, int, int], int],
    list[set[int]],
    list[set[int]],
    list[set[int]],
]:
    """Places a node only if the parity pattern (x % 2, y % 2, z % 2) of the integer coordinates (x, y, z)
    is included in `allowed_parities`, and returns the corresponding GraphState and a coordinate-to-node-index mapping.

    Returns:
    - graphstate: GraphState
        RHG graphstate
    - coord2node: dict[tuple[int,int,int], int]
        { (x, y, z): node_index }
    - x_parity_check_groups: list[set[int]]
        List of sets of nodes that form X parity check groups.
    - z_parity_check_groups: list[set[int]]
        List of sets of nodes that form Z parity check groups.
    - grouping: list[set[int]]
        The measurement order grouping, where each set contains nodes that can be measured together.

    """
    gs = GraphState()
    coord2node: dict[tuple[int, int, int], int] = {}
    x_parity_check_groups: list[set[int]] = []  # tuple means a directed edge
    z_parity_check_groups: list[set[int]] = []

    coord2qindex: dict[tuple[int, int], int] = {}

    grouping: list[set[int]] = []

    for z in range(offsets[2], Lz + offsets[2]):
        data_qubits = set()
        ancilla_qubits = set()
        for y in range(offsets[1], Ly + offsets[1]):
            for x in range(offsets[0], Lx + offsets[0]):
                parity = (x % 2, y % 2, z % 2)
                if parity not in allowed_parities:
                    continue

                if (z == Lz + offsets[2] - 1) and parity not in data_parities:
                    # skip output layer if not in data_parities
                    continue
                node_idx = gs.add_physical_node()
                coord2node[(x, y, z)] = node_idx
                if z == Lz + offsets[2] - 1:  # output layer
                    if parity in data_parities:
                        gs.register_output(node_idx, coord2qindex[(x, y)])
                    else:
                        gs.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, 0.0))
                else:
                    if z == offsets[2]:  # input layer
                        if parity in data_parities:
                            q_index = gs.register_input(node_idx)
                            coord2qindex[(x, y)] = q_index
                    gs.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, 0.0))

                if parity in data_parities:
                    data_qubits.add(node_idx)
                else:
                    ancilla_qubits.add(node_idx)
        grouping.append(ancilla_qubits)
        grouping.append(data_qubits)

    # add edges
    for (x, y, z), u in coord2node.items():
        for dx, dy, dz in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (nx, ny, nz) in coord2node:
                v = coord2node[(nx, ny, nz)]
                try:
                    gs.add_physical_edge(u, v)
                except ValueError:
                    pass
        parity = (x % 2, y % 2, z % 2)
        next_ancilla = coord2node.get((x, y, z + 2), None)
        if parity == ancilla_x_check_parity:
            if next_ancilla:
                x_parity_check_groups.append({u, next_ancilla})
        elif parity == ancilla_z_check_parity:
            if next_ancilla:
                z_parity_check_groups.append({u, next_ancilla})
            if z == 1:
                z_parity_check_groups.append({u})

    # add data qubit stabilizers
    for i in range((Lx - 1) // 2):
        for j in range((Ly + 1) // 2):
            group: set[int] = set()
            pos0 = (2 * i, 2 * j, Lz - 1)
            pos1 = (2 * i + 1, 2 * j - 1, Lz - 1)
            pos2 = (2 * i + 2, 2 * j, Lz - 1)
            pos3 = (2 * i + 1, 2 * j + 1, Lz - 1)
            if node0 := coord2node.get(pos0):
                group.add(node0)
            if node1 := coord2node.get(pos1):
                group.add(node1)
            if node2 := coord2node.get(pos2):
                group.add(node2)
            if node3 := coord2node.get(pos3):
                group.add(node3)

            # add the previous stabilizer measurement
            group.add(coord2node[2 * i + 1, 2 * j, Lz - 2])
            x_parity_check_groups.append(group)

    return gs, coord2node, x_parity_check_groups, z_parity_check_groups, grouping
