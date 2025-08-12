from typing import NamedTuple

import matplotlib.pyplot as plt
from graphix_zx.common import Plane, PlannerMeasBasis
from graphix_zx.graphstate import GraphState

allowed_parities = [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
data_parities = [(0, 0, 0), (1, 1, 0), (0, 0, 1), (1, 1, 1)]
ancilla_x_check_parity = (0, 1, 0)
ancilla_z_check_parity = (1, 0, 1)


class RHGResult(NamedTuple):
    """Result of RHG lattice creation.

    Attributes
    ----------
    graph_state : GraphState
        The RHG graph state.
    coord_to_node : dict[tuple[int, int, int], int]
        Mapping from (x, y, z) coordinates to node indices.
    x_parity_checks : list[set[int]]
        List of X parity check groups, where each set contains node indices
        that form an X parity check.
    z_parity_checks : list[set[int]]
        List of Z parity check groups, where each set contains node indices
        that form a Z parity check.
    measurement_groups : list[set[int]]
        Measurement order grouping, where each set contains nodes that can
        be measured together (alternating between ancilla and data qubits).
    """

    graph_state: GraphState
    coord_to_node: dict[tuple[int, int, int], int]
    x_parity_checks: list[set[int]]
    z_parity_checks: list[set[int]]
    measurement_groups: list[set[int]]


def create_rhg(
    dx: int,
    dy: int,
    rounds: int,
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
) -> RHGResult:
    """Create a Raussendorf lattice (RHG) with the specified distance `d`.

    Parameters
    ----------
    dx : int
        The code distance of the RHG lattice in the x direction.
    dy : int
        The code distance of the RHG lattice in the y direction.
    rounds : int
        The number of rounds for the RHG lattice.
    allowed_parities : list[tuple[int, int, int]], optional
        The allowed parity patterns, by default allowed_parities

    Returns
    -------
    RHGResult
        The created RHG lattice and its associated data.

    """
    length_x = 2 * dx - 1
    length_y = 2 * dy - 1
    length_z = 2 * rounds + 1
    gs, coord2node, x_checks, z_checks, grouping = _create_rhg(
        length_x,
        length_y,
        length_z,
        allowed_parities=allowed_parities,
    )
    return RHGResult(
        graph_state=gs,
        coord_to_node=coord2node,
        x_parity_checks=x_checks,
        z_parity_checks=z_checks,
        measurement_groups=grouping,
    )


def _create_rhg(
    Lx: int,
    Ly: int,
    Lz: int,
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
) -> tuple[
    GraphState,
    dict[tuple[int, int, int], int],
    list[set[int]],
    list[set[int]],
    list[set[int]],
]:
    """Places a node only if the parity pattern (x % 2, y % 2, z % 2) of the integer coordinates (x, y, z)
    is included in `allowed_parities`, and returns the corresponding GraphState and a coordinate-to-node-index mapping.

    Returns
    -------
    tuple[GraphState, dict[tuple[int, int, int], int], list[set[int]], list[set[int]], list[set[int]]]
        A tuple containing:
        - graphstate: RHG graphstate
        - coord2node: Mapping from (x, y, z) to node_index
        - x_parity_check_groups: X parity check groups
        - z_parity_check_groups: Z parity check groups
        - grouping: Measurement order grouping

    """
    gs = GraphState()
    coord2node: dict[tuple[int, int, int], int] = {}
    x_parity_check_groups: list[set[int]] = []  # tuple means a directed edge
    z_parity_check_groups: list[set[int]] = []

    coord2qindex: dict[tuple[int, int], int] = {}

    grouping: list[set[int]] = []

    for z in range(Lz):
        data_qubits = set()
        ancilla_qubits = set()
        for y in range(Ly):
            for x in range(Lx):
                parity = (x % 2, y % 2, z % 2)
                if parity not in allowed_parities:
                    continue

                if (z == Lz - 1) and parity not in data_parities:
                    # skip output layer if not in data_parities
                    continue
                node_idx = gs.add_physical_node()
                coord2node[(x, y, z)] = node_idx
                if z == Lz - 1:  # output layer
                    if parity in data_parities:
                        gs.register_output(node_idx, coord2qindex[(x, y)])
                    else:
                        gs.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, 0.0))
                else:
                    if z == 0:  # input layer
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


def visualize_rhg(
    lattice_state: GraphState,
    coord2node: dict[tuple[int, int, int], int],
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
) -> None:
    """Visualizes the Raussendorf lattice with nodes colored based on their parity.
    Nodes with allowed parities are colored white, others are red.
    Physical edges are drawn in gray.

    Parameters
    ----------
    - lattice_state: GraphState
        The Raussendorf lattice state to visualize.
    - coord2node: dict[tuple[int,int,int], int]
        Mapping from coordinates to node indices.
    - allowed_parities: list[tuple[int, int, int]]
        List of allowed parity patterns for nodes.

    """
    node2coord: dict[int, tuple[int, int, int]] = {
        node: coord for coord, node in coord2node.items()
    }

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))  # Set aspect ratio to be equal
    ax.grid(False)
    ax.set_axis_off()

    xs, ys, zs = [], [], []
    colors = []
    for _node, (x, y, z) in node2coord.items():
        xs.append(x)
        ys.append(y)
        zs.append(z)

        parity = (x % 2, y % 2, z % 2)
        if parity in allowed_parities[:3]:
            colors.append("white")
        else:
            colors.append("red")

    ax.scatter(
        xs,
        ys,
        zs,
        c=colors,
        edgecolors="black",
        s=50,
        depthshade=True,
        label="nodes",
    )
    for u, v in lattice_state.physical_edges:
        # Extract coordinates from coord2node
        x1, y1, z1 = node2coord[u]
        x2, y2, z2 = node2coord[v]
        ax.plot([x1, x2], [y1, y2], [z1, z2], c="gray", linewidth=1, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Raussendorf lattice (allowed parity nodes)")
    plt.legend()
    plt.tight_layout()
    plt.show()
