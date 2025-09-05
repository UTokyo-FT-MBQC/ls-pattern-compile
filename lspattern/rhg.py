"""RHG (Raussendorf-Harrington-Goyal) lattice creation and visualization."""

import contextlib
import os
from pathlib import Path
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
    lx: int,
    ly: int,
    lz: int,
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
) -> tuple[
    GraphState,
    dict[tuple[int, int, int], int],
    list[set[int]],
    list[set[int]],
    list[set[int]],
]:
    """Create RHG lattice by orchestrating helper functions.

    Places a node only if the parity pattern (x % 2, y % 2, z % 2) of the integer
    coordinates (x, y, z) is included in `allowed_parities`, and returns the
    corresponding GraphState and a coordinate-to-node-index mapping.

    Parameters
    ----------
    lx : int
        Lattice size in x direction
    ly : int
        Lattice size in y direction
    lz : int
        Lattice size in z direction
    allowed_parities : list[tuple[int, int, int]]
        List of allowed parity patterns for node placement

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
    # 1. Create nodes and measurement groups
    gs, coord2node, grouping = _create_nodes_and_groups(lx, ly, lz, allowed_parities)

    # 2. Add physical edges
    _add_physical_edges(gs, coord2node)

    # 3. Create parity check groups
    x_parity_check_groups, z_parity_check_groups = _create_parity_check_groups(coord2node)

    # 4. Add data qubit stabilizers
    _add_data_qubit_stabilizers(coord2node, x_parity_check_groups, lx, ly, lz)

    return gs, coord2node, x_parity_check_groups, z_parity_check_groups, grouping


def _create_nodes_and_groups(
    lx: int, ly: int, lz: int, allowed_parities: list[tuple[int, int, int]]
) -> tuple[GraphState, dict[tuple[int, int, int], int], list[set[int]]]:
    """Create nodes and measurement groups for RHG lattice.

    Parameters
    ----------
    lx : int
        Lattice size in x direction
    ly : int
        Lattice size in y direction
    lz : int
        Lattice size in z direction
    allowed_parities : list[tuple[int, int, int]]
        List of allowed parity patterns for node placement

    Returns
    -------
    tuple[GraphState, dict[tuple[int, int, int], int], list[set[int]]]
        Graph state, coordinate to node mapping, and measurement groups
    """
    gs = GraphState()
    coord2node: dict[tuple[int, int, int], int] = {}
    coord2qindex: dict[tuple[int, int], int] = {}
    grouping: list[set[int]] = []

    for z in range(lz):
        data_qubits = set()
        ancilla_qubits = set()
        for y in range(ly):
            for x in range(lx):
                parity = (x % 2, y % 2, z % 2)
                if parity not in allowed_parities:
                    continue

                if (z == lz - 1) and parity not in data_parities:
                    # skip output layer if not in data_parities
                    continue
                node_idx = gs.add_physical_node()
                coord2node[x, y, z] = node_idx
                if z == lz - 1:  # output layer
                    if parity in data_parities:
                        gs.register_output(node_idx, coord2qindex[x, y])
                    else:
                        gs.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, 0.0))
                else:
                    if z == 0 and parity in data_parities:  # input layer
                        q_index = gs.register_input(node_idx)
                        coord2qindex[x, y] = q_index
                    gs.assign_meas_basis(node_idx, PlannerMeasBasis(Plane.XY, 0.0))

                if parity in data_parities:
                    data_qubits.add(node_idx)
                else:
                    ancilla_qubits.add(node_idx)
        grouping.extend((ancilla_qubits, data_qubits))

    return gs, coord2node, grouping


def _add_physical_edges(gs: GraphState, coord2node: dict[tuple[int, int, int], int]) -> None:
    """Add physical edges between adjacent nodes.

    Parameters
    ----------
    gs : GraphState
        Graph state to add edges to
    coord2node : dict[tuple[int, int, int], int]
        Mapping from coordinates to node indices
    """
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
                v = coord2node[nx, ny, nz]
                with contextlib.suppress(ValueError):
                    gs.add_physical_edge(u, v)


def _create_parity_check_groups(
    coord2node: dict[tuple[int, int, int], int],
) -> tuple[list[set[int]], list[set[int]]]:
    """Create X and Z parity check groups.

    Parameters
    ----------
    coord2node : dict[tuple[int, int, int], int]
        Mapping from coordinates to node indices

    Returns
    -------
    tuple[list[set[int]], list[set[int]]]
        X parity check groups and Z parity check groups
    """
    x_parity_check_groups: list[set[int]] = []
    z_parity_check_groups: list[set[int]] = []

    for (x, y, z), u in coord2node.items():
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

    return x_parity_check_groups, z_parity_check_groups


def _add_data_qubit_stabilizers(
    coord2node: dict[tuple[int, int, int], int],
    x_parity_check_groups: list[set[int]],
    lx: int,
    ly: int,
    lz: int,
) -> None:
    """Add data qubit stabilizer groups to X parity checks.

    Parameters
    ----------
    coord2node : dict[tuple[int, int, int], int]
        Mapping from coordinates to node indices
    x_parity_check_groups : list[set[int]]
        List of X parity check groups to append to
    lx : int
        Lattice size in x direction
    ly : int
        Lattice size in y direction
    lz : int
        Lattice size in z direction
    """
    for i in range((lx - 1) // 2):
        for j in range((ly + 1) // 2):
            group: set[int] = set()
            pos0 = (2 * i, 2 * j, lz - 1)
            pos1 = (2 * i + 1, 2 * j - 1, lz - 1)
            pos2 = (2 * i + 2, 2 * j, lz - 1)
            pos3 = (2 * i + 1, 2 * j + 1, lz - 1)

            if node0 := coord2node.get(pos0):
                group.add(node0)
            if node1 := coord2node.get(pos1):
                group.add(node1)
            if node2 := coord2node.get(pos2):
                group.add(node2)
            if node3 := coord2node.get(pos3):
                group.add(node3)

            # add the previous stabilizer measurement
            group.add(coord2node[2 * i + 1, 2 * j, lz - 2])
            x_parity_check_groups.append(group)


def visualize_rhg(  # noqa: PLR0913, PLR0917
    lattice_state: GraphState,
    coord2node: dict[tuple[int, int, int], int],
    allowed_parities: list[tuple[int, int, int]] = allowed_parities,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[int, int] = (6, 6),
    dpi: int = 120,
) -> None:
    """Visualizes the Raussendorf lattice with nodes colored based on their parity.

    Nodes with allowed parities are colored white, others are red.
    Physical edges are drawn in gray.

    Parameters
    ----------
    lattice_state : GraphState
        The Raussendorf lattice state to visualize.
    coord2node : dict[tuple[int, int, int], int]
        Mapping from coordinates to node indices.
    allowed_parities : list[tuple[int, int, int]], optional
        List of allowed parity patterns for nodes, by default allowed_parities
    save_path : Optional[str], optional
        Path to save the figure. If None, the figure is not saved.
        Directory will be created if it doesn't exist, by default None
    show : bool, optional
        Whether to display the figure. Set to False for non-interactive environments
        or when only saving is desired, by default True
    figsize : tuple[int, int], optional
        Figure size in inches (width, height), by default (6, 6)
    dpi : int, optional
        Figure resolution in dots per inch, by default 120

    """
    node2coord: dict[int, tuple[int, int, int]] = {node: coord for coord, node in coord2node.items()}

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.grid(False)
    ax.set_axis_off()

    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []
    colors = []
    for x, y, z in node2coord.values():
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
        zs,  # pyright: ignore[reportArgumentType]
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

    # Save figure if path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    # Show figure if requested and in interactive mode
    if show:
        # Check if we're in a Jupyter notebook
        try:
            get_ipython()  # type: ignore[name-defined]
            # In Jupyter, just display the plot
            plt.show()
        except NameError:
            # Not in Jupyter, check if display is available
            if os.environ.get("DISPLAY") or os.name == "nt":
                plt.show()
    else:
        plt.close(fig)
