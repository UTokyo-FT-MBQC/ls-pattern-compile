from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt

from lspattern.canvas_loader import CanvasCubeSpec, CanvasPipeSpec, load_canvas_spec
from lspattern.layout import PatchCoordinates, RotatedSurfaceCodeLayoutBuilder
from lspattern.mytype import Coord2D, Coord3D, NodeRole

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from matplotlib.figure import Figure

    from lspattern.canvas import Canvas


class NodeStyleSpec(TypedDict):
    """Style specification for node visualization.

    Attributes
    ----------
    color : str
        Fill color of the node marker.
    line_color : str
        Border color of the node marker.
    size : int
        Size of the node marker in pixels.
    label : str
        Display label for the node type in the legend.
    """

    color: str
    line_color: str
    size: int
    label: str


class NodeGroup2D(TypedDict):
    """Grouped node data for 2D visualization.

    Attributes
    ----------
    x : list[int]
        List of x coordinates.
    y : list[int]
        List of y coordinates.
    coords : list[Coord3D]
        List of Coord3D objects.
    """

    x: list[int]
    y: list[int]
    coords: list[Coord3D]


_COLOR_MAP: dict[NodeRole, NodeStyleSpec] = {
    NodeRole.DATA: {"color": "white", "line_color": "black", "size": 100, "label": "Data"},
    NodeRole.ANCILLA_X: {"color": "green", "line_color": "darkgreen", "size": 80, "label": "X ancilla"},
    NodeRole.ANCILLA_Z: {"color": "blue", "line_color": "darkblue", "size": 80, "label": "Z ancilla"},
}

_HIGHLIGHT_STYLE: NodeStyleSpec = {
    "color": "red",
    "line_color": "darkred",
    "size": 150,
    "label": "Highlighted",
}


def _group_nodes_2d(
    nodes: Iterable[Coord3D],
    coord2role: Mapping[Coord3D, NodeRole],
) -> dict[NodeRole, NodeGroup2D]:
    """Group nodes by their role for 2D visualization traces.

    Parameters
    ----------
    nodes : Iterable[Coord3D]
        Iterable of 3D coordinates representing nodes in the canvas.
    coord2role : Mapping[Coord3D, NodeRole]
        Mapping from coordinates to their roles. Nodes not in the mapping
        default to NodeRole.DATA.

    Returns
    -------
    dict[NodeRole, NodeGroup2D]
        Dictionary mapping each node role to its grouped 2D coordinate data.
        Each NodeGroup2D contains separate lists for x, y coordinates
        and the original Coord3D objects.
    """
    groups: dict[NodeRole, NodeGroup2D] = {role: {"x": [], "y": [], "coords": []} for role in _COLOR_MAP}
    for coord in nodes:
        role = coord2role.get(coord, NodeRole.DATA)
        groups[role]["x"].append(coord.x)
        groups[role]["y"].append(coord.y)
        groups[role]["coords"].append(coord)
    return groups


def visualize_canvas_matplotlib_2d(
    canvas: Canvas,
    target_z: int,
    *,
    highlight_nodes: Iterable[Coord3D] | None = None,
    show_edges: bool = True,
    edge_color: str = "gray",
    edge_width: float = 0.5,
    edge_alpha: float = 0.5,
    figsize: tuple[int, int] = (8, 8),
    show_grid: bool = True,
    reverse_axes: bool = False,
) -> Figure:
    """Create a 2D visualization of a Canvas at a specific Z-slice using Matplotlib.

    This function renders nodes at a specific Z coordinate, colored by their role
    (data, X ancilla, Z ancilla) and optionally displays edges connecting them.
    The visualization uses 2D scatter plots with customizable styling.

    Parameters
    ----------
    canvas : Canvas
        The Canvas object to visualize, containing nodes, edges, and role information.
    target_z : int
        The Z coordinate of the slice to visualize.
    highlight_nodes : Iterable[Coord3D] | None, optional
        Nodes to highlight in red color. These nodes will be drawn on top
        of regular nodes with a larger marker size. By default None.
    show_edges : bool, optional
        Whether to display edges between nodes, by default True.
    edge_color : str, optional
        Color string for edges, by default "gray".
    edge_width : float, optional
        Width of edge lines, by default 0.5.
    edge_alpha : float, optional
        Transparency of edge lines (0.0 to 1.0), by default 0.5.
    figsize : tuple[int, int], optional
        Figure size in inches (width, height), by default (8, 8).
    show_grid : bool, optional
        Whether to show grid lines, by default True.
    reverse_axes : bool, optional
        Reverse X and Y axes to match quantum circuit layout convention,
        by default False.

    Returns
    -------
    Figure
        Matplotlib Figure object ready for display or further customization.

    Examples
    --------
    >>> fig = visualize_canvas_matplotlib_2d(canvas, target_z=0)
    >>> plt.show()
    """
    # Filter nodes at target Z coordinate
    nodes_at_z = {node for node in canvas.nodes if node.z == target_z}

    # Group nodes by role
    coord2role = canvas.coord2role
    groups = _group_nodes_2d(nodes_at_z, coord2role)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot nodes grouped by role
    for role, pts in groups.items():
        if not pts["coords"]:  # Skip empty groups
            continue
        spec = _COLOR_MAP[role]
        ax.scatter(
            pts["x"],
            pts["y"],
            c=spec["color"],
            edgecolors=spec["line_color"],
            s=spec["size"],
            label=spec["label"],
            alpha=0.9,
            linewidths=1.5,
            zorder=2,  # Draw nodes on top of edges
        )

    # Draw highlighted nodes on top
    if highlight_nodes is not None:
        highlight_set = set(highlight_nodes)
        highlight_at_z = [n for n in highlight_set if n in nodes_at_z]
        if highlight_at_z:
            hx = [n.x for n in highlight_at_z]
            hy = [n.y for n in highlight_at_z]
            ax.scatter(
                hx,
                hy,
                c=_HIGHLIGHT_STYLE["color"],
                edgecolors=_HIGHLIGHT_STYLE["line_color"],
                s=_HIGHLIGHT_STYLE["size"],
                label=_HIGHLIGHT_STYLE["label"],
                alpha=1.0,
                linewidths=2.0,
                zorder=3,  # Draw on top of regular nodes
            )

    # Draw edges if requested
    if show_edges:
        edges = canvas.edges
        for start, end in edges:
            # Only draw edges where both endpoints are in the current Z slice
            if start not in nodes_at_z or end not in nodes_at_z:
                continue
            ax.plot(
                [start.x, end.x],
                [start.y, end.y],
                color=edge_color,
                linewidth=edge_width,
                alpha=edge_alpha,
                zorder=1,  # Draw edges behind nodes
            )

    # Configure axes
    ax.set_aspect("equal", "box")
    ax.grid(show_grid, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Canvas 2D Slice at Z={target_z}")
    ax.legend(loc="best")

    # Reverse axes if requested (quantum circuit convention)
    if reverse_axes:
        # Reverse setting: invert X axis only
        ax.invert_xaxis()
    else:
        # Default: invert Y axis only
        ax.invert_yaxis()

    return fig


# =============================================================================
# Color map for 2D qubit coordinate types
# =============================================================================

_COORD_COLOR_MAP: dict[str, NodeStyleSpec] = {
    "data": {"color": "white", "line_color": "black", "size": 100, "label": "Data"},
    "ancilla_x": {"color": "green", "line_color": "darkgreen", "size": 80, "label": "X ancilla"},
    "ancilla_z": {"color": "blue", "line_color": "darkblue", "size": 80, "label": "Z ancilla"},
}


# =============================================================================
# Helper functions for layer visualization
# =============================================================================


def _collect_blocks_at_z(
    cubes: Sequence[CanvasCubeSpec],
    pipes: Sequence[CanvasPipeSpec],
    target_z: int,
) -> tuple[list[CanvasCubeSpec], list[CanvasPipeSpec]]:
    """Collect cubes and pipes at a specific z-layer.

    Parameters
    ----------
    cubes : Sequence[CanvasCubeSpec]
        All cube specifications from the canvas.
    pipes : Sequence[CanvasPipeSpec]
        All pipe specifications from the canvas.
    target_z : int
        The patch z-coordinate to filter by.

    Returns
    -------
    tuple[list[CanvasCubeSpec], list[CanvasPipeSpec]]
        (cubes_at_z, pipes_at_z) - cubes and spatial pipes at the target z.
    """
    cubes_at_z = [c for c in cubes if c.position.z == target_z]
    # Filter spatial pipes: both start.z and end.z should equal target_z
    pipes_at_z = [p for p in pipes if p.start.z == target_z and p.end.z == target_z]
    return cubes_at_z, pipes_at_z


def _generate_cube_coordinates(
    cube: CanvasCubeSpec,
    code_distance: int,
) -> PatchCoordinates:
    """Generate PatchCoordinates for a cube.

    Parameters
    ----------
    cube : CanvasCubeSpec
        The cube specification.
    code_distance : int
        Code distance of the surface code.

    Returns
    -------
    PatchCoordinates
        Coordinate sets for the cube.
    """
    global_pos = Coord2D(cube.position.x, cube.position.y)
    return RotatedSurfaceCodeLayoutBuilder.cube(code_distance, global_pos, cube.boundary)


def _generate_pipe_coordinates(
    pipe: CanvasPipeSpec,
    code_distance: int,
) -> PatchCoordinates:
    """Generate PatchCoordinates for a pipe.

    Parameters
    ----------
    pipe : CanvasPipeSpec
        The pipe specification.
    code_distance : int
        Code distance of the surface code.

    Returns
    -------
    PatchCoordinates
        Coordinate sets for the pipe.
    """
    return RotatedSurfaceCodeLayoutBuilder.pipe(code_distance, pipe.start, pipe.end, pipe.boundary)


def _merge_patch_coordinates(coords_list: Sequence[PatchCoordinates]) -> PatchCoordinates:
    """Merge multiple PatchCoordinates into one.

    Parameters
    ----------
    coords_list : Sequence[PatchCoordinates]
        Sequence of PatchCoordinates to merge.

    Returns
    -------
    PatchCoordinates
        Merged coordinate sets.
    """
    if not coords_list:
        return PatchCoordinates(frozenset(), frozenset(), frozenset())

    data: set[Coord2D] = set()
    ancilla_x: set[Coord2D] = set()
    ancilla_z: set[Coord2D] = set()

    for coords in coords_list:
        data.update(coords.data)
        ancilla_x.update(coords.ancilla_x)
        ancilla_z.update(coords.ancilla_z)

    return PatchCoordinates(frozenset(data), frozenset(ancilla_x), frozenset(ancilla_z))


# =============================================================================
# Public visualization functions
# =============================================================================


def visualize_patch_coordinates_2d(
    coords: PatchCoordinates,
    *,
    title: str = "Patch Coordinates",
    figsize: tuple[int, int] = (10, 10),
    show_grid: bool = True,
) -> Figure:
    """Visualize PatchCoordinates as a 2D scatter plot.

    This is a lower-level function for direct PatchCoordinates visualization.
    Useful for testing individual blocks.

    Parameters
    ----------
    coords : PatchCoordinates
        The coordinate sets to visualize.
    title : str, optional
        Title of the plot, by default "Patch Coordinates".
    figsize : tuple[int, int], optional
        Figure size in inches (width, height), by default (10, 10).
    show_grid : bool, optional
        Whether to show grid lines, by default True.

    Returns
    -------
    Figure
        Matplotlib Figure object ready for display or saving.

    Examples
    --------
    >>> from lspattern.layout import RotatedSurfaceCodeLayoutBuilder
    >>> from lspattern.consts import BoundarySide, EdgeSpecValue
    >>> from lspattern.mytype import Coord2D
    >>> boundary = {
    ...     BoundarySide.TOP: EdgeSpecValue.X,
    ...     BoundarySide.BOTTOM: EdgeSpecValue.X,
    ...     BoundarySide.LEFT: EdgeSpecValue.Z,
    ...     BoundarySide.RIGHT: EdgeSpecValue.Z,
    ... }
    >>> coords = RotatedSurfaceCodeLayoutBuilder.cube(3, Coord2D(0, 0), boundary)
    >>> fig = visualize_patch_coordinates_2d(coords, title="Single Cube")
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data qubits
    if coords.data:
        data_x = [c.x for c in coords.data]
        data_y = [c.y for c in coords.data]
        spec = _COORD_COLOR_MAP["data"]
        ax.scatter(
            data_x,
            data_y,
            c=spec["color"],
            edgecolors=spec["line_color"],
            s=spec["size"],
            label=spec["label"],
            alpha=0.9,
            linewidths=1.5,
            zorder=2,
        )

    # Plot X ancillas
    if coords.ancilla_x:
        ax_x = [c.x for c in coords.ancilla_x]
        ax_y = [c.y for c in coords.ancilla_x]
        spec = _COORD_COLOR_MAP["ancilla_x"]
        ax.scatter(
            ax_x,
            ax_y,
            c=spec["color"],
            edgecolors=spec["line_color"],
            s=spec["size"],
            label=spec["label"],
            alpha=0.9,
            linewidths=1.5,
            zorder=2,
        )

    # Plot Z ancillas
    if coords.ancilla_z:
        az_x = [c.x for c in coords.ancilla_z]
        az_y = [c.y for c in coords.ancilla_z]
        spec = _COORD_COLOR_MAP["ancilla_z"]
        ax.scatter(
            az_x,
            az_y,
            c=spec["color"],
            edgecolors=spec["line_color"],
            s=spec["size"],
            label=spec["label"],
            alpha=0.9,
            linewidths=1.5,
            zorder=2,
        )

    # Configure axes
    ax.set_aspect("equal", "box")
    ax.grid(show_grid, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    # Only show legend if there are labeled artists
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    ax.invert_yaxis()

    return fig


def visualize_canvas_layout(
    yaml_path: str | Path,
    code_distance: int,
    target_z: int,
    *,
    figsize: tuple[int, int] = (10, 10),
    show_grid: bool = True,
    show_boundary_labels: bool = True,
) -> Figure:
    """Visualize all cubes and pipes at a specific z-layer for boundary debugging.

    This function loads a YAML canvas file and renders all blocks (cubes and
    spatial pipes) at the specified patch z-coordinate in a single 2D figure.
    Useful for debugging boundary conditions and qubit placements.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML canvas file.
    code_distance : int
        Code distance of the surface code.
    target_z : int
        Patch z-coordinate to visualize (as specified in YAML).
    figsize : tuple[int, int], optional
        Figure size in inches (width, height), by default (10, 10).
    show_grid : bool, optional
        Whether to show grid lines, by default True.
    show_boundary_labels : bool, optional
        Whether to show block boundary labels (currently unused, reserved for future),
        by default True.

    Returns
    -------
    Figure
        Matplotlib Figure object ready for display or saving.

    Examples
    --------
    >>> from lspattern.visualizer_2d import visualize_canvas_layout
    >>> import matplotlib.pyplot as plt

    >>> # Visualize all blocks at z=1
    >>> fig = visualize_canvas_layout(
    ...     "examples/design/cnot.yml",
    ...     code_distance=3,
    ...     target_z=1,
    ... )
    >>> plt.show()  # doctest: +SKIP

    >>> # Visualize z=0 layer (init blocks)
    >>> fig = visualize_canvas_layout(
    ...     "examples/design/short_z_memory_canvas.yml",
    ...     code_distance=3,
    ...     target_z=0,
    ... )
    >>> plt.savefig("/tmp/layer_z0.png")  # doctest: +SKIP
    """
    # Silence unused parameter warning - reserved for future use
    _ = show_boundary_labels

    # Load canvas spec from YAML
    spec = load_canvas_spec(yaml_path)

    # Collect blocks at target z
    cubes_at_z, pipes_at_z = _collect_blocks_at_z(spec.cubes, spec.pipes, target_z)

    # Generate coordinates for each block
    all_coords: list[PatchCoordinates] = []

    for cube in cubes_at_z:
        coords = _generate_cube_coordinates(cube, code_distance)
        all_coords.append(coords)

    for pipe in pipes_at_z:
        coords = _generate_pipe_coordinates(pipe, code_distance)
        all_coords.append(coords)

    # Merge all coordinates
    merged = _merge_patch_coordinates(all_coords)

    # Build title
    num_cubes = len(cubes_at_z)
    num_pipes = len(pipes_at_z)
    title = f"Layer z={target_z}: {num_cubes} cube(s), {num_pipes} pipe(s)"

    # Use the lower-level function to render
    return visualize_patch_coordinates_2d(
        merged,
        title=title,
        figsize=figsize,
        show_grid=show_grid,
    )
