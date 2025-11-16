"""The base definition for RHG unit layers"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lspattern.consts import BoundarySide, EdgeSpecValue, NodeRole
from lspattern.new_blocks import coord_utils
from lspattern.new_blocks.accumulator import (
    CoordFlowAccumulator,
    CoordParityAccumulator,
    CoordScheduleAccumulator,
)
from lspattern.new_blocks.layer_data import CoordBasedLayerData
from lspattern.new_blocks.layout.rotated_surface_code import rotated_surface_code_layout
from lspattern.new_blocks.mytype import Coord2D, Coord3D


@dataclass
class UnitLayer:
    """Generic unit layer built from YAML configuration.

    This class represents a unit layer in the RHG lattice, constructed from
    YAML configuration files. It contains metadata about the layer structure
    and provides methods to build coordinate-based layer data.

    Attributes
    ----------
    name : str
        Name of the unit layer (e.g., "MemoryUnit", "InitZero").
    description : str
        Human-readable description of the layer's purpose.
    layout_type : str
        Type of layout (e.g., "rotated_surface_code").
    boundary : dict[BoundarySide, EdgeSpecValue]
        Boundary specifications for each side of the patch.
    layers_config : list[dict]
        Configuration for each physical layer (layer1, layer2).
        Each dict contains: {"basis": str, "ancilla": bool, "init": bool}
    """

    name: str
    description: str
    layout_type: str
    boundary: dict[BoundarySide, EdgeSpecValue]
    layers_config: list[dict[str, Any]]

    def build_metadata(
        self,
        code_distance: int,
        global_pos: Coord3D,
    ) -> CoordBasedLayerData:
        """Build coordinate-based metadata from YAML configuration.

        This method constructs a memory unit layer with two physical layers:
        - Layer z (even): Data qubits + Z-check ancillas
        - Layer z+1 (odd): Data qubits + X-check ancillas

        Parameters
        ----------
        code_distance : int
            Code distance for the surface code.
        global_pos : Coord3D
            Global (x, y, z) position offset for this unit layer.

        Returns
        -------
        CoordBasedLayerData
            Coordinate-based layer data with coordinates, edges, and accumulators.
        """
        # Get base layout at z=0 (will shift to global_pos.z later)
        base_data, base_x_ancilla, base_z_ancilla = rotated_surface_code_layout(
            code_distance=code_distance,
            global_pos=Coord3D(global_pos.x, global_pos.y, 0),
            boundary=self.boundary,
        )

        coords_by_z: dict[int, set[Coord3D]] = {}
        coord2role: dict[Coord3D, NodeRole] = {}

        # Build two layers
        for height in (0, 1):
            z = global_pos.z + height

            # Shift base coordinates to correct z-layer
            # Note: rotated_surface_code_layout returns Coord3D, so access via .x, .y, .z
            data_coords = {Coord3D(c[0], c[1], z) for c in base_data}

            if height == 0:
                # Even layer: Data + Z-check ancillas
                ancilla_coords = {Coord3D(c[0], c[1], z) for c in base_z_ancilla}
                ancilla_role = NodeRole.ANCILLA_Z
            else:
                # Odd layer: Data + X-check ancillas
                ancilla_coords = {Coord3D(c[0], c[1], z) for c in base_x_ancilla}
                ancilla_role = NodeRole.ANCILLA_X

            all_coords = data_coords | ancilla_coords
            coords_by_z[z] = all_coords

            # Assign roles
            for coord in data_coords:
                coord2role[coord] = NodeRole.DATA
            for coord in ancilla_coords:
                coord2role[coord] = ancilla_role

        # Build spatial edges (ancilla -> 4 diagonal data neighbors)
        spatial_edges: set[tuple[Coord3D, Coord3D]] = set()
        for coords in coords_by_z.values():
            data_coords_z = {c for c in coords if coord2role[c] == NodeRole.DATA}
            ancilla_coords_z = {c for c in coords if coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}}
            spatial_edges.update(coord_utils.build_spatial_edges(data_coords_z, ancilla_coords_z))

        # Build temporal edges (data qubit vertical connections)
        z0 = global_pos.z
        z1 = global_pos.z + 1
        data_coords_z0 = {c for c in coords_by_z[z0] if coord2role[c] == NodeRole.DATA}
        data_coords_z1 = {c for c in coords_by_z[z1] if coord2role[c] == NodeRole.DATA}
        temporal_edges = coord_utils.build_temporal_edges(data_coords_z0, data_coords_z1)

        # Build schedule (ancillas measured at their layer)
        coord_schedule = CoordScheduleAccumulator()
        for z, coords in coords_by_z.items():
            ancilla_coords_z = {c for c in coords if coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}}
            coord_schedule.add_at_time(z, ancilla_coords_z)

        # Build flow (temporal edges: lower layer -> upper layer)
        coord_flow = CoordFlowAccumulator()
        for coord_z0, coord_z1 in temporal_edges:
            coord_flow.add_flow(coord_z0, coord_z1)

        # Build parity checks (one check per ancilla)
        coord_parity = CoordParityAccumulator()
        for z, coords in coords_by_z.items():
            ancilla_coords_z = {c for c in coords if coord2role[c] in {NodeRole.ANCILLA_X, NodeRole.ANCILLA_Z}}
            for ancilla in ancilla_coords_z:
                xy = Coord2D(ancilla.x, ancilla.y)
                coord_parity.add_check(xy, z, {ancilla})

        return CoordBasedLayerData(
            coords_by_z=coords_by_z,
            coord2role=coord2role,
            spatial_edges=spatial_edges,
            temporal_edges=temporal_edges,
            coord_schedule=coord_schedule,
            coord_flow=coord_flow,
            coord_parity=coord_parity,
        )


class CustomUnitLayer:
    """Custom unit layer defined by user-provided metadata."""

    def __init__(self, global_pos: Coord3D, layer_data: CoordBasedLayerData) -> None:
        """Initialize the custom unit layer with its global offset and metadata."""
        self._global_pos = global_pos
        self._layer_data = layer_data

    @property
    def global_pos(self) -> Coord3D:
        """Return the global (x, y, z) position of the unit layer."""
        return self._global_pos

    def build_metadata(
        self,
        z_offset: int,  # noqa: ARG002
    ) -> CoordBasedLayerData:
        """Return the pre-defined coordinate-based metadata for this custom unit layer."""
        return self._layer_data


def load_unit_layer_from_yaml(yaml_path: str | Path) -> UnitLayer:
    """Load unit layer configuration from YAML file.

    This function reads a YAML configuration file and constructs a UnitLayer
    instance with the specified parameters. The YAML file should define the
    layer name, description, layout type, boundary conditions, and layer
    configurations.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    UnitLayer
        A UnitLayer instance configured according to the YAML file.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is malformed or missing required fields.

    Examples
    --------
    >>> from pathlib import Path
    >>> yaml_file = Path("lspattern/new_blocks/patch_layout/layers/memory.yml")
    >>> unit_layer = load_unit_layer_from_yaml(yaml_file)
    >>> unit_layer.name
    'MemoryUnit'
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        msg = f"YAML file not found: {yaml_path}"
        raise FileNotFoundError(msg)

    with yaml_path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)

    # Validate required fields
    if "name" not in data:
        msg = "YAML file missing required field: name"
        raise ValueError(msg)
    if "params" not in data:
        msg = "YAML file missing required field: params"
        raise ValueError(msg)

    params = data["params"]

    # Parse boundary
    boundary_data = params.get("boundary", {})
    boundary: dict[BoundarySide, EdgeSpecValue] = {}
    for side_str, value_str in boundary_data.items():
        side = BoundarySide[side_str.upper()]
        boundary[side] = EdgeSpecValue[value_str]

    # Parse layers configuration
    # Each layer_dict in YAML is like {"layer1": None, "basis": "X", "ancilla": true, "init": false}
    # We need to extract only the layer config (basis, ancilla, init)
    layers_config: list[dict[str, Any]] = []
    for layer_dict in params.get("layers", []):
        # Filter out layer name keys (layer1, layer2, etc.) and keep only config params
        config = {k: v for k, v in layer_dict.items() if k not in {"layer1", "layer2"} and v is not None}
        if config:  # Only add if there's actual config
            layers_config.append(config)

    return UnitLayer(
        name=data["name"],
        description=data.get("description", ""),
        layout_type=params.get("layout", "rotated_surface_code"),
        boundary=boundary,
        layers_config=layers_config,
    )
