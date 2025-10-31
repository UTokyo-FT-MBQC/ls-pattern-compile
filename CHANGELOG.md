# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `compile_to_stim()` function in `lspattern.compile` to streamline CompiledRHGCanvas to stim.Circuit compilation ([#63](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/63))
  - Unified API that internally handles scheduler setup, parity extraction, pattern compilation, and logical observable resolution
  - Supports both PatchCoordGlobal3D and PipeCoordGlobal3D for logical observable coordinates
  - Reduces boilerplate code from ~30-40 lines to ~3-5 lines in example files
  - Updated all example files (`memory_error_sim.py`, `plus_initialization.py`, `zero_initialization.py`, `merge_split_error_sim.py`) to use the new API

---

## Version [0.0.4] - 2025-10-31

### Added
- CI status badges (pytest, type checking, ruff) to README.md
- Complete logical error rate simulation for merge and split operations ([#19](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/19))
  - Cout port setting functionality for pipes (`InitPlusPipe`, `MeasureXPipe`, `MeasureZPipe`)
  - Full multi-observable evaluation: Z₁, Z₂, Z₁Z₂ and X₁X₂ observables
  - New example file `examples/merge_split_error_sim_xxinit.py` for XX-initialized merge/split error simulation
  - Enhanced `examples/merge_split_error_sim.py` with comprehensive error pattern correlation analysis
  - Per-observable error rates and standard errors calculation
  - Statistical analysis tools for observable error combinations using Sinter's `count_observable_error_combos`
  - Correlation coefficient calculation between observables (Pearson correlation)
  - Multi-panel visualization showing total error rate, per-observable rates, and correlation matrices
- Pipe-specific cout_port unit tests in `tests/canvas/test_ports.py` (12 new tests)
  - `test_register_cout_group_pipe_basic`: Basic pipe cout group registration
  - `test_register_multiple_pipe_cout_groups`: Multiple groups per pipe
  - `test_register_empty_pipe_cout_group`: Empty group handling
  - `test_register_pipe_cout_group_with_none_values`: None value filtering
  - `test_get_cout_group_by_node_pipe`: Node-based pipe group retrieval
  - `test_cube_and_pipe_cout_groups_separated`: Cube/pipe separation verification
  - `test_cube_and_pipe_at_same_sink_coordinate`: **Key test verifying cube/pipe disambiguation at same sink coordinate**
  - `test_remap_pipe_cout_groups`: Pipe cout group remapping
  - `test_rebuild_pipe_cout_group_cache`: Pipe cache rebuilding
  - `test_copy_includes_pipe_cout_ports`: Copy method pipe handling
  - `test_merge_pipe_cout_groups`: Merge method pipe handling
  - `test_multiple_pipes_different_coords`: Multiple pipe management

### Changed
- Refactored test file naming and documentation for improved clarity
  - Removed `Txx_` prefixes from test file names for better semantic naming
  - Renamed root-level test files to reflect their actual testing purpose:
    - `test_T37_seam_edges_same_z.py` → `test_seam_edges_horizontal_vertical.py`
    - `test_T39_memory.py` → `test_memory_blocks_ports_boundaries.py`
    - `test_T41_blocks_basic.py` → `test_basic_block_types_representation.py`
    - `test_T43_compile_smoke.py` → `test_canvas_compilation_smoke.py`
    - `test_mockup.py` → `test_merge_split_mockup_snapshot.py`
    - `test_temporal_and_spatial.py` → `test_canvas_snapshot_temporal_spatial.py`
  - Renamed canvas subdirectory test files to match tested class names:
    - `test_composition.py` → `test_graph_composer.py`
    - `test_coordinates.py` → `test_coordinate_mapper.py`
    - `test_ports.py` → `test_port_manager.py`
    - `test_seams.py` → `test_seam_generator.py`
  - Translated all Japanese comments in test files to English
  - Translated snapshot testing guide from Japanese to English
    - `tests/snapshotの手引き.md` → `tests/snapshot_testing_guide.md`
- Tightened typing across tests and visualization helpers to satisfy strict `mypy`/`pyright` runs
  - Swapped ad-hoc edge spec dictionaries for `BoundarySide`/`EdgeSpecValue` enums in block tests
  - Standardized seam generator fixtures and layered-block lookups on `NodeIdLocal`/`PhysCoordGlobal3D`
  - Replaced tuple literals with `AxisMeasBasis` in graph utils tests to align with `GraphState.assign_meas_basis`
  - Added optional `Axes` guards in Matplotlib-based visualizers to silence optional-member checks
  - Converted logical observable gathering in merge/split examples to use typed port maps
- Expanded CI test matrix to cover Python 3.10, 3.11, 3.12, and 3.13 ([#61](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/61))
- Unified `x_parity` and `z_parity` parameters into single `parity` parameter in `compile_canvas()`
  - Updated `lspattern.compile.compile_canvas()` function signature to match graphqomb's unified `parity_check_group` API
  - Removed separate `z_parity` parameter (was unused in all examples)
  - Updated all example files to use unified `parity` variable naming
  - Updated README.md documentation with new API usage
- **BREAKING CHANGE**: Separated cube and pipe cout_port management in `PortManager` to eliminate ambiguity
  - Pipes are now uniquely identified by `PipeCoordGlobal3D` (source, sink) tuple instead of sink coordinate only
  - Renamed methods:
    - `register_cout_group()` → `register_cout_group_cube()` for cube cout_ports
    - Added new `register_cout_group_pipe()` for pipe cout_ports
  - Split unified `cout_portset` into separate dictionaries:
    - `cout_portset_cube: dict[PatchCoordGlobal3D, list[NodeIdLocal]]` for cubes
    - `cout_portset_pipe: dict[PipeCoordGlobal3D, list[NodeIdLocal]]` for pipes
  - Split unified `cout_port_groups` into:
    - `cout_port_groups_cube: dict[PatchCoordGlobal3D, list[list[NodeIdLocal]]]`
    - `cout_port_groups_pipe: dict[PipeCoordGlobal3D, list[list[NodeIdLocal]]]`
  - Split lookup caches into:
    - `cout_group_lookup_cube: dict[NodeIdLocal, tuple[PatchCoordGlobal3D, int]]`
    - `cout_group_lookup_pipe: dict[NodeIdLocal, tuple[PipeCoordGlobal3D, int]]`
  - Updated `get_cout_group_by_node()` return type to `tuple[PatchCoordGlobal3D | PipeCoordGlobal3D, list[NodeIdLocal]]`
  - Updated `lspattern/canvas/composition.py` to use `pipe_coord` for pipe cout_port registration
  - Updated `lspattern/canvas/compiled.py` to expose separate `cout_portset_cube` and `cout_portset_pipe` properties
  - Updated `lspattern/canvas/layer.py` to expose separate cube/pipe cout_port properties
  - Updated all 8 example files to use `cout_portset_cube` instead of `cout_portset`:
    - `examples/plus_initialization.py`
    - `examples/zero_initialization.py`
    - `examples/memory_error_sim.py`
    - `examples/layered_unit_memory_demo.py`
    - `examples/merge_split_error_sim.py`
    - `examples/merge_split_mockup.py`
    - `examples/merge_split_xx_mockup.py`
    - `examples/merge_split_xx_error_sim.py`
  - Updated `tests/canvas/test_ports.py` to use new API
  - Updated `tests/test_mockup.py` and `tests/test_temporal_and_spatial.py` to use `cout_portset_cube`
- Improved error handling in `GraphComposer` to raise exceptions instead of suppressing errors
  - Better diagnostics for missing node mappings and coordinate system mismatches
- Removed redundant methods from internal APIs for cleaner codebase
- Switched to install `graphqomb` from PyPI server
- Switch from `DIRECTION3D` to `DIRECTION2D`

### Fixed
- Fixed pipe port management to use `PipeCoordGlobal3D` consistently ([#74](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/74))
  - Added `in_portset_pipe` and `out_portset_pipe` dictionaries using `PipeCoordGlobal3D` keys
  - Added `add_in_ports_pipe()` and `add_out_ports_pipe()` methods to `PortManager`
  - Updated `GraphComposer.process_pipe_ports()` to use pipe-specific methods instead of converting to `PatchCoordGlobal3D`
  - All pipe ports (in/out/cout) now consistently use `PipeCoordGlobal3D` coordinate system
  - Enables proper retrieval of all ports associated with a specific pipe
  - Resolves ambiguity when multiple pipes share the same sink coordinate
  - Added 11 comprehensive unit tests for pipe in/out port management to `tests/canvas/test_ports.py`
- Fixed critical duplicated node remapping bug in `CompiledRHGCanvas` ([#19](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/19))
  - Eliminated duplicate calls to `_remap_graph_nodes()` that caused incorrect node ID mappings
  - Resolved issue where `node_map_global` was being remapped twice during compilation
  - This fix was essential for correct multi-observable evaluation in error simulations
- Fixed `node_map_global` composition in pipe connection handling
  - Ensured proper node ID mapping across pipe-cube boundaries
  - Added proper merging of node maps from different coordinate systems

---

## Version [0.0.3]  - 2025-10-16

### Added
- Layer-by-layer construction architecture for RHG blocks ([#23](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/23))
  - `UnitLayer` abstract base class for 2-layer unit (1 X-check + 1 Z-check cycle)
  - `LayerData` dataclass for encapsulating layer metadata
  - `MemoryUnitLayer` for standard memory layers
  - `InitPlusUnitLayer` for |+⟩ state initialization layers
  - `InitZeroUnitLayer` for |0⟩ state initialization layers
  - `EmptyUnitLayer` for empty placeholder layers (no nodes)
  - `LayeredRHGCube` base class for layer-by-layer cube construction
  - `LayeredMemoryCube`, `LayeredInitPlusCube`, `LayeredInitZeroCube` concrete implementations
  - `LayeredRHGPipe` base class for layer-by-layer pipe construction
  - `LayeredMemoryPipe`, `LayeredInitPlusPipe` concrete implementations
  - Support for flexible composition of different layer types within blocks
  - Enables customization at 2-layer granularity instead of full 2*d layer blocks
  - Automatic temporal edge connection across empty layers
  - Validation to ensure `unit_layers` length does not exceed code distance `d`
- `TemporalBoundarySpecValue` enum in `lspattern/consts/consts.py` for type-safe temporal boundary specification ([#23](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/23))
  - `O`: Open boundary (add final data layer)
  - `MX`: X-basis measurement at final layer
  - `MZ`: Z-basis measurement at final layer

### Removed
- Legacy `lspattern.geom` package remnants (`__init__.py`, `rhg_parity.py`, `tiler.py`) and unused visualizer stubs (`visualize.py`, `template.py`) as part of cleanup for [#64](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/64).
- Skipped legacy inactive tests `tests/test_T42_skip.py` and `tests/test_T48_skip.py`.

### Changed
- Refined temporal-layer visualizers to remove dependencies on deleted geom helpers, harmonize color palettes between Matplotlib and Plotly variants, and improve axis handling and input/output highlighting.
- `RHGBlock.final_layer` field type changed from `str | None` to `TimeBoundarySpecValue` for improved type safety ([#23](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/23))
- Updated dependency from `graphix-zx` to `graphqomb` following upstream library rename
  - Updated all import statements across library code, examples, and tests
  - Updated installation documentation in README.md
  - Updated requirements-dev.txt dependency URL


---

## Version [0.0.2] - 2025-10-10

### Added
- Apache License 2.0 (LICENSE file)
- License information in pyproject.toml
- License badge in README.md
- Acknowledgment of tqec design inspiration in README.md
- Issue templates (bug report, feature request, enhancement)
- Pull request template
- CHANGELOG.md file
- Comprehensive README.md with installation guide, usage examples, project status, and citation format
- `PortManager` class for managing input/output/cout ports ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33) Phase 1.1)
- `CoordinateMapper` class for managing node-coordinate bidirectional mappings ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33) Phase 1.2)
- `GraphComposer` class for handling graph composition logic ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33) Phase 1.3)
- `SeamGenerator` class for generating CZ edges across cube-pipe seams ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33) Phase 1.4)
- Unit tests for port management functionality (`tests/canvas/test_ports.py`)
- Unit tests for coordinate mapping functionality (`tests/canvas/test_coordinates.py`)
- Unit tests for graph composition functionality (`tests/canvas/test_composition.py`)
- Enum classes in `lspattern/consts/consts.py` for type-safe constants ([#36](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/36))
  - `EdgeSpecValue`: Edge specification values (X, Z, O)
  - `BoundarySide`: Spatial boundary sides (TOP, BOTTOM, LEFT, RIGHT, UP, DOWN)
  - `NodeRole`: Node roles (DATA, ANCILLA_X, ANCILLA_Z)
  - `CoordinateSystem`: Coordinate system identifiers (TILING_2D, PHYS_3D, PATCH_3D)
  - `VisualizationKind`: Visualization kind options (BOTH, X, Z)
  - `VisualizationMode`: Visualization mode options (HIST, SLICES)
- Unit tests for seam edge generation functionality (`tests/canvas/test_seams.py`)
 - Plotly visualization option `hilight_nodes` in `visualize_compiled_canvas_plotly` to highlight specific nodes for review/debug (PR #55)
 - Example: `examples/merge_split_xx_error_sim.py` for XX merge/split error simulation (PR #55)
- Unit tests for graph remapping utilities (`tests/canvas/test_graph_utils.py`) ([#52](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/52))

### Changed
- Extracted graph remapping utilities to separate module `lspattern/canvas/graph_utils.py` ([#52](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/52))
  - Moved `_remap_graph_nodes()`, `_remap_measurement_bases()`, `_remap_graph_edges()`, and `_create_remapped_graphstate()` from `CompiledRHGCanvas` to module-level functions
  - Improved code modularity and testability
- Unified into absolute coordinate ([#24](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/pull/24))
- Refactored `lspattern/canvas.py` into modular structure ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33))
  - Created `lspattern/canvas/` package with separate modules
  - Moved `MixedCodeDistanceError` to `lspattern/canvas/exceptions.py`
  - Extracted port management logic into `lspattern/canvas/ports.py` (Phase 1.1)
  - Extracted coordinate mapping logic into `lspattern/canvas/coordinates.py` (Phase 1.2)
  - Extracted graph composition logic into `lspattern/canvas/composition.py` (Phase 1.3)
  - Extracted seam edge generation logic into `lspattern/canvas/seams.py` (Phase 1.4)
  - Moved main canvas implementation to `lspattern/canvas/_canvas_impl.py`
  - Maintained backward compatibility through `lspattern/canvas/__init__.py`
- Refactored `TemporalLayer` to use `GraphComposer` for graph building operations ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33) Phase 1.3)
- Replaced string literals with type-safe enums for improved type safety ([#36](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/36))
  - Updated `lspattern/mytype.py` to use enum-based type aliases
  - Refactored all library code to use enum values instead of string literals
  - Updated function signatures to accept enum types
  - Using `str` mixin (`class X(str, Enum)`) for backward compatibility
- Refactored `TemporalLayer` to use `SeamGenerator` for seam edge generation ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33) Phase 1.4)
 - Removed `@overload` type stubs from pipe skeletons’ `to_block` methods for simplicity (PR #55 review)
 - Standardized example filenames to include `_xx` suffix (e.g., `merge_split_xx_mockup.py`) (PR #55 review)

### Fixed

- Switched to `typing_extensions.assert_never` from `typing.assert_never` since `py310` doesn't support the latter.
 - Deterministic X-seam detector pairing in `InitZeroPipe` (prevents singleton detectors at seam; fixes non-deterministic groups around nodes like `{363}` by pairing e.g. `{363, 375}`) (PR #55, fixes #20)

### Examples
- Regenerated `examples/merge_split_xx_mockup.txt` parity dump to reflect corrected detector groups (PR #55)

### Removed
- Deprecated modules `rhg.py` and `ops.py` ([#21](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/21))

---

## Version [0.0.1] - 2025-10-02

### Added
- Fault-tolerant merge and split functionality for MBQC error simulation ([#18](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/pull/18))
- Zero state initialization implementation with thin layer architecture ([#16](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/pull/16))
- Logical error rate evaluation with merge and split operations ([#15](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/pull/15))
- RHG memory logical error rate simulation ([#14](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/pull/14))
- TQEC knowledge integration and enhanced RHG Canvas architecture ([#9](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/pull/9))

### Changed
- Enhanced memory error simulation to support both plus and zero initialization
- Updated parity check handling with proper detector group structure
- Improved coordinate system consistency across cubes and pipes
- Refactored parity structure from list to dict for better detector organization

### Fixed
- Coordinate system fixes for proper temporal layer composition
- Z-coordinate calculations in measurement blocks for proper layer composition
