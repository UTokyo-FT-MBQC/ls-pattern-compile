# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Expanded CI test matrix to cover Python 3.10, 3.11, 3.12, and 3.13 ([#61](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/61))

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
