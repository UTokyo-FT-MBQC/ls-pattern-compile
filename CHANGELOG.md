# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Changed
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

### Removed
- Deprecated modules `rhg.py` and `ops.py` ([#21](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/21))

---

## Version [0.1.0] - 2025-10-02

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
