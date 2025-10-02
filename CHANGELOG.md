# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Issue templates (bug report, feature request, enhancement)
- Pull request template
- CHANGELOG.md file

## [0.1.0] - 2025-10-02

### Added
- Fault-tolerant merge and split functionality for MBQC error simulation (#18)
- Zero state initialization implementation with thin layer architecture (#16)
- Logical error rate evaluation with merge and split operations (#15)
- RHG memory logical error rate simulation (#14)
- TQEC knowledge integration and enhanced RHG Canvas architecture (#9)

### Changed
- Enhanced memory error simulation to support both plus and zero initialization
- Updated parity check handling with proper detector group structure
- Improved coordinate system consistency across cubes and pipes
- Refactored parity structure from list to dict for better detector organization

### Fixed
- Coordinate system fixes for proper temporal layer composition
- Z-coordinate calculations in measurement blocks for proper layer composition

[Unreleased]: https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/releases/tag/v0.1.0
