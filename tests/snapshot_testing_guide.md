# Snapshot Testing Guide

This document outlines the operational procedures for saving and comparing CompiledRHGCanvas as snapshots (JSON) to detect regressions. The approach is to stabilize and compare results without modifying the circuit logic from notebooks.

## Purpose
- Migrate notebook circuits (e.g., visualize_T43.ipynb, merge_split_mockup.ipynb) to pytest, serialize CompiledRHGCanvas to dictionary → save as JSON
- Generate execution results in the same format for subsequent runs and verify complete JSON match (regression testing)

## Snapshot Location
- `tests/snapshots/T43_compiled_canvas.json`
- `tests/snapshots/mockup_compiled_canvas.json`

Each test file references the JSON with the same name.

## What is Saved (Field Specification)
- `meta`: Metadata
  - `layers`: Number of layers (TemporalLayer count)
  - `zlist`: List of z-coordinates constructed so far
  - `coord_map`: Number of entries in `coord2node`
  - `nodes`: Number of nodes in GraphState
  - `edges`: Number of edges in GraphState
- `coords`: All node coordinates `(x,y,z)` sorted in ascending order (stabilized) as an array
- `edges_coords`: Each edge mapped to "coordinates of both endpoint nodes", sorted in (min, max) order as an array
- `inputs`/`outputs`: GraphState input/output nodes converted to "coordinate string → logical index"
- `in_ports`/`out_ports`/`cout_ports`: Port nodes per patch coordinate converted to "array of coordinate strings"

Note: Everything is normalized to coordinate-based representation, so it is not affected by non-determinism in node IDs.

## Initial Generation / Update Method
If the snapshot is not yet generated, or if you want to update the snapshot due to intentional specification changes, run pytest with the environment variable `UPDATE_SNAPSHOTS=1`.

- PowerShell (Windows)
```
$env:UPDATE_SNAPSHOTS=1; pytest -k test_T43_temporal_and_spatial_snapshot -q
$env:UPDATE_SNAPSHOTS=1; pytest -k test_merge_split_mockup_snapshot -q
Remove-Item Env:UPDATE_SNAPSHOTS
```

- Bash (Linux/macOS, Git Bash, etc.)
```
UPDATE_SNAPSHOTS=1 pytest -k test_T43_temporal_and_spatial_snapshot -q
UPDATE_SNAPSHOTS=1 pytest -k test_merge_split_mockup_snapshot -q
```

Upon success, JSON files will be created (or overwritten) in `tests/snapshots/`.

## Normal Execution (Comparison Only)
```
pytest -k test_T43_temporal_and_spatial_snapshot -q
pytest -k test_merge_split_mockup_snapshot -q
```
Verifies complete match with the snapshot. If there is a mismatch, the test fails, indicating a difference in circuit generation.

## Verification Points on Failure
- First check if the difference in `meta.nodes`/`meta.edges`/`coords` is large or small
- Which coordinates were added or removed (`coords` diff)
- Which connections changed (`edges_coords` diff)
- Differences in input/output nodes or ports (`inputs`/`outputs`/`in_ports`/`out_ports`/`cout_ports`)

Example of viewing JSON diff (Bash):
```
jq . tests/snapshots/T43_compiled_canvas.json > exp.json
jq . /tmp/got.json > got.json
diff -u exp.json got.json | less
```

## Procedure for Creating Snapshots from New Notebooks
1. Migrate the notebook's circuit configuration (block placement, edgespec, pipes) "as-is" to pytest
   - Example: `tests/test_temporal_and_spatial.py` / `tests/test_mockup.py`
2. Snapshot CompiledRHGCanvas to dictionary (reuse functions from this repository)
3. For the first run, generate snapshot with `UPDATE_SNAPSHOTS=1`
4. Thereafter, verify complete match with normal execution

Template (pseudocode):
```python
compiled = build_from_notebook_logic()
got = snapshot_compiled_canvas(compiled)
# First run: save with UPDATE_SNAPSHOTS=1, thereafter assert match
```

## Operational Notes / Common Pitfalls
- `RHGCanvas.to_temporal_layers()` can only be called once (to prevent duplicate shifting). To reconstruct, call `to_canvas()` again from `RHGCanvasSkeleton`.
- `materialize()` does not re-run `to_tiling()` if template coordinates are already set (to prevent XY shift overwriting).
- In temporal layer seam connections, there may be cases where gid cannot be retrieved for non-existent coordinates. In such cases, the implementation safely does not connect (`is_allowed_pair` is None-safe).
- If there are differences due to execution environment (dependency package versions/OS) in rendering or floating-point ordering, the snapshot stage normalizes everything to coordinates/sorted order, so they should generally match. If differences still appear, verify that the `snapshot` generation logic in the relevant test is deterministic.

## Existing Snapshot Tests
- T43: `tests/test_temporal_and_spatial.py` → `tests/snapshots/T43_compiled_canvas.json`
- Mockup: `tests/test_mockup.py` → `tests/snapshots/mockup_compiled_canvas.json`

End of guide. Please update this guide if there are any questions during operation.
