# Experimental: Graph-based block YAML (`graph:`)

This branch adds an *experimental* block format where users can provide the full graph information (nodes/edges/bases/flow/scheduler/detector candidates) as **JSON referenced from block YAML**, without writing Python.

## Where it plugs in

- `canvas.yml` still places `cube` / `pipe` entries and references a `block:` name.
- If the referenced block YAML contains `graph:`, `Canvas.add_cube/add_pipe` merges the user graph into the global canvas instead of generating a rotated-surface-code patch.

## Block YAML schema (draft)

```yml
name: MyGraphBlock
boundary: XXZZ   # still required by current loader; unused for graph blocks
graph: my_graph.json
```

`graph:` accepts **only** a JSON filename/path (inline YAML mappings are rejected).

The referenced `my_graph.json` must contain the graph spec:

```json
{
  "coord_mode": "local",
  "time_mode": "local",
  "nodes": [
    {"coord": [0, 0, 0], "basis": "X", "role": "DATA"}
  ],
  "edges": [
    [[0, 0, 0], [2, 0, 0]]
  ],
  "xflow": [
    {"from": [0, 0, 0], "to": [[2, 0, 0]]}
  ],
  "schedule": {
    "prep": [{"time": 0, "nodes": [[0, 0, 0], [2, 0, 0]]}],
    "entangle": [{"time": 1, "edges": [[[0, 0, 0], [2, 0, 0]]]}],
    "meas": [{"time": 2, "nodes": [[0, 0, 0], [2, 0, 0]]}]
  },
  "detector_candidates": {
    "syndrome_meas": [
      {"id": [1, 1], "rounds": [{"z": 0, "nodes": [[0, 0, 0], [2, 0, 0]]}]}
    ],
    "remaining_parity": [],
    "non_deterministic": []
  }
}
```

## Coordinate/time conventions

### `coord_mode: local`

Coordinates in `graph.nodes/edges/xflow/schedule/detector_candidates` are translated when the block is placed on the canvas.

- **Cube placement** at `position: [px, py, pz]` with code distance `d`:
  - `dx = 2*(d+1)*px`
  - `dy = 2*(d+1)*py`
  - `dz = 2*d*pz`
  - global coord = `[x+dx, y+dy, z+dz]`
- **Pipe placement** at `start/end` uses the same offset convention as `RotatedSurfaceCodeLayoutBuilder`:
  - base: `dx = 2*(d+1)*start.x`, `dy = 2*(d+1)*start.y`
  - then shift depending on direction (`start -> end`):
    - RIGHT: `dx += 2*d`
    - LEFT: `dx -= 2`
    - TOP: `dy -= 2`
    - BOTTOM: `dy += 2*d`
  - `dz = 2*d*start.z`

### `time_mode: local`

Schedule times are shifted based on the block z position, matching the existing convention:

- `time_offset = z_slot * (2*d*(_PHYSICAL_CLOCK + ANCILLA_LENGTH))`
  - cubes: `z_slot = cube.position.z`
  - pipes: `z_slot = pipe.start.z`

If you want absolute times, set `time_mode: global`.

## Current limitations

- `cube.logical_observables` / `pipe.logical_observables` token form (`TB/LR/X/Z`) is not supported for `graph:` blocks; use an explicit `nodes:` list.
- Boundary-based non-determinism removal (`remove_non_deterministic_det`) is skipped for `graph:` blocks (they don't register `bgraph` entries); use `detector_candidates.non_deterministic` instead.

## Canvas YAML: attach logical observables to cubes/pipes

For graph-based blocks, `cube[].logical_observables` / `pipe[].logical_observables` must be specified as an explicit node list (not a `TB/LR/X/Z` token).

```yml
cube:
  - position: [1, 0, 2]
    block: graph_block
    logical_observables:
      nodes:
        - [0, 0, 0]   # interpreted in the same coord_mode as the block's graph

pipe:
  - start: [0, 0, 2]
    end: [1, 0, 2]
    block: graph_pipe_block
    logical_observables:
      nodes:
        - [0, 0, 0]

logical_observables:
  - cube: [[1, 0, 2]]  # include the cube's node set into OBSERVABLE(0)
```
