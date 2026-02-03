# Patch Layout YAML Configuration Reference

This directory contains YAML configuration files for defining MBQC patch layouts.
The configuration system is hierarchical: **Canvas** → **Block** → **Unit Layer**.

## Directory Structure

```
patch_layout/
├── blocks/          # Block definitions (composed of unit layers)
├── layers/          # Unit layer definitions (atomic building blocks)
└── README.md        # This file
```

## Unit Layer Configuration (`layers/*.yml`)

Unit layers are the atomic building blocks that define a single temporal slice
of the surface code patch.

### Schema

```yaml
name: MemoryUnit                    # Required: Unique identifier
description: Memory unit layer      # Required: Human-readable description

layer1:                             # Required: Configuration for sublayer 1 (even z)
  basis: X                          # Optional: Measurement basis (X/Y/Z/null), default: X
  ancilla: true                     # Optional: Whether to place ancilla qubits, default: true
  skip_syndrome: false              # Optional: Skip syndrome registration when ancilla=false, default: false
  init: false                       # Optional: Add flow for ancilla qubits (for init layers), default: false

layer2:                             # Required: Configuration for sublayer 2 (odd z)
  basis: X
  ancilla: true
  skip_syndrome: false
  init: false
```

### Field Details

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `basis` | `X` \| `Y` \| `Z` \| `null` | `X` | Measurement basis for data qubits. Use `null` for no qubits. |
| `ancilla` | `bool` | `true` | Whether ancilla qubits are prepared in this sublayer. |
| `skip_syndrome` | `bool` | `false` | When `ancilla: false`, skip inferring syndrome measurements from neighbors. |
| `init` | `bool` | `false` | Add correction flow for ancilla qubits (used in initialization layers). |

### Examples

**Memory Unit** (standard stabilizer measurement):
```yaml
name: MemoryUnit
description: Standard memory unit with X-basis measurement
layer1:
  basis: X
  ancilla: true
layer2:
  basis: X
  ancilla: true
```

**Init Zero Unit** (|0⟩ state preparation):
```yaml
name: InitZeroUnit
description: Initialize logical |0⟩ state
layer1:
  basis: null      # No qubits in first sublayer
  ancilla: false
layer2:
  basis: X
  ancilla: true
  init: true       # Enable flow for initialization
```

---

## Block Configuration (`blocks/*.yml`)

Blocks are temporal sequences of unit layers that define complete operations
(e.g., memory, measurement, initialization).

### Schema

```yaml
name: MemoryBlock                   # Required: Unique identifier
description: Memory block           # Required: Human-readable description

# Option 1: Layer-based definition
layers:                             # List of unit layers in temporal order
  - type: MemoryUnit                # Required: Unit layer name (from layers/)
    num_layers: 3                   # Option A: Fixed number of repetitions
    # OR
    num_layers_from_distance: rest  # Option B: Derive from code distance

# Option 2: Graph-based definition (mutually exclusive with layers)
graph: custom_graph.json            # Path to JSON file with explicit graph structure
```

### Layer Repetition Options

| Option | Type | Description |
|--------|------|-------------|
| `num_layers` | `int` | Fixed number of unit layer repetitions |
| `num_layers_from_distance` | `"rest"` \| `int` \| `{scale, offset}` | Dynamic calculation based on code distance |

**Dynamic calculation examples:**
```yaml
# Fill remaining layers up to code distance
num_layers_from_distance: rest

# Fixed value
num_layers_from_distance: 2

# Formula: scale * d + offset
num_layers_from_distance:
  scale: 1
  offset: -1
```

### Graph-Based Blocks

For custom graph structures, reference a JSON file instead of using layers:

```yaml
name: CustomGraphBlock
description: Block with explicit graph definition
graph: my_graph.json          # Path relative to block YAML or search paths
```

The JSON file must define nodes, edges, flow, schedule, and detector candidates.
See `lspattern/fragment.py` for the `GraphSpec` schema.

---

## Canvas Configuration

Canvas YAML files define the spatial arrangement of cubes and pipes.
These are typically placed in `examples/` or user directories.

### Schema

```yaml
name: MyCanvas                      # Required: Canvas identifier
description: Example canvas         # Optional: Description
layout: rotated_surface_code        # Optional: Layout type, default: rotated_surface_code

cube:                               # List of cube (patch) definitions
  - position: [0, 0, 0]             # Required: 3D position [x, y, z]
    block: MemoryBlock              # Required: Block name (from blocks/)
    boundary: XXZZ                  # Optional: Boundary types, default: XXZZ
    invert_ancilla_order: false     # Optional: Invert ancilla placement order
    logical_observables: TB         # Optional: Logical observable specification

pipe:                               # List of pipe (connection) definitions
  - start: [0, 0, 0]                # Required: Start cube position
    end: [1, 0, 0]                  # Required: End cube position
    block: MemoryBlock              # Required: Block name
    boundary: XXZZ                  # Optional: Boundary types, default: XXZZ
    invert_ancilla_order: false     # Optional: Invert ancilla placement order
    logical_observables: null       # Optional: Logical observable specification

logical_observables:                # Optional: Composite observables spanning multiple cubes/pipes
  - cube: [[0, 0, 0], [1, 0, 0]]
    pipe: []
    label: X_obs
```

### Boundary Specification

Boundaries define the edge types for each side of the patch.
Specified at the **cube/pipe level** in Canvas YAML.

**String format** (order: Top, Bottom, Left, Right):
```yaml
boundary: XXZZ    # Top=X, Bottom=X, Left=Z, Right=Z (default)
boundary: ZZZZ    # All Z boundaries
```

**Mapping format**:
```yaml
boundary:
  top: X
  bottom: X
  left: Z
  right: Z
```

### Ancilla Order Inversion

The `invert_ancilla_order` flag controls ancilla placement.
Specified at the **cube/pipe level** in Canvas YAML.

| Flag | layer1 (even z) | layer2 (odd z) |
|------|-----------------|----------------|
| `false` (default) | Z-ancilla | X-ancilla |
| `true` | X-ancilla | Z-ancilla |

```yaml
cube:
  - position: [0, 0, 0]
    block: MemoryBlock
    invert_ancilla_order: true   # Swap ancilla types between sublayers
```

### Logical Observable Specification

Logical observables can be specified in several formats:

**Token format** (for patch-based observables):
```yaml
logical_observables: TB             # Top-Bottom observable
logical_observables: X              # X-type observable
logical_observables: Z              # Z-type observable
logical_observables: LR             # Left-Right observable
```

**Detailed format** (with layer/sublayer control):
```yaml
logical_observables:
  token: TB
  layer: 0                          # Unit layer index (0-based, supports negative)
  sublayer: 1                       # Physical sublayer (1 or 2)
  label: my_observable              # Optional label for identification
```

**Multiple observables**:
```yaml
logical_observables:
  - token: TB
    label: "0"
  - token: X
    layer: -1
    label: "1"
```

**Coordinate-based** (for graph blocks):
```yaml
logical_observables:
  nodes:
    - [0, 0, 1]
    - [1, 0, 1]
    - [2, 0, 1]
  label: custom_obs
```

---

## File Resolution

The loader searches for YAML files in the following order:

1. Explicit path (if provided as absolute/relative path)
2. User-provided search paths (`extra_paths` parameter)
3. Parent directory of the referencing YAML file
4. Package resources (`lspattern.patch_layout.blocks`, `lspattern.patch_layout.layers`)

File name matching is flexible:
- `MemoryBlock` → `memory_block.yml`, `MemoryBlock.yml`, `memory_block.yaml`, etc.
- `memory_unit` → `memory_unit.yml`, `MemoryUnit.yml`, etc.

---

## Quick Reference

### Minimal Memory Block

```yaml
# blocks/my_memory_block.yml
name: MyMemoryBlock
description: Simple memory block
layers:
  - type: MemoryUnit
    num_layers_from_distance: rest
```

### Minimal Init Block

```yaml
# blocks/my_init_block.yml
name: MyInitBlock
description: Initialize |0⟩ state
layers:
  - type: InitZeroUnit
    num_layers: 1
  - type: MemoryUnit
    num_layers_from_distance: rest
```

### Minimal Canvas

```yaml
# examples/my_canvas.yml
name: SinglePatchMemory
description: Single patch memory experiment
cube:
  - position: [0, 0, 0]
    block: MemoryBlock
    logical_observables: TB
```

### Canvas with Ancilla Inversion

```yaml
# examples/inverted_ancilla.yml
name: InvertedAncillaExample
description: Example with inverted ancilla order
cube:
  - position: [0, 0, 0]
    block: MemoryBlock
    boundary: XXZZ
    invert_ancilla_order: true
    logical_observables: TB
```
