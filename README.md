# ls-pattern-compile

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

**ls-pattern-compile** is an experimental MBQC (Measurement-Based Quantum Computing) compiler that converts lattice surgery commands to quantum patterns using the RHG (Raussendorf-Harrington-Goyal) lattice blocks-and-pipes architecture. This library implements fault-tolerant quantum computing compilation through spatial and temporal composition of modular quantum building blocks.

NOTE: This project is an independent, from-scratch implementation inspired by publicly known ideas in [TQEC](https://github.com/tqec/tqec). It is not affiliated with or endorsed by the TQEC team.

## Overview

This project provides a modular framework for constructing fault-tolerant MBQC circuits by composing quantum blocks (cubes) and connectors (pipes) in 3D space and time. The key features include:

- **Modular Blocks-and-Pipes Architecture**: RHG cubes represent stationary quantum patches, while pipes connect them spatially and temporally
- **Fault-Tolerant Operations**: Support for merge and split operations, initialization (|0⟩, |+⟩), memory, and measurement (X/Z basis)
- **Temporal Layer Composition**: Automatic management of temporal layers with proper coordinate systems and graph state composition
- **Error Simulation**: Integration with Stim for logical error rate evaluation using pymatching decoder
- **Visualization Tools**: Interactive 3D visualization using Plotly for graph states, blocks, and compiled canvases

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Install graphix-zx dependency

Since `graphix-zx` is not yet available on PyPI, you need to install it from source:

```bash
git clone https://github.com/TeamGraphix/graphix-zx.git
cd graphix-zx
uv pip install -e .
cd ..
```

### Install ls-pattern-compile

```bash
git clone https://github.com/UTokyo-FT-MBQC/ls-pattern-compile.git
cd ls-pattern-compile
uv pip install -e .
```

For development dependencies (pytest, ruff, mypy, etc.):

```bash
uv pip install -e .[dev]
```

## Quick Start

Here's a minimal example demonstrating the basic workflow:

```python
from lspattern.blocks.cubes.initialize import InitZeroCubeThinLayerSkeleton
from lspattern.blocks.cubes.measure import MeasureZSkeleton
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.compile import compile_canvas
from lspattern.mytype import PatchCoordGlobal3D

# Set code distance (controls fault tolerance and temporal height = 2*d)
d = 3

# Create canvas skeleton
canvas_skeleton = RHGCanvasSkeleton("Simple Example")

# Define edge specifications (X/Z/O for open boundaries)
edgespec = {"LEFT": "Z", "RIGHT": "Z", "TOP": "X", "BOTTOM": "X"}

# Add blocks: initialization and measurement
canvas_skeleton.add_cube(
    PatchCoordGlobal3D((0, 0, 0)),
    InitZeroCubeThinLayerSkeleton(d=d, edgespec=edgespec)
)
canvas_skeleton.add_cube(
    PatchCoordGlobal3D((0, 0, 1)),
    MeasureZSkeleton(d=d, edgespec=edgespec)
)

# Materialize skeleton into canvas and compile
canvas = canvas_skeleton.to_canvas()
compiled_canvas = canvas.compile()

# Extract graph state and metadata
graph = compiled_canvas.global_graph
xflow = {int(src): {int(dst) for dst in dsts}
         for src, dsts in compiled_canvas.flow.flow.items()}
x_parity = [
    {int(node) for node in group}
    for group_dict in compiled_canvas.parity.checks.values()
    for group in group_dict.values()
]

# Compile to MBQC pattern
pattern = compile_canvas(graph, xflow=xflow, x_parity=x_parity, z_parity=[])
```

For more complete examples including merge/split operations and error simulation, see the `examples/` directory:

- `examples/merge_split_mockup.py` - Fault-tolerant merge and split with error simulation
- `examples/merge_split_xx_mockup.py` - Fault-tolerant merge and split with error simulation
- `examples/memory_error_sim.py` - Logical error rate evaluation for memory
- `examples/plus_initialization.py` / `zero_initialization.py` - Initialization examples
- `examples/compiled_canvas_visualization.py` - 3D visualization of compiled canvases

## Usage Guide

### 1. Define Block Topology with Skeletons

Use skeletons to define block positions before materializing full graph states:

```python
from lspattern.canvas import RHGCanvasSkeleton
from lspattern.blocks.cubes.memory import MemoryCubeSkeleton
from lspattern.blocks.pipes.initialize import InitPlusPipeSkeleton
from lspattern.mytype import PatchCoordGlobal3D

canvas_skeleton = RHGCanvasSkeleton("My Circuit")

# Add cubes at specific 3D patch coordinates
canvas_skeleton.add_cube(
    PatchCoordGlobal3D((0, 0, 0)),
    MemoryCubeSkeleton(d=3, edgespec={"LEFT": "Z", "RIGHT": "O", ...})
)

# Add pipes connecting cubes
canvas_skeleton.add_pipe(
    PatchCoordGlobal3D((0, 0, 1)),  # from coordinate
    PatchCoordGlobal3D((1, 0, 1)),  # to coordinate
    InitPlusPipeSkeleton(d=3, edgespec={"LEFT": "O", "RIGHT": "O", ...})
)
```

### 2. Materialize and Compile

```python
# Materialize skeletons into full RHGBlocks with graph states
canvas = canvas_skeleton.to_canvas()

# Compile: compose all temporal layers into final graph state
compiled_canvas = canvas.compile()
```

### 3. Extract Compilation Metadata

```python
# Graph state
graph = compiled_canvas.global_graph

# X-correction flow (feedforward dependencies)
xflow = {int(src): {int(dst) for dst in dsts}
         for src, dsts in compiled_canvas.flow.flow.items()}

# Stabilizer parity checks (detectors)
x_parity = [
    {int(node) for node in group}
    for group_dict in compiled_canvas.parity.checks.values()
    for group in group_dict.values()
]

# Measurement schedule
schedule = compiled_canvas.schedule.compact()
```

### 4. Generate MBQC Pattern and Simulate

```python
from lspattern.compile import compile_canvas
from graphix_zx.scheduler import Scheduler
from graphix_zx.stim_compiler import stim_compile
import stim
import pymatching

# Create pattern
pattern = compile_canvas(graph, xflow=xflow, x_parity=x_parity, z_parity=[])

# Compile to Stim circuit
stim_str = stim_compile(
    pattern,
    logical_observables={0: output_node_set},
    before_measure_flip_probability=0.001  # noise model
)
circuit = stim.Circuit(stim_str)

# Generate detector error model and decode
dem = circuit.detector_error_model()
matching = pymatching.Matching.from_detector_error_model(dem)
```

### 5. Visualize

```python
from lspattern.visualizers import visualize_compiled_canvas_plotly

fig = visualize_compiled_canvas_plotly(compiled_canvas, show_edges=True)
fig.show()
```

## Project Status

### Implemented Features ✅

- **Initialization Blocks**
  - Zero state initialization (`InitZeroCubeThinLayerSkeleton`)
  - Plus state initialization (`InitPlusCubeThinLayerSkeleton`, `InitPlusPipeSkeleton`)
- **Memory Operations**
  - RHG memory cubes and pipes (`MemoryCubeSkeleton`, `MemoryPipeSkeleton`)
- **Measurement**
  - X-basis measurement (`MeasureXSkeleton`, `MeasureXPipeSkeleton`)
  - Z-basis measurement (`MeasureZSkeleton`)
- **Lattice Surgery Operations**
  - Fault-tolerant merge and split (ZZ measurement) ([#18](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/pull/18))
- **Compilation Pipeline**
  - Skeleton-based block definition
  - Automatic temporal layer management
  - Graph state composition with coordinate mapping
  - Accumulator system (Schedule, Flow, Parity)
- **Error Simulation**
  - Logical error rate evaluation framework
  - Stim integration for circuit-level simulation
  - PyMatching decoder for error correction

### In Progress 🚧

- **Complete logical error rate evaluation for merge/split** ([#19](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/19))
  - Enable ancilla qubit specification for ZZ measurement
  - Evaluate Z₁ ⊕ Z₂ ⊕ (Z₁Z₂) and X₁ ⊕ X₂ ⊕ (X₁X₂) observables
- **Layer-by-layer construction refactoring** ([#23](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/23))
  - Redesign blocks from fundamental layers for code reuse
- **Module refactoring**
  - Canvas module improvements ([#33](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/33))
  - Visualizers module refactoring ([#32](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/32))
- **PyPI publication preparation** ([#30](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/30))
- **Sphinx documentation** ([#31](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/31))

### Planned Features 📋

- **Lattice Surgery Instructions** ([#3](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/3))
  - Hadamard gate
  - S gate
  - Magic state preparation
  - Boundary rotation
- **Additional Measurements**
  - Logical XX measurement ([#20](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/20))
- **Patch Operations**
  - Patch deformation (expansion, shrinkage, rotation) ([#22](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/22))
- **Quality Improvements**
  - Type checking for examples and tests ([#29](https://github.com/UTokyo-FT-MBQC/ls-pattern-compile/issues/29))

## Development

### Running Tests

```bash
pytest                    # Run all tests
pytest tests/test_*.py    # Run specific test file
pytest -v                 # Verbose output
```

### Code Quality Checks

```bash
ruff check               # Linting
ruff format              # Code formatting
mypy .                   # Type checking with mypy
pyright                  # Type checking with pyright
```

## Citation

If you use ls-pattern-compile in your research, please cite:

### BibTeX
```bibtex
@software{lspattern2025,
  title = {ls-pattern-compile: MBQC Lattice Surgery Compiler},
  author = {{UTokyo FT-MBQC Team}},
  year = {2025},
  url = {https://github.com/UTokyo-FT-MBQC/ls-pattern-compile},
  version = {0.1.0}
}
```

### APA
```
UTokyo FT-MBQC Team. (2025). ls-pattern-compile: MBQC Lattice Surgery Compiler (Version 0.1.0) [Computer software]. https://github.com/UTokyo-FT-MBQC/ls-pattern-compile
```

## Related Projects

- [graphix-zx](https://github.com/TeamGraphix/graphix-zx) - ZX-calculus based MBQC framework

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project's design is inspired by [tqec](https://github.com/tqec/tqec), though no code from tqec is used directly.
