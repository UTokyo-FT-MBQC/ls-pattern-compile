# Distillation Factory Template

The liblsqecc importer can optionally expand distillation factory regions
into cube/pipe entries using a user-supplied template function.

## Background

`lsqecc_slicer` outputs `DistillationQubit` cells that occupy rectangular
regions of the grid. By default the importer ignores these cells, but
when a **distillation template** is provided, the importer:

1. **Detects** factory regions via connected-component analysis on the
   first time slice.
2. **Calls** the template function for each detected factory.
3. **Merges** the returned cubes and pipes into the main canvas output.

## Quick Start

```python
from lspattern.importer import (
    convert_slices_file_to_canvas_yaml,
    default_distillation_template,
)

# Without distillation (default — same as before)
yaml_text = convert_slices_file_to_canvas_yaml(
    "slices.json",
    "canvas.yml",
    name="my_circuit",
)

# With the built-in placeholder template
yaml_text = convert_slices_file_to_canvas_yaml(
    "slices.json",
    "canvas_with_distill.yml",
    name="my_circuit",
    distillation_template=default_distillation_template,
)
```

The `convert_slices_to_canvas_yaml()` function accepts the same
`distillation_template` keyword argument.

## DistillationFactory dataclass

Each detected factory is represented as a frozen dataclass:

```python
@dataclass(frozen=True)
class DistillationFactory:
    origin: Coord2D                  # (x, y) top-left of bounding box
    width: int                       # bounding box width  (3 or 5)
    height: int                      # bounding box height (5 or 3)
    z_period: int                    # distillation round length (slices)
    outer_ring: frozenset[Coord2D]   # relative (dx, dy) of active cells
    inner_cells: frozenset[Coord2D]  # relative (dx, dy) of inactive cells
```

- **origin** — absolute grid coordinate of the bounding box top-left corner.
- **width / height** — dimensions of the bounding box. A 3×5 factory has
  `width=3, height=5`; a 5×3 factory has `width=5, height=3`.
- **z_period** — number of time slices per distillation round, detected from
  `"Time to next magic state:N"` countdown text in the JSON.
- **outer_ring** — coordinates (relative to origin) of active boundary cells.
- **inner_cells** — coordinates (relative to origin) of inactive interior cells.

## DistillationTemplateFn

The template function type is:

```python
DistillationTemplateFn = Callable[
    [DistillationFactory, int],       # (factory, total_slices)
    tuple[list[dict], list[dict]],    # (cubes, pipes)
]
```

It receives a single factory and the total number of time slices, and
returns `(cubes, pipes)` in the same dict format used by canvas YAML:

- **cube dict**: `{"position": [x, y, z], "block": str, "boundary": str}`
- **pipe dict**: `{"start": [x,y,z], "end": [x,y,z], "block": str, "boundary": str}`

## Writing a Custom Template

```python
from lspattern.importer import (
    DistillationFactory,
    convert_slices_file_to_canvas_yaml,
)


def my_template(
    factory: DistillationFactory,
    total_slices: int,
) -> tuple[list[dict], list[dict]]:
    cubes = []
    pipes = []
    ox, oy = factory.origin

    for round_start in range(0, total_slices, factory.z_period):
        for dz in range(factory.z_period):
            z = round_start + dz
            if z >= total_slices:
                break

            # --- Customize block assignment per dz here ---
            if dz == 0:
                block = "InitZeroBlock"
            elif dz == factory.z_period - 1:
                block = "MeasureZBlock"
            else:
                block = "MemoryBlock"

            for dx, dy in sorted(factory.outer_ring):
                cubes.append({
                    "position": [ox + dx, oy + dy, z],
                    "block": block,
                    "boundary": "XXZZ",
                })

            # --- Add pipe entries between adjacent cells here ---

    return cubes, pipes


yaml_text = convert_slices_file_to_canvas_yaml(
    "slices.json",
    "canvas.yml",
    name="my_circuit",
    distillation_template=my_template,
)
```

### Tips

- `factory.outer_ring` and `factory.inner_cells` use **relative**
  coordinates. Add `factory.origin` to get absolute grid positions.
- For a 5×3 (landscape) factory, `factory.width > factory.height`.
  The built-in `default_distillation_template` transposes `(dx, dy)`
  for such factories; a custom template can handle orientation freely.
- Pipes connect adjacent cubes. The `"boundary"` string follows
  Top-Bottom-Left-Right order with `O` for open (connected) sides.
- The template is called once per factory. All returned cubes/pipes are
  merged into the final canvas and re-sorted by coordinate.

## Default Template Behavior

`default_distillation_template` is a placeholder that fills each factory
as follows:

| dz position          | Block assigned  |
|----------------------|-----------------|
| `dz == 0`            | `InitZeroBlock` |
| `0 < dz < period-1`  | `MemoryBlock`   |
| `dz == period-1`     | `MeasureZBlock` |

- Only outer-ring cells are populated; inner cells are left empty.
- No pipes are generated (empty list).
- The pattern repeats every `z_period` slices until `total_slices`.

This is intended as a starting point — replace it with a template that
matches your actual distillation protocol (e.g., 15-to-1 or other
magic state factory designs).

## Public API Summary

All symbols are exported from `lspattern.importer`:

| Symbol                          | Kind       |
|---------------------------------|------------|
| `DistillationFactory`           | dataclass  |
| `DistillationTemplateFn`        | type alias |
| `default_distillation_template` | function   |
