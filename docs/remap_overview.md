# Node Remapping Overview

This note documents how node remapping is handled inside `ls-pattern-compile`,
which modules participate, and the invariants each component expects. The goal
is to make it easier to reason about the current design and to highlight places
where the logic may require additional safeguards or tuning.

## Compilation Stages and Node Maps

Node identifiers are rewritten several times while a lattice-surgery pattern is
compiled. Each stage produces a mapping from an old namespace to a new one:

- **Block to temporal layer** – `GraphComposer` materialises cube/pipe blocks
  into a `TemporalLayer` and records `node_map_global` on every block. All block
  accumulators (`schedule`, `flow`, `parity`) are remapped into the layer-wide
  namespace.
- **Temporal layer to compiled canvas** – `CompiledRHGCanvas.add_temporal_layer`
  composes successive layers. When `compose` renumbers existing nodes it emits
  `node_map1`, which is pushed through every cached structure before the layer
  is appended. The freshly added layer is remapped by `node_map2`.
- **Global graph export** – When the assembled canvas needs to be cloned or
  serialised, `create_remapped_graphstate` migrates `GraphState` nodes, edges,
  and measurement bases into a brand-new object.

Understanding which direction a particular `node_map` travels (local→global vs
old_global→new_global) is critical; mismatches tend to surface as stale indices
inside coordinate maps or port caches.

## Accumulator-Level Remapping (`lspattern/accumulator.py`)

- `_remap_flow(flow, node_map)`
  Remaps every flow edge using direct lookups. It assumes `node_map` contains
  all participating node ids; missing entries raise immediately. This is used
  by `FlowAccumulator.remap_nodes`.
- `_remap_groups(groups, node_map)`
  Applies `node_map.get` to every parity group member, keeping unknown ids for
  robustness. The helper underpins parity/dangling remaps.
- `ScheduleAccumulator.remap_nodes(node_map)`
  Produces a new schedule by mapping each node id per time slot. Unknown nodes
  stay untouched so upstream callers can supply sparse maps.
- `ParityAccumulator.remap_nodes(node_map)`
  Remaps parity checks and dangling sets while copying `ignore_dangling`
  verbatim. Empty groups are filtered out.
- `FlowAccumulator.remap_nodes(node_map)`
  Returns a new accumulator after delegating to `_remap_flow`.

**Notable expectations**
- All flow edges must be covered by the provided map. Unlike the parity and
  schedule cases there is no fallback.
- `node_map` arguments are raw `dict[int, int]` even when the public API is
  typed with `NodeIdLocal`; callers perform the conversions.

## Coordinate Mapping (`lspattern/canvas/coordinates.py`)

- `CoordinateMapper.remap_nodes(node_map)`
  Rewrites `node2coord`, `coord2node`, and `node2role` in place by rebuilding
  each dictionary with the supplied mapping. Keys not listed in `node_map` are
  preserved. No collision detection is performed; if two old nodes are mapped
  to the same new id the later entry wins silently.
- `CoordinateMapper.merge(self, other, self_node_map, other_node_map)`
  Uses remappings while merging two mappers during temporal composition, and
  raises on coordinate collisions.

`GraphComposer` and `TemporalLayer._remap_node_mappings` rely on these
operations whenever `compose` renumbers an existing node.

## Port Management (`lspattern/canvas/ports.py`)

- `PortManager.remap_ports(node_map)`
  Applies the mapping to cube and pipe in/out ports as well as grouped cout
  structures. After remapping it rebuilds the flat caches via
  `rebuild_cout_group_cache`, so reverse lookups remain consistent.
- `PortManager.merge(self, other, self_node_map, other_node_map, ...)`
  Builds superficial copies of both managers, remaps them independently, and
  combines the results according to the temporal composition strategy.

The remapping code expects `node_map` to use bare integers; callers convert
`NodeIdLocal` before passing the mapping in.

## Temporal Layer Composition (`lspattern/canvas/composition.py` and `layer.py`)

- `GraphComposer.compose_*`
  Each call to `compose` returns `(node_map1, node_map2)`. `node_map1` renumbers
  nodes that already lived in the accumulated graph; `node_map2` maps the new
  block’s local ids into the expanded graph.
- After every composition step, `GraphComposer` performs:
  - `coord_mapper.remap_nodes(node_map1)` to keep spatial caches aligned.
  - `port_manager.remap_ports(node_map1)` so port lists continue to point at
    the right global ids.
  - `block.node_map_global = node_map2` to seed later accumulator remaps.
  - Back-propagation of `node_map1` through every previously stored
    `node_map_global` so earlier blocks stay valid.
- `TemporalLayer._remap_node_mappings(node_map)`
  Convenience hook that remaps coordinates and accumulators when the entire
  layer must be shifted.
- `TemporalLayer.compile()`
  Applies each block’s `node_map_global` to its accumulators, merging them into
  the layer-wide `schedule`, `flow`, and `parity`.

## Compiled Canvas (`lspattern/canvas/compiled.py`)

- `_remap_layer(layer, node_map)`
  Clones a `TemporalLayer`, remapping its coordinate mapper, port manager, and
  accumulators. Used when pre-existing layers need to follow a new global
  numbering.
- `_remap_layer_portsets(layer, remapped_layer, node_map)`
  Helper that remaps port-based caches during the clone.
- `CompiledRHGCanvas.remap_nodes(node_map)`
  Deep-remaps every stored layer, the global port manager, and the aggregated
  schedule/flow/parity before also rewriting `coord2node` and `node2role`.
- `_remap_layer_mappings(next_layer, node_map2)`
  Adjusts the layer just added by temporal composition.
- `add_temporal_layer(cgraph, next_layer)`
  Calls `compose` on the global graph. If existing nodes are renumbered
  (`node_map1` non-identity), the current canvas is remapped in place before
  merging in the new layer (`node_map2`).

## GraphState Utilities (`lspattern/canvas/graph_utils.py`)

- `remap_graph_nodes(gsrc, nmap)`
  Builds a new `GraphState` by creating fresh physical nodes according to
  `nmap`. Throws if two old nodes map to the same new id.
- `remap_measurement_bases(gsrc, gdst, nmap, created)`
  Copies measurement bases for every `old -> new` pair present in `nmap`.
  Nodes missing from `nmap` keep the default basis in the destination state.
- `remap_graph_edges(gsrc, gdst, nmap, created)`
  Adds remapped edges to the destination graph using the `created` lookup,
  falling back to the raw ids when a mapping is absent.
- `create_remapped_graphstate(gsrc, nmap)`
  Convenience wrapper that applies the three helpers sequentially.

## Robustness and Performance Opportunities

The following ideas could help harden the remap pipeline and reduce friction
during debugging:

1. **Guard against identity maps** – `CoordinateMapper.remap_nodes` and
   `PortManager.remap_ports` are called even when `node_map1` is identity. An
   early return on empty/identity mappings would keep large layers from doing
   needless work.
2. **Flow coverage checks** – before running `_remap_flow`, verify that the
   mapping contains every referenced source and destination. A lightweight
   assertion or explicit error message would localise issues faster than the
   implicit `KeyError`.
3. **Collision detection in coordinate remaps** – `CoordinateMapper.remap_nodes`
   could compare the number of entries before/after or keep a temporary set to
   catch accidental many-to-one remappings.
4. **Shared remap helpers** – centralising repeated patterns (e.g., building
   `{NodeIdGlobal(k): ...}` maps) would reduce the chance of inconsistent
   fallback behaviour across modules.
5. **GraphState bases fallback** – `remap_measurement_bases` currently skips
   nodes absent from `nmap`; consider defaulting to the source node id to ensure
   measurement bases survive even when the map is sparse.
6. **Instrumentation and tests** – adding targeted unit tests that compose two
   layers with overlapping nodes (forcing non-trivial `node_map1`) would help
   validate that ports, coordinates, and accumulators stay in sync. Optional
  logging when `node_map1` renumbers nodes can also aid field debugging.

Together these guardrails should make the remap surface easier to reason about
and less prone to hidden mismatches between cached structures.
