"""Utility script to inspect the compiled CNOT mockup canvas."""
from __future__ import annotations

import argparse
import pathlib
import sys
from pprint import pformat

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_coord2node_integrity import _build_cnot_canvas


def _format_graph_summary(compiled_canvas) -> str:
    graph = compiled_canvas.global_graph
    if graph is None:
        return "Global graph is None"

    summary: dict[str, object] = {
        "physical_nodes": graph.physical_nodes,
        "physical_edges": graph.physical_edges,
    }
    if hasattr(graph, "detectors"):
        summary["detectors"] = graph.detectors
    if hasattr(graph, "boundary_nodes"):
        summary["boundary_nodes"] = graph.boundary_nodes
    return pformat(summary)


def _format_physical_nodes(compiled_canvas) -> str:
    graph = compiled_canvas.global_graph
    if graph is None:
        return "Global graph is None"

    return pformat(graph.physical_nodes)


def _format_coord2node(compiled_canvas) -> str:
    formatted_items = sorted(
        ((tuple(coord), node) for coord, node in compiled_canvas.coord2node.items()),
        key=lambda item: item[1],
    )
    return pformat(formatted_items)


def _write_debug_files(compiled_canvas, output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    physical_nodes_path = output_dir / "cnot_physical_nodes.txt"
    physical_nodes_path.write_text(_format_physical_nodes(compiled_canvas) + "\n", encoding="utf-8")

    coord2node_path = output_dir / "cnot_coord2node.txt"
    coord2node_path.write_text(_format_coord2node(compiled_canvas) + "\n", encoding="utf-8")

    return physical_nodes_path, coord2node_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-files",
        action="store_true",
        help="Write physical node and coord2node debug files alongside console output.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / ".local" / "test",
        help="Directory where debug txt files should be written (defaults to .local/test).",
    )
    args = parser.parse_args(argv)

    skeleton = _build_cnot_canvas()
    compiled_canvas = skeleton.to_canvas().compile()

    print("Global graph summary:")
    print(_format_graph_summary(compiled_canvas))

    print("\ncoord2node mapping (sorted by node id):")
    print(_format_coord2node(compiled_canvas))

    if args.write_files:
        physical_path, coord2node_path = _write_debug_files(compiled_canvas, args.output_dir)
        print(
            "\nWritten debug files:\n - {}\n - {}".format(
                physical_path.relative_to(REPO_ROOT), coord2node_path.relative_to(REPO_ROOT)
            )
        )


if __name__ == "__main__":
    main()
