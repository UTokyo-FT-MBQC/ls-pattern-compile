"""Standalone lassynth demo script for testing specifications.

Usage:
    python lassynth_demo.py --spec distillation_spec.json --print-detail --text-diagram
    python lassynth_demo.py --spec cnot_spec.json --allow-y --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lassynth import LatticeSurgerySynthesizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LaSSynth synthesis standalone")
    parser.add_argument("--spec", type=Path, required=True, help="Input specification JSON")
    parser.add_argument("--output-dir", type=Path, default=Path(), help="Output directory")
    parser.add_argument("--allow-y", action="store_true", help="Allow Y cubes")
    parser.add_argument("--solver", choices=["z3", "kissat"], default="z3")
    parser.add_argument("--optimize-depth", action="store_true", help="Run depth optimization")
    parser.add_argument("--print-detail", action="store_true", help="Print detailed solving info")
    parser.add_argument("--verify", action="store_true", help="Verify stabilizers")
    parser.add_argument("--text-diagram", action="store_true", help="Print text diagram")
    parser.add_argument("--gltf", action="store_true", help="Export GLTF 3D model")
    parser.add_argument("--zigxag", action="store_true", help="Generate ZigXag URL")
    parser.add_argument("--save-lasre", action="store_true", help="Save LaSRe JSON")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    name = args.spec.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Specification: {args.spec}")
    print(f"  max_i={spec['max_i']}, max_j={spec['max_j']}, max_k={spec['max_k']}")
    print(f"  ports: {len(spec.get('ports', []))}")
    print(f"  stabilizers: {len(spec.get('stabilizers', []))}")

    synth = LatticeSurgerySynthesizer(solver=args.solver)

    given_arrs = None
    if not args.allow_y:
        zeros_nodey = [[[0 for _ in range(spec["max_k"])] for _ in range(spec["max_j"])] for _ in range(spec["max_i"])]
        given_arrs = {"NodeY": zeros_nodey}
        print("Y cubes: forbidden")
    else:
        print("Y cubes: allowed")

    print(f"\nSolving with {args.solver}...")

    if args.optimize_depth:
        result = synth.optimize_depth(
            specification=spec,
            given_arrs=given_arrs,
            start_depth=spec["max_k"],
            print_detail=args.print_detail,
        )
    else:
        result = synth.solve(
            specification=spec,
            given_arrs=given_arrs,
            print_detail=args.print_detail,
        )

    if result is None:
        print("\n*** UNSAT - No solution found ***")
        print("\nPossible causes:")
        print("  1. Specification is over-constrained")
        print("  2. Spacetime volume (max_i * max_j * max_k) too small")
        print("  3. Try --allow-y if Y cubes might be needed")
        print("  4. Check stabilizer format matches port count")
        return

    print("\n*** SAT - Solution found! ***")
    print(f"Depth: {result.get_depth()}")

    result = result.after_default_optimizations()

    if args.save_lasre:
        lasre_path = args.output_dir / f"{name}.lasre.json"
        result.save_lasre(str(lasre_path))
        print(f"Saved LaSRe: {lasre_path}")

    if args.gltf:
        gltf_path = args.output_dir / f"{name}.gltf"
        result.to_3d_model_gltf(str(gltf_path), attach_axes=True)
        print(f"Saved GLTF: {gltf_path}")

    if args.text_diagram:
        print("\n--- Text Diagram ---")
        print(result.to_text_diagram())

    if args.zigxag:
        url = result.to_zigxag_url(io_spec=None)
        print(f"\nZigXag URL: {url}")

    if args.verify:
        print("\n--- Stabilizer Verification ---")
        result.verify_stabilizers_stimzx(specification=spec, print_stabilizers=True)


if __name__ == "__main__":
    main()
