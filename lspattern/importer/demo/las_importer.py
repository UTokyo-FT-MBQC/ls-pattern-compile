"""End-to-end demo: LaSSynth lasre -> lspattern YAMLs.

Usage:
    python las_importer.py --spec cnot_spec.json [--name-prefix cnot] [--allow-y]

Notes:
    * By default we forbid Y cubes (NodeY=0) to match current importer limits.
    * Requires lassynth package installed in the active environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lassynth import LatticeSurgerySynthesizer

from lspattern.importer.las import convert_lasre_to_yamls


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LaSSynth and emit lspattern YAMLs")
    parser.add_argument("--spec", type=Path, default=Path(__file__).parent / "cnot_spec.json")
    parser.add_argument("--name-prefix", type=str, default=None, help="Prefix for generated YAML names")
    parser.add_argument("--description", type=str, default=None, help="Description in YAML")
    parser.add_argument("--allow-y", action="store_true", help="Allow Y cubes (otherwise force NodeY=0)")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    name_prefix = args.name_prefix or args.spec.stem

    synth = LatticeSurgerySynthesizer()

    given_arrs = None
    if not args.allow_y:
        zeros_nodey = [[[0 for _ in range(spec["max_k"])] for _ in range(spec["max_j"])] for _ in range(spec["max_i"])]
        given_arrs = {"NodeY": zeros_nodey}

    result = synth.solve(specification=spec, given_arrs=given_arrs)
    if result is None:
        print("Spec is UNSAT")
        raise SystemExit(1)

    result = result.after_default_optimizations()
    yamls = convert_lasre_to_yamls(
        result.lasre,
        spec,
        name_prefix=name_prefix,
        description=args.description or f"Imported from {args.spec.name}",
    )

    for name, text in yamls:
        out_path = args.spec.parent / f"{name}.yml"
        out_path.write_text(text)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
