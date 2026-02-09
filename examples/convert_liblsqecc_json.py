"""Convert liblsqecc slices JSON to canvas YAML.

This script converts liblsqecc output (slices JSON format) to the
canvas YAML format used by ls-pattern-compile.
"""

from __future__ import annotations

from pathlib import Path

from lspattern.importer.liblsqecc import (
    convert_slices_file_to_canvas_yaml,
    default_distillation_template,
)

# =============================================================================
# Configuration - Edit these values
# =============================================================================
input_json = Path(
    "path2json"
)  # Path to liblsqecc slices JSON
output_dir = Path(__file__).parent / "output"

# =============================================================================

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

yaml_path = output_dir / f"{input_json.stem}.yml"

print(f"Input JSON: {input_json}")
print(f"Output YAML: {yaml_path}")
print()

print("Converting liblsqecc slices JSON to canvas YAML...")
yaml_text = convert_slices_file_to_canvas_yaml(
    input_json,
    yaml_path,
    name=input_json.stem,
    description=f"Imported from {input_json.name}",
    distillation_template=default_distillation_template,
)

print(f"  Generated YAML: {yaml_path}")
print(f"  YAML size: {len(yaml_text):,} characters")
print()
print("Conversion completed successfully!")
