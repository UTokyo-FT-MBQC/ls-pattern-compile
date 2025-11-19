"""Manual test script for the YAML loader."""

from pathlib import Path
import sys
import importlib.util

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

# Import loader module directly without going through __init__.py
loader_path = Path(__file__).parent / "lspattern" / "new_blocks" / "loader.py"
spec = importlib.util.spec_from_file_location("loader", loader_path)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)

# Test loading different YAML files
test_files = [
    "lspattern/new_blocks/patch_layout/layers/memory.yml",
    "lspattern/new_blocks/patch_layout/layers/init_plus.yml",
    "lspattern/new_blocks/patch_layout/layers/measure.yml",
    "lspattern/new_blocks/patch_layout/layers/init_zero.yml",
    "lspattern/new_blocks/patch_layout/layers/empty.yml",
    "lspattern/new_blocks/patch_layout/layers/zz_measure.yml",
]

print("Testing YAML Loader\n" + "=" * 60)

for file_path in test_files:
    print(f"\nTesting {file_path}...")
    try:
        config = loader.load_patch_layout_from_yaml(Path(file_path))
        print(f"  ✓ Name: {config.name}")
        print(f"  ✓ Description: {config.description}")
        print(f"  ✓ Layer 1: basis={config.layer1.basis}, ancilla={config.layer1.ancilla}")
        print(f"  ✓ Layer 2: basis={config.layer2.basis}, ancilla={config.layer2.ancilla}")
        print("  SUCCESS!")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
