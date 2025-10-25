# Testing Guide

- Unit tests: `pytest -q`.
- Integration tests: `tests/integration/` validate end-to-end compilation and guard against regressions using circuit fingerprints.
- Slow tests: marked with `slow` and include quick error simulations via sinter/pymatching.

## Circuit Fingerprinting

Fingerprints are stored at `tests/integration/fixtures/circuit_fingerprints.json` as a map of test name to metadata and a SHA256 of the circuit text form.

### Running Integration Tests

```bash
pytest tests/integration -v
pytest tests/integration -v -m slow  # include simulations
```

### Updating Fingerprints After Breaking Changes

When you introduce intentional breaking changes to the compilation pipeline (e.g., optimizations, bug fixes, or algorithm changes), the circuit fingerprints will no longer match. To update the golden fingerprints:

#### Option 1: Using UPDATE_FINGERPRINTS Environment Variable (Recommended)

Set the `UPDATE_FINGERPRINTS=1` environment variable to automatically update fingerprints when tests run:

```bash
# On Unix/Linux/macOS:
UPDATE_FINGERPRINTS=1 pytest tests/integration -v

# On Windows (PowerShell):
$env:UPDATE_FINGERPRINTS=1; pytest tests/integration -v

# On Windows (cmd):
set UPDATE_FINGERPRINTS=1 && pytest tests/integration -v
```

This will:
1. Run all integration tests
2. Automatically update `tests/integration/fixtures/circuit_fingerprints.json` with new fingerprints
3. Show which fingerprints were updated in the test output

#### Option 2: Manual Update Script

Generate fingerprints manually by running the circuits and saving results:

```python
from pathlib import Path
from lspattern.testing.fingerprints import CircuitFingerprint, FingerprintRegistry

# Load the registry
registry_path = Path("tests/integration/fixtures/circuit_fingerprints.json")
registry = FingerprintRegistry(registry_path)
registry.load()

# Generate a circuit (example)
from examples.merge_split_error_sim import create_circuit
circuit = create_circuit(d=3, noise=0.0)

# Create and save fingerprint
fp = CircuitFingerprint.from_circuit("merge_split_d3", circuit)
registry.set(fp)
registry.save()
```

#### Important Notes

- **Review changes carefully**: Always review the diff of `circuit_fingerprints.json` before committing to ensure changes are intentional.
- **Document breaking changes**: Update the CHANGELOG or commit message to explain why fingerprints changed.
- **Run all tests**: Make sure to run both fast and slow tests to update all fingerprints:
  ```bash
  UPDATE_FINGERPRINTS=1 pytest tests/integration -v -m slow
  ```
- **CI/CD**: Never set `UPDATE_FINGERPRINTS=1` in CI - fingerprint mismatches in CI should always fail the build.

